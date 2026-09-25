"""Federated arms: FedAvg over LoRA, every client every round, local loss L_QA (+L_ICL) + L_REG."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from .config import partition_dir
from .data.io import load_split
from .evaluate import Evaluator
from .matching import Matcher
from .modeling import (fedavg, get_adapter_state, load_adapter_state, load_lora_model,
                       load_tokenizer, save_adapter, set_adapter_state, trainable_params)
from .prompt import PromptBuilder
from .results import append_results, result_rows, save_eval_outputs
from .train_centralized import finalize
from .train_common import (EarlyStopper, StepRunner, append_log, build_train_items, eval_views,
                           load_assign, load_examples, lr_at, make_optimizer, micro_batches,
                           monitored_loss, run_evaluation, steps_per_pass)
from .utils import read_json, rng_for, set_seed, write_json

LOG = logging.getLogger("fedicl.federated")


def _latest_round(rounds_root: Path) -> int:
    done = [int(p.name.split("_")[1]) for p in rounds_root.glob("round_*")
            if (p / "metrics.json").exists() and (p / "server_state" / "state.json").exists()]
    return max(done, default=0)


def run(cfg, out: Path, extend_to: int | None = None) -> dict:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(cfg.seed), bool(cfg.runtime.deterministic))
    tok = load_tokenizer(cfg)
    model = load_lora_model(cfg, dev)
    builder = PromptBuilder(tok, cfg)
    ex, ids = load_examples(cfg)
    pdir = partition_dir(cfg)
    if not pdir.exists():
        raise SystemExit(f"{pdir} missing: run `python -m fedicl.data.partition` first")

    K = int(cfg.federated.num_clients)
    clients = []
    for i in range(1, K + 1):
        qids = load_split(pdir / f"client_{i}.csv")["id"].tolist()
        assign = load_assign(pdir / f"demo_assignment_client_{i}_train.json") if cfg.icl.enabled else None
        clients.append(build_train_items(cfg, builder, ex, qids, assign))
        LOG.info("client_%d: %d train items", i, len(qids))
    LOG.info("demos dropped to fit max_seq_len: %d", builder.truncated)

    evaluator = Evaluator(model, tok, builder, Matcher.from_config(cfg, dev), ex, cfg, dev)
    views = eval_views(cfg, ids)
    params = trainable_params(model)
    runner = StepRunner(model, cfg, dev, tok.pad_token_id, params, use_reg=True)
    es = cfg.federated.early_stopping
    stopper = EarlyStopper(int(es.patience), float(es.min_delta))
    max_rounds = int(extend_to or cfg.federated.rounds)
    local_epochs = int(cfg.federated.local_epochs)
    rounds_root = out / "rounds"

    global_state = get_adapter_state(model)  # round 0: fresh LoRA (B = 0 => identical to base)
    start = 1
    last = _latest_round(rounds_root)
    if last:
        rd = rounds_root / f"round_{last}"
        global_state = load_adapter_state(rd / "global_adapter")
        st = read_json(rd / "server_state" / "state.json")
        stopper = EarlyStopper.from_state(st["early_stop"])
        if extend_to and stopper.stopped and stopper.reason != "overfit":
            stopper.stopped = False
        start = last + 1
        LOG.info("resumed after round %d", last)

    for r in range(start, max_rounds + 1):
        if stopper.stopped:
            break
        rd = rounds_root / f"round_{r}"
        states, weights, agg = [], [], []
        for i, items in enumerate(clients, start=1):
            t0 = time.time()
            set_adapter_state(model, global_state)
            runner.set_global()
            opt = make_optimizer(cfg, params)  # fresh local optimizer every round (standard FedAvg)
            S = steps_per_pass(len(items), cfg) * local_epochs
            step, logs = 0, []
            for le in range(local_epochs):
                order = rng_for(int(cfg.seed), "round", r, "client", i, "epoch", le).permutation(len(items))
                for micro in micro_batches(items, order, int(cfg.train.batch_size), int(cfg.train.grad_accum)):
                    progress = ((r - 1) + step / S) / max_rounds
                    log = runner.step(micro, opt, lr_at(cfg, progress))
                    step += 1
                    logs.append({"round": r, "client_id": i, "step": step, **log})
                    if step % 20 == 0:
                        LOG.info("round %d client %d step %d/%d loss=%.4f (qa %.4f icl %.4f reg %.5f)",
                                 r, i, step, S, log["loss_total"], log["loss_qa"], log["loss_icl"],
                                 log["loss_reg"])
            state = get_adapter_state(model)
            states.append(state)
            weights.append(len(items))
            cdir = rd / "clients" / f"client_{i}"
            if cfg.save.save_client_adapters:
                save_adapter(model, cdir / "adapter")
            append_log(cdir / "train_log.csv", logs)
            append_log(out / "logs" / "train_log.csv", logs)
            cm = {k: float(np.mean([l[k] for l in logs])) for k in ("loss_qa", "loss_icl", "loss_reg", "loss_total")}
            write_json(cdir / "metrics.json", {"n_samples": len(items), "steps": step,
                                               "train_seconds": round(time.time() - t0, 1), **cm})
            agg.append({"client_id": i, "n_samples": len(items), "train_loss": cm["loss_total"]})
            del opt
            torch.cuda.empty_cache() if dev == "cuda" else None

        global_state = fedavg(states, weights)
        set_adapter_state(model, global_state)
        save_adapter(model, rd / "global_adapter", state=global_state)
        total = sum(weights)
        write_json(rd / "aggregation.json", {"method": cfg.federated.aggregation, "clients": [
            {**a, "weight": a["n_samples"] / total} for a in agg]})

        metrics, preds = run_evaluation(evaluator, views)
        vl = monitored_loss(metrics)
        stopper.update(vl, f"round_{r}", str(rd))
        train_loss = sum(a["train_loss"] * a["n_samples"] for a in agg) / total
        append_log(out / "logs" / "val_curve.csv", [{
            "at": f"round_{r}", "round": r, "demo_pool": pool, "val_loss": m["val_loss"],
            "val_accuracy": m.get("accuracy"), "train_loss": train_loss}
            for pool, m in metrics["validation"].items()])
        save_eval_outputs(rd, metrics, preds)
        write_json(rd / "server_state" / "state.json", {"round": r, "early_stop": stopper.state()})
        append_results(out / "result.csv", result_rows(
            cfg, f"round_{r}", r, metrics, str((rd / "global_adapter").relative_to(out))))
        LOG.info("round %d: monitored val_loss=%.4f best=%.4f (%s)", r, vl, stopper.best, stopper.best_at)

    return finalize(cfg, out, model, evaluator, views, stopper)
