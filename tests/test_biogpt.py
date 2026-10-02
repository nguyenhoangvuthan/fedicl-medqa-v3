"""Offline BioGPT integration: real tokenizer/model classes with tiny random weights."""
import json
import string

import pytest
import torch
from omegaconf import OmegaConf
from transformers import BioGptConfig, BioGptForCausalLM, BioGptTokenizer

from fedicl.config import ARMS, CONFIG_DIR, arm_dir, config_hash, load_config
from fedicl.data.io import Example
from fedicl.evaluate import Evaluator, EvalView, collate
from fedicl.losses import answer_logits
from fedicl.modeling import (
    get_adapter_state,
    load_adapter,
    load_lora_model,
    load_tokenizer,
    save_adapter,
    trainable_params,
)
from fedicl.prompt import PromptBuilder
from fedicl.train_common import StepRunner, TrainItem, make_optimizer


def biogpt_config(arm="centralized_icl", overrides=None):
    return load_config(arm, extra=[str(CONFIG_DIR / "biogpt.yaml")], overrides=overrides)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("biogpt-base")
    chars = string.ascii_letters + string.digits + string.punctuation
    tokens = ["<s>", "<pad>", "</s>", "<unk>"] + list(chars) + [c + "</w>" for c in chars]
    (path / "vocab.json").write_text(json.dumps(dict(zip(tokens, range(len(tokens))))))
    (path / "merges.txt").write_text("#version: 0.2\n")
    tok = BioGptTokenizer(str(path / "vocab.json"), str(path / "merges.txt"), model_max_length=1024)
    tok.save_pretrained(path)
    model = BioGptForCausalLM(BioGptConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=1024,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0))
    model.save_pretrained(path)
    return path


def example(i, words=1):
    return Example(f"q{i}", "What helps? " * words,
                   ("aspirin", "water", "insulin", "rest"), "C", "step1", f"h{i}")


def test_biogpt_keeps_data_protocol_and_separates_outputs():
    for arm in ARMS:
        bio, qwen = biogpt_config(arm), load_config(arm)
        assert bio.data == qwen.data and bio.retrieval == qwen.retrieval
        assert bio.federated == qwen.federated and bio.train == qwen.train
        assert bio.icl == qwen.icl and bio.loss == qwen.loss and bio.eval == qwen.eval
        assert arm_dir(bio) != arm_dir(qwen)
        assert config_hash(bio) != config_hash(qwen)
    smoke = load_config(extra=[str(CONFIG_DIR / "smoke.yaml"), str(CONFIG_DIR / "biogpt.yaml")])
    assert smoke.model.max_seq_len == 1024
    assert smoke.save.root == "outputs_smoke/biogpt/seed42"


@pytest.fixture
def cfg(tiny_checkpoint):
    return biogpt_config(overrides=[f"model.name={tiny_checkpoint}", "model.dtype=float32"])


def test_plain_prompt_eos_masking_and_demo_fitting(cfg):
    tok = load_tokenizer(cfg)
    b = PromptBuilder(tok, cfg)
    q = example(1)
    enc = b.encode(q, [example(2)], with_target=True)
    assert enc.input_ids[0] == tok.sep_token_id
    assert enc.input_ids[-1] == tok.eos_token_id
    assert tok.decode(enc.input_ids[enc.n_prompt:-1]) == q.gold_text
    prompt = tok.decode(enc.input_ids[:enc.n_prompt])
    assert "Answer: insulin" in prompt and "Target Question:" in prompt
    assert "<|im_start|>" not in prompt and "<think>" not in prompt
    batch = collate([enc], tok.pad_token_id, "cpu")
    assert (batch["labels"][0, :enc.n_prompt] == -100).all()
    assert batch["labels"][0, -1] == tok.eos_token_id
    demos = [example(2), example(3, 100), example(4, 100)]
    fitted = b.encode(q, demos, with_target=False, reserve=64)
    assert fitted.n_demos == 1
    assert len(fitted.input_ids) + 64 <= 1024
    with pytest.raises(ValueError, match="no prompt space"):
        b.encode(q, [], with_target=False, reserve=1024)


@pytest.mark.parametrize("arm", ARMS)
def test_biogpt_train_generate_and_reload(tiny_checkpoint, tmp_path, arm):
    cfg = biogpt_config(arm, [f"model.name={tiny_checkpoint}", "model.dtype=float32",
                              "eval.generation.max_new_tokens=3"])
    tok = load_tokenizer(cfg)
    b = PromptBuilder(tok, cfg)
    model = load_lora_model(cfg, "cpu")
    assert model.is_gradient_checkpointing
    for name in cfg.lora.target_modules:
        assert any(f".{name}.lora_A" in n for n, _ in model.named_parameters())
    q, demo = example(1), example(2, words=2)
    main = b.encode(q, [demo] if cfg.icl.enabled else [], with_target=True)
    alt = b.encode(q, [], with_target=True) if cfg.icl.enabled else None
    batch = collate([main], tok.pad_token_id, "cpu")
    model.eval()
    with torch.no_grad():
        logits, targets = answer_logits(model, batch)
        full = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    mask = batch["labels"][:, 1:] != -100
    torch.testing.assert_close(logits, full[:, :-1][mask].float())
    assert targets.tolist() == main.input_ids[main.n_prompt:]
    before = get_adapter_state(model)
    params = trainable_params(model)
    runner = StepRunner(model, cfg, "cpu", tok.pad_token_id, params, use_reg=cfg.setting == "federated")
    runner.set_global()
    logs = runner.step([[TrainItem(main, alt)]], make_optimizer(cfg, params), lr=1e-2)
    assert all(torch.isfinite(torch.tensor(v)) for v in logs.values())
    assert any(not torch.equal(before[k], v) for k, v in get_adapter_state(model).items())
    view = EvalView("test", "none", [q.id, demo.id], None)
    evaluator = Evaluator(model, tok, b, None, {q.id: q, demo.id: demo}, cfg, "cpu")
    assert len(evaluator.generate(view)) == 2
    out = tmp_path / arm
    out.mkdir()
    OmegaConf.save(cfg, out / "config.yaml")
    save_adapter(model, out / "best" / "adapter")
    for path in (out, out / "best" / "adapter"):
        restored, restored_tok = load_adapter(path)
        assert restored.config.model_type == "biogpt"
        assert restored_tok.eos_token_id == tok.eos_token_id
        for k, v in get_adapter_state(model).items():
            torch.testing.assert_close(get_adapter_state(restored)[k], v)
    # A copied BioGPT adapter must not silently fall back to the default Qwen backbone.
    (out / "config.yaml").unlink()
    with pytest.raises(ValueError, match="pass its training cfg"):
        load_adapter(out)
    restored, _ = load_adapter(out, cfg=cfg)
    assert restored.config.model_type == "biogpt"


def test_invalid_context_and_lora_targets_fail(cfg):
    cfg.model.max_seq_len = 2048
    with pytest.raises(ValueError, match="exceeds tokenizer limit"):
        PromptBuilder(load_tokenizer(cfg), cfg)
    with pytest.raises(ValueError, match="exceeds model limit"):
        load_lora_model(cfg, "cpu")
    cfg.model.max_seq_len = 1024
    cfg.lora.target_modules = ["q_proj", "o_proj"]
    with pytest.raises(ValueError, match="o_proj"):
        load_lora_model(cfg, "cpu")
