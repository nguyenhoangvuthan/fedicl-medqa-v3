import numpy as np
import pytest
import torch

from fedicl.config import load_config
from fedicl.data.io import Example
from fedicl.evaluate import collate
from fedicl.losses import answer_logits, icl_consistency_loss, proximal_loss, qa_loss
from fedicl.modeling import fedavg
from fedicl.prompt import Encoded, PromptBuilder, option_order


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    return Qwen3ForCausalLM(cfg).eval()


def test_answer_logits_match_full_forward(tiny_model):
    encs = [Encoded([5, 6, 7, 8, 9, 10], 4, 0, [0, 1, 2, 3]), Encoded([11, 12, 13, 14], 2, 0, [0, 1, 2, 3])]
    batch = collate(encs, pad_id=0, dev="cpu")
    logits, tgt = answer_logits(tiny_model, batch)
    full = tiny_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    # answer tokens: seq0 positions 4,5 (predicted from 3,4); seq1 positions 2,3 (from 1,2)
    expected = torch.stack([full[0, 3], full[0, 4], full[1, 1], full[1, 2]])
    assert torch.allclose(logits, expected, atol=1e-5)
    assert tgt.tolist() == [9, 10, 13, 14]
    assert qa_loss(logits, tgt) > 0


def test_icl_loss_zero_for_identical_context_and_positive_otherwise():
    a = torch.randn(5, 11)
    assert icl_consistency_loss(a, a).item() == pytest.approx(0.0, abs=1e-6)
    assert icl_consistency_loss(a, torch.randn(5, 11)).item() > 0


def test_icl_loss_gradient_only_flows_to_student():
    t = torch.randn(3, 7, requires_grad=True)
    s = torch.randn(3, 7, requires_grad=True)
    icl_consistency_loss(t, s).backward()
    assert t.grad is None and s.grad is not None


def test_proximal_loss():
    p = [torch.ones(2, 2), torch.zeros(3)]
    g = [torch.ones(2, 2), torch.ones(3)]
    assert proximal_loss(p, p, 0.1).item() == 0.0
    assert proximal_loss(p, g, 0.1).item() == pytest.approx(0.5 * 0.1 * 3)


def test_fedavg_weighted():
    s1, s2 = {"w": torch.zeros(2)}, {"w": torch.ones(2)}
    assert fedavg([s1, s2], [1, 3])["w"].tolist() == [0.75, 0.75]


def test_step_runner_full_objective_updates_lora_and_sets_train_mode():
    """Regression: from_pretrained returns eval mode; the runner must switch to train mode."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from fedicl.modeling import trainable_params
    from fedicl.train_common import StepRunner, TrainItem, make_optimizer

    torch.manual_seed(0)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=8))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                            init_lora_weights=False)).eval()
    cfg = load_config("federated_icl", overrides=["loss.reg.mu=0.1"])
    params = trainable_params(model)
    runner = StepRunner(model, cfg, "cpu", 0, params, use_reg=True)
    runner.set_global()
    before = [p.detach().clone() for p in params]
    main = Encoded([5, 6, 7, 8, 9, 10], 4, 3, [0, 1, 2, 3])
    alt = Encoded([11, 7, 8, 9, 10], 3, 3, [0, 1, 2, 3])   # same answer tokens [9, 10]
    logs = runner.step([[TrainItem(main, alt)]], make_optimizer(cfg, params), lr=1e-2)
    assert model.training
    assert all(np.isfinite(v) for v in logs.values())
    assert logs["loss_icl"] > 0 and logs["loss_reg"] == 0.0   # reg is 0 at the global point
    assert any(not torch.equal(a, b) for a, b in zip(before, params))
    logs2 = runner.step([[TrainItem(main, alt)]], make_optimizer(cfg, params), lr=1e-2)
    assert logs2["loss_reg"] > 0                               # drifted from theta^(t)


def test_option_order_deterministic_and_permutation():
    o = option_order(42, "q1", True)
    assert sorted(o) == [0, 1, 2, 3] and o == option_order(42, "q1", True)
    assert option_order(42, "q1", False) == [0, 1, 2, 3]


@pytest.fixture(scope="module")
def builder():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception:
        pytest.skip("Qwen3 tokenizer not available offline")
    return PromptBuilder(tok, load_config("centralized_icl", overrides=["model.max_seq_len=400"])), tok


def _ex(i, n_words=20):
    return Example(f"q{i}", " ".join(["word"] * n_words) + f" question {i}?",
                   (f"opt a{i}", f"opt b{i}", f"opt c{i}", f"opt d{i}"), "C", "step1", f"h{i}")


def test_prompt_has_no_letters_and_target_is_option_text(builder):
    b, tok = builder
    enc = b.encode(_ex(1), [_ex(2), _ex(3)], with_target=True)
    prompt = tok.decode(enc.input_ids[:enc.n_prompt])
    target = tok.decode(enc.input_ids[enc.n_prompt:])
    assert target == "opt c1<|im_end|>"
    assert "- opt a1" in prompt and "A." not in prompt and "(A)" not in prompt
    assert "Answer: opt c2" in prompt and "Target Question:" in prompt
    assert "<think>\n\n</think>" in prompt  # enable_thinking=False => empty think block, then answer
    # most relevant demo (first in retrieval order) sits right before the target question
    assert prompt.index("question 3") < prompt.index("question 2") < prompt.index("question 1")


def test_prompt_drops_least_relevant_demos_to_fit(builder):
    b, _ = builder
    long_demos = [_ex(i, n_words=60) for i in range(2, 6)]
    enc = b.encode(_ex(1), long_demos, with_target=True)
    assert len(enc.input_ids) <= 400 and 0 < enc.n_demos < 4
