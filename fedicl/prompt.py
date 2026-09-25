"""Spec §4.1 In-Context Prompt Construction, shared by training and evaluation for every arm.

Options are shown WITHOUT letters and in a per-question shuffled order, and the target is the
option text + end-of-turn, so the model never picks a letter (no letter / position bias).
"""
from __future__ import annotations

from dataclasses import dataclass

from .data.io import Example
from .utils import rng_for


def option_order(seed: int, qid: str, shuffle: bool) -> list[int]:
    """Permutation of original option indices, fixed per question across epochs/rounds."""
    return rng_for(seed, qid, "options").permutation(4).tolist() if shuffle else [0, 1, 2, 3]


def _block(ex: Example, order: list[int]) -> str:
    opts = "\n".join(f"- {ex.options[j]}" for j in order)
    return f"Question: {ex.question}\nOptions:\n{opts}"


@dataclass
class Encoded:
    input_ids: list[int]
    n_prompt: int          # labels are input_ids[n_prompt:]
    n_demos: int           # demos kept after length fitting
    order: list[int]       # displayed option order of the target question


class PromptBuilder:
    def __init__(self, tokenizer, cfg):
        self.tok = tokenizer
        self.system = cfg.prompt.system
        self.shuffle = bool(cfg.prompt.shuffle_options)
        self.most_relevant_last = bool(cfg.prompt.most_relevant_last)
        self.enable_thinking = bool(cfg.model.enable_thinking)
        self.max_len = int(cfg.model.max_seq_len)
        self.seed = int(cfg.seed)
        self.end_id = tokenizer.convert_tokens_to_ids(cfg.model.end_of_turn_token)
        if self.end_id is None or self.end_id == tokenizer.unk_token_id:
            raise ValueError(f"tokenizer has no token {cfg.model.end_of_turn_token!r}")
        self.truncated = 0  # number of examples that lost demos to fit max_seq_len

    def order(self, ex: Example) -> list[int]:
        return option_order(self.seed, ex.id, self.shuffle)

    def user_text(self, ex: Example, demos: list[Example]) -> str:
        """demos arrive most-relevant-first (retrieval order)."""
        demos = list(reversed(demos)) if self.most_relevant_last else list(demos)
        parts = [f"Example {i}\n{_block(d, self.order(d))}\nAnswer: {d.gold_text}"
                 for i, d in enumerate(demos, start=1)]
        target = _block(ex, self.order(ex))
        parts.append(f"Target Question: {target[len('Question: '):]}" if demos else target)
        parts[-1] += "\nAnswer:"
        return "\n\n".join(parts)

    def prompt_ids(self, ex: Example, demos: list[Example]) -> list[int]:
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": self.user_text(ex, demos)}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=self.enable_thinking)
        return self.tok(text, add_special_tokens=False).input_ids

    def target_ids(self, ex: Example) -> list[int]:
        return self.tok(ex.gold_text, add_special_tokens=False).input_ids + [self.end_id]

    def encode(self, ex: Example, demos: list[Example], with_target: bool,
               reserve: int = 0) -> Encoded:
        """Fit into max_seq_len by dropping the least relevant demos first.

        with_target=False (generation) reserves `reserve` tokens for the generated answer.
        """
        target = self.target_ids(ex) if with_target else []
        budget = self.max_len - (len(target) if with_target else reserve)
        kept = list(demos)
        while True:
            ids = self.prompt_ids(ex, kept)
            if len(ids) <= budget or not kept:
                break
            kept = kept[:-1]  # retrieval order => last = least relevant
        if len(kept) < len(demos):
            self.truncated += 1
        if len(ids) > budget:  # a single question longer than the budget: keep its tail
            ids = ids[-budget:]
        return Encoded(ids + target, len(ids), len(kept), self.order(ex))
