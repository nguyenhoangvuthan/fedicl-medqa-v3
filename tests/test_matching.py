import numpy as np

from fedicl.matching import Matcher, normalize, token_f1


class StubEncoder:
    """Deterministic fake SapBERT: synonyms share a vector, everything else is random-orthogonal."""
    SYN = {"low thyroid hormone": "hypothyroidism", "low thyroid hormone levels": "hypothyroidism"}

    def encode_cached(self, texts):
        vecs = []
        for t in texts:
            key = self.SYN.get(t, t)
            v = np.random.default_rng(abs(hash(key)) % (2**32)).normal(size=64)
            vecs.append(v / np.linalg.norm(v))
        return np.stack(vecs)


OPTS = ["Aspirin", "Aspirin and clopidogrel", "Heparin", "Warfarin"]


def lexical():
    return Matcher(w_lex=1.0, w_sem=0.0)


def test_normalize_strips_prefixes_and_punctuation():
    assert normalize("  The answer is: Heparin. ") == "heparin"
    assert normalize("- Answer: Warfarin") == "warfarin"


def test_exact():
    m = lexical().match("Heparin", OPTS)
    assert (m.index, m.match_type) == (2, "exact")


def test_substring_option_prefers_longest():
    m = lexical().match("aspirin and clopidogrel", OPTS)
    assert (m.index, m.match_type) == (1, "exact")
    m = lexical().match("I would give aspirin and clopidogrel now", OPTS)
    assert (m.index, m.match_type) == (1, "contains")


def test_prefix_the_answer_is():
    m = lexical().match("The answer is Warfarin", OPTS)
    assert (m.index, m.match_type) == (3, "exact")


def test_contains_needs_word_boundary():
    # "heparin" inside "heparinoid" must not count as containing the option
    m = lexical().match("heparinoid", OPTS)
    assert m.match_type == "nearest"


def test_synonym_without_shared_words_uses_semantic():
    opts = ["Hyperthyroidism", "Hypothyroidism", "Cushing syndrome", "Addison disease"]
    m = Matcher(0.5, 0.5, StubEncoder()).match("Low thyroid hormone", opts)
    assert (m.index, m.match_type) == (1, "nearest")


def test_hypo_vs_hyper_decided_by_lexical():
    opts = ["Hyperthyroidism", "Hypothyroidism", "Graves disease", "Thyroid storm"]
    m = Matcher(0.5, 0.5, StubEncoder()).match("primary hypothyroidism", opts)
    assert m.index == 1


def test_empty_output():
    m = lexical().match("   ...  ", OPTS)
    assert (m.index, m.match_type) == (None, "empty")


def test_garbage_still_picks_argmax_but_flags_low_confidence():
    m = lexical().match("qwerty zxcv", OPTS)
    assert m.match_type == "nearest" and m.index is not None and m.low_confidence


def test_two_non_nested_options_are_ambiguous_not_positional():
    m = lexical().match("aspirin or heparin", OPTS)
    assert m.match_type == "nearest" and m.low_confidence


def test_small_margin_flags_low_confidence():
    opts = ["red blood cell", "red blood vessel", "white matter", "grey matter"]
    m = Matcher(1.0, 0.0, min_score=0.0, min_margin=0.2).match("red blood", opts)
    assert m.match_type == "nearest" and m.margin < 0.2 and m.low_confidence


def test_token_f1():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a", "b") == 0.0
