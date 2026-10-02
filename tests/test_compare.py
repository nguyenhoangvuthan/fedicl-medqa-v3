import numpy as np
import pytest

from fedicl.compare import Run, compare, mcnemar_exact, paired_bootstrap, run_pair


def test_mcnemar_exact_matches_hand_computation():
    # 10 questions only A gets right, 2 only B gets right: p = 2 * (C(12,0)+C(12,1)+C(12,2)) / 2^12
    a = np.array([1] * 10 + [0] * 2 + [1] * 30 + [0] * 30)
    b = np.array([0] * 10 + [1] * 2 + [1] * 30 + [0] * 30)
    n10, n01, p = mcnemar_exact(a, b)
    assert (n10, n01) == (10, 2)
    assert p == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_mcnemar_identical_is_one():
    a = np.array([1, 0, 1, 1])
    assert mcnemar_exact(a, a)[2] == 1.0


def test_bootstrap_ci_contains_true_gain_and_detects_no_difference():
    rng = np.random.default_rng(0)
    a = (rng.random(2000) < 0.45).astype(float)
    b = (rng.random(2000) < 0.35).astype(float)
    lo, hi, p = paired_bootstrap(a, b)
    assert lo < 100 * 0 + (a.mean() - b.mean()) < hi and p < 0.01
    lo, hi, p = paired_bootstrap(a, a.copy())
    assert lo == hi == 0.0 and p == 1.0


def test_macro_mean_and_per_client_rows():
    ids = [f"q{i}" for i in range(6)]
    fl = Run("fl", {"client_1": dict(zip(ids, [1, 1, 1, 0, 0, 0])),
                    "client_2": dict(zip(ids, [1, 1, 0, 0, 0, 0]))})
    base = Run("base", {"none": dict(zip(ids, [1, 0, 0, 0, 0, 0]))})
    rows = run_pair("ICL gain", fl, base)
    assert rows[0]["acc_A"] == pytest.approx((3 / 6 + 2 / 6) / 2)   # macro mean over client views
    assert rows[0]["p_mcnemar"] is None                             # not a single 0/1 view
    per_client = {r["A"]: r for r in rows[1:]}
    assert per_client["fl[client_1]"]["acc_A"] == pytest.approx(0.5)
    assert per_client["fl[client_1]"]["p_mcnemar"] is not None
    assert compare("x", base, base)["gain_pp"] == 0.0
