import multiprocessing as mp

import pandas as pd
import pytest

from fedicl.config import config_hash, load_config, requested_gpu, select_gpu
from fedicl.utils import file_lock, write_csv_df


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("FEDICL_GPU", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_gpu_from_config_and_env_override(clean_env):
    cfg = load_config("federated_icl", overrides=["runtime.gpu=1"])
    assert requested_gpu(cfg) == "1"
    clean_env.setenv("FEDICL_GPU", "0")
    assert requested_gpu(cfg) == "0"  # env wins over config


def test_select_gpu_sets_visibility_in_nvidia_smi_order(clean_env):
    select_gpu(load_config(overrides=["runtime.gpu=1"]))
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


@pytest.mark.parametrize("value", [None, "all", "null", ""])
def test_gpu_all_leaves_visibility_untouched(clean_env, value):
    ov = ["runtime.gpu=null"] if value is None else [f"runtime.gpu='{value}'"]
    cfg = load_config(overrides=ov)
    assert requested_gpu(cfg) is None
    select_gpu(cfg)
    import os

    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_invalid_gpu_rejected(clean_env):
    clean_env.setenv("FEDICL_GPU", "cuda:1")
    with pytest.raises(SystemExit):
        requested_gpu(load_config())


def test_gpu_choice_does_not_change_config_hash():
    a = config_hash(load_config("centralized_icl", overrides=["runtime.gpu=0"]))
    b = config_hash(load_config("centralized_icl", overrides=["runtime.gpu=1"]))
    c = config_hash(load_config("centralized_icl", overrides=["train.lr=1e-4"]))
    assert a == b != c  # resuming on the other GPU is allowed; a real change is still detected


def _bump(path, n):
    for _ in range(n):
        with file_lock(path):
            df = pd.read_csv(path)
            df.loc[0, "count"] += 1
            write_csv_df(path, df)


def test_file_lock_serializes_read_modify_write_across_processes(tmp_path):
    path = tmp_path / "runs_index.csv"
    write_csv_df(path, pd.DataFrame({"count": [0]}))
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_bump, args=(path, 25)) for _ in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert pd.read_csv(path).loc[0, "count"] == 75  # no lost update
    assert not (tmp_path / "runs_index.csv.lock").exists()
