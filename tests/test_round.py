# -*- coding: utf-8 -*-
"""Test bước 1 (CLAUDE.md mục 10.1): chạy đủ 7 bước của một vòng.

Chạy bs + uav + iot1 + iot2 TRONG MỘT TIẾN TRÌNH (thread) trên 127.0.0.1,
KHÔNG cần Docker. Đạt khi: 3 vòng không lỗi, đủ log, byte > 0.
Docker chỉ cần khi chạy "như thật" (docker-compose), test logic thì không.
Từ bước 2 iot huấn luyện thật nên cần data/ và weights/ (tools/prep_data.py).
`run_rounds` dùng lại cho test các bước sau.
"""
import os
import sys
import time
import json
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # weights/, data/ là đường dẫn tương đối

import yaml
from bs.server import run_bs
from uav.relay import run_uav
from iot.client import run_iot

BS_PORT = 6000
IOT_PORTS = [7001, 7002]
ROUNDS = 3
# test ghi vào thư mục RIÊNG (không đụng results/, capture/ mà docker-compose / thí nghiệm bước 8 đang dùng)
RESULTS_DIR = os.path.join(ROOT, "tests", "_out", "results")
CAPTURE_DIR = os.path.join(ROOT, "tests", "_out", "capture")


def _runner(fn, name, errors):
    try:
        fn()
    except Exception as e:   # noqa: BLE001
        errors.append((name, repr(e)))


def load_config():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_rounds(config, rounds, timeout=120):
    """Chạy `rounds` vòng với 4 thành phần bằng thread; trả về danh sách bản ghi log từng vòng."""
    config = dict(config, rounds=rounds)
    results_dir = RESULTS_DIR
    capture_dir = CAPTURE_DIR
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(capture_dir, exist_ok=True)
    # dọn log cũ
    for d in (results_dir, capture_dir):
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith((".json", ".jsonl", ".npz")):
                    os.remove(os.path.join(d, fn))

    with open(os.path.join(ROOT, "secrets", "perm_seed"), "rb") as f:
        perm_seed = f.read()

    errors = []
    threads = []

    # bs chỉ có K với fas10/fas50 (như docker-compose.fas.yml)
    bs_K = perm_seed if str(config.get("policy", "")).startswith("fas") else None
    t_bs = threading.Thread(target=_runner, name="bs", args=(
        lambda: run_bs("127.0.0.1", BS_PORT, rounds, config, results_dir, capture_dir, perm_seed=bs_K),
        "bs", errors))
    t_bs.start(); threads.append(t_bs)
    time.sleep(0.6)  # chờ bs lắng nghe

    t_uav = threading.Thread(target=_runner, name="uav", args=(
        lambda: run_uav("127.0.0.1", BS_PORT, "127.0.0.1", IOT_PORTS, rounds, config),
        "uav", errors))
    t_uav.start(); threads.append(t_uav)
    time.sleep(0.4)  # chờ uav mở cổng

    t_i1 = threading.Thread(target=_runner, name="iot1", args=(
        lambda: run_iot(1, "127.0.0.1", IOT_PORTS[0], config, perm_seed, True),
        "iot1", errors))
    t_i2 = threading.Thread(target=_runner, name="iot2", args=(
        lambda: run_iot(2, "127.0.0.1", IOT_PORTS[1], config, perm_seed, False),
        "iot2", errors))
    t_i1.start(); t_i2.start(); threads += [t_i1, t_i2]

    for t in threads:
        t.join(timeout=timeout)

    alive = [t.name for t in threads if t.is_alive()]
    assert not alive, f"thread treo: {alive}"
    assert not errors, f"có lỗi trong thread: {errors}"

    records = []
    for t in range(1, rounds + 1):
        path = os.path.join(results_dir, f"round_{t:04d}.json")
        assert os.path.isfile(path), f"thiếu log {path}"
        records.append(json.load(open(path, encoding="utf-8")))
        assert os.path.isfile(os.path.join(capture_dir, f"upload_{t:04d}.npz"))
    jsonl = os.path.join(results_dir, "log.jsonl")
    n = sum(1 for _ in open(jsonl, encoding="utf-8"))
    assert n == rounds, f"log.jsonl có {n} dòng, mong {rounds}"
    return records


def main():
    records = run_rounds(load_config(), ROUNDS, timeout=600)
    for rec in records:
        t = rec["t"]
        assert rec["bytes"]["UPLOAD"] > 0, f"UPLOAD byte = 0 ở vòng {t}"
        assert rec["decision"]["n_ciphertext"] > 0, f"n_ciphertext = 0 ở vòng {t}"
        assert rec["metrics"]["accuracy"] is not None, f"thiếu METRICS ở vòng {t}"

    print(f"\n[TEST] ĐẠT: {ROUNDS} vòng chạy đủ 7 bước, log đầy đủ, byte > 0.")


if __name__ == "__main__":
    main()
