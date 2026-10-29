# -*- coding: utf-8 -*-
"""Test bước 2 (CLAUDE.md mục 10.2): FL thật chưa bảo vệ (`no_he`), IID, 15 vòng.

Đạt khi: độ chính xác vòng cuối ≈ 80% và lệch ≤ 1,5 điểm so với reference_sim chạy cùng cấu hình
(fl.py --schedule 15×p=0 --lam 0 --he 0 trên cùng đặc trưng, cùng phân hoạch) — kết quả lưu ở tests/ref/.
Cần data/ và weights/ (tools/prep_data.py). Chạy: python tests/test_step2_fl.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_round import ROOT, load_config, run_rounds

ROUNDS = 15
TOL = 0.015
REF = os.path.join(ROOT, "tests", "ref", "fl_no_he_iid_seed0.json")


def main():
    assert os.path.isfile(REF), f"thiếu kết quả reference_sim {REF}"
    config = load_config()
    config["policy"] = "no_he"
    config["devices"] = dict(config["devices"], split="iid")
    meta = json.load(open(os.path.join(ROOT, "data", "train", "client1", "meta.json"), encoding="utf-8"))
    assert meta["split"] == "iid" and meta["seed"] == config["seed"], \
        f"data/ đang chia {meta}, chạy lại tools/prep_data.py --split-only với split: iid"

    records = run_rounds(config, ROUNDS, timeout=3600)
    accs = [r["metrics"]["accuracy"] for r in records]
    for r in records:
        assert r["decision"]["p"] == 0.0 and r["decision"]["n_ciphertext"] == 0, "no_he phải có p = 0"
    print("[TEST] độ chính xác theo vòng:", " ".join(f"{a:.4f}" for a in accs))

    ref = json.load(open(REF, encoding="utf-8"))
    ref_accs = [r["acc"] for r in ref["rounds"][1:ROUNDS + 1]]
    print("[TEST] reference_sim         :", " ".join(f"{a:.4f}" for a in ref_accs))
    diff = abs(accs[-1] - ref_accs[-1])
    print(f"[TEST] vòng {ROUNDS}: testbed {accs[-1]:.4f}, reference_sim {ref_accs[-1]:.4f}, lệch {diff * 100:.2f} điểm")
    assert 0.75 <= accs[-1] <= 0.85, f"độ chính xác {accs[-1]:.4f} không ≈ 80%"
    assert diff <= TOL, f"lệch {diff * 100:.2f} điểm > {TOL * 100} điểm"
    print(f"\n[TEST] ĐẠT: no_he IID {ROUNDS} vòng, độ chính xác {accs[-1]:.4f}, lệch reference_sim {diff * 100:.2f} điểm.")


if __name__ == "__main__":
    main()
