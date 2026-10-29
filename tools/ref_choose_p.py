# -*- coding: utf-8 -*-
# [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
# Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
#   D(p) <= R_min * (T_max - T_comp)
# với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
# Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
# Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
"""Sinh lịch p tham chiếu bằng CHÍNH reference_sim/integrate.choose_p, dùng bảng chi phí ĐO TRÊN MÁY NÀY (costs/)
và CÙNG ngưỡng chọn với testbed: T_max − δ (bs.policy.t_select) -> tests/ref/choose_p_ref.json (test bước 7).

  python tools/ref_choose_p.py --ref-sim <đường dẫn>/reference_sim
Chép integrate.py, channel.py, plotstyle.py sang thư mục tạm có results/ = costs/ của testbed (không sửa reference_sim).
Phải chạy lại sau mỗi lần đo lại bảng chi phí (test kiểm tra mã băm).
"""
import os
import sys
import json
import shutil
import hashlib
import argparse
import tempfile
import importlib

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bs import policy                     # noqa: E402
from uav import channel as tb_channel     # noqa: E402

MODES = ("uav_noma_pc", "direct_noma")


def costs_hash():
    h = hashlib.sha256()
    for fn in ("cost_table.csv", "cost_fixed.json"):
        h.update(open(os.path.join(ROOT, "costs", fn), "rb").read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-sim", required=True)
    ap.add_argument("--n", type=int, default=2000, help="số mẫu kênh mỗi chế độ")
    ap.add_argument("--rounds", type=int, default=20, help="thêm γ của các vòng testbed 1..rounds")
    args = ap.parse_args()
    config = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    cm = policy.cost_model(os.path.join(ROOT, "costs"))
    t_max = policy.resolve_t_max(config, cm)
    t_sel = policy.t_select(config, cm)        # T_max − safety_margin_s

    tmp = tempfile.mkdtemp()
    for fn in ("integrate.py", "channel.py", "plotstyle.py"):
        shutil.copy(os.path.join(args.ref_sim, fn), tmp)
    shutil.copytree(os.path.join(ROOT, "costs"), os.path.join(tmp, "results"))
    sys.path.insert(0, tmp)
    I = importlib.import_module("integrate")
    rch = importlib.import_module("channel")

    out = {"t_max": t_max, "t_select": t_sel, "safety_margin_s": float(config.get("safety_margin_s", 0.0)),
           "costs_sha256": costs_hash(), "T_train_ref": float(I.T_TRAIN),
           "grid_ref": I.GRID.tolist(), "cases": {}}
    for mode in MODES:
        g_rand = rch.draw(I.P, np.random.default_rng(11), args.n)[mode]
        cfg = dict(config, channel=dict(config["channel"], mode=mode))
        g_rounds = np.array([tb_channel.channel_round(cfg, t)[:2] for t in range(1, args.rounds + 1)])
        gam = np.vstack([g_rand, g_rounds])
        p, late = I.choose_p(gam, t_sel, mode)       # late ở đây = không có p nào ≤ t_select (giữ p_min)
        out["cases"][mode] = {"gamma": gam.tolist(), "p": p.tolist(), "late": late.tolist(),
                              "n_random": args.n, "rounds": args.rounds}
        vals, cnt = np.unique(p, return_counts=True)
        print(f"[ref] {mode}: t_max={t_max:.2f}s, chọn với {t_sel:.2f}s, p trung bình {p.mean():.3f}, trễ hạn {late.mean() * 100:.1f}%, "
              f"phân bố p {dict(zip(np.round(vals, 1).tolist(), cnt.tolist()))}")
    os.makedirs(os.path.join(ROOT, "tests", "ref"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "tests", "ref", "choose_p_ref.json"), "w"))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
