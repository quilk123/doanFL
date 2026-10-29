# -*- coding: utf-8 -*-
"""Xuất số liệu cho MATLAB (matlab/plot_results.m): CSV phẳng, tên cột ASCII, mỗi dòng = 1 vòng của 1 cấu hình.

  python tools/export_matlab.py          (máy không có python: sh tools/export_matlab.sh)
Đọc: experiments/step8/*/round_*.json (+ config.yaml từng thư mục), experiments/demo/*/*/ nếu có,
     experiments/ref_compare/compare.csv, experiments/attack/attack_round1.csv nếu có.
Ghi: experiments/matlab/{rounds.csv, ref_compare.csv, attack.csv}
"""
import os
import csv
import glob
import json
import shutil

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "matlab")
COLS = ["source", "config", "mode", "split", "policy", "time_scale", "t", "p", "n_ct", "n_he",
        "R1_Mbps", "R2_Mbps", "thr1_Mbps", "thr2_Mbps", "upload_MB_per_dev", "tx1_s", "tx2_s",
        "T_pred_s", "T_pi4_s", "cum_T_pi4_s", "t_max_s", "t_select_s", "late", "late_pred",
        "acc", "loss", "meas_ok", "train_slowdown_pct"]


def mbps(bps):
    return bps * 8 / 1e6 if bps else float("nan")


def upload(r, k):
    return next((it for it in r["uplink"][f"dev{k}"] if it["type"] == "UPLOAD"), {})


def rows_of(d, source):
    recs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(d, "round_*.json")))]
    if not recs:
        return []
    cfg = yaml.safe_load(open(os.path.join(d, "config.yaml"), encoding="utf-8")) if os.path.exists(
        os.path.join(d, "config.yaml")) else {}
    split = cfg.get("devices", {}).get("split", "?")
    name = os.path.relpath(os.path.normpath(d), os.path.join(ROOT, "experiments")).replace("\\", "/")
    out, cum = [], 0.0
    for r in recs:
        u1, u2 = upload(r, 1), upload(r, 2)
        cum += r["time"]["round_pi4_s"]
        sl = r.get("sim_artifact", {}).get("slowdown_pct") or [float("nan")]
        out.append(dict(
            source=source, config=name, mode=r.get("channel_mode", "?"), split=split, policy=r["policy"],
            time_scale=r.get("time_scale", 1) or 1, t=r["t"], p=r["decision"]["p"],
            n_ct=r["decision"]["n_ciphertext"], n_he=r["decision"]["n_he"],
            R1_Mbps=mbps(r["channel"]["R1"]), R2_Mbps=mbps(r["channel"]["R2"]),
            thr1_Mbps=mbps(u1.get("throughput_Bps")), thr2_Mbps=mbps(u2.get("throughput_Bps")),
            upload_MB_per_dev=r["bytes"]["UPLOAD"] / 2 / 1e6,
            tx1_s=u1.get("tx_s", float("nan")), tx2_s=u2.get("tx_s", float("nan")),
            T_pred_s=r["decision"]["T_pred_s"], T_pi4_s=r["time"]["round_pi4_s"], cum_T_pi4_s=cum,
            t_max_s=r["t_max"], t_select_s=r.get("t_select", r["t_max"]),
            late=int(r["late"]), late_pred=int(r["decision"]["late_pred"]),
            acc=r["metrics"]["accuracy"], loss=r["metrics"]["loss"],
            meas_ok=int(r.get("measurement_ok", True)), train_slowdown_pct=max(sl)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "experiments", "step8", "*", ""))):
        rows += rows_of(d, "step8")
    for d in sorted(glob.glob(os.path.join(ROOT, "experiments", "demo", "*", "*", ""))):
        rows += rows_of(d, "demo")
    write_csv("rounds.csv", COLS, rows)
    print(f"[matlab] rounds.csv: {len(rows)} dòng, {len({r['config'] for r in rows})} cấu hình")
    p = os.path.join(ROOT, "experiments", "ref_compare", "compare.csv")
    if os.path.exists(p):
        shutil.copy(p, os.path.join(OUT, "ref_compare.csv"))   # cột đã ASCII: config,t,p,acc_testbed,acc_ref,diff_pt
        print("[matlab] ref_compare.csv <- ref_compare/compare.csv")
    else:
        print("[matlab] chưa có ref_compare/compare.csv (chạy sau khi có)")
    p = os.path.join(ROOT, "experiments", "attack", "attack_round1.csv")
    if os.path.exists(p):
        a = list(csv.DictReader(open(p, encoding="utf-8")))
        write_csv("attack.csv", ATTACK_COLS, a)               # chỉ cột ASCII cần vẽ
        print(f"[matlab] attack.csv: {len(a)} dòng <- attack/attack_round1.csv")
    else:
        print("[matlab] chưa có attack/attack_round1.csv (chạy sau khi có)")


ATTACK_COLS = ["config", "policy", "split", "device", "p", "recovery_corr", "auc_bs", "auc_null_halfwidth95",
               "deshuffle_exact", "deshuffle_corr", "pair_corr_same_pi"]


def write_csv(name, cols, rows):
    """Ô số NaN/rỗng -> ô trống (textscan EmptyValue = NaN trong MATLAB/Octave)."""
    def cell(v):
        if isinstance(v, float) and v != v:
            return ""
        return {"True": 1, "False": 0}.get(v, v) if isinstance(v, str) else v
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([cell(r.get(c, "")) for c in cols])


if __name__ == "__main__":
    main()
