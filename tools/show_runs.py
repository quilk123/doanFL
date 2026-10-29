# -*- coding: utf-8 -*-
"""Xem số liệu chi tiết các lượt chạy (round_*.json do bs ghi): từng vòng, tóm tắt từng cấu hình, so sánh có/không UAV.

  python tools/show_runs.py [thư mục ...] [--rounds]
    thư mục: experiments/step8/<cấu hình>, experiments/demo/<thời điểm>/<policy>, ... (mặc định: mọi experiments/step8/*)
    --rounds: in bảng từng vòng của mỗi cấu hình
Máy không có python: sh tools/show.sh [...] (chạy trong image testbed-tools).
Tốc độ quy về Mbit/s. "đo" = số byte UPLOAD thật / thời gian truyền thật qua uav (lượt nén kênh: thời gian quy về kênh thật).
"""
import os
import sys
import glob
import json
import argparse
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mbps(bps):
    return bps * 8 / 1e6


def load(d):
    return [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(d, "round_*.json")))]


def upload(r, k):
    return next((it for it in r["uplink"][f"dev{k}"] if it["type"] == "UPLOAD"), None)


def row(r):
    u = [upload(r, k) for k in (1, 2)]
    R = [r["channel"]["R1"], r["channel"]["R2"]]
    return dict(
        t=r["t"], p=r["decision"]["p"], n_ct=r["decision"]["n_ciphertext"],
        R=[mbps(x) for x in R],
        thr=[mbps(x["throughput_Bps"]) if x and x.get("throughput_Bps") else float("nan") for x in u],
        dev_pct=[(x["ratio_to_R"] - 1) * 100 if x and x.get("ratio_to_R") else float("nan") for x in u],
        up_mb=r["bytes"]["UPLOAD"] / 2 / 1e6,
        tx=[x["tx_s"] if x else float("nan") for x in u],
        pred=r["decision"]["T_pred_s"], pi4=r["time"]["round_pi4_s"],
        wall=r["time"].get("round_s", 0.0), t_max=r["t_max"],
        late=r["late"], acc=r["metrics"]["accuracy"], ok=r.get("measurement_ok", True),
        mode=r.get("channel_mode", "?"), ts=r.get("time_scale", 1) or 1, policy=r["policy"])


def print_rounds(name, rows):
    print(f"\n=== {name} — từng vòng ===")
    print(f"{'vòng':>4} {'p':>4} {'n_ct':>4} | {'R1 lý thuyết':>12} {'R2':>7} | {'đo 1':>7} {'đo 2':>7} {'lệch%':>6} | "
          f"{'UPLOAD/tb':>9} {'tx1 s':>7} {'tx2 s':>7} | {'dự báo':>6} {'đo Pi4':>6} {'t_max':>5} {'trễ':>3} | {'acc':>6} {'đo ok':>5}")
    for x in rows:
        dev = max((abs(v) for v in x["dev_pct"] if v == v), default=float("nan"))
        print(f"{x['t']:>4} {x['p']:>4.1f} {x['n_ct']:>4} | {x['R'][0]:>9.3f} Mb {x['R'][1]:>7.3f} | "
              f"{x['thr'][0]:>7.3f} {x['thr'][1]:>7.3f} {dev:>6.2f} | {x['up_mb']:>6.2f} MB {x['tx'][0]:>7.1f} {x['tx'][1]:>7.1f} | "
              f"{x['pred']:>6.1f} {x['pi4']:>6.1f} {x['t_max']:>5.1f} {'CÓ' if x['late'] else '-':>3} | "
              f"{x['acc']:>6.4f} {'✓' if x['ok'] else '✗':>5}")


def summary(name, rows):
    R = [v for x in rows for v in x["R"]]
    thr = [v for x in rows for v in x["thr"] if v == v]
    tx = [v for x in rows for v in x["tx"] if v == v]
    return dict(name=name, mode=rows[0]["mode"], policy=rows[0]["policy"], ts=rows[0]["ts"], n=len(rows),
                R_med=st.median(R), R_min=min(R), R_max=max(R), thr_med=st.median(thr) if thr else float("nan"),
                tx_med=st.median(tx) if tx else float("nan"), tx_max=max(tx) if tx else float("nan"),
                p_mean=st.mean(x["p"] for x in rows), pi4_med=st.median(x["pi4"] for x in rows),
                wall_total=sum(x["wall"] for x in rows), pi4_total=sum(x["pi4"] for x in rows),
                late=sum(x["late"] for x in rows), ok=sum(x["ok"] for x in rows),
                err=max(abs(x["pi4"] - x["pred"]) for x in rows), acc=rows[-1]["acc"], rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--rounds", action="store_true")
    a = ap.parse_args()
    dirs = a.dirs or sorted(glob.glob(os.path.join(ROOT, "experiments", "step8", "*", "")))
    S = []
    for d in dirs:
        rows = [row(r) for r in load(d)]
        if not rows:
            continue
        name = os.path.relpath(os.path.normpath(d), os.path.join(ROOT, "experiments"))
        if a.rounds:
            print_rounds(name, rows)
        S.append(summary(name, rows))
    if not S:
        print("không có round_*.json trong thư mục đã chọn"); sys.exit(1)

    print("\n=== Tóm tắt từng cấu hình (tốc độ Mbit/s, thời gian s, thang Pi 4) ===")
    print(f"{'cấu hình':<40} {'vòng':>4} {'R trung vị':>10} {'R min':>7} {'R max':>6} {'đo tv':>6} | {'tx tv':>6} {'tx max':>7} | "
          f"{'p tb':>4} {'vòng tv':>7} {'trễ':>5} {'|đo−dự|max':>10} {'đo ok':>6} | {'acc cuối':>8}")
    for s in S:
        print(f"{s['name']:<40} {s['n']:>4} {s['R_med']:>10.3f} {s['R_min']:>7.3f} {s['R_max']:>6.2f} {s['thr_med']:>6.3f} | "
              f"{s['tx_med']:>6.1f} {s['tx_max']:>7.1f} | {s['p_mean']:>4.2f} {s['pi4_med']:>7.1f} {s['late']:>2}/{s['n']:<2} "
              f"{s['err']:>10.2f} {s['ok']:>3}/{s['n']:<2} | {s['acc']:>8.4f}")
        print(f"    Tổng thời gian {s['n']} vòng: thực tế {s['wall_total']:.1f}s "
              f"({s['wall_total'] / 60:.1f} phút); quy đổi Pi 4 {s['pi4_total']:.1f}s "
              f"({s['pi4_total'] / 60:.1f} phút).")

    print("\n=== So sánh theo chế độ kênh (uav_noma_pc = CÓ UAV, direct_noma = KHÔNG UAV) ===")
    print(f"{'chế độ':<12} {'số vòng':>7} {'R trung vị':>10} {'R min':>8} {'R max':>7} | {'tx UPLOAD tv':>12} {'tx max':>9} | {'trễ hạn':>8}")
    for m in sorted({s["mode"] for s in S}):
        rows = [x for s in S if s["mode"] == m for x in s["rows"]]
        R = [v for x in rows for v in x["R"]]
        tx = [v for x in rows for v in x["tx"] if v == v]
        late = sum(x["late"] for x in rows)
        print(f"{m:<12} {len(rows):>7} {st.median(R):>10.3f} {min(R):>8.4f} {max(R):>7.2f} | {st.median(tx):>12.1f} {max(tx):>9.1f} | "
              f"{late:>3}/{len(rows):<4}")
    print("\nR: tốc độ lý thuyết của kênh (Mbit/s); đo: tốc độ thật qua uav; tx: thời gian truyền UPLOAD (s, thời gian kênh thật);"
          "\nvòng tv: thời gian vòng theo thang Pi 4; trễ: vòng vượt t_max; đo ok: lệch tốc độ đo ≤ 5% (measurement_ok).")


if __name__ == "__main__":
    main()
