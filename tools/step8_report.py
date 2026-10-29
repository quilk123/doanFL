# -*- coding: utf-8 -*-
"""Bước 8: gộp kết quả testbed (experiments/step8/<mode>_<split>_<policy>/round_*.json) thành bảng và hình
(như reference_sim plot_fl.py E8–E9: độ chính xác theo vòng và theo thời gian).

  python tools/step8_report.py
Kết quả: experiments/step8/{rounds.csv, summary.csv, acc_vs_round.png, acc_vs_time.png, p_schedule.png}
Thời gian theo thang Pi 4 (round_pi4_s, mục 1f README); thô (wall-clock) để tham khảo.
Vòng có measurement_ok = False (máy giả lập quá tải làm lệch R_k > 5%) được đếm riêng (bad_rounds) và loại khỏi late_pct_ok.
"""
import os
import glob
import json

import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bs.server import check_measurement   # noqa: E402
OUT = os.path.join(ROOT, "experiments", "step8")
POLICIES = ["no_he", "fixed10", "full_he", "fas10", "fas50", "adaptive"]
LABEL = {"no_he": "Không HE (p = 0)", "fixed10": "FAS p = 10%, khóa xáo ở thiết bị (chính)",
         "full_he": "HE toàn phần (p = 100%)", "fas10": "FAS gốc p = 10% (bs giữ khóa xáo)",
         "fas50": "FAS gốc p = 50%", "adaptive": "p thích nghi (hướng phát triển)"}
COLOR = {"no_he": "#888780", "fixed10": "#D85A30", "full_he": "#534AB7", "fas10": "#BA7517", "fas50": "#8C6D1F",
         "adaptive": "#1D9E75"}


def load():
    import pandas as pd
    rows = []
    for d in sorted(glob.glob(os.path.join(OUT, "*_*_*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        mode = split = pol = None
        parts = name.split("_")
        # <mode có thể chứa '_'>_<split>_<policy có thể chứa '_'>
        suffix = ""
        if parts[-1].startswith("ts-"):        # hậu tố nén thời gian kênh (run_step8.sh TIME_SCALE)
            suffix = parts[-1]
            parts = parts[:-1]
        for i, s in enumerate(parts):
            if s in ("iid", "noniid"):
                mode, split, pol = "_".join(parts[:i]) + (f" ({suffix})" if suffix else ""), s, "_".join(parts[i + 1:])
                break
        for fn in sorted(glob.glob(os.path.join(d, "round_*.json"))):
            r = json.load(open(fn, encoding="utf-8"))
            tm = r["time"]
            if "measurement_ok" not in r:          # log cũ: tính lại từ uplink
                check_measurement(r)
            rows.append(dict(mode=mode, split=split, policy=pol, t=r["t"], p=r["decision"]["p"],
                             n_ct=r["decision"]["n_ciphertext"], acc=r["metrics"]["accuracy"], loss=r["metrics"]["loss"],
                             T_pi4=tm["round_pi4_s"], T_pred=r["decision"]["T_pred_s"], pred_err=r.get("pred_error_s"),
                             T_wall=tm["round_s"], late=r["late"], t_max=r["t_max"],
                             up_bytes=r["bytes"]["UPLOAD"] / 2, pd_bytes=r["bytes"]["PARTIAL_DEC"] / 2,
                             up_tx=max(tm["upload_tx_s"]),
                             slowdown=float(np.mean(r["sim_artifact"]["slowdown_pct"])),
                             meas_ok=bool(r["measurement_ok"]),
                             time_scale=float(r.get("time_scale", 1.0))))   # log trước khi có nén: k = 1
    return pd.DataFrame(rows)


def main():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load()
    assert len(df), "chưa có kết quả bước 8"
    df = df.sort_values(["mode", "split", "policy", "t"])
    df["T_cum_min"] = df.groupby(["mode", "split", "policy"])["T_pi4"].cumsum() / 60
    df.to_csv(os.path.join(OUT, "rounds.csv"), index=False)
    g = df.groupby(["mode", "split", "policy"])
    s = g.agg(rounds=("t", "count"), acc_final=("acc", "last"), acc_best=("acc", "max"), mean_p=("p", "mean"),
              mean_T_pi4=("T_pi4", "mean"), total_T_pi4_min=("T_pi4", lambda x: x.sum() / 60),
              late_pct=("late", lambda x: 100 * x.mean()), bad_rounds=("meas_ok", lambda x: int((~x).sum())),
              up_MB=("up_bytes", lambda x: x.mean() / 1e6),
              pred_err_mean=("pred_err", "mean"), pred_err_max=("pred_err", "max"),
              slowdown_pct=("slowdown", "mean"), wall_min=("T_wall", lambda x: x.sum() / 60),
              time_scale_max=("time_scale", "max")).reset_index()
    # trễ hạn chỉ tính trên các vòng đo tin cậy (measurement_ok)
    ok = df[df["meas_ok"]].groupby(["mode", "split", "policy"])["late"].mean().mul(100).rename("late_pct_ok")
    s = s.merge(ok.reset_index(), on=["mode", "split", "policy"], how="left")
    s["policy"] = pd.Categorical(s["policy"], POLICIES, ordered=True)
    s = s.sort_values(["mode", "split", "policy"])
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string(index=False))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    for fig_name, xcol, xlab in (("acc_vs_round", "t", "Vòng FL"), ("acc_vs_time", "T_cum_min", "Thời gian tích lũy (phút, thang Pi 4)")):
        keys = sorted({(m, sp) for m, sp in zip(df["mode"], df["split"])})
        fig, axs = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 3.8), squeeze=False)
        for ax, (m, sp) in zip(axs[0], keys):
            for pol in POLICIES:
                d = df[(df["mode"] == m) & (df["split"] == sp) & (df["policy"] == pol)]
                if len(d):
                    ax.plot(d[xcol], d["acc"] * 100, "o-", ms=3, color=COLOR.get(pol, "#2a78d6"), label=LABEL.get(pol, f"FAS p = {pol[5:]}%, khóa ở thiết bị"))
            ax.set_title(f"{m}, {sp.upper() if sp == 'iid' else 'non-IID (Dirichlet α=0,5)'}", fontsize=9)
            ax.set_xlabel(xlab)
            ax.set_ylabel("Độ chính xác test (%)")
        h, l = axs[0][0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fig_name + ".png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
    d = df[df["policy"] == "adaptive"]
    if len(d):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for (m, sp), x in d.groupby(["mode", "split"]):
            ax.step(x["t"], x["p"] * 100, where="mid", label=f"{m}, {sp}")
        ax.set_xlabel("Vòng FL"); ax.set_ylabel("p được chọn (%)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "p_schedule.png"), dpi=160); plt.close(fig)
    print(f"[step8] bảng: {os.path.join(OUT, 'summary.csv')}; hình: acc_vs_round.png, acc_vs_time.png, p_schedule.png")


if __name__ == "__main__":
    main()
