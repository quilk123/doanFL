# -*- coding: utf-8 -*-
# [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
# Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
#   D(p) <= R_min * (T_max - T_comp)
# với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
# Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
# Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
"""Thí nghiệm quét T_max 50–90 s (sau bước 7).

  python tools/tmax_sweep.py analytic     # mô hình (bs/policy.py + bảng chi phí máy này), 1000 mẫu kênh / chế độ
  python tools/tmax_sweep.py summarize    # gộp số đo testbed (tools/run_tmax_sweep.sh) + vẽ hình
Kết quả: experiments/KET_QUA_CU/tmax_sweep_adaptive/{analytic.csv, measured.csv, summary.csv, tmax_sweep.png}
Giống reference_sim E6 (integrate.py timing: quét T_max), nhưng bảng chi phí đo trên máy này.
"""
import os
import sys
import glob
import json

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bs import policy                     # noqa: E402
from uav import channel as ch             # noqa: E402

OUT = os.path.join(ROOT, "experiments", "KET_QUA_CU", "tmax_sweep_adaptive")
TMAX = list(range(50, 91, 5))
POLICIES = ["no_he", "full_he", "fas10", "fas50", "adaptive"]
LABEL = {"no_he": "Không bảo vệ (p = 0)", "full_he": "HE toàn phần (p = 100%)", "fas10": "FAS p = 10%",
         "fas50": "FAS p = 50%", "adaptive": "p thích nghi (đề xuất)"}
COLOR = {"no_he": "#888780", "full_he": "#534AB7", "fas10": "#D85A30", "fas50": "#BA7517", "adaptive": "#1D9E75"}


def load_config():
    return yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))


def analytic(n=1000):
    import pandas as pd
    config = load_config()
    cm = policy.cost_model(os.path.join(ROOT, "costs"))
    rows = []
    for mode in ("uav_noma_pc", "direct_noma"):
        cfg = dict(config, channel=dict(config["channel"], mode=mode))
        gam = ch.draw(ch.Params(cfg), np.random.default_rng(11), n)[mode]   # như reference E6 (rng 11)
        for tm in TMAX:
            for pol in POLICIES:
                if pol == "adaptive":
                    p = policy.adaptive_p(cfg, gam, tm, cm)[0]
                else:
                    p = np.full(n, policy._FIXED[pol])
                T = np.array([policy.round_time(cfg, q, g[None], he=(pol != "no_he"), cm=cm)[0] for q, g in zip(p, gam)])
                rows.append(dict(mode=mode, T_max=tm, policy=pol, mean_p=p.mean(), miss=(T > tm).mean(),
                                 mean_T=T.mean(), median_T=np.median(T)))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "analytic.csv"), index=False)
    d = df[(df["mode"] == "uav_noma_pc")].pivot(index="T_max", columns="policy", values=["mean_p", "miss"])
    print("[sweep] uav_noma_pc (mô hình, 1000 mẫu kênh): p trung bình / tỷ lệ trễ hạn")
    print((d * 100).round(1).to_string())
    print(f"[sweep] T_train_Pi4 = {cm.T_train:.2f} s; t_max mặc định (auto) = {policy.resolve_t_max(config, cm):.2f} s")


def summarize():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for d in sorted(glob.glob(os.path.join(OUT, "measured", "tmax_*"))):
        for fn in sorted(glob.glob(os.path.join(d, "round_*.json"))):
            r = json.load(open(fn, encoding="utf-8"))
            tm = r["time"]
            rows.append(dict(T_max=r["t_max"], t=r["t"], p=r["decision"]["p"], T_pred=r["decision"]["T_pred_s"],
                             late_pred=r["decision"]["late_pred"], T_pi4=tm["round_pi4_s"], late=r["late"],
                             T_pi4_measured_train=tm["round_pi4_measured_train_s"], T_wall=tm["round_s"],
                             late_wall=r["late_wallclock"], acc=r["metrics"]["accuracy"],
                             upload_tx_max=max(tm["upload_tx_s"]),
                             slowdown_pct=float(np.mean(r["sim_artifact"]["slowdown_pct"]))))
    if not rows:
        sys.exit("chưa có số đo: chạy tools/run_tmax_sweep.sh")
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(OUT, "measured.csv"), index=False)
    s = m.groupby("T_max").agg(rounds=("t", "count"), mean_p=("p", "mean"), late_pred=("late_pred", "mean"),
                               late=("late", "mean"), late_wall=("late_wall", "mean"),
                               mean_T_pred=("T_pred", "mean"), mean_T_pi4=("T_pi4", "mean"),
                               mean_err=("T_pi4", "mean"), mean_slowdown=("slowdown_pct", "mean")).reset_index()
    s["mean_err"] = (m["T_pi4"] - m["T_pred"]).groupby(m["T_max"]).mean().values
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print("[sweep] testbed đo thật (adaptive, uav_noma_pc):")
    print(m[["T_max", "t", "p", "T_pred", "T_pi4", "late_pred", "late", "T_wall", "slowdown_pct", "acc"]]
          .round(2).to_string(index=False))
    print(s.round(3).to_string(index=False))

    a = pd.read_csv(os.path.join(OUT, "analytic.csv"))
    a = a[a["mode"] == "uav_noma_pc"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for pol in POLICIES:
        d = a[a.policy == pol]
        axs[0].plot(d.T_max, d.miss * 100, "o-", ms=3, color=COLOR[pol], label=LABEL[pol])
        axs[1].plot(d.T_max, d.mean_p * 100, "o-", ms=3, color=COLOR[pol], label=LABEL[pol])
    axs[0].set_xlabel("T_max (s, thang Pi 4)"); axs[0].set_ylabel("Tỷ lệ vòng trễ hạn (%)")
    axs[0].set_title("Mô hình, 1000 mẫu kênh UAV NOMA-PC", fontsize=9)
    axs[1].plot(s.T_max, s.mean_p * 100, "s", ms=7, mfc="none", color=COLOR["adaptive"], label="adaptive — testbed đo")
    axs[1].set_xlabel("T_max (s, thang Pi 4)"); axs[1].set_ylabel("p trung bình (%)")
    axs[1].set_title("p chọn được", fontsize=9)
    for _, r in m.iterrows():
        axs[2].plot(r.T_max, r.T_pred, "o", color="#888780", ms=4)
        axs[2].plot(r.T_max, r.T_pi4, "x", color=COLOR["adaptive"], ms=6)
    axs[2].plot(TMAX, TMAX, "--", color="#D85A30", lw=1)
    axs[2].plot([], [], "o", color="#888780", label="dự báo (bảng chi phí)")
    axs[2].plot([], [], "x", color=COLOR["adaptive"], label="đo trên testbed (Pi 4)")
    axs[2].plot([], [], "--", color="#D85A30", label="T = T_max")
    axs[2].set_xlabel("T_max (s)"); axs[2].set_ylabel("Thời gian vòng (s)")
    axs[2].set_title("adaptive: dự báo vs đo, từng vòng", fontsize=9)
    axs[2].legend(fontsize=7)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tmax_sweep.png"), dpi=160, bbox_inches="tight")
    print(f"[sweep] hình: {os.path.join(OUT, 'tmax_sweep.png')}")


if __name__ == "__main__":
    {"analytic": analytic, "summarize": summarize}[sys.argv[1]]()
