# -*- coding: utf-8 -*-
"""So sánh các phương án (đổi hướng FAS, 28/10/2026) — hình kiểu reference_sim plot_compare.py (E18) / e20_he_vs_fas,
và bảng tổng hợp testbed so với mô phỏng.

Phương án: Không HE (no_he) | FAS p = 10%, khóa xáo ở thiết bị (fixed10 — CHÍNH) | FAS gốc p = 10%, bs giữ khóa xáo
(fas10, chỉ có UAV) | HE toàn phần (full_he); mỗi phương án: có UAV (uav_noma_pc) và không UAV (direct_noma, nén kênh).
Đọc:
  experiments/step8/<cấu hình>/round_*.json         (testbed: 15 vòng IID, seed 0)
  experiments/ref_compare/compare.csv                (reference_sim fl.py, cùng lịch p)
  experiments/attack/attack_round1.csv               (tấn công vòng 1: góc nhìn bs / kẻ nghe lén)
Mô phỏng thời gian: bs/policy.round_time (= integrate.round_time, bảng chi phí máy này) trên 2.000 lần sinh kênh
(uav.channel.draw, seed 0), T_max như testbed — cùng cách reference_sim E18 (e18_compare_timing).
Độ chính xác: trung bình 3 vòng cuối của 15 vòng (như E18).
Rò rỉ: testbed đo độ khôi phục cập nhật (tương quan Δ̂ của bên tấn công với Δ thật, vòng 1) — KHÔNG phải tấn công nhận
dạng ảnh (gradient 1 ảnh) của reference_sim: tấn công đó không tái hiện được với cập nhật 1 epoch (CLAUDE.md mục 9).
Kết quả: experiments/fas/{summary.csv, so_voi_mo_phong.csv, e20_he_vs_fas.png}
  python tools/fas_report.py
"""
import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bs import policy                     # noqa: E402
from uav import channel                   # noqa: E402

OUT = os.path.join(ROOT, "experiments", "fas")
STEP8 = os.path.join(ROOT, "experiments", "step8")
SCHEMES = [("no_he", "Không HE"), ("fixed10", "FAS 10% (khóa ở thiết bị)"),
           ("fas10", "FAS gốc 10% (bs giữ khóa)"), ("full_he", "HE toàn phần")]
MODES = [("uav_noma_pc", "Có UAV", ""), ("direct_noma", "Không UAV", "_ts-auto")]
COLS = {"Có UAV": "#1D9E75", "Không UAV": "#888780"}
N_DRAW = 2000


def tb_rows(pol, mode, suf):
    d = os.path.join(STEP8, f"{mode}_iid_{pol}{suf}")
    recs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(d, "round_*.json")))]
    return d, recs


def model_timing(config, mode, p, he):
    """Mô phỏng (như reference_sim E18): thời gian vòng dự báo trên N_DRAW lần sinh kênh -> (T trung bình, % trễ hạn)."""
    cfg = dict(config, channel=dict(config["channel"], mode=mode), devices=dict(config["devices"], split="iid"))
    P = channel.Params(cfg)
    gam = channel.draw(P, np.random.default_rng(0), N_DRAW)[mode]
    T = policy.round_time(cfg, p, gam, he=he)
    t_max = policy.resolve_t_max(cfg)
    return float(T.mean()), float(100 * (T > t_max).mean())


def main():
    config = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    ref = pd.read_csv(os.path.join(ROOT, "experiments", "ref_compare", "compare.csv")) \
        if os.path.exists(os.path.join(ROOT, "experiments", "ref_compare", "compare.csv")) else None
    att = pd.read_csv(os.path.join(ROOT, "experiments", "attack", "attack_round1.csv")) \
        if os.path.exists(os.path.join(ROOT, "experiments", "attack", "attack_round1.csv")) else None
    rows = []
    for pol, label in SCHEMES:
        for mode, mlabel, suf in MODES:
            d, recs = tb_rows(pol, mode, suf)
            if not recs:
                continue
            name = os.path.basename(d)
            acc = [r["metrics"]["accuracy"] for r in recs]                       # tập TEST (báo cáo)
            vacc = [r["metrics"].get("val_accuracy") for r in recs]               # tập VALIDATION (log từ 3 tập)
            tacc = [r["metrics"].get("train_accuracy") or [None, None] for r in recs]   # train từng thiết bị
            T = [r["time"]["round_pi4_s"] for r in recs]
            up = [np.mean(r["summary"]["upload_MB"]) if "summary" in r else r["bytes"]["UPLOAD"] / 2e6 for r in recs]
            p = recs[0]["decision"]["p"]
            row = dict(scheme=label, policy=pol, kenh=mlabel, config=name, rounds=len(recs),
                       acc=100 * np.mean(acc[-3:]), acc_final=100 * acc[-1],
                       acc_val=100 * np.mean(vacc[-3:]) if vacc[-1] is not None else np.nan,
                       acc_train_dev1=100 * np.mean([x[0] for x in tacc[-3:]]) if tacc[-1][0] is not None else np.nan,
                       acc_train_dev2=100 * np.mean([x[1] for x in tacc[-3:]]) if tacc[-1][1] is not None else np.nan,
                       mean_T=np.mean(T),
                       miss=100 * np.mean([r["late"] for r in recs]), mean_p=100 * p, upload_MB=np.mean(up),
                       upload_tx_s=np.mean([max(r["time"]["upload_tx_s"]) for r in recs]),
                       time_scale_max=max(r.get("time_scale", 1.0) for r in recs))
            if att is not None:
                a = att[(att["config"] == name) & att["device"].astype(str).isin(["1", "2"])]
                if len(a):
                    row.update(bs=100 * a["recovery_corr"].mean(), eav=100 * a["recovery_eav"].mean(),
                               auc_bs=a["auc_bs"].mean(), auc_eav=a["auc_eav"].mean())
            # so với mô phỏng: (a) cùng 15 kênh testbed gặp (dự báo round_time của bs mỗi vòng = integrate.round_time);
            # (b) 2.000 lần sinh kênh như E18 (khác mẫu kênh => khác cả do ngẫu nhiên, vd. vòng 10 kênh xấu)
            row.update(mean_T_model_same=np.mean([r["decision"]["T_pred_s"] for r in recs]),
                       miss_model_same=100 * np.mean([r["decision"]["late_pred"] for r in recs]))
            Tm, missm = model_timing(config, mode, p, he=(pol != "no_he"))
            row.update(mean_T_model=Tm, miss_model=missm)
            if ref is not None:
                rr = ref[ref["config"] == name].sort_values("t")
                if len(rr):
                    row.update(acc_ref=100 * rr["acc_ref"].tail(3).mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    assert len(df), "chưa có kết quả bước 8 (experiments/step8)"
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    cmp_cols = ["scheme", "kenh", "acc", "acc_val", "acc_train_dev1", "acc_train_dev2", "acc_ref", "mean_T", "mean_T_model_same", "mean_T_model", "miss",
                "miss_model_same", "miss_model", "mean_p", "upload_MB", "bs", "eav"]
    cmp = df[[c for c in cmp_cols if c in df]].rename(columns={
        "kenh": "kênh", "acc": "acc_test_%", "acc_val": "acc_validation_%", "acc_train_dev1": "acc_train_iot1_%", "acc_train_dev2": "acc_train_iot2_%", "acc_ref": "acc_mô_phỏng_%", "mean_T": "T_vòng_testbed_s",
        "mean_T_model_same": "T_vòng_mô_phỏng_cùng_kênh_s", "mean_T_model": "T_vòng_mô_phỏng_2000_kênh_s",
        "miss": "trễ_hạn_testbed_%", "miss_model_same": "trễ_hạn_mô_phỏng_cùng_kênh_%",
        "miss_model": "trễ_hạn_mô_phỏng_2000_kênh_%",
        "mean_p": "p_%", "upload_MB": "gửi_lên_MB/thiết_bị", "bs": "bs_khôi_phục_%", "eav": "nghe_lén_khôi_phục_%"})
    cmp.to_csv(os.path.join(OUT, "so_voi_mo_phong.csv"), index=False)
    pd.set_option("display.width", 250)
    print(cmp.round(2).to_string(index=False))
    t_max = policy.resolve_t_max(config)
    plot_main(df, t_max)
    plot_uav(df, t_max)
    plot(df, t_max)


# --- hình dễ đọc (28/10/2026): mỗi hình một câu hỏi, kết luận ghi ngay trên hình -----------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"                 # 2 slot đầu bảng màu tham chiếu (đã kiểm CVD, tương phản)
GOOD, BAD = "#0ca30c", "#d03b3b"                    # trạng thái: kịp hạn / trễ hạn (luôn kèm chữ + dấu)
BOX = dict(facecolor="#fcfcfb", edgecolor="none", pad=1.5)   # nền chữ số liệu (không bị đường hạn chót đè)
SHORT = {"no_he": "Không mã hóa\n(Không HE)", "fixed10": "★ FAS 10%\nkhóa ở thiết bị (đề xuất)",
         "fas10": "FAS gốc 10%\ntrạm gốc giữ khóa", "full_he": "Mã hóa toàn bộ\n(HE toàn phần)"}


def _style(plt):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK, "ytick.left": False,
                         "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.axisbelow": True})   # lưới nằm DƯỚI thanh


def _human_time(s):
    """Giây -> chữ dễ đọc: '47 giây', '51 phút', '4,8 giờ'."""
    if s < 120:
        return f"{s:.0f} giây"
    if s < 7200:
        return f"{s / 60:.0f} phút"
    return f"{s / 3600:.1f} giờ".replace(".", ",")


def plot_main(df, t_max):
    """Hình chính, CÓ UAV: thời gian 1 vòng so với hạn chót | bị lộ bao nhiêu | độ chính xác."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    d = df[df["kenh"] == "Có UAV"].set_index("policy")
    pols = [p for p, _ in SCHEMES if p in d.index]
    y = np.arange(len(pols))[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.25, 1.25, 0.9]})

    # (1) thời gian một vòng
    ax = axs[0]
    T = [d.loc[p, "mean_T"] for p in pols]
    ok = [t <= t_max for t in T]
    ax.barh(y, T, height=0.55, color=[GOOD if o else BAD for o in ok], alpha=0.85)
    ax.axvline(t_max, color=INK, ls="--", lw=1.2)
    ax.text(t_max, len(pols) - 0.45, f" hạn chót {t_max:.0f} s", color=INK, fontsize=9, va="bottom")
    for yy, t, p in zip(y, T, pols):
        late = int(round(d.loc[p, "miss"] * d.loc[p, "rounds"] / 100))
        lab = f"{t:.0f} s   " + (f"✓ kịp hạn {d.loc[p, 'rounds'] - late}/{d.loc[p, 'rounds']} vòng" if late < d.loc[p, "rounds"] / 2
                                  else f"✗ trễ hạn {late}/{d.loc[p, 'rounds']} vòng")
        ax.text(t + 2, yy, lab, va="center", fontsize=9, color=INK, zorder=4, bbox=BOX)   # nền che đường hạn chót
    ax.set_xlim(0, max(T) * 1.75)
    ax.set_xlabel("giây cho 1 vòng (quy về Raspberry Pi 4)")
    ax.set_title("① Mỗi vòng mất bao lâu?", loc="left", fontsize=12, fontweight="bold", color=INK)

    # (2) bị lộ bao nhiêu cập nhật — trạm gốc / kẻ nghe lén
    ax = axs[1]
    h = 0.34
    for j, (k, lab, col) in enumerate((("bs", "trạm gốc (bs)", BLUE), ("eav", "kẻ nghe lén", ORANGE))):
        v = [d.loc[p, k] if k in d else np.nan for p in pols]
        yy = y + (h / 2 + 0.02 if j == 0 else -h / 2 - 0.02)
        ax.barh(yy, v, height=h, color=col, label=lab)
        for a, b in zip(yy, v):
            ax.text(b + 1.5, a, f"{b:.1f}%" if b < 10 else f"{b:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 125)
    ax.set_xlabel("% cập nhật bên tấn công khôi phục được (vòng 1) — thấp là tốt")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.legend(loc="lower right", fontsize=9, frameon=False, title="bên tấn công", title_fontsize=9)
    ax.set_title("② Bị lộ bao nhiêu?", loc="left", fontsize=12, fontweight="bold", color=INK)

    # (3) độ chính xác — biểu đồ điểm (khác biệt nhỏ, không dùng cột cắt trục)
    ax = axs[2]
    acc = [d.loc[p, "acc"] for p in pols]
    ax.scatter(acc, y, s=70, color=INK, zorder=3)
    for a, yy in zip(acc, y):
        ax.text(a, yy + 0.22, f"{a:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_xlim(75, 85)
    ax.set_xlabel("% ảnh test đoán đúng (TB 3 vòng cuối)")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_title("③ Mô hình học tốt không?", loc="left", fontsize=12, fontweight="bold", color=INK)

    axs[0].set_yticks(y)
    axs[0].set_yticklabels([SHORT[p] for p in pols], fontsize=10)
    for t in axs[0].get_yticklabels():
        if t.get_text().startswith("★"):
            t.set_fontweight("bold")
    for ax in axs:
        ax.set_ylim(-0.7, len(pols) - 0.3)
    fig.suptitle("Có UAV: FAS 10% (khóa ở thiết bị) gần nhanh như không mã hóa, kịp hạn gần như mọi vòng, "
                 "mà trạm gốc và kẻ nghe lén chỉ khôi phục được <1% cập nhật",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK, y=1.02)
    fig.text(0.01, -0.06, "Testbed 4 container thật, dữ liệu IID, 15 vòng. ② đo trên đúng dữ liệu trạm gốc nhận được; "
             "FAS gốc: trạm gốc có khóa xáo nên đọc được phần không mã hóa (chỉ còn nhiễu). "
             "③ các phương án chênh ≤ 1,4 điểm.", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fn = os.path.join(OUT, "e20_he_vs_fas.png")
    fig.savefig(fn, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"[fas] -> {fn}")


def plot_uav(df, t_max):
    """Có UAV và không UAV: thời gian 1 vòng (đơn vị dễ đọc)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    pols = [p for p, _ in SCHEMES if p in set(df["policy"]) and p != "fas10"]
    y = np.arange(len(pols))[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    h = 0.36
    for j, (ml, col) in enumerate((("Có UAV", BLUE), ("Không UAV", ORANGE))):
        for yy, p in zip(y, pols):
            r = df[(df["policy"] == p) & (df["kenh"] == ml)]
            if not len(r):
                continue
            t = float(r["mean_T"].iloc[0])
            a = yy + (h / 2 + 0.02 if j == 0 else -h / 2 - 0.02)
            ax.barh(a, t, height=h, color=col, label=ml if yy == y[0] else None)
            ax.text(t * 1.12, a, _human_time(t), va="center", fontsize=9.5, color=INK, zorder=4, bbox=BOX)
    ax.set_xscale("log")
    ax.axvline(t_max, color=INK, ls="--", lw=1.2)
    ax.text(t_max * 1.05, len(pols) - 0.45, f"hạn chót {t_max:.0f} s", fontsize=9, color=INK, va="bottom")
    ticks = [10, 60, 600, 3600, 6 * 3600]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["10 giây", "1 phút", "10 phút", "1 giờ", "6 giờ"])
    ax.set_xlim(10, 3e5)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[p] for p in pols])
    ax.set_ylim(-0.7, len(pols) - 0.2)
    ax.set_xlabel("thời gian 1 vòng (thang chia nhân 10 — mỗi vạch dài gấp nhiều lần vạch trước)")
    ax.legend(loc="lower right", fontsize=9.5, frameon=False)
    fig.suptitle("Không có UAV, tín hiệu yếu nên mỗi vòng chậm hơn 35–180 lần → mọi phương án đều trễ hạn",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, -0.04, "Có UAV: tốc độ kênh trung vị ~1,8 Mbit/s; không UAV (thiết bị gửi thẳng tới trạm gốc): ~0,05 Mbit/s. "
             "Thời gian quy về Raspberry Pi 4 và kênh thật.", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fn = os.path.join(OUT, "e20_co_va_khong_uav.png")
    fig.savefig(fn, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"[fas] -> {fn}")


def plot(df, t_max):
    """Hình chi tiết 6 ô (bố cục reference_sim E18) -> e20_chi_tiet_6_o.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    schemes = [s for _, s in SCHEMES if s in set(df["scheme"])]
    panels = [("acc", "Độ chính xác (%)\ncao là tốt", "%.1f", False),
              ("mean_T", "Thời gian vòng TB, thang Pi 4 (s)\nthấp là tốt — thang log", "%.0f", True),
              ("miss", f"Vòng trễ hạn T_max = {t_max:.1f} s (%)\nthấp là tốt", "%.0f", False),
              ("upload_MB", "Gửi lên mỗi thiết bị / vòng (MB)\nthấp là tốt", "%.2f", False),
              ("bs", "Trạm gốc khôi phục cập nhật (%)\nthấp là tốt", "%.1f", False),
              ("eav", "Kẻ nghe lén khôi phục cập nhật (%)\nthấp là tốt", "%.1f", False)]
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.6))
    y = np.arange(len(schemes))[::-1]
    h = 0.36
    for ax, (k, title, fmt, logx) in zip(axs.flat, panels):
        if k not in df:
            ax.set_visible(False)
            continue
        for j, (_, ml, _) in enumerate(MODES):
            vals = [df[(df["scheme"] == s) & (df["kenh"] == ml)][k] for s in schemes]
            v = np.array([x.iloc[0] if len(x) and pd.notna(x.iloc[0]) else np.nan for x in vals])
            yy = y + (h / 2 if j == 0 else -h / 2)
            b = ax.barh(yy[~np.isnan(v)], v[~np.isnan(v)], height=h, color=COLS[ml], label=ml)
            ax.bar_label(b, fmt=fmt, fontsize=7, padding=2)
        if k == "acc":
            ax.set_xlim(60, 86)
        elif logx:
            ax.set_xscale("log")
            ax.axvline(t_max, color="#C0392B", ls="--", lw=1)        # T_max
        else:
            ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.18)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(schemes if ax in axs[:, 0] else [], fontsize=8.5)
        ax.grid(axis="x", alpha=0.25)
    hs, ls = axs[0][0].get_legend_handles_labels()       # chú thích chung ở trên cùng (không che cột)
    fig.legend(hs, ls, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.01, -0.03, "Testbed 4 container, IID, seed 0, 15 vòng; độ chính xác = TB 3 vòng cuối (như E18). Không UAV: "
             "nén thời gian kênh, thời gian quy về kênh thật. Rò rỉ: độ khôi phục cập nhật vòng 1 (tương quan với Δ thật), "
             "KHÔNG phải tấn công nhận dạng ảnh 1-gradient của reference_sim. FAS gốc chỉ chạy có UAV.",
             fontsize=7.5, color="0.35", wrap=True)
    fig.tight_layout()
    fn = os.path.join(OUT, "e20_chi_tiet_6_o.png")
    fig.savefig(fn, dpi=160, bbox_inches="tight")
    print(f"[fas] -> {fn}")


if __name__ == "__main__":
    main()
