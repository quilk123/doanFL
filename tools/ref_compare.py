# -*- coding: utf-8 -*-
"""Kiểm chứng TOÀN BỘ thiết kế với reference_sim: chạy reference_sim/fl.py (không sửa) với ĐÚNG lịch p testbed đã dùng
(round_*.json của bước 8) và cờ bảo vệ tương ứng policy, cùng đặc trưng / phân hoạch / seed; so độ chính xác theo vòng.

  python tools/ref_compare.py run <experiments/step8/cấu hình> [...] --ref-sim /ref   (trong image testbed-tools)
  python tools/ref_compare.py report
Cờ reference_sim theo policy (mục 8):
  no_he, full_he: --lam 0          fixed10/fas10/fas50: --lam λ --perm 1 (nhiễu λ·RMS)          adaptive: --clip C --lam λ --perm 1
--he 0: tổng phần HE tính rõ (HE thật chỉ thêm sai số ~5e-7, bước 5). Nhiễu ngẫu nhiên khác nhau giữa hai bên => so theo
khoảng lệch, không trùng từng bit. Kết quả: experiments/ref_compare/{<cấu hình>.json, compare.csv, compare.png}
reference_sim cần ROOT/cache (= data/full), ROOT/../weights, ROOT/results: chép fl.py, common.py sang thư mục tạm.
"""
import os
import sys
import glob
import json
import shutil
import argparse
import tempfile
import subprocess

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "ref_compare")


def ref_flags(cfg):
    pol = cfg["policy"]
    lam, C = str(cfg["dp"]["lam"]), str(cfg["dp"]["clip"])
    if pol in ("no_he", "full_he"):
        return ["--lam", "0"]
    if pol.startswith("fixed") or pol in ("fas10", "fas50"):   # cùng cách bảo vệ; ai giữ khóa xáo không đổi độ chính xác
        return ["--lam", lam, "--perm", "1"]
    if pol == "adaptive":
        return ["--clip", C, "--lam", lam, "--perm", "1"]
    raise ValueError(pol)


def run(dirs, ref_sim):
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp()
    rs = os.path.join(tmp, "reference_sim")
    os.makedirs(os.path.join(rs, "results"))
    for fn in ("fl.py", "common.py"):
        shutil.copy(os.path.join(ref_sim, fn), rs)
    os.symlink(os.path.join(ROOT, "data", "full"), os.path.join(rs, "cache"))
    os.symlink(os.path.join(ROOT, "weights"), os.path.join(tmp, "weights"))
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        cfg = yaml.safe_load(open(os.path.join(d, "config.yaml"), encoding="utf-8"))
        recs = sorted(glob.glob(os.path.join(d, "round_*.json")))
        sched = [json.load(open(f, encoding="utf-8"))["decision"]["p"] for f in recs]
        sp = os.path.join(tmp, f"sched_{name}.json")
        json.dump(sched, open(sp, "w"))
        cmd = [sys.executable, "fl.py", "--name", name, "--schedule", sp, "--he", "0",
               "--split", cfg["devices"]["split"], "--seed", str(cfg.get("seed", 0)),
               "--lr", str(cfg["train"]["lr"]), "--bs", str(cfg["train"]["batch"])] + ref_flags(cfg)
        print(f"[ref] {name}: p = {sched}; {' '.join(cmd[2:])}", flush=True)
        subprocess.run(cmd, cwd=rs, check=True)
        shutil.copy(os.path.join(rs, "results", f"fl_{name}.json"), os.path.join(OUT, f"{name}.json"))
    shutil.rmtree(tmp, ignore_errors=True)


def report():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for fn in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        name = os.path.splitext(os.path.basename(fn))[0]
        ref = json.load(open(fn, encoding="utf-8"))["rounds"][1:]
        tb = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(ROOT, "experiments", "step8", name, "round_*.json")))]
        for a, b in zip(tb, ref):
            rows.append(dict(config=name, t=a["t"], p=a["decision"]["p"], acc_testbed=a["metrics"]["accuracy"], acc_ref=b["acc"]))
    df = pd.DataFrame(rows)
    df["diff_pt"] = (df.acc_testbed - df.acc_ref) * 100
    df.to_csv(os.path.join(OUT, "compare.csv"), index=False)
    s = df.groupby("config").agg(rounds=("t", "count"), acc_testbed_final=("acc_testbed", "last"),
                                 acc_ref_final=("acc_ref", "last"), diff_final_pt=("diff_pt", "last"),
                                 mean_abs_diff_pt=("diff_pt", lambda x: x.abs().mean()),
                                 max_abs_diff_pt=("diff_pt", lambda x: x.abs().max())).reset_index()
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(s.round(3).to_string(index=False))
    names = list(s.config)
    fig, axs = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2), squeeze=False)
    for ax, n in zip(axs[0], names):
        d = df[df.config == n]
        ax.plot(d.t, d.acc_testbed * 100, "o-", ms=3, label="testbed")
        ax.plot(d.t, d.acc_ref * 100, "s--", ms=3, label="reference_sim")
        ax.set_title(n.replace("uav_noma_pc_", ""), fontsize=8); ax.set_xlabel("Vòng"); ax.grid(alpha=0.25)
    axs[0][0].set_ylabel("Độ chính xác (%)"); axs[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--ref-sim", default="/ref")
    a = ap.parse_args()
    run(a.dirs, a.ref_sim) if a.cmd == "run" else report()


if __name__ == "__main__":
    main()
