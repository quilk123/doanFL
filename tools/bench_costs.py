# -*- coding: utf-8 -*-
"""Đo bảng chi phí phía thiết bị TRÊN MÁY NÀY (bước 3; logic reference_sim/bench_costs.py).

Chạy trong container giới hạn như Raspberry Pi 4 (giống iot1/iot2 trong docker-compose):
  docker run --rm --cpus 1.6 --memory 4g --memory-swap 4g -e NTHREADS=2 -e OMP_NUM_THREADS=2 -v <testbed>:/tb -w /tb \
    testbed-tools python tools/bench_costs.py
Khác reference_sim: T_train đo đúng đường chạy của iot (iot.train.local_train: đặc trưng fp16 đọc qua mmap,
SGD momentum + weight_decay theo config) thay vì tensor float32 nạp sẵn.
Mọi thời gian thiết bị nhân `fidelity` (config, mặc định 1,4) để quy về Pi 4; thời gian thô cũng được ghi.

Kết quả: costs/cost_table.csv (theo p) và costs/cost_fixed.json (không phụ thuộc p).

Thời gian huấn luyện RIÊNG từng thiết bị (non-IID: số ảnh khác nhau, vd. 21.619 / 28.381 thay vì 25.000):
  ... python tools/bench_costs.py --train-only     (dữ liệu data/train/client{1,2} phải đang chia theo config `devices.split`)
  -> costs/train_time.json[split][k] (đo LẦN LƯỢT iot1 rồi iot2 trong cùng container giới hạn Pi 4 — không tranh CPU, như
  bảng chính). Không đụng cost_table.csv / cost_fixed.json. bs/policy.py dùng file này khi có mục cho split đang chạy.
"""
import os
import sys
import csv
import json
import time
import platform
import resource
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml
import torch
from common import model
from common.mkckks import MKCKKS, n_ct, ser_size
from iot import train

OUT = os.path.join(ROOT, "costs")


def med(f, reps):
    ts = []
    r = None
    for _ in range(reps):
        t = time.perf_counter()
        r = f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), r


def cgroup_limits():
    """Đọc giới hạn cgroup v2 của container: (số lõi, RAM byte, RAM+swap byte); None nếu không giới hạn."""
    def rd(path):
        try:
            return open(path).read().split()
        except OSError:
            return None
    cpu = rd("/sys/fs/cgroup/cpu.max")
    cpus = None if not cpu or cpu[0] == "max" else int(cpu[0]) / int(cpu[1])
    mem = rd("/sys/fs/cgroup/memory.max")
    mem = None if not mem or mem[0] == "max" else int(mem[0])
    swp = rd("/sys/fs/cgroup/memory.swap.max")
    swp = None if not swp or swp[0] == "max" else int(swp[0])
    return cpus, mem, (mem + swp) if mem is not None and swp is not None else None


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def bench_train(config, reps, device=1):
    """Thời gian thô từng lần huấn luyện cục bộ (local_train, `local_epochs` epoch) trên dữ liệu của thiết bị `device`
    (mặc định thiết bị 1; IID = 25.000 mẫu).
    Trả (lần khởi động, [reps lần ổn định]): lần đầu chậm hơn do đọc lạnh đặc trưng (mmap) và nạp thư viện."""
    w = model.init_vec(config)
    t0 = time.perf_counter()
    train.local_train(w, config, device, 0)
    first = time.perf_counter() - t0
    print(f"[bench] thiết bị {device} — epoch khởi động: {first:.2f}s (thô, không tính trung vị)", flush=True)
    times = []
    for t in range(1, reps + 1):
        t0 = time.perf_counter()
        train.local_train(w, config, device, t)
        times.append(time.perf_counter() - t0)
        print(f"[bench] thiết bị {device} — epoch {t}: {times[-1]:.2f}s (thô)", flush=True)
    return first, times


def bench_train_devices(config, reps, F):
    """Đo LẦN LƯỢT iot1, iot2 trên dữ liệu thật của từng thiết bị -> costs/train_time.json[split] (×fidelity)."""
    split = config["devices"]["split"]
    meta_split = {k: json.load(open(os.path.join(train.DATA_DIR, "train", f"client{k}", "meta.json"), encoding="utf-8"))
                  for k in (1, 2)}
    for k, m in meta_split.items():
        if m["split"] != split:
            raise SystemExit(f"data/train/client{k} đang chia '{m['split']}', config là '{split}' — chạy "
                             f"tools/prep_data.py --split-only trước")
    out = {}
    for k in (1, 2):
        first, raw = bench_train(config, reps, device=k)
        out[str(k)] = {"n_samples": int(meta_split[k]["n"]), "T_train_s": float(np.median(raw)) * F,
                       "T_train_raw_s": raw, "T_train_first_raw_s": first}
        print(f"[bench] thiết bị {k}: {out[str(k)]['n_samples']} ảnh, T_train = {out[str(k)]['T_train_s']:.2f}s (Pi 4)",
              flush=True)
    path = os.path.join(OUT, "train_time.json")
    allt = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    allt[split] = dict(out, fidelity_factor=F, local_epochs=int(config["train"].get("local_epochs", 1)),
                       machine_cpu=cpu_model())
    json.dump(allt, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[bench] -> {path} [{split}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="số lần lặp mỗi phép đo (lấy trung vị)")
    ap.add_argument("--train-reps", type=int, default=5, help="số epoch đo thời gian huấn luyện")
    ap.add_argument("--allow-unlimited", action="store_true", help="cho chạy khi container không bị giới hạn")
    ap.add_argument("--train-only", action="store_true",
                    help="chỉ đo thời gian huấn luyện riêng từng thiết bị -> costs/train_time.json")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"), help="file cấu hình (vd. bản đặt split: noniid)")
    ap.add_argument("--out", default=OUT, help="thư mục ghi kết quả (mặc định costs/; demo tùy chỉnh: costs_ep<N>_<split>/)")
    args = ap.parse_args()
    globals()["OUT"] = args.out
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    F = float(config.get("fidelity", 1.4))
    cpus, mem, memsw = cgroup_limits()
    want_cpus = float(config["devices"]["cpus"])
    print(f"[bench] cgroup: cpus={cpus} mem={mem} mem+swap={memsw}; NTHREADS={torch.get_num_threads()}", flush=True)
    if not args.allow_unlimited:
        assert cpus is not None and abs(cpus - want_cpus) < 1e-6, f"container phải --cpus {want_cpus} (đang {cpus})"
        assert mem is not None and memsw == mem, "container phải --memory 4g --memory-swap 4g (không swap)"
        assert os.environ.get("OMP_NUM_THREADS") == str(torch.get_num_threads()), \
            "đặt -e OMP_NUM_THREADS bằng NTHREADS (như image iot), nếu không OpenFHE mở quá nhiều luồng"
    if args.train_only:
        bench_train_devices(config, args.train_reps, F)
        return

    fixed = {"machine_cpu": cpu_model(), "cgroup_cpus": cpus, "cgroup_mem_bytes": mem,
             "torch_threads": torch.get_num_threads(), "omp_threads": os.environ.get("OMP_NUM_THREADS"),
             "fidelity_factor": F, "M_trainable": model.NUM_PARAMS}

    # --- huấn luyện cục bộ 1 epoch (đúng code iot) ---
    first, raw = bench_train(config, args.train_reps)
    fixed["T_train_first_epoch_raw_s"] = first
    fixed["T_train_epoch_raw_s"] = raw
    fixed["T_train_epoch_s"] = float(np.median(raw)) * F

    # --- trích đặc trưng thân mạng (một lần cho mỗi ảnh) ---
    from torchvision.models import squeezenet1_1
    net = squeezenet1_1(weights=None)
    net.load_state_dict(torch.load(model.WEIGHTS))
    trunk = net.features[:12].eval()
    xb = torch.randn(64, 3, 112, 112)
    with torch.no_grad():
        t, _ = med(lambda: trunk(xb), args.reps)
    fixed["T_trunk_per_image_ms"] = t / 64 * 1000 * F
    fixed["T_trunk_once_25k_s"] = t / 64 * 25000 * F

    # --- HE ---
    slots = int(config["he"]["slots"])
    t0 = time.perf_counter()
    he = MKCKKS(config, 2)
    fixed["T_keygen_2party_s"] = (time.perf_counter() - t0) * F
    probe = he.encrypt(np.zeros(slots))
    fixed["ct_bytes"] = ser_size(probe[0])
    fixed["share_bytes"] = ser_size(he.partial_decrypt(probe, 1)[0])
    fixed["plain_bytes_per_param"] = 4
    M = model.NUM_PARAMS
    rng = np.random.default_rng(0)
    rows = []
    for k in range(0, 11):
        p = k / 10
        m_enc = int(round(p * M))
        nc = n_ct(m_enc, slots)
        x1 = rng.normal(0, 1e-2, m_enc)
        x2 = rng.normal(0, 1e-2, m_enc)
        if nc:
            te, c1 = med(lambda: he.encrypt(x1), args.reps)
            c2 = he.encrypt(x2)
            tadd, cs = med(lambda: he.add(c1, c2), args.reps)
            tl, sh0 = med(lambda: he.partial_decrypt(cs, 0), args.reps)
            tm, sh1 = med(lambda: he.partial_decrypt(cs, 1), args.reps)
            tf, dec = med(lambda: he.fuse([sh0, sh1], m_enc), args.reps)
            err = float(np.abs(dec - (x1 + x2)).max())
        else:
            te = tadd = tl = tm = tf = err = 0.0
        row = dict(p=p, m_enc=m_enc, n_ct=nc,
                   T_enc_s=te * F, T_pdec_s=max(tl, tm) * F,
                   T_add_bs_s=tadd, T_fuse_bs_s=tf,
                   D_up_bytes=nc * fixed["ct_bytes"] + (M - m_enc) * 4,
                   D_share_bytes=nc * fixed["share_bytes"], max_abs_err=err)
        rows.append(row)
        print("[bench]", {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()}, flush=True)
    fixed["peak_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cost_table.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    json.dump(fixed, open(os.path.join(OUT, "cost_fixed.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(fixed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
