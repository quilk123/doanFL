# -*- coding: utf-8 -*-
"""Test bước 3 (CLAUDE.md mục 10.3): giới hạn Pi 4 + bảng chi phí đo trên máy này.

Chạy TRONG container giới hạn như iot (cần data/, weights/, costs/ từ tools/bench_costs.py):
  docker run --rm --cpus 1.6 --memory 4g --memory-swap 4g -e NTHREADS=2 -e OMP_NUM_THREADS=2 -v <testbed>:/tb -w /tb \
    testbed-tools python tests/test_step3_pi4.py
Đạt khi:
  - docker-compose: iot1, iot2 có cpus = devices.cpus, mem_limit = memswap_limit = devices.mem; tổng cpus ≤ số lõi;
  - container đang chạy bị giới hạn đúng như vậy (cgroup);
  - thời gian huấn luyện 1 epoch ổn định: mọi epoch lệch trung vị ≤ 10%;
  - bảng chi phí (costs/) đo trên máy này và khớp lần đo lại (trung vị lệch ≤ 10%).
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from bench_costs import ROOT, OUT, cgroup_limits, cpu_model, bench_train

import yaml
import numpy as np

REPS = 5
TOL = 0.10


def _bytes(s):
    s = str(s).lower()
    mult = {"k": 2 ** 10, "m": 2 ** 20, "g": 2 ** 30}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def main():
    config = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    want_cpus = float(config["devices"]["cpus"])
    want_mem = _bytes(config["devices"]["mem"])

    # 1) cấu hình docker-compose
    dc = yaml.safe_load(open(os.path.join(ROOT, "docker-compose.yml"), encoding="utf-8"))["services"]
    for name in ("iot1", "iot2"):
        s = dc[name]
        assert float(s["cpus"]) == want_cpus, f"{name}: cpus {s['cpus']} ≠ {want_cpus}"
        assert _bytes(s["mem_limit"]) == want_mem and _bytes(s["memswap_limit"]) == want_mem, \
            f"{name}: mem_limit/memswap_limit phải = {config['devices']['mem']} (không swap)"
    total = sum(float(s.get("cpus", 0)) for s in dc.values())
    assert total <= os.cpu_count(), f"tổng cpus {total} > {os.cpu_count()} lõi"
    print(f"[TEST] compose: iot cpus={want_cpus}, mem={config['devices']['mem']}, tổng cpus={total} ≤ {os.cpu_count()} lõi")

    # 2) container hiện tại bị giới hạn như Pi 4
    cpus, mem, memsw = cgroup_limits()
    assert cpus is not None and abs(cpus - want_cpus) < 1e-6, f"container: cpus={cpus}, cần --cpus {want_cpus}"
    assert mem == want_mem and memsw == want_mem, f"container: mem={mem}, mem+swap={memsw}, cần {want_mem} và không swap"
    print(f"[TEST] cgroup: cpus={cpus}, mem={mem / 2 ** 30:.1f} GiB, không swap")

    # 3) thời gian huấn luyện 1 epoch ổn định
    first, times = bench_train(config, REPS)
    times = np.array(times)
    m = float(np.median(times))
    dev = np.abs(times - m) / m
    print(f"[TEST] epoch khởi động {first:.2f} s (đọc lạnh, không tính); ổn định (thô): "
          f"{' '.join(f'{t:.2f}' for t in times)} s; trung vị {m:.2f} s; lệch tối đa {dev.max() * 100:.1f}%")
    assert dev.max() <= TOL, f"thời gian epoch lệch {dev.max() * 100:.1f}% > {TOL * 100:.0f}%"

    # 4) bảng chi phí đo trên máy này, khớp lần đo lại
    fixed = json.load(open(os.path.join(OUT, "cost_fixed.json"), encoding="utf-8"))
    assert fixed["machine_cpu"] == cpu_model(), "costs/ không đo trên máy này — chạy lại tools/bench_costs.py"
    assert abs(fixed["cgroup_cpus"] - want_cpus) < 1e-6, "costs/ không đo với giới hạn Pi 4"
    assert fixed.get("omp_threads") == "2", "costs/ đo khi chưa giới hạn luồng OpenMP — chạy lại với -e OMP_NUM_THREADS=2"
    # bench_train đo thiết bị 1 trên dữ liệu ĐANG chia (3 tập từ 28/10/2026: 22.500 ảnh) -> so với T_train riêng của
    # thiết bị 1 cho đúng phân hoạch đó (costs/train_time.json); chưa có thì so bảng chính (25.000 ảnh, 2 tập)
    split = config["devices"]["split"]
    tt_path = os.path.join(OUT, "train_time.json")
    tt = json.load(open(tt_path, encoding="utf-8")).get(split) if os.path.exists(tt_path) else None
    if tt:
        m_tab = float(np.median(tt["1"]["T_train_raw_s"]))
        src = f"train_time.json[{split}] thiết bị 1 ({tt['1']['n_samples']} ảnh)"
    else:
        m_tab = float(np.median(fixed["T_train_epoch_raw_s"]))
        src = "cost_fixed.json"
    d = abs(m - m_tab) / m_tab
    print(f"[TEST] bảng chi phí ({src}): T_train thô {m_tab:.2f} s, lần đo lại lệch {d * 100:.1f}%")
    assert d <= TOL, f"T_train trong costs/ lệch lần đo lại {d * 100:.1f}% > {TOL * 100:.0f}%"
    print(f"\n[TEST] ĐẠT: giới hạn Pi 4 đúng, epoch ổn định (lệch ≤ {dev.max() * 100:.1f}%), bảng chi phí đo trên máy này.")


if __name__ == "__main__":
    main()
