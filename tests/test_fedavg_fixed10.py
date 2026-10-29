# -*- coding: utf-8 -*-
"""Kiểm tra đổi hướng FAS (28/10/2026): FedAvg tại thiết bị + policy fixed10.

  A. FedAvg: mỗi iot nhân Δ (sau clipping, trước khi tách) với n_k/(n_1+n_2), bs chỉ cộng, iot w <- w + tổng.
     IID (n_1 = n_2 = 25.000 => trọng số ½): cập nhật mô hình TRÙNG TỪNG BIT công thức cũ (Δ1 + Δ2)/2 trên cùng dữ liệu
     truyền (no_he: phần rõ float32; ½·x chính xác trong dấu phẩy động). Log có aggregation (FedAvg, n_k, trọng số) và
     summary (policy, p, MB gửi lên, thời gian truyền, vòng Pi 4, trễ hạn, acc).
     ĐỘ CHÍNH XÁC trùng trước khi sửa kiểm bằng 4 container thật (tất định): xem README — KHÔNG so ở đây vì 2 iot chạy
     chung một tiến trình (test_round) dùng chung bộ sinh ngẫu nhiên của torch (Dropout) => không tất định (cùng mã cũ,
     2 lần chạy: vòng 1 = 0,6836 và 0,6719; container: 0,6769 mọi lần).
  B. fixed10: p = 0,1 mọi vòng (không đọc SINR), bs KHÔNG có K (cộng mù vector đã xáo), 2 iot giữ mô hình giống hệt.
  python tests/test_fedavg_fixed10.py     (trong image testbed-tools, dữ liệu đang chia IID)
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_round import ROOT, load_config, run_rounds, CAPTURE_DIR   # noqa: E402
import iot.protect as protect_mod                                     # noqa: E402
import iot.train as train_mod                                         # noqa: E402

info_log = {}
w_log = {}


def instrument():
    orig_train = train_mod.local_train

    def local_train(w, config, device_id, t, **kw):   # ghi mô hình đầu vào mỗi vòng (= mô hình toàn cục sau vòng t − 1)
        w_log[(config.get("policy"), t, device_id)] = w.copy()
        return orig_train(w, config, device_id, t, **kw)
    train_mod.local_train = local_train

    orig = protect_mod.protect

    def protect(dhe, delta, p, config, perm_seed, t, device_id, pol, noise_rng, weight=1.0):
        out = orig(dhe, delta, p, config, perm_seed, t, device_id, pol, noise_rng, weight=weight)
        info_log[(pol, t, device_id)] = (out[4], out[1], out[3], out[2], p)
        return out
    protect_mod.protect = protect


def part_a(config):
    meta = [json.load(open(os.path.join(ROOT, "data", "train", f"client{k}", "meta.json"), encoding="utf-8")) for k in (1, 2)]
    assert all(m["split"] == "iid" for m in meta), "dữ liệu phải đang chia IID (tools/prep_data.py --split-only)"
    recs = run_rounds(dict(config, policy="no_he"), 3, timeout=1800)
    ag = recs[0]["aggregation"]
    n_meta = [m["n"] for m in meta]                      # số ảnh train thật của 2 thiết bị (3 tập: 22.500)
    assert ag["n_samples"] == n_meta and ag["weights"] == [0.5, 0.5], f"aggregation sai: {ag} (meta {n_meta})"
    s = recs[0]["summary"]
    for key in ("policy", "p", "upload_MB", "upload_tx_s", "round_pi4_s", "t_max_s", "late", "accuracy"):
        assert key in s, f"summary thiếu {key}"
    for t in (1, 2):
        w_prev, w_next = w_log[("no_he", t, 1)], w_log[("no_he", t + 1, 1)]
        assert np.array_equal(w_next, w_log[("no_he", t + 1, 2)]), "2 iot giữ mô hình khác nhau"
        (i1, x1, ip, _, _), (i2, x2, _, _, _) = info_log[("no_he", t, 1)], info_log[("no_he", t, 2)]
        assert i1["weight"] == i2["weight"] == 0.5
        d1, d2 = i1["delta_clipped"] / 0.5, i2["delta_clipped"] / 0.5        # Δ_k gốc (chia ½ chính xác)
        # mã cũ: phần rõ float32 của Δ_k gốc, bs cộng float32, iot chia 2 (float64) rồi đặt về đúng vị trí ip
        old = w_prev.copy()
        old[ip] = w_prev[ip] + (d1[ip].astype(np.float32) + d2[ip].astype(np.float32)).astype(np.float64) / 2.0
        assert np.array_equal(w_next, old), "FedAvg mới khác công thức cũ (Δ1 + Δ2)/2"
    print(f"[TEST] A. FedAvg tại thiết bị: n_k = {ag['n_samples']}, trọng số {ag['weights']}; cập nhật mô hình 2 vòng "
          f"trùng TỪNG BIT công thức cũ w + (Δ1 + Δ2)/2 (no_he, {len(ip)} tham số); summary vòng 1: "
          f"{s['upload_MB'][0]:.2f} MB/thiết bị, truyền {s['upload_tx_s'][0]:.1f} s, Pi 4 {s['round_pi4_s']:.1f} s, "
          f"trễ hạn {s['late']}, acc {s['accuracy']:.4f}")


def part_b(config):
    recs = run_rounds(dict(config, policy="fixed10"), 3, timeout=1800)
    for r in recs:
        assert r["decision"]["p"] == 0.1 and r["summary"]["policy"] == "fixed10"
        t = r["t"]
        (i1, x1, ip, ih, p1), (i2, x2, _, _, p2) = info_log[("fixed10", t, 1)], info_log[("fixed10", t, 2)]
        assert p1 == p2 == 0.1 and i1["pi"] is not None and np.array_equal(i1["pi"], i2["pi"]), "fixed10: phải xáo, cùng π_t"
        cap = np.load(os.path.join(CAPTURE_DIR, f"upload_{t:04d}.npz"))
        assert "bs_view_dev1" not in cap.files, "fixed10: bs KHÔNG được giải xáo (không có K)"
        assert np.array_equal(cap["shuffled_dev1"], x1)
        same_pos = float((cap["shuffled_dev1"] == i1["delta_clipped"][ip].astype(np.float32)).mean())
        assert same_pos < 0.01, "fixed10: bs thấy phần rõ đúng vị trí"
    print(f"[TEST] B. fixed10: p = 0,1 cả 3 vòng (không đọc SINR); bs không có K, chỉ thấy vector đã xáo "
          f"({len(x1)} giá trị); 2 iot cùng π_t; acc {[r['metrics']['accuracy'] for r in recs]}")


def main():
    config = load_config()
    instrument()
    part_a(config)
    part_b(config)
    print("\n[TEST] ĐẠT: FedAvg tại thiết bị (IID trùng từng bit công thức cũ) + policy fixed10 (p = 0,1, K chỉ ở thiết bị).")


if __name__ == "__main__":
    main()
