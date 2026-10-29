# -*- coding: utf-8 -*-
"""Test bước 6 (CLAUDE.md mục 10.6): clipping + nhiễu + xáo vị trí π_t.

Đạt khi:
  A. HKDF đúng RFC 5869; hai iot sinh CÙNG π_t (cùng K, t); π_t đổi giữa các vòng; K khác -> π khác.
  B. Bảo vệ theo policy đúng bảng mục 8 (clipping / kiểu nhiễu / xáo).
  C. Vòng FL thật, `adaptive`, λ = 0: mô hình sau giải xáo trùng FedAvg thường trên cùng dữ liệu truyền (sai khác < 1e-9);
     2 iot giữ mô hình giống hệt; ‖Δ‖ ≤ C; vector bs nhận là hoán vị (không cùng thứ tự) của phần rõ.
  D. λ = 3: độ lệch chuẩn nhiễu ≈ λ·C/√M (±5%), nhiễu 2 thiết bị độc lập.
  E. docker-compose: secrets/perm_seed chỉ cấp cho iot1, iot2 (bs, uav không có); không Dockerfile nào chép secrets/.
     Kiểm tra trên container thật: tests/check_secrets_docker.sh.
Chạy: python tests/test_step6_protect.py  (cần data/, weights/)
"""
import os
import sys
import yaml

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_round import ROOT, load_config, run_rounds, CAPTURE_DIR
import iot.protect as protect_mod
import iot.train as train_mod
from common import crypto_setup, model

TOL = 1e-9

info_log = {}    # (lần chạy, t, thiết bị) -> (info, phần rõ gửi đi, idx_plain, idx_he)
w_log = {}       # (lần chạy, t, thiết bị) -> w đầu vòng t
fused = []       # tổng phần HE bs ghép (theo thứ tự)
run_id = [0]


def instrument():
    orig = protect_mod.protect

    def protect(dhe, delta, p, config, perm_seed, t, device_id, policy, noise_rng, weight=1.0):
        out = orig(dhe, delta, p, config, perm_seed, t, device_id, policy, noise_rng, weight=weight)
        info_log[(run_id[0], t, device_id)] = (out[4], out[1], out[3], out[2])
        return out
    protect_mod.protect = protect

    orig_train = train_mod.local_train

    def local_train(w, config, device_id, t, **kw):
        w_log[(run_id[0], t, device_id)] = np.array(w, copy=True)
        return orig_train(w, config, device_id, t, **kw)
    train_mod.local_train = local_train

    orig_fuse = crypto_setup.BsHE.fuse

    def fuse(self, s1, s2, n):
        out = orig_fuse(self, s1, s2, n)
        fused.append((run_id[0], out))
        return out
    crypto_setup.BsHE.fuse = fuse


class FakeHE:
    def encrypt(self, v):
        return [b"ct"] * int(np.ceil(len(v) / 4096))


def part_a(K):
    okm = protect_mod.hkdf_sha256(bytes([0x0b] * 22), b"", 42)
    assert okm.hex() == ("8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d"
                         "9d201395faa4b61a96c8"), "HKDF sai so với RFC 5869 (ca 3)"
    n = 101157
    p1, p2 = protect_mod.perm_for(K, 5, n), protect_mod.perm_for(bytes(K), 5, n)   # như 2 thiết bị độc lập
    assert np.array_equal(np.sort(p1), np.arange(n)), "π không phải hoán vị"
    assert np.array_equal(p1, p2), "hai thiết bị ra π khác nhau"
    p3 = protect_mod.perm_for(K, 6, n)
    p4 = protect_mod.perm_for(bytes(32), 5, n)
    assert (p1 == p3).mean() < 0.01 and (p1 == p4).mean() < 0.01, "π không đổi theo vòng / theo K"
    print(f"[TEST] A. HKDF đúng RFC 5869; π_t trùng giữa 2 thiết bị; vị trí trùng giữa vòng 5 và 6: "
          f"{(p1 == p3).mean() * 100:.3f}%, giữa K khác: {(p1 == p4).mean() * 100:.3f}%")


def part_b(config, K):
    M = model.NUM_PARAMS
    C = float(config["dp"]["clip"])
    lam = float(config["dp"]["lam"])
    rng = np.random.default_rng(1)
    delta = rng.normal(0, 0.02, M)                     # ‖Δ‖ ≈ 9 > C
    norm = np.linalg.norm(delta)
    for policy, p in (("no_he", 0.0), ("full_he", 1.0), ("fixed10", 0.1), ("fas50", 0.5), ("adaptive", 0.5)):
        cts, x, idx_he, idx_plain, info = protect_mod.protect(
            FakeHE(), delta, p, config, K, 3, 1, policy, np.random.default_rng(2))
        prof = protect_mod.PROFILES[policy]
        dc = info["delta_clipped"]
        if prof["clip"]:
            assert abs(np.linalg.norm(dc) - C) < 1e-9, "clipping sai"
        else:
            assert np.array_equal(dc, delta), f"{policy} không được clipping"
        plain = dc[idx_plain]
        if prof["shuffle"]:
            assert info["pi"] is not None
            back = protect_mod.unshuffle(x.astype(np.float64), info["pi"])
        else:
            assert info["pi"] is None
            back = x.astype(np.float64)
        noise = back - plain
        want_sd = 0.0 if not prof["noise"] else \
            lam * (C / np.sqrt(M) if prof["noise"] == "clip" else np.sqrt(np.mean(plain ** 2)))
        got_sd = float(noise.std()) if len(noise) else 0.0
        if want_sd == 0:
            assert got_sd < 1e-8, f"{policy} không được có nhiễu"
        else:
            assert abs(got_sd / want_sd - 1) < 0.02, f"{policy}: sd nhiễu {got_sd} ≠ {want_sd}"
        print(f"[TEST] B. {policy:8s} p={p}: clipping={'có' if prof['clip'] else 'không'} "
              f"(‖Δ‖ {norm:.2f} -> {np.linalg.norm(dc):.2f}), sd nhiễu {got_sd:.5f} (mong {want_sd:.5f}), "
              f"xáo={'có' if info['pi'] is not None else 'không'}, {len(cts)} ciphertext")


def part_c(config):
    run_id[0] = 1
    cfg = dict(config, policy="adaptive", dp=dict(config["dp"], lam=0))
    C = float(cfg["dp"]["clip"])
    recs = run_rounds(cfg, 3, timeout=1800)
    worst_plain = worst_he = worst_f64 = 0.0
    for t in (1, 2):
        w_prev = [w_log[(1, t, k)] for k in (1, 2)]
        w_next = [w_log[(1, t + 1, k)] for k in (1, 2)]
        assert np.array_equal(w_prev[0], w_prev[1]) and np.array_equal(w_next[0], w_next[1]), \
            f"vòng {t}: 2 iot giữ mô hình khác nhau"
        (i1, x1, ip, ih), (i2, x2, _, _) = info_log[(1, t, 1)], info_log[(1, t, 2)]
        assert np.array_equal(i1["pi"], i2["pi"]), "2 iot khác π_t"
        for i in (i1, i2):
            assert np.linalg.norm(i["delta_clipped"]) <= C * (1 + 1e-12), "‖Δ‖ > C"
        assert i1["weight"] == i2["weight"] == 0.5, "IID: trọng số FedAvg phải là n_k/n = ½"
        d1, d2 = i1["delta_clipped"], i2["delta_clipped"]   # đã ×n_k/n (FedAvg tại thiết bị)
        # FedAvg trên CÙNG dữ liệu truyền (phần rõ float32, không xáo): d_k đã ×n_k/n nên chỉ cộng theo vị trí
        base_plain = (d1[ip].astype(np.float32) + d2[ip].astype(np.float32)).astype(np.float64)
        upd = w_next[0] - w_prev[0]
        worst_plain = max(worst_plain, float(np.abs(upd[ip] - base_plain).max()))
        worst_he = max(worst_he, float(np.abs(upd[ih] - (d1[ih] + d2[ih])).max()))
        worst_f64 = max(worst_f64, float(np.abs(upd - (d1 + d2)).max()))
        # bs thấy gì: vector đã xáo (capture/), cùng tập giá trị nhưng khác thứ tự
        cap = np.load(os.path.join(CAPTURE_DIR, f"upload_{t:04d}.npz"))
        seen = cap["shuffled_dev1"]
        assert np.array_equal(seen, x1), "capture khác vector thiết bị gửi"
        same_pos = float((seen == d1[ip].astype(np.float32)).mean())
        assert same_pos < 0.01 and np.array_equal(np.sort(seen), np.sort(d1[ip].astype(np.float32)))
        print(f"[TEST] C. vòng {t}: ‖Δ‖ trước clipping {i1['norm']:.2f} / {i2['norm']:.2f} -> C={C}; "
              f"bs nhận {len(seen)} giá trị rõ, chỉ {same_pos * 100:.3f}% ở đúng vị trí; acc {recs[t - 1]['metrics']['accuracy']:.4f}")
    assert not np.array_equal(info_log[(1, 1, 1)][0]["pi"], info_log[(1, 2, 1)][0]["pi"]), "π_t không đổi giữa các vòng"
    print(f"[TEST] C. λ=0: sai khác phần rõ so với FedAvg thường {worst_plain:.1e} (< {TOL:g}); phần HE {worst_he:.1e}; "
          f"so với FedAvg float64 (gồm làm tròn float32 + HE) {worst_f64:.1e}")
    assert worst_plain < TOL, f"sai khác {worst_plain:.2e} ≥ {TOL}"
    assert worst_he < 1e-6


def part_d(config):
    run_id[0] = 2
    cfg = dict(config, policy="adaptive", dp=dict(config["dp"], lam=3))
    recs = run_rounds(cfg, 1, timeout=1800)
    M = model.NUM_PARAMS
    noises = []
    for k in (1, 2):
        info, x, ip, _ = info_log[(2, 1, k)]
        back = protect_mod.unshuffle(x.astype(np.float64), info["pi"])
        noises.append(back - info["delta_clipped"][ip])
    # FedAvg tại thiết bị: nhiễu adaptive nhân cùng trọng số n_k/n (IID: ½) => λ·w·C/√M
    w = info_log[(2, 1, 1)][0]["weight"]
    want = 3 * w * float(cfg["dp"]["clip"]) / np.sqrt(M)
    sds = [float(n.std()) for n in noises]
    corr = float(np.corrcoef(noises[0], noises[1])[0, 1])
    print(f"[TEST] D. λ=3: sd nhiễu {sds[0]:.5f} / {sds[1]:.5f} (mong λ·w·C/√M = {want:.5f}, w = {w}); "
          f"tương quan nhiễu 2 thiết bị {corr:+.4f}; "
          f"acc vòng 1 {recs[0]['metrics']['accuracy']:.4f}")
    assert all(abs(s / want - 1) < 0.05 for s in sds), "sd nhiễu sai"
    assert abs(corr) < 0.02, "nhiễu 2 thiết bị không độc lập"


def part_e():
    dc = yaml.safe_load(open(os.path.join(ROOT, "docker-compose.yml"), encoding="utf-8"))
    for name, s in dc["services"].items():
        has = "perm_seed" in (s.get("secrets") or []) or any("perm_seed" in str(v) for v in s.get("volumes") or [])
        if name in ("iot1", "iot2"):
            assert has, f"{name} phải có perm_seed"
        else:
            assert not has, f"{name} KHÔNG được có perm_seed"
            assert not any("perm_seed" in str(v) for v in (s.get("environment") or {}).values())
    for svc in ("bs", "uav", "iot"):
        df = open(os.path.join(ROOT, svc, "Dockerfile"), encoding="utf-8").read()
        assert "secrets" not in df, f"{svc}/Dockerfile chép secrets/"
    cfg_text = open(os.path.join(ROOT, "config.yaml"), encoding="utf-8").read()
    K = open(os.path.join(ROOT, "secrets", "perm_seed"), "rb").read()
    assert K.hex() not in cfg_text, "K không được nằm trong config.yaml"
    print("[TEST] E. compose: perm_seed chỉ cấp cho iot1, iot2; bs, uav không có; Dockerfile không chép secrets/; K không ở config")


def main():
    config = load_config()
    K = open(os.path.join(ROOT, "secrets", "perm_seed"), "rb").read()
    assert len(K) == 32
    instrument()
    part_a(K)
    part_b(config, K)
    part_e()
    part_c(config)
    part_d(config)
    print("\n[TEST] ĐẠT: clipping + nhiễu + xáo π_t đúng thiết kế; λ=0 trùng FedAvg (< 1e-9); 2 iot cùng π_t, π_t đổi theo vòng; "
          "bs/uav không có K (cấu hình). Chạy thêm tests/check_secrets_docker.sh trên máy có Docker.")


if __name__ == "__main__":
    main()
