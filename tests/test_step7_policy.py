# -*- coding: utf-8 -*-
"""Test bước 7 (CLAUDE.md mục 10.7): policies (mục 8).

Đạt khi:
  A. t_max = T_train_Pi4 + t_max_margin_s (config `t_max: auto`).
  A2. safety_margin_s = phân vị 95% của (đo − dự báo) ở bước 7, làm tròn lên 0,5 s (tools/calib_margin.py).
  B. Lịch p của `adaptive` (bs/policy.py) TRÙNG reference_sim/integrate.choose_p với CÙNG γ, CÙNG bảng chi phí (costs/,
     kiểm mã băm) và CÙNG ngưỡng chọn T_max − δ — tests/ref/choose_p_ref.json (tools/ref_choose_p.py),
     cả uav_noma_pc và direct_noma.
  C. Policy cố định: no_he 0, full_he 1, fas10 0,1, fas50 0,5.
  D. Vòng FL thật (adaptive, 2 vòng): p trong log = choose_p(γ trong log); log có t_max, T dự báo, round_pi4_s, late,
     sim_artifact (độ chậm thêm khi 2 iot chạy song song).
  E. fas50 (bs CÓ K): bs giải xáo từng thiết bị — capture bs_view = phần rõ đã nhiễu của thiết bị, đúng thứ tự;
     mô hình 2 iot vẫn giống hệt và đúng FedAvg trên dữ liệu truyền.
Chạy: python tests/test_step7_policy.py  (cần data/, weights/, costs/, tests/ref/choose_p_ref.json)
"""
import os
import sys
import json
import hashlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_round import ROOT, load_config, run_rounds, CAPTURE_DIR
from bs import policy
from common import protocol as P
import iot.protect as protect_mod
import iot.train as train_mod

sys.path.insert(0, os.path.join(ROOT, "tools"))
import calib_margin                       # noqa: E402

info_log = {}
w_log = {}


def instrument():
    orig = protect_mod.protect

    def protect(dhe, delta, p, config, perm_seed, t, device_id, pol, noise_rng, weight=1.0):
        out = orig(dhe, delta, p, config, perm_seed, t, device_id, pol, noise_rng, weight=weight)
        info_log[(pol, t, device_id)] = (out[4], out[1], out[3], out[2])
        return out
    protect_mod.protect = protect
    orig_train = train_mod.local_train

    def local_train(w, config, device_id, t, **kw):
        w_log[(config.get("policy"), t, device_id)] = np.array(w, copy=True)
        return orig_train(w, config, device_id, t, **kw)
    train_mod.local_train = local_train


def costs_hash():
    h = hashlib.sha256()
    for fn in ("cost_table.csv", "cost_fixed.json"):
        h.update(open(os.path.join(ROOT, "costs", fn), "rb").read())
    return h.hexdigest()


def part_a(config, cm):
    t_max = policy.resolve_t_max(config, cm)
    want = cm.T_train + float(config["t_max_margin_s"])
    assert str(config["t_max"]).lower() == "auto" and abs(t_max - want) < 1e-12
    print(f"[TEST] A. t_max = T_train_Pi4 {cm.T_train:.2f} s + {config['t_max_margin_s']} s = {t_max:.2f} s")
    errs = [r["err"] for r in calib_margin.pred_errors()]
    d = calib_margin.margin(errs)
    assert float(config["safety_margin_s"]) == d, f"safety_margin_s {config['safety_margin_s']} ≠ {d} (tools/calib_margin.py)"
    t_sel = policy.t_select(config, cm)
    print(f"[TEST] A2. δ = safety_margin_s {d} s (p95 của {len(errs)} lệch đo − dự báo bước 7, {min(errs):.2f}…{max(errs):.2f} s "
          f"= {np.percentile(errs, 95):.2f} s, làm tròn lên 0,5 s); adaptive chọn p với {t_sel:.2f} s")
    return t_max, t_sel


def part_b(config, cm, t_sel):
    ref = json.load(open(os.path.join(ROOT, "tests", "ref", "choose_p_ref.json"), encoding="utf-8"))
    assert ref["costs_sha256"] == costs_hash(), "costs/ đã đổi — chạy lại tools/ref_choose_p.py"
    assert abs(ref["t_select"] - t_sel) < 1e-12 and abs(ref["T_train_ref"] - cm.T_train) < 1e-12
    assert np.allclose(ref["grid_ref"], policy.grid(config))
    for mode, case in ref["cases"].items():
        cfg = dict(config, channel=dict(config["channel"], mode=mode))
        gam = np.array(case["gamma"])
        p, late = policy.adaptive_p(cfg, gam, t_sel, cm)
        diff = int((p != np.array(case["p"])).sum())
        diff_late = int((late != np.array(case["late"])).sum())
        sched = [round(float(x), 1) for x in p[case["n_random"]:]]
        print(f"[TEST] B. {mode}: {len(gam)} trường hợp γ, khác reference ở {diff} giá trị p, {diff_late} cờ trễ hạn; "
              f"lịch vòng 1–{case['rounds']}: {sched}")
        assert diff == 0 and diff_late == 0, f"{mode}: lịch p khác integrate.choose_p"


def part_c(config):
    for pol, want in (("no_he", 0.0), ("full_he", 1.0), ("fixed10", 0.1), ("fas10", 0.1), ("fas50", 0.5)):
        assert policy.choose_p(dict(config, policy=pol), 5.0, 5.0, 1) == want
    print("[TEST] C. policy cố định (không đọc SINR): no_he 0, full_he 1, fixed10 0,1, fas10 0,1, fas50 0,5")


def part_d(config, cm, t_max, t_sel):
    cfg = dict(config, policy="adaptive")
    recs = run_rounds(cfg, 2, timeout=1800)
    for r in recs:
        g = np.array([[r["channel"]["g1"], r["channel"]["g2"]]])
        p_ref = float(policy.adaptive_p(cfg, g, t_sel, cm)[0][0])
        tm = r["time"]
        assert r["decision"]["p"] == p_ref, "p trong log khác choose_p"
        assert abs(r["t_max"] - t_max) < 1e-12 and abs(r["t_select"] - t_sel) < 1e-12
        assert abs(r["pred_error_s"] - (tm["round_pi4_s"] - r["decision"]["T_pred_s"])) < 1e-12
        for k in ("round_pi4_s", "round_pi4_measured_train_s"):
            assert k in tm
        assert r["late"] == (tm["round_pi4_s"] > t_max)
        sa = r["sim_artifact"]
        print(f"[TEST] D. vòng {r['t']}: γ = {(10 * np.log10(g[0])).round(1).tolist()} dB -> p = {r['decision']['p']}; "
              f"Pi 4: dự báo {r['decision']['T_pred_s']:.1f} s, đo {tm['round_pi4_s']:.1f} s "
              f"(lệch {r['pred_error_s']:+.2f} s; nếu lấy huấn luyện đo được: {tm['round_pi4_measured_train_s']:.1f} s), "
              f"chọn với {t_sel:.1f} s, t_max {t_max:.1f} s, "
              f"trễ hạn {r['late']}; thô {tm['round_s']:.1f} s; huấn luyện chậm thêm "
              f"{[round(x) for x in sa['slowdown_pct']]}% so với 1 thiết bị một mình (giả lập)")


def part_e(config):
    cfg = dict(config, policy="fas50")
    recs = run_rounds(cfg, 2, timeout=1800)
    w1, w2 = w_log[("fas50", 1, 1)], w_log[("fas50", 2, 1)]
    assert np.array_equal(w2, w_log[("fas50", 2, 2)]), "fas50: 2 iot giữ mô hình khác nhau"
    info1, x1, ip, ih = info_log[("fas50", 1, 1)]
    info2, x2, _, _ = info_log[("fas50", 1, 2)]
    cap = np.load(os.path.join(CAPTURE_DIR, "upload_0001.npz"))
    for k, (info, x) in ((1, (info1, x1)), (2, (info2, x2))):
        assert np.array_equal(cap[f"shuffled_dev{k}"], x)
        noisy_true_order = protect_mod.unshuffle(x, info["pi"])
        assert np.array_equal(cap[f"bs_view_dev{k}"], noisy_true_order), "bs (FAS) không giải xáo đúng"
        err = float((noisy_true_order.astype(np.float64) - info["delta_clipped"][ip]).std())
        print(f"[TEST] E. fas50 thiết bị {k}: bs giải xáo được {len(x)} giá trị rõ đúng vị trí; "
              f"nhiễu sd {err:.5f} (λ·RMS = {info['sd']:.5f})")
    want = (cap["bs_view_dev1"] + cap["bs_view_dev2"]).astype(np.float64)   # bs cộng float32 sau giải xáo (Δ đã ×n_k/n)
    err = float(np.abs((w2 - w1)[ip] - want).max())
    assert err < 1e-9, f"fas50: cập nhật phần rõ lệch FedAvg trên dữ liệu truyền {err:.2e}"
    print(f"[TEST] E. fas50: 2 iot giữ mô hình giống hệt; phần rõ lệch FedAvg trên dữ liệu truyền {err:.1e}; "
          f"acc {recs[0]['metrics']['accuracy']:.4f} -> {recs[1]['metrics']['accuracy']:.4f}, p = {recs[0]['decision']['p']}")


def part_f(config, cm, cm_int):
    """T_train riêng từng thiết bị (costs/train_time.json — từ 28/10/2026 đo cho IID 3 tập, 22.500 ảnh/thiết bị):
    bảng chung (cm_int, không có train_time) -> round_time trùng TỪNG BIT integrate (T_train + truyền);
    T riêng (cm) -> max_k(T_train_k + truyền_k); pi4_timing dùng T_train của đúng thiết bị."""
    from bs.server import pi4_timing
    rng = np.random.default_rng(7)
    gam = 10 ** rng.uniform(-1, 2, (500, 2))
    for mode in ("uav_noma_pc", "direct_noma"):
        cfg = dict(config, channel=dict(config["channel"], mode=mode), devices=dict(config["devices"], split="iid"))
        for p in (0.0, 0.3, 1.0):
            r = cm_int.row(p)
            old = cm_int.T_train + policy.tx_time(cfg, gam, r["D_up_bytes"] * 8)          # integrate.round_time
            if p > 0:
                old = old + r["T_enc_s"] + r["T_pdec_s"] + policy.tx_time(cfg, gam, r["D_share_bytes"] * 8)
            assert np.array_equal(policy.round_time(cfg, p, gam, cm=cm_int), old), f"{mode} p={p}: khác integrate"
    split = config["devices"]["split"]
    assert split in cm.train_time, f"chưa đo T_train riêng từng thiết bị cho '{split}' (bench_costs.py --train-only)"
    T1, T2 = cm.train_times(split)
    cfg = dict(config, devices=dict(config["devices"], split=split))
    r = cm.row(0.4)
    t_up = r["D_up_bytes"] * 8 / policy.rates(cfg, gam)
    want = np.maximum(T1 + t_up[:, 0], T2 + t_up[:, 1]) + r["T_enc_s"] + r["T_pdec_s"]         + policy.tx_time(cfg, gam, r["D_share_bytes"] * 8)
    assert np.allclose(policy.round_time(cfg, 0.4, gam, cm=cm), want, rtol=0, atol=1e-9)
    p_int = policy.adaptive_p(cfg, gam, 61.43, cm_int)[0]
    p_dev = policy.adaptive_p(cfg, gam, 61.43, cm)[0]
    # bản ghi giả: thiết bị 1 truyền chậm hơn 1 s -> thời gian vòng theo thiết bị muộn nhất
    rec = {"time": {"dev1": {"train_s": 1.0, "enc_s": 0.1, "pdec_s": 0.1, "update_s": 0.0},
                    "dev2": {"train_s": 1.0, "enc_s": 0.1, "pdec_s": 0.1}, "he_add_s": 0.0, "he_fuse_s": 0.0},
           "uplink": {"dev1": [{"type": P.UPLOAD, "tx_s": 5.0}], "dev2": [{"type": P.UPLOAD, "tx_s": 4.0}]}}
    pi4_timing(rec, 1.4, cm, split)
    assert rec["time"]["train_pi4_table_s"] == [T1, T2]
    assert abs(rec["time"]["round_pi4_s"] - (max(T1 + 0.14 + 5.0, T2 + 0.14 + 4.0) + 0.14)) < 1e-9
    print(f"[TEST] F. bảng chung trùng integrate từng bit (2 chế độ × 3 p × 500 γ); T_train riêng ({split}) = "
          f"{T1:.2f} / {T2:.2f} s ({cm.train_time[split]['1']['n_samples']} / {cm.train_time[split]['2']['n_samples']} ảnh) "
          f"-> p adaptive trung bình {p_int.mean():.2f} (bảng chung) -> {p_dev.mean():.2f} (T riêng); "
          f"pi4_timing theo thiết bị muộn nhất")


def main():
    config = load_config()
    cm = policy.cost_model(os.path.join(ROOT, "costs"))
    # bảng chung (như integrate.round_time của reference_sim): bỏ T_train riêng từng thiết bị -> so lịch p từng bit
    cm_int = policy.CostModel(os.path.join(ROOT, "costs"))
    cm_int.train_time = {}
    instrument()
    t_max, t_sel = part_a(config, cm_int)
    part_b(config, cm_int, t_sel)
    part_c(config)
    part_f(config, cm, cm_int)
    part_d(config, cm, t_max, t_sel)
    part_e(config)
    print("\n[TEST] ĐẠT: lịch p của adaptive trùng integrate.choose_p (cùng bảng chi phí, cùng ngưỡng t_max − δ); "
          "policies đúng mục 8.")


if __name__ == "__main__":
    main()
