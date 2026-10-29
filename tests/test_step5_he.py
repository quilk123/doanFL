# -*- coding: utf-8 -*-
"""Test bước 5 (CLAUDE.md mục 10.5): HE đa khóa OpenFHE thật giữa bs và 2 iot.

Chạy các vòng FL đủ 7 bước (thread, như test_round) với p = 0,5 và p = 1. Đạt khi:
  1. Sai số tổng phần HE sau khi bs ghép 2 phần giải mã < 1e-6 (so với Δ_1 + Δ_2 thật của vùng HE).
  2. bs KHÔNG có khóa bí mật: mọi tin nhắn bs nhận chỉ gồm loại/trường công khai; đối tượng HE của bs không giữ khóa bí mật;
     khóa bí mật chỉ nằm ở 2 thiết bị và khác nhau.
  3. Phần giải mã của MỘT bên (iot1 hoặc iot2) KHÔNG giải được tổng.
  4. Byte UPLOAD thật = n_ct × kích thước ciphertext + phần rõ (lệch ≤ 1%), số ciphertext = ceil(m/4096).
Chạy: python tests/test_step5_he.py  (cần data/, weights/, costs/)
"""
import os
import sys
import json
import math
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_round import ROOT, load_config, run_rounds, RESULTS_DIR
from common import protocol as P
from common import crypto_setup
import bs.server as bs_server
import iot.protect as protect_mod
import iot.he as he_mod

TOL_HE = 1e-6
# loại tin nhắn và trường bs được phép nhận (tất cả đều công khai / đã mã hóa)
ALLOWED = {
    P.HELLO: {"type", "role", "device_id", "n_samples"},   # n_samples: số ảnh (trọng số FedAvg) — công khai
    P.HE_PK1: {"type", "pk"},
    P.HE_PK: {"type", "pk"},
    # time_scale: hệ số nén thời gian kênh do uav đặt (tham số kênh công khai, không phải dữ liệu thiết bị)
    P.CHANNEL_REPORT: {"type", "t", "g1", "g2", "R1", "R2", "mode", "time_scale", "warning"},
    P.UPLOAD: {"type", "t", "ciphertexts", "shuffled", "n_ciphertext", "n_he", "timing"},
    P.PARTIAL_DEC: {"type", "t", "delta", "timing"},
    P.METRICS: {"type", "t", "accuracy", "loss", "val_accuracy", "val_loss", "train_accuracy", "train_loss", "timing"},
    P.UAV_STATS: {"type", "t", "dev1", "dev2"},
}

truth = {}      # (t, device_id) -> Δ vùng HE thật (phía thiết bị)
fused = {}      # t -> tổng phần HE bs ghép được
bs_seen = []    # mọi tin nhắn bs nhận
devices = []    # các DeviceHE (để kiểm tra khóa bí mật nằm ở thiết bị)
shares = {}     # t -> các phần giải mã bs nhận (thứ tự iot1/iot2 không cố định)


def instrument():
    # phía thiết bị: ghi Δ vùng HE thật
    orig_protect = protect_mod.protect

    def protect(dhe, delta, p, config, perm_seed, t, device_id, *args, **kw):
        out = orig_protect(dhe, delta, p, config, perm_seed, t, device_id, *args, **kw)
        truth[(t, device_id)] = out[4]["delta_clipped"][out[2]].copy()   # Δ (sau clipping nếu có, đã ×n_k/n) vùng HE
        if dhe not in devices:
            devices.append(dhe)
        return out
    protect_mod.protect = protect

    # phía bs: ghi mọi tin nhắn nhận được + kết quả ghép (P riêng cho bs, không ảnh hưởng iot/uav)
    bsP = types.ModuleType("bs_protocol_spy")
    bsP.__dict__.update(P.__dict__)

    def recv_msg(sock):
        msg, n = P.recv_msg(sock)
        bs_seen.append(msg)
        if msg.get("type") == P.PARTIAL_DEC and msg.get("delta"):
            shares.setdefault(msg["t"], []).append(msg["delta"])
        return msg, n
    bsP.recv_msg = recv_msg
    bs_server.P = bsP

    orig_fuse = crypto_setup.BsHE.fuse

    def fuse(self, s1, s2, n):
        out = orig_fuse(self, s1, s2, n)
        fused[len(fused) + 1] = out
        return out
    crypto_setup.BsHE.fuse = fuse


def check_round(rec, slots, cost):
    t = rec["t"]
    p = rec["decision"]["p"]
    n_he = rec["decision"]["n_he"]
    want = truth[(t, 1)] + truth[(t, 2)]
    got = fused_by_t[t]
    err = float(np.abs(got - want).max())
    nct = rec["decision"]["n_ciphertext"]
    assert nct == math.ceil(n_he / slots), f"n_ct {nct} ≠ ceil({n_he}/{slots})"
    up = [it for it in rec["uplink"]["dev1"] if it["type"] == P.UPLOAD][0]
    expect = nct * cost["ct_bytes"] + (rec_M - n_he) * 4
    dev = abs(up["bytes"] - expect) / expect
    print(f"[TEST] vòng {t} p={p}: {n_he} giá trị HE, {nct} ciphertext; sai số tổng sau giải mã {err:.2e}; "
          f"UPLOAD {up['bytes']} B (dự kiến từ bảng chi phí {expect} B, lệch {dev * 100:.2f}%); "
          f"acc {rec['metrics']['accuracy']:.4f}")
    assert err < TOL_HE, f"sai số {err:.2e} ≥ {TOL_HE}"
    assert dev <= 0.01, f"byte UPLOAD lệch {dev * 100:.2f}%"
    return err


def main():
    global fused_by_t, rec_M
    from common import model
    rec_M = model.NUM_PARAMS
    config = load_config()
    slots = int(config["he"]["slots"])
    cost = json.load(open(os.path.join(ROOT, "costs", "cost_fixed.json"), encoding="utf-8"))
    instrument()

    records = []
    for policy, rounds in (("fas50", 2), ("full_he", 1)):   # p = 0,5 và p = 1 (bước 6/7 mới có nhiễu, xáo)
        cfg = dict(config, policy=policy)
        n0 = len(fused)
        recs = run_rounds(cfg, rounds, timeout=1800)
        fused_by_t = {t: fused[n0 + t] for t in range(1, rounds + 1)}
        for rec in recs:
            check_round(rec, slots, cost)
        records += recs
        shares_last = shares

    # 2. bs không có khóa bí mật
    for msg in bs_seen:
        mt = msg.get("type")
        assert mt in ALLOWED, f"bs nhận loại tin nhắn lạ: {mt}"
        extra = set(msg) - ALLOWED[mt]
        assert not extra, f"bs nhận trường lạ trong {mt}: {extra}"
    kinds = sorted({m.get('type') for m in bs_seen})
    print(f"[TEST] bs nhận {len(bs_seen)} tin nhắn, chỉ các loại công khai: {', '.join(kinds)}")
    bhe_attrs = {k: type(v).__name__ for k, v in vars(crypto_setup.BsHE(config)).items()}
    assert "PrivateKey" not in bhe_attrs.values(), f"BsHE giữ khóa bí mật: {bhe_attrs}"
    assert len(devices) == 4, f"mong 2 thiết bị × 2 lần chạy, có {len(devices)}"
    devs = devices[-2:]                      # 2 thiết bị của lần chạy cuối
    assert all(d._sk is not None for d in devs)
    sk = [crypto_setup.ser(d._sk) for d in devs]
    assert sk[0] != sk[1], "hai thiết bị trùng khóa bí mật"
    assert not any(s in m.get("pk", b"") for s in sk for m in bs_seen if m.get("type") in (P.HE_PK1, P.HE_PK))
    print(f"[TEST] khóa bí mật chỉ ở 2 thiết bị (khác nhau); BsHE chỉ có {sorted(set(bhe_attrs.values()))}")

    # 3. một phần giải mã đơn lẻ không đủ
    bhe = crypto_setup.BsHE(config)
    t_last = max(shares_last)
    one = shares_last[t_last][0]
    try:
        pt = bhe.cc.MultipartyDecryptFusion([crypto_setup.load_ct(one[0])])
        pt.SetLength(slots)
        vals = np.array(pt.GetRealPackedValue())
        want = (truth[(t_last, 1)] + truth[(t_last, 2)])[:slots]
        err1 = float(np.abs(vals[:len(want)] - want).max())
        assert err1 > 1e-2, f"chỉ với phần của một bên mà vẫn giải được (sai số {err1:.2e})"
        print(f"[TEST] chỉ với phần giải mã của một bên: sai số {err1:.2e} — không giải được")
    except RuntimeError as e:
        print(f"[TEST] chỉ với phần giải mã của một bên: OpenFHE từ chối ({str(e).splitlines()[0][-80:]})")

    setup = json.load(open(os.path.join(RESULTS_DIR, "setup.json"), encoding="utf-8"))
    print(f"[TEST] thiết lập HE: {setup['setup_s']:.2f} s, context {setup['size']['context']} B, "
          f"pk1 {setup['size']['pk1']} B, pk chung {setup['size']['pk_joint']} B")
    print(f"\n[TEST] ĐẠT: HE đa khóa giữa các bên — sai số < {TOL_HE:g}, bs không có khóa bí mật, "
          f"một phần giải mã không đủ, byte gói khớp bảng chi phí.")


if __name__ == "__main__":
    main()
