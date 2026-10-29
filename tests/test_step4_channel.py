# -*- coding: utf-8 -*-
"""Test bước 4 (CLAUDE.md mục 10.4): kênh thật + bóp tốc độ, cả `uav_noma_pc` và `direct_noma`.

Đạt khi:
  A. γ của uav/channel.py trùng reference_sim/channel.py (cùng rng, mọi chế độ, los_mode mean/random) — tests/ref/channel_ref.json.
  B. Mục 4.1: 2 thiết bị CÙNG LÚC gửi 5 MB qua uav ở 1 Mbit/s -> mỗi thiết bị ≈ 40 s (sai lệch ≤ 5%).
  C. R_k thật của vòng t từ channel_round, chế độ uav_noma_pc và direct_noma: gửi R_k × 10 s byte -> ≈ 10 s (≤ 5%).
  D. Một vòng FL đủ 7 bước với uav_noma_pc: thời gian truyền UPLOAD = byte / R_k (≤ 5%), log có thông lượng,
     measurement_ok = True.
  E. Máy giả lập quá tải: luồng chuyển tiếp của uav bị dừng 3 s giữa tin nhắn 10 s -> thời gian tin nhắn VẪN ≈ bytes/R_k
     (≤ 5%, định nhịp neo theo đầu tin nhắn gửi bù); check_measurement đánh dấu vòng lệch > 5%.
  F. Nén thời gian kênh (channel.time_scale): k cố định 50 và auto (kênh direct_noma rất chậm) -> thời gian uav báo
     (đã ×k) ≈ bytes/R_k và thời gian máy ≈ bytes/(R_k·k) (≤ 5%); auto chọn k = sàn / min R_k.
Chạy: python tests/test_step4_channel.py  (cần data/, weights/ cho phần D)
"""
import os
import sys
import json
import time
import socket
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_round import ROOT, load_config, run_rounds
from common import protocol as P
from uav import channel
from uav.relay import run_uav

TOL = 0.05
BS_PORT = 6100
IOT_PORTS = [7101, 7102]


def check_reference(config):
    ref = json.load(open(os.path.join(ROOT, "tests", "ref", "channel_ref.json"), encoding="utf-8"))
    worst = 0.0
    for key, rows in ref["gamma"].items():
        los_mode, mode = key.split("/")
        cfg = dict(config, channel=dict(config["channel"], los_mode=los_mode))
        Pm = channel.Params(cfg)
        for t, want in zip(ref["rounds"], rows):
            got = channel.draw(Pm, np.random.default_rng([ref["seed"], t]), 1)[mode][0]
            rel = float(np.max(np.abs(got - want) / np.abs(want)))
            worst = max(worst, rel)
            assert rel < 1e-9, f"{key} vòng {t}: γ {got} ≠ reference {want}"
    print(f"[TEST] A. γ trùng reference_sim: {len(ref['gamma'])} tổ hợp × {len(ref['rounds'])} vòng, sai khác tương đối tối đa {worst:.1e}")


def relay_transfer(config, rates, nbytes):
    """Hai thiết bị cùng lúc gửi một tin nhắn nbytes[k] qua uav (R_k = rates[k] byte/s).
    Trả về [(thời gian đầu-cuối đo tại bs, byte khung, tx_s theo uav)] cho 2 thiết bị."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", BS_PORT))
    srv.listen(8)
    errors = []

    def uav():
        try:
            run_uav("127.0.0.1", BS_PORT, "127.0.0.1", IOT_PORTS, 1, config,
                    channel_fn=lambda c, t: (0.0, 0.0, float(rates[0]), float(rates[1])))
        except Exception as e:   # noqa: BLE001
            errors.append(repr(e))
    tu = threading.Thread(target=uav, daemon=True)
    tu.start()

    # bs giả: control + 2 pipe
    conns = {}
    while len(conns) < 3:
        s, _ = srv.accept()
        m, _ = P.recv_msg(s)
        conns["uav" if m["role"] == "uav" else int(m["device_id"])] = s
        if m["role"] == "uav":
            # iot nối tới uav sau khi uav mở cổng
            iots = {}
            for k, port in enumerate(IOT_PORTS, 1):
                for _ in range(100):
                    try:
                        iots[k] = socket.create_connection(("127.0.0.1", port))
                        break
                    except OSError:
                        time.sleep(0.05)
                P.send_msg(iots[k], P.HELLO, {"role": "device", "device_id": k, "n_samples": 25000})
    srv.close()
    P.recv_msg(conns["uav"])                 # CHANNEL_REPORT: uav đã đặt R_k

    start = {}
    res = {}

    def send(k):
        start[k] = time.monotonic()
        P.send_msg(iots[k], P.UPLOAD, {"blob": b"\0" * nbytes[k - 1]})

    def recv(k):
        _, nb = P.recv_msg(conns[k])
        res[k] = (time.monotonic(), nb)
    th = [threading.Thread(target=f, args=(k,)) for k in (1, 2) for f in (recv, send)]
    for x in th:
        x.start()
    for x in th:
        x.join(timeout=600)
    P.send_msg(conns["uav"], P.ROUND_DONE, {"t": 1})
    st, _ = P.recv_msg(conns["uav"])
    for s in list(conns.values()) + list(iots.values()):
        s.close()
    tu.join(timeout=10)
    assert not errors, errors
    out = []
    for k in (1, 2):
        up = [it for it in st[f"dev{k}"] if it["type"] == P.UPLOAD]
        assert len(up) == 1, f"uav không ghi UPLOAD của thiết bị {k}: {st}"
        out.append((res[k][0] - start[k], res[k][1], up[0]["tx_s"]))
    return out


def check_timing(label, rates, results):
    for k, ((dt, nb, tx), R) in enumerate(zip(results, rates), 1):
        want = nb / R
        err = abs(dt - want) / want
        print(f"[TEST] {label} thiết bị {k}: {nb} B ở {R * 8 / 1e6:.3f} Mbit/s -> {dt:.2f} s "
              f"(lý thuyết {want:.2f} s, lệch {err * 100:.2f}%; uav đo {tx:.2f} s)")
        assert err <= TOL, f"{label} thiết bị {k}: lệch {err * 100:.2f}% > {TOL * 100:.0f}%"


def part_f(config):
    """Nén thời gian kênh: thời gian quy đổi (×k) = bytes/R_k; thời gian máy = bytes/(R_k·k)."""
    from uav.shaper import time_scale_for
    cases = [("k = 50", 50, [2500.0, 5000.0], 100.0),                       # 0,02 / 0,04 Mbit/s
             ("auto", "auto", [125.0, 6125.0], 400.0)]                      # direct_noma vòng 1: 0,001 / 0,049 Mbit/s
    for label, ts, rates, real_s in cases:
        cfg = dict(config, channel=dict(config["channel"], time_scale=ts))
        k = time_scale_for(cfg["channel"], *rates)
        if ts == "auto":
            want_k = float(cfg["channel"]["time_scale_floor_bps"]) / 8 / min(rates)
            assert abs(k - want_k) < 1e-9, f"auto chọn k = {k} ≠ {want_k}"
        nbytes = [int(r * real_s) for r in rates]
        res = relay_transfer(cfg, rates, nbytes)
        for i, ((wall, nb, tx), R) in enumerate(zip(res, rates), 1):
            err_tx = abs(tx - nb / R) / (nb / R)
            err_wall = abs(wall - nb / (R * k)) / (nb / (R * k))
            print(f"[TEST] F. {label} (k = {k:.0f}) thiết bị {i}: {nb} B ở {R * 8 / 1e6:.4f} Mbit/s -> uav báo {tx:.1f} s "
                  f"(thật {nb / R:.1f} s, lệch {err_tx * 100:.2f}%), máy chạy {wall:.2f} s (lệch {err_wall * 100:.2f}%)")
            assert err_tx <= TOL and err_wall <= TOL, f"F. {label}: lệch quá {TOL * 100:.0f}%"


def main():
    config = load_config()

    # A. mô hình kênh
    check_reference(config)

    # B. mục 4.1: 5 MB ở 1 Mbit/s, hai thiết bị cùng lúc
    R = 1e6 / 8
    check_timing("B. 5 MB @ 1 Mbit/s", [R, R], relay_transfer(config, [R, R], [5_000_000, 5_000_000]))

    # C. R_k thật của từng chế độ
    for mode, t in (("uav_noma_pc", 1), ("direct_noma", 3)):
        cfg = dict(config, channel=dict(config["channel"], mode=mode))
        g1, g2, R1, R2 = channel.channel_round(cfg, t)
        print(f"[TEST] C. {mode} vòng {t}: SINR {10 * np.log10(g1):.1f} / {10 * np.log10(g2):.1f} dB")
        check_timing(f"C. {mode}", [R1, R2], relay_transfer(cfg, [R1, R2], [int(R1 * 10), int(R2 * 10)]))

    # D. một vòng FL đủ 7 bước qua kênh uav_noma_pc
    cfg = dict(config, policy="no_he", channel=dict(config["channel"], mode="uav_noma_pc"))
    rec = run_rounds(cfg, 1, timeout=900)[0]
    for k in (1, 2):
        Rk = rec["channel"][f"R{k}"]
        up = next(it for it in rec["uplink"][f"dev{k}"] if it["type"] == P.UPLOAD)
        err = abs(up["ratio_to_R"] - 1)
        print(f"[TEST] D. vòng FL thiết bị {k}: UPLOAD {up['bytes']} B, {up['tx_s']:.2f} s, "
              f"thông lượng {up['throughput_Bps'] * 8 / 1e6:.3f} Mbit/s / R_k {Rk * 8 / 1e6:.3f} Mbit/s (lệch {err * 100:.2f}%)")
        assert err <= TOL, f"thông lượng lệch R_k {err * 100:.2f}%"
    assert rec["measurement_ok"] and not rec["measurement_issues"], rec["measurement_issues"]
    print("[TEST] D. measurement_ok = True")

    # E. máy quá tải: dừng luồng uav 3 s giữa tin nhắn
    import types
    import uav.relay as relay_mod
    from bs.server import check_measurement
    orig_P = relay_mod.P
    spy = types.ModuleType("protocol_stall")
    spy.__dict__.update(P.__dict__)
    state = {"seen": 0, "stalled": False}

    def recv_exactly(sock, n):
        b = P.recv_exactly(sock, n)
        state["seen"] += len(b)
        if not state["stalled"] and state["seen"] > 2 * 125_000:     # ~2 s vào tin nhắn: luồng uav bị "bỏ đói" 3 s
            state["stalled"] = True
            time.sleep(3.0)
        return b
    spy.recv_exactly = recv_exactly
    relay_mod.P = spy
    try:
        R = 1e6 / 8
        res = relay_transfer(config, [R, R], [1_250_000, 1_250_000])
    finally:
        relay_mod.P = orig_P
    assert state["stalled"], "không chèn được lần dừng"
    check_timing("E. dừng 3 s giữa tin nhắn 10 s", [R, R], res)
    bad = {"uplink": {"dev1": [{"type": P.UPLOAD, "bytes": 814784, "ratio_to_R": 0.879}]}}
    assert check_measurement(bad) is False and len(bad["measurement_issues"]) == 1
    ok = {"uplink": {"dev1": [{"type": P.UPLOAD, "bytes": 814784, "ratio_to_R": 0.99},
                              {"type": P.METRICS, "bytes": 53, "ratio_to_R": 21.8}]}}
    assert check_measurement(ok) is True
    print("[TEST] E. check_measurement: tỉ lệ 0,879 -> không tin cậy; 0,99 (+ METRICS nhỏ) -> tin cậy")

    # F. nén thời gian kênh
    part_f(config)
    print(f"\n[TEST] ĐẠT: kênh trùng reference_sim; bóp tốc độ đúng R_k (≤ {TOL * 100:.0f}%) cho uav_noma_pc và direct_noma, "
          f"kể cả khi uav bị dừng giữa tin nhắn.")


if __name__ == "__main__":
    main()
