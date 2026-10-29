# -*- coding: utf-8 -*-
"""Test độ bền giao thức (không huấn luyện, chạy vài giây):
  a. Không ai kết nối -> bs báo ProtocolError sau `round_timeout_min_s`, không treo.
  b. Thiết bị hoàn tất thiết lập HE (OpenFHE thật) rồi "treo" không gửi UPLOAD -> bs báo ProtocolError (timeout vòng).
  c. Sai giao thức (thiết bị gửi sai loại tin nhắn) -> ProtocolError, kể cả khi chạy python -O (không dựa vào assert).
Chạy: python tests/test_robustness.py   (và python -O tests/test_robustness.py)
"""
import os
import sys
import time
import socket
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_round import ROOT, load_config, RESULTS_DIR, CAPTURE_DIR
from common import protocol as P
from bs.server import run_bs, _accept_all
from iot.he import DeviceHE

PORT = 6200


def fake_peers(config, mode):
    """uav giả (gửi CHANNEL_REPORT) + 2 thiết bị giả: làm thiết lập HE thật, rồi treo (mode='hang')
    hoặc gửi sai loại tin nhắn ở bước thiết lập (mode='wrong')."""
    def connect():
        for _ in range(100):
            try:
                return socket.create_connection(("127.0.0.1", PORT))
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("không nối được bs")
    socks = []
    ctl = connect()
    socks.append(ctl)
    P.send_msg(ctl, P.HELLO, {"role": "uav"})
    devs = {}
    for k in (1, 2):
        devs[k] = connect()
        socks.append(devs[k])
        P.send_msg(devs[k], P.HELLO, {"role": "device", "device_id": k, "n_samples": 25000})

    def device(k):
        s = devs[k]
        try:
            cc = P.recv_msg(s)[0]["cc"]
            dhe = DeviceHE(cc, config["he"]["slots"], lead=(k == 1))
            if k == 1:
                if mode == "wrong":
                    P.send_msg(s, P.METRICS, {"t": 0})          # sai: bs mong HE_PK1
                    return
                P.send_msg(s, P.HE_PK1, {"pk": dhe.keygen_lead()})
                P.recv_msg(s)
            else:
                pk = dhe.keygen_main(P.recv_msg(s)[0]["pk"])
                P.send_msg(s, P.HE_PK, {"pk": pk})
                P.recv_msg(s)
            P.recv_msg(s)                                        # ROUND_START ... rồi treo, không gửi UPLOAD
            time.sleep(30)
        except (OSError, ConnectionError):
            pass
    for k in (1, 2):
        threading.Thread(target=device, args=(k,), daemon=True).start()
    if mode == "hang":
        P.send_msg(ctl, P.CHANNEL_REPORT, {"t": 1, "g1": 10.0, "g2": 10.0, "R1": 2e5, "R2": 2e5, "mode": "uav_noma_pc"})
    return socks


def run_case(config, mode):
    err = []
    t0 = time.monotonic()

    def bs():
        try:
            run_bs("127.0.0.1", PORT, 1, config, RESULTS_DIR, CAPTURE_DIR)
        except Exception as e:   # noqa: BLE001
            err.append(e)
    tb = threading.Thread(target=bs, daemon=True)
    tb.start()
    time.sleep(0.3)
    socks = fake_peers(config, mode)
    tb.join(timeout=60)
    for s in socks:
        s.close()
    return err, time.monotonic() - t0, tb.is_alive()


def main():
    config = load_config()
    # a. không ai kết nối
    t0 = time.monotonic()
    try:
        _accept_all("127.0.0.1", PORT, timeout=1.0)
        raise SystemExit("[TEST] LỖI: _accept_all không báo timeout")
    except P.ProtocolError as e:
        print(f"[TEST] a. không ai kết nối -> ProtocolError sau {time.monotonic() - t0:.1f} s: {e}")

    # b. thiết bị treo giữa vòng
    cfg = dict(config, policy="no_he", protocol={"round_timeout_min_s": 3, "round_timeout_factor": 0.01})
    err, dt, alive = run_case(cfg, "hang")
    if alive or not err or not isinstance(err[0], P.ProtocolError):
        raise SystemExit(f"[TEST] LỖI b: alive={alive} err={err}")
    print(f"[TEST] b. thiết bị treo sau ROUND_START -> bs dừng sau {dt:.1f} s: {str(err[0])[:90]}…")

    # c. sai giao thức
    cfg = dict(config, policy="no_he", protocol={"round_timeout_min_s": 10, "round_timeout_factor": 1})
    err, dt, alive = run_case(cfg, "wrong")
    if alive or not err or not isinstance(err[0], P.ProtocolError):
        raise SystemExit(f"[TEST] LỖI c: alive={alive} err={err}")
    print(f"[TEST] c. sai loại tin nhắn -> ProtocolError sau {dt:.1f} s: {err[0]} "
          f"(assert {'TẮT' if not __debug__ else 'bật'})")
    print("\n[TEST] ĐẠT: bs không treo khi thiếu kết nối / thiết bị treo / sai giao thức.")


if __name__ == "__main__":
    main()
