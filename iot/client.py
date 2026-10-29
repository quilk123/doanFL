# -*- coding: utf-8 -*-
"""Thiết bị IoT (client FL) — thiết lập HE (mục 5) rồi trình tự một vòng (mục 6).

Huấn luyện thật (bước 2), HE đa khóa OpenFHE thật (bước 5), clipping + nhiễu + xáo π_t theo policy (bước 6, iot/protect.py).
Mọi iot ↔ bs đi qua uav (iot chỉ nối tới uav). Thời gian từng pha gửi kèm tin nhắn cho bs ghi log (mục 9).
"""
import os
import time
import socket
import numpy as np

from common import protocol as P
from common import model
from iot import train, protect, he, evaluate


def _connect(host, port, timeout=20.0):
    """Nối tới uav, thử lại cho tới khi uav mở cổng (dùng khi khởi động đồng thời).

    docker compose khởi động 4 container gần như cùng lúc nên iot có thể chạy trước khi uav mở cổng: thử lại mỗi
    0,1 s trong tối đa `timeout` giây. TCP_NODELAY: gửi tin nhắn nhỏ ngay, không gom gói (đo thời gian chính xác hơn).
    """
    t0 = time.monotonic()
    while True:
        try:
            s = socket.create_connection((host, port), timeout=5.0)
            s.settimeout(None)   # timeout 5 s chỉ cho lúc nối; đọc thì chờ (vòng có thể lâu: huấn luyện, kênh xấu)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except OSError:
            if time.monotonic() - t0 > timeout:
                raise
            time.sleep(0.1)


def _expect(sock, mtype):
    """Nhận một tin nhắn và bắt buộc nó đúng loại `mtype` (giao thức đi theo trình tự cố định); sai thì dừng."""
    msg, _ = P.recv_msg(sock)
    if msg.get("type") != mtype:
        raise RuntimeError(f"mong {mtype}, nhận {msg.get('type')}")
    return msg


def he_setup(sock, device_id, config):
    """Thiết lập HE đa khóa (mục 5.1–5.2): nhận context của bs; iot1 KeyGen -> pk1; iot2 MultipartyKeyGen(pk1) -> pk chung.
    Khóa bí mật ở lại trong DeviceHE.

    Chạy MỘT lần trước vòng 1. Trình tự tin nhắn (mọi tin nhắn đi qua uav, bs chuyển tiếp giữa 2 iot):
      bs -> iot1, iot2 : HE_CONTEXT (tham số CKKS)
      iot1 -> bs -> iot2: HE_PK1 (khóa công khai riêng của iot1)
      iot2 -> bs        : HE_PK  (khóa công khai CHUNG, sinh từ pk1 và khóa bí mật của iot2)
      bs -> iot1, iot2  : HE_PK  (phát lại khóa chung để cả hai cùng mã hóa bằng nó)
    Trả về DeviceHE chứa khóa bí mật của thiết bị này và khóa công khai chung.
    """
    t0 = time.monotonic()
    msg = _expect(sock, P.HE_CONTEXT)
    dhe = he.DeviceHE(msg["cc"], config["he"]["slots"], lead=(device_id == 1))
    if device_id == 1:
        P.send_msg(sock, P.HE_PK1, {"pk": dhe.keygen_lead()})
        dhe.set_joint_pk(_expect(sock, P.HE_PK)["pk"])
    else:
        pk = dhe.keygen_main(_expect(sock, P.HE_PK1)["pk"])
        P.send_msg(sock, P.HE_PK, {"pk": pk})
        _expect(sock, P.HE_PK)                 # bs phát lại pk chung cho cả hai
    print(f"[iot{device_id}] thiết lập HE xong ({time.monotonic() - t0:.2f}s)", flush=True)
    return dhe


def run_iot(device_id, uav_host, uav_port, config, perm_seed, is_evaluator):
    """Vòng đời của một thiết bị IoT: nối uav -> thiết lập HE -> lặp các vòng FL cho tới khi bs đóng kết nối.

    Đầu vào:
      device_id   : 1 hoặc 2 (iot1 là "lead" trong HE đa bên và là bên đánh giá độ chính xác)
      uav_host/port: địa chỉ uav — iot KHÔNG nối thẳng tới bs, mọi tin nhắn đi qua uav (uav bóp tốc độ theo kênh)
      perm_seed   : K (32 byte, secrets/perm_seed) để sinh π_t — chỉ iot có
      is_evaluator: True với iot1 (chỉ iot1 được mount val + test; cả 2 iot đo trên train của mình)
    Mỗi vòng t (bs quyết p, iot làm theo):
      ROUND_START{t, p, policy, n_samples} -> huấn luyện 1 epoch -> Δ -> ×n_k/n (FedAvg) -> bảo vệ -> UPLOAD
      CSUM (tổng ciphertext của 2 iot) -> giải mã từng phần -> PARTIAL_DEC
      AGGREGATE{tổng phần HE, tổng phần rõ đã xáo} -> giải xáo -> w = w + Σ n_k/n·Δ_k -> METRICS (cả 2 iot: acc trên train của mình; iot1 thêm val + test)
    Số ảnh n_k của thiết bị gửi cho bs trong HELLO; bs gửi lại [n_1, n_2] trong ROUND_START để mỗi iot tự tính trọng số.
    Thời gian từng pha (train/enc/pdec/update) gửi kèm tin nhắn để bs quy đổi sang thang Pi 4.
    Không biết trước số vòng: bs đóng kết nối sau vòng cuối -> recv lỗi -> thoát vòng lặp.
    """
    M = model.num_params(config)
    w = model.init_vec(config)   # w0 tự dựng, giống hệt nhau ở 2 iot (mục 5.4)

    n_k = int(len(train.load_client(device_id)[1]))   # số ảnh của thiết bị (trọng số FedAvg) — chỉ con số, không gửi dữ liệu
    sock = _connect(uav_host, uav_port)
    P.send_msg(sock, P.HELLO, {"role": "device", "device_id": device_id, "n_samples": n_k})
    print(f"[iot{device_id}] đã nối uav {uav_host}:{uav_port}", flush=True)

    # nguồn nhiễu RIÊNG của thiết bị: entropy hệ điều hành (không từ K, không từ seed trong config mà bs cũng có)
    noise_rng = np.random.default_rng(np.random.SeedSequence())

    try:
        dhe = he_setup(sock, device_id, config)
        while True:
            try:
                msg, _ = P.recv_msg(sock)
            except (ConnectionError, OSError):
                break  # bs đã kết thúc các vòng và đóng kết nối

            if msg.get("type") != P.ROUND_START:
                continue
            t = msg["t"]; p = float(msg["p"]); policy = msg.get("policy", config.get("policy", "fixed10"))
            ns = msg.get("n_samples")
            P.expect(ns is not None and len(ns) == 2, f"[iot{device_id}] ROUND_START thiếu n_samples (FedAvg)")
            weight = ns[device_id - 1] / float(sum(ns))   # trọng số FedAvg n_k / (n_1 + n_2)

            log = lambda s: print(f"[iot{device_id}] {s}", flush=True)   # noqa: E731
            log(f"vòng {t}: nhận ROUND_START — policy {policy}, p = {p} (mã hóa {p * 100:.0f}%), trọng số FedAvg {weight:.3f}")
            # Bước 3: huấn luyện -> Δ -> chia HE/phần rõ -> mã hóa
            t0 = time.monotonic()
            delta = train.local_train(w, config, device_id, t, log=log)
            train_s = time.monotonic() - t0
            log(f"vòng {t}: huấn luyện xong {train_s:.1f} s (thô); ‖Δ‖ = {np.linalg.norm(delta):.3f}")
            cts, shuf, idx_he, idx_plain, info = protect.protect(
                dhe, delta, p, config, perm_seed, t, device_id, policy, noise_rng, weight=weight)
            log(f"vòng {t}: bảo vệ — mã hóa HE {len(idx_he)} tham số = {len(cts)} ciphertext ({info['enc_s']:.2f} s); "
                f"phần rõ {len(idx_plain)} tham số" + (f", nhiễu sd {info['sd']:.5f}" if info["sd"] else "")
                + (", xáo vị trí π_t" if info["pi"] is not None else "") + " -> gửi UPLOAD qua uav")
            P.send_msg(sock, P.UPLOAD, {
                "t": t,
                "ciphertexts": cts,                       # bytes OpenFHE Serialize(BINARY)
                "shuffled": P.pack_array(shuf),
                "n_ciphertext": len(cts),
                "n_he": int(len(idx_he)),                 # số giá trị đã mã hóa (bs cần để cắt sau khi ghép)
                "timing": {"train_s": train_s, "enc_s": info["enc_s"]},
            })

            # Bước 4->5: nhận CSUM -> giải mã từng phần bằng khóa bí mật của mình
            cmsg = _expect(sock, P.CSUM)
            t0 = time.monotonic()
            shares = dhe.partial_decrypt(cmsg["ciphertexts"])
            pdec_s = time.monotonic() - t0
            if shares:
                log(f"vòng {t}: nhận CSUM (tổng ciphertext 2 thiết bị) -> giải mã từng phần {len(shares)} khối ({pdec_s:.2f} s)")
            P.send_msg(sock, P.PARTIAL_DEC, {"t": t, "delta": shares, "timing": {"pdec_s": pdec_s}})

            # Bước 6->7: nhận AGGREGATE -> giải xáo -> cập nhật
            amsg = _expect(sock, P.AGGREGATE)
            t0 = time.monotonic()
            he_sum = P.unpack_array(amsg["he_part"])
            shuffled_sum = P.unpack_array(amsg["shuffled_sum"])
            if amsg.get("unshuffled"):   # fas: bs (có K) đã giải xáo rồi cộng
                plain_sum = shuffled_sum
            else:                        # giải xáo MỘT lần bằng π_t⁻¹
                plain_sum = protect.unshuffle(shuffled_sum, info["pi"])

            # ráp lại tổng cập nhật đầy đủ: vị trí phần HE lấy từ tổng đã giải mã, vị trí phần rõ lấy từ tổng đã giải xáo
            update = np.zeros(M, dtype=np.float64)
            update[idx_he] = he_sum
            update[idx_plain] = plain_sum
            # FedAvg: mỗi thiết bị đã nhân Δ_k với n_k/n trước khi gửi, bs chỉ cộng => tổng là trung bình có trọng số
            # w_mới = w + Σ n_k/n · Δ_k  (IID n_1 = n_2: trùng w + (Δ1 + Δ2)/2 trước đây)
            w = w + update
            update_s = time.monotonic() - t0
            log(f"vòng {t}: nhận AGGREGATE -> " + ("" if amsg.get("unshuffled") else "giải xáo, ")
                + f"cập nhật mô hình chung w <- w + tổng (‖tổng‖ = {np.linalg.norm(update):.3f})")

            # đánh giá trên CẢ 2 thiết bị: mỗi iot đo mô hình chung trên tập train của mình; iot1 thêm val + test
            m = evaluate.evaluate_train(w, config, device_id)
            if is_evaluator:
                m.update(evaluate.evaluate(w, config, t))   # accuracy/loss: TEST; val_accuracy/val_loss: VALIDATION
            P.send_msg(sock, P.METRICS, dict(m, t=t, timing={"update_s": update_s}))
            extra = f" test={m['accuracy']:.4f} val={m['val_accuracy']:.4f}" if is_evaluator else ""
            print(f"[iot{device_id}] xong vòng {t} (p={p}) acc train(thiết bị {device_id})={m['train_accuracy']:.4f}{extra}",
                  flush=True)
    finally:
        try:
            sock.close()
        except OSError:
            pass
    print(f"[iot{device_id}] kết thúc", flush=True)


def main():
    """Điểm vào của container iot: đọc biến môi trường (DEVICE_ID, UAV_HOST, UAV_PORT, CONFIG, PERM_SEED do
    docker-compose.yml đặt), đọc config.yaml và khóa xáo K (Docker secret), rồi chạy run_iot."""
    import yaml
    device_id = int(os.environ["DEVICE_ID"])
    uav_host = os.environ.get("UAV_HOST", "uav")
    uav_port = int(os.environ.get("UAV_PORT", "7001"))
    with open(os.environ.get("CONFIG", "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    seed_path = os.environ.get("PERM_SEED", "secrets/perm_seed")
    with open(seed_path, "rb") as f:
        perm_seed = f.read()
    run_iot(device_id, uav_host, uav_port, config, perm_seed,
            is_evaluator=(device_id == 1))


if __name__ == "__main__":
    main()
