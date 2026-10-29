# -*- coding: utf-8 -*-
"""Trạm gốc (BS) — điều phối vòng FL (CLAUDE.md mục 6, 9).

Nhận 3 kết nối (qua chang2): control của uav + 2 pipe thiết bị (uav mở hộ).
Mỗi vòng ghi một bản ghi JSON (byte thật, thời gian thật).
bs KHÔNG giữ mô hình đầy đủ, khóa bí mật HE, hay khóa xáo.
HE (bước 5): bs tạo CryptoContext, chuyển khóa công khai giữa 2 iot, cộng đồng cấu, ghép 2 phần giải mã
(chỉ ra TỔNG phần HE — giới hạn đã chấp nhận ở mục 2).
"""
import os
import time
import socket
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from common import protocol as P
from common.crypto_setup import BsHE
from common.perm import perm_for, unshuffle
from bs import policy, aggregate
from bs.logger import Logger


def _accept_all(bs_host, bs_port, timeout=None):
    """Lắng nghe và nhận: control (uav) + device 1 + device 2 (quá `timeout` s thì báo lỗi).

    Cả 3 kết nối đều do uav mở (bs chỉ nằm trên mạng chang2 cùng uav, không thấy iot). Mỗi kết nối gửi HELLO đầu
    tiên để bs biết đó là kênh điều khiển của uav hay đường ống của thiết bị nào. HELLO của thiết bị có số ảnh n_k
    (trọng số FedAvg). Trả (control, dev1, dev2, [n_1, n_2]).
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bs_host, bs_port))
    srv.listen(8)
    srv.settimeout(timeout)
    print(f"[bs] lắng nghe {bs_host}:{bs_port}", flush=True)

    control = None
    devices = {}
    n_samples = {}
    while control is None or len(devices) < 2:
        try:
            sock, _ = srv.accept()
        except TimeoutError as e:
            raise P.ProtocolError(f"[bs] quá {timeout:.0f} s chờ kết nối: uav={'có' if control else 'CHƯA'}, "
                                  f"thiết bị {sorted(devices)}") from e
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        msg, _ = P.recv_msg(sock)   # HELLO
        if msg.get("role") == "uav":
            control = sock
            print("[bs] control (uav) đã nối", flush=True)
        elif msg.get("role") == "device":
            did = int(msg["device_id"])
            P.expect(int(msg.get("n_samples") or 0) > 0, f"[bs] HELLO thiết bị {did} thiếu n_samples (FedAvg)")
            devices[did] = sock
            n_samples[did] = int(msg["n_samples"])
            print(f"[bs] device {did} đã nối ({n_samples[did]} ảnh)", flush=True)
    srv.close()
    return control, devices[1], devices[2], [n_samples[1], n_samples[2]]


_pool = ThreadPoolExecutor(max_workers=2)


def _recv_both(dev1, dev2):
    """Nhận một tin nhắn từ MỖI thiết bị SONG SONG (như trạm gốc thật). Đọc lần lượt sẽ làm ứ bộ đệm TCP
    uav -> bs của thiết bị gửi trước khi thiết bị kia chậm, uav phải dừng chuyển tiếp => sai thời gian truyền."""
    f1 = _pool.submit(P.recv_msg, dev1)
    f2 = _pool.submit(P.recv_msg, dev2)
    return f1.result(), f2.result()


def he_setup(bhe, dev1, dev2, log, extra=None):
    """Thiết lập HE đa khóa (mục 5.1–5.2). bs chỉ thấy context và khóa công khai.

    Phía bs của trình tự trong iot/client.py he_setup: gửi HE_CONTEXT cho 2 iot -> nhận HE_PK1 của iot1, chuyển cho
    iot2 -> nhận HE_PK (khóa chung) của iot2, phát cho cả hai. Ghi thời gian + số byte vào setup.json.
    """
    t0 = time.monotonic()
    b_cc = [P.send_msg(d, P.HE_CONTEXT, {"cc": bhe.cc_bytes}) for d in (dev1, dev2)]
    m1, b_pk1 = P.recv_msg(dev1)
    P.expect(m1.get("type") == P.HE_PK1, f"[bs] mong HE_PK1, nhận {m1.get('type')}")
    b_pk1_fwd = P.send_msg(dev2, P.HE_PK1, {"pk": m1["pk"]})
    m2, b_pk = P.recv_msg(dev2)
    P.expect(m2.get("type") == P.HE_PK, f"[bs] mong HE_PK, nhận {m2.get('type')}")
    b_pk_fwd = [P.send_msg(d, P.HE_PK, {"pk": m2["pk"]}) for d in (dev1, dev2)]
    rec = {"setup_s": time.monotonic() - t0,
           "bytes": {"HE_CONTEXT": sum(b_cc), "HE_PK1": b_pk1 + b_pk1_fwd, "HE_PK": b_pk + sum(b_pk_fwd)},
           "size": {"context": len(bhe.cc_bytes), "pk1": len(m1["pk"]), "pk_joint": len(m2["pk"])}}
    rec.update(extra or {})                 # vd. aggregation: FedAvg, số ảnh, trọng số
    log.write_setup(rec)
    print(f"[bs] thiết lập HE xong ({rec['setup_s']:.2f}s): context {len(bhe.cc_bytes)} B, "
          f"pk {len(m2['pk'])} B", flush=True)


def check_measurement(record, tol=0.05, min_bytes=65536):
    """Cờ tin cậy của phép đo đường truyền: mọi tin nhắn chiều lên ≥ min_bytes phải có thông lượng trong R_k·(1 ± tol).
    Lệch hơn => máy giả lập quá tải làm sai phép đo (không phải kênh) -> vòng đánh dấu measurement_ok = False."""
    issues = []
    for dev, items in record["uplink"].items():
        for it in items:
            r = it.get("ratio_to_R")
            if it["bytes"] >= min_bytes and r is not None and abs(r - 1) > tol:
                issues.append({"dev": dev, "type": it["type"], "bytes": it["bytes"], "ratio_to_R": r})
    record["measurement_ok"] = not issues
    record["measurement_issues"] = issues
    return record["measurement_ok"]


def _uplink_tx(record, k, mtype):
    """Thời gian truyền chiều lên (s, trên kênh) của tin nhắn loại `mtype` từ thiết bị k trong vòng; 0 nếu không có."""
    return next((it["tx_s"] for it in record["uplink"][f"dev{k}"] if it["type"] == mtype), 0.0)


def pi4_timing(record, F, cm, split="iid"):
    """Thời gian vòng quy về Pi 4 (quyết định 2a), cùng thang với choose_p:
      max_k(F·(train_k + enc_k) + tx UPLOAD_k) + bs cộng + max_k(F·pdec_k + tx PARTIAL_DEC_k) + bs ghép + F·update.
    Theo quyết định 1a, huấn luyện lấy T_train của bảng chi phí (1 thiết bị chạy một mình) thay cho thời gian đo được khi
    2 iot chạy song song trên MỘT máy; phần chậm thêm ghi riêng ở `sim_artifact` (sai lệch do giả lập).
    T_train RIÊNG từng thiết bị theo phân hoạch `split` (cm.train_times: non-IID khác số ảnh => khác thời gian).
    Không tính: đường xuống (lý tưởng), đánh giá trên tập test (đo lường, không thuộc giao thức)."""
    tm = record["time"]
    Tk = cm.train_times(split)
    bench_raw = [T / F for T in Tk]
    tm["train_pi4_table_s"] = list(Tk)                   # T_train từng thiết bị đã dùng (bảng chi phí, thang Pi 4)
    up = [F * (tm[f"dev{k}"]["train_s"] + tm[f"dev{k}"]["enc_s"]) + _uplink_tx(record, k, P.UPLOAD) for k in (1, 2)]
    up_c = [Tk[k - 1] + F * tm[f"dev{k}"]["enc_s"] + _uplink_tx(record, k, P.UPLOAD) for k in (1, 2)]
    pdec = [F * tm[f"dev{k}"].get("pdec_s", 0.0) + _uplink_tx(record, k, P.PARTIAL_DEC) for k in (1, 2)]
    rest = tm["he_add_s"] + max(pdec) + tm["he_fuse_s"] + F * tm["dev1"].get("update_s", 0.0)
    train = [tm[f"dev{k}"]["train_s"] for k in (1, 2)]
    tm["round_pi4_s"] = max(up_c) + rest                 # dùng để xét trễ hạn
    tm["round_pi4_measured_train_s"] = max(up) + rest    # nếu lấy thời gian huấn luyện đo được (gồm sai lệch giả lập)
    record["sim_artifact"] = {
        "note": "2 iot huấn luyện song song trên một máy: chậm hơn 1 thiết bị chạy một mình (bảng chi phí) — "
                "sai lệch do giả lập, không phải của thiết bị thật",
        "train_raw_s": train,
        "train_bench_raw_s": bench_raw,
        "slowdown_pct": [100.0 * (x / b - 1) for x, b in zip(train, bench_raw)],
    }


def run_bs(bs_host, bs_port, rounds, config, results_dir, capture_dir, perm_seed=None):
    """Điều phối toàn bộ lượt chạy FL: nhận kết nối -> thiết lập HE -> `rounds` vòng -> đóng kết nối (iot tự thoát).

    Mỗi vòng (7 bước, CLAUDE.md mục 6):
      1. CHANNEL_REPORT từ uav: SINR, R_k của vòng
      2. policy.decide: p (cố định theo policy; adaptive: theo SINR) -> ROUND_START{t, p, policy, n_samples} (không gửi mô hình)
      3. nhận 2 UPLOAD song song (ciphertext phần HE + phần rõ đã xáo); lưu capture/ để đánh giá tấn công
      4. cộng đồng cấu 2 phần HE -> CSUM cho 2 iot
      5. nhận 2 PARTIAL_DEC (phần giải mã của từng iot)
      6. ghép -> tổng phần HE; cộng mù phần rõ -> AGGREGATE cho 2 iot
      7. METRICS từ CẢ 2 iot (acc trên train từng thiết bị; iot1 thêm validation + test)
    Sau đó ROUND_DONE -> uav trả UAV_STATS (byte, thời gian truyền thật) -> tính thời gian thang Pi 4, cờ trễ hạn,
    cờ tin cậy phép đo -> ghi results/round_XXXX.json.
    perm_seed (K) chỉ được có với fas10/fas50 — kiểm tra ngay đầu hàm (fixed10: K chỉ ở thiết bị).
    FedAvg: iot nhân Δ_k với n_k/n (n_k gửi trong HELLO); bs giữ nguyên phép cộng, ghi "aggregation" vào log.
    """
    os.makedirs(capture_dir, exist_ok=True)
    F = float(config.get("fidelity", 1.4))
    log = Logger(results_dir, fidelity=F)
    cm = policy.cost_model()
    t_max = policy.resolve_t_max(config, cm)
    pol = config.get("policy", "fixed10")
    fas = pol.startswith("fas")
    # bs chỉ có K với fas10/fas50 (mô phỏng FAS gốc: máy chủ giữ khóa xáo, giải xáo từng thiết bị rồi cộng).
    # fixed10 (phương án chính) và adaptive: K CHỈ ở iot1/iot2 — bs cộng mù vector đã xáo.
    P.expect((perm_seed is not None) == fas,
             f"[bs] policy {pol}: bs {'PHẢI' if fas else 'KHÔNG ĐƯỢC'} có K (docker-compose.fas.yml chỉ dùng cho fas)")
    sel = (f", chọn p với t_max − δ = {policy.t_select(config, cm):.2f}s (adaptive — hướng phát triển)"
           if pol == "adaptive" else f", p cố định = {policy.fixed_p(pol)}")
    print(f"[bs] policy={pol} t_max={t_max:.2f}s (thang Pi 4){sel}{' — bs CÓ K (FAS gốc)' if fas else ''}", flush=True)

    proto = config.get("protocol", {})
    t_min = float(proto.get("round_timeout_min_s", 600))
    t_fac = float(proto.get("round_timeout_factor", 5))
    control, dev1, dev2, n_samples = _accept_all(bs_host, bs_port, timeout=t_min)
    # FedAvg: mỗi iot tự nhân Δ với n_k/n trước khi gửi; bs CHỈ CỘNG (HE + vector đã xáo) => tổng = trung bình có trọng số
    fedavg = {"method": "FedAvg (iot nhân Δ_k với n_k/n trước khi gửi; bs cộng; iot: w <- w + tổng)",
              "n_samples": n_samples, "weights": [n / float(sum(n_samples)) for n in n_samples]}
    bhe = BsHE(config)
    for s in (control, dev1, dev2):
        s.settimeout(t_min)                 # thiết lập HE + chờ CHANNEL_REPORT

    try:
        he_setup(bhe, dev1, dev2, log, extra={"aggregation": fedavg})
        for _ in range(rounds):
            # Bước 1: nhận CHANNEL_REPORT
            cr, b_cr = P.recv_msg(control)
            t = cr["t"]; g1 = cr["g1"]; g2 = cr["g2"]; R1 = cr["R1"]; R2 = cr["R2"]
            tr0 = time.monotonic()

            # Bước 2: chọn p, gửi ROUND_START (KHÔNG gửi mô hình)
            dec = policy.decide(config, g1, g2, t)
            p = dec["p"]
            # timeout cả vòng: max(min, hệ số × thời gian vòng dự báo) — kênh chậm thì tự nới; một bên treo => lỗi rõ
            t_out = max(t_min, t_fac * dec["T_pred"])
            for s in (control, dev1, dev2):
                s.settimeout(t_out)
            rs = {"t": t, "p": p, "policy": pol,   # policy -> cách bảo vệ ở iot (mục 8)
                  "n_samples": n_samples}          # số ảnh 2 thiết bị -> iot tự tính trọng số FedAvg n_k/n
            b_rs1 = P.send_msg(dev1, P.ROUND_START, rs)
            b_rs2 = P.send_msg(dev2, P.ROUND_START, rs)

            # Bước 3: chờ đủ 2 UPLOAD
            tw0 = time.monotonic()
            (up1, b_up1), (up2, b_up2) = _recv_both(dev1, dev2)
            t_wait = time.monotonic() - tw0

            cts1 = up1["ciphertexts"]; shuf1 = P.unpack_array(up1["shuffled"])
            cts2 = up2["ciphertexts"]; shuf2 = P.unpack_array(up2["shuffled"])
            n_ct = len(cts1)
            n_he = int(up1["n_he"])
            P.expect(len(cts2) == n_ct and int(up2["n_he"]) == n_he, "[bs] hai thiết bị khác số ciphertext")

            # Lưu nguyên UPLOAD để phân tích tấn công offline (mục 9, capture/)
            cap = dict(shuffled_dev1=shuf1, shuffled_dev2=shuf2)
            if fas:   # FAS: bs giải xáo TỪNG thiết bị -> đọc được phần rõ (đã nhiễu) của từng thiết bị
                pi = perm_for(perm_seed, t, len(shuf1))
                cap.update(bs_view_dev1=unshuffle(shuf1, pi), bs_view_dev2=unshuffle(shuf2, pi))
            np.savez(os.path.join(capture_dir, f"upload_{t:04d}.npz"), **cap)

            # Bước 4: cộng đồng cấu phần HE -> CSUM
            tc0 = time.monotonic()
            csum = bhe.add(cts1, cts2)
            t_add = time.monotonic() - tc0
            b_cs1 = P.send_msg(dev1, P.CSUM, {"t": t, "ciphertexts": csum})
            b_cs2 = P.send_msg(dev2, P.CSUM, {"t": t, "ciphertexts": csum})

            # Bước 5: nhận PARTIAL_DEC
            (pd1, b_pd1), (pd2, b_pd2) = _recv_both(dev1, dev2)

            # Bước 6: ghép δ_1, δ_2 -> tổng phần HE; cộng mù vector đã xáo -> AGGREGATE
            tf0 = time.monotonic()
            he_sum = bhe.fuse(pd1["delta"], pd2["delta"], n_he)
            t_fuse = time.monotonic() - tf0
            # bs BIẾT tổng phần HE sau khi ghép (giới hạn đã chấp nhận, mục 2) -> lưu để đánh giá rò rỉ (tools/attack_eval.py)
            np.savez(os.path.join(capture_dir, f"upload_{t:04d}.npz"), he_sum=he_sum.astype(np.float32), **cap)
            if fas:   # cộng sau khi giải xáo, gửi tổng ĐÚNG thứ tự (thiết bị không giải xáo nữa)
                shuf_sum = aggregate.shuffled_sum([cap["bs_view_dev1"], cap["bs_view_dev2"]])
            else:     # cộng mù theo vị trí
                shuf_sum = aggregate.shuffled_sum([shuf1, shuf2])
            agg = {"t": t, "he_part": P.pack_array(he_sum, np.float64),
                   "shuffled_sum": P.pack_array(shuf_sum), "unshuffled": fas}
            b_ag1 = P.send_msg(dev1, P.AGGREGATE, agg)
            b_ag2 = P.send_msg(dev2, P.AGGREGATE, agg)

            # Bước 7: nhận METRICS từ CẢ 2 iot — mỗi iot đo mô hình chung trên dữ liệu train của mình; iot1 thêm val + test
            (mt, b_mt1), (mt2, b_mt2) = _recv_both(dev1, dev2)
            P.expect(mt.get("type") == P.METRICS and mt2.get("type") == P.METRICS, f"[bs] mong METRICS từ 2 iot vòng {t}")
            b_mt = b_mt1 + b_mt2

            t_round = time.monotonic() - tr0
            record = {
                "t": t,
                "channel": {"g1": g1, "g2": g2, "R1": R1, "R2": R2},
                "policy": pol,
                "decision": {"p": p, "n_ciphertext": int(n_ct), "n_he": n_he,
                             "T_pred_s": dec["T_pred"], "late_pred": dec["late_pred"]},
                "t_max": t_max,
                "t_select": dec["t_select"],   # adaptive chọn p với t_max − δ (safety_margin_s)
                "bytes": {
                    "CHANNEL_REPORT": b_cr,
                    "ROUND_START": b_rs1 + b_rs2,
                    "UPLOAD": b_up1 + b_up2,
                    "CSUM": b_cs1 + b_cs2,
                    "PARTIAL_DEC": b_pd1 + b_pd2,
                    "AGGREGATE": b_ag1 + b_ag2,
                    "METRICS": b_mt,
                },
                # accuracy/loss: tập TEST (báo cáo); val_accuracy/val_loss: tập VALIDATION (theo dõi)
                "metrics": {"accuracy": mt.get("accuracy"), "loss": mt.get("loss"),
                            "val_accuracy": mt.get("val_accuracy"), "val_loss": mt.get("val_loss"),
                            # mô hình chung đo trên tập TRAIN của từng thiết bị [iot1, iot2]
                            "train_accuracy": [mt.get("train_accuracy"), mt2.get("train_accuracy")],
                            "train_loss": [mt.get("train_loss"), mt2.get("train_loss")]},
                "time": {
                    "wait_uploads_s": t_wait,
                    "he_add_s": t_add,
                    "he_fuse_s": t_fuse,
                    "compute_s": t_add + t_fuse,   # pha tính toán tại bs (×fidelity -> Pi4)
                    "round_s": t_round,
                    # mốc thời gian pha của từng thiết bị (thô; logger thêm bản ×fidelity)
                    "dev1": dict(up1.get("timing", {}), **pd1.get("timing", {}), **mt.get("timing", {})),
                    "dev2": dict(up2.get("timing", {}), **pd2.get("timing", {}), **mt2.get("timing", {})),
                },
                "channel_mode": cr.get("mode"),
                "time_scale": float(cr.get("time_scale", 1.0)),   # k: nén thời gian kênh (1 = thời gian thật); tx_s đã ×k
                "warnings": [w for w in [cr.get("warning")] if w],   # ⚠️ vd. uav_oma chạy song song, không phải TDMA
                "aggregation": fedavg,             # FedAvg: trọng số n_k/n đã nhân ở iot, bs chỉ cộng
            }

            # Đồng bộ nhịp với uav, nhận số liệu truyền chiều lên thật (byte, thời gian qua bộ định nhịp MessagePacer)
            P.send_msg(control, P.ROUND_DONE, {"t": t})
            st, _ = P.recv_msg(control)
            P.expect(st.get("type") == P.UAV_STATS and st["t"] == t, f"[bs] mong UAV_STATS vòng {t}")
            record["uplink"] = {}
            for k, R in ((1, R1), (2, R2)):
                items = []
                for it in st[f"dev{k}"]:
                    thr = it["bytes"] / it["tx_s"] if it["tx_s"] > 0 else None
                    items.append(dict(it, throughput_Bps=thr, ratio_to_R=(thr / R) if thr else None))
                record["uplink"][f"dev{k}"] = items
            up_tx = [next((it["tx_s"] for it in record["uplink"][f"dev{k}"] if it["type"] == P.UPLOAD), None)
                     for k in (1, 2)]
            record["time"]["upload_tx_s"] = up_tx
            check_measurement(record, float(config["channel"].get("shaping_tol", 0.05)))
            pi4_timing(record, F, cm, config.get("devices", {}).get("split", "iid"))
            record["late"] = bool(record["time"]["round_pi4_s"] > t_max)          # trễ hạn (mục 9, thang Pi 4)
            # kiểm chứng mô hình: lệch đo − dự báo (round_time bỏ qua bs cộng/ghép, cập nhật, phần đầu gói; bước 7: 0,4–1,1 s)
            record["pred_error_s"] = record["time"]["round_pi4_s"] - dec["T_pred"]
            # tham khảo: thời gian thô; vô nghĩa khi nén thời gian kênh (k > 1) -> null
            record["late_wallclock"] = bool(t_round > t_max) if record["time_scale"] == 1 else None
            # tóm tắt vòng (đọc nhanh / so sánh policy): policy, p, dung lượng gửi lên mỗi thiết bị (MB, UPLOAD), thời gian
            # truyền UPLOAD (s, trên kênh thật), thời gian vòng quy đổi Pi 4, trễ hạn so với T_max, độ chính xác
            up_b = [next((it["bytes"] for it in record["uplink"][f"dev{k}"] if it["type"] == P.UPLOAD), 0)
                    for k in (1, 2)]
            record["summary"] = {
                "policy": pol, "p": p,
                "upload_MB": [b / 1e6 for b in up_b],
                "upload_tx_s": up_tx,
                "round_pi4_s": record["time"]["round_pi4_s"],
                "t_max_s": t_max, "late": record["late"],
                "accuracy": record["metrics"]["accuracy"], "val_accuracy": record["metrics"]["val_accuracy"],
                "train_accuracy": record["metrics"]["train_accuracy"],
            }

            path = log.write(record)
            tm = record["time"]
            print(f"[bs] vòng {t}: p={p} n_ct={n_ct} R=({R1:.0f},{R2:.0f})B/s UPLOAD={record['bytes']['UPLOAD']}B "
                  f"tx={[round(x, 2) for x in up_tx]} | Pi4: dự báo {dec['T_pred']:.1f}s, đo {tm['round_pi4_s']:.1f}s "
                  f"(t_max {t_max:.1f}s{', TRỄ HẠN' if record['late'] else ''}) | thô {t_round:.1f}s{f' (NÉN kênh k={record["time_scale"]:.1f})' if record['time_scale'] != 1 else ''} | "
                  f"{'' if record['measurement_ok'] else 'ĐO KHÔNG TIN CẬY (máy quá tải) | '}"
                  f"huấn luyện chậm thêm {[round(x) for x in record['sim_artifact']['slowdown_pct']]}% (giả lập) "
                  f"-> {os.path.basename(path)}", flush=True)
    except TimeoutError as e:
        raise P.ProtocolError("[bs] quá thời gian chờ trong vòng (protocol.round_timeout_*): một container treo "
                              "hoặc quá chậm — xem log iot/uav") from e
    finally:
        for s in (control, dev1, dev2):
            try:
                s.close()
            except OSError:
                pass
    print("[bs] kết thúc", flush=True)


def main():
    """Điểm vào của container bs: đọc biến môi trường (BS_BIND, BS_PORT, CONFIG, RESULTS, CAPTURE; PERM_SEED chỉ với
    docker-compose.fas.yml) và config.yaml, rồi chạy run_bs."""
    import yaml
    bs_host = os.environ.get("BS_BIND", "0.0.0.0")
    bs_port = int(os.environ.get("BS_PORT", "6000"))
    with open(os.environ.get("CONFIG", "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    perm_seed = None
    if os.environ.get("PERM_SEED"):   # chỉ có trong docker-compose.fas.yml
        with open(os.environ["PERM_SEED"], "rb") as f:
            perm_seed = f.read()
    run_bs(bs_host, bs_port, int(config["rounds"]), config,
           os.environ.get("RESULTS", "results"),
           os.environ.get("CAPTURE", "capture"), perm_seed=perm_seed)


if __name__ == "__main__":
    main()
