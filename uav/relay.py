# -*- coding: utf-8 -*-
"""UAV relay AF — proxy TCP tầng ứng dụng + sinh báo cáo kênh (CLAUDE.md mục 4.1, 6).

- Mỗi iot nối tới uav qua MỘT CỔNG RIÊNG; uav mở kết nối tương ứng tới bs.
- Chiều lên bị bóp tốc độ R_k (định nhịp theo tin nhắn, uav/shaper.py; đặt lại mỗi vòng); chiều xuống không giới hạn.
  Chuyển tiếp theo khối `channel.shaper_chunk` byte, giữ ranh giới tin nhắn.
- uav là nhịp vòng: mỗi vòng sinh kênh -> đặt R_k -> CHANNEL_REPORT cho bs -> chờ ROUND_DONE
  -> gửi UAV_STATS (byte, thời gian truyền thật của từng tin nhắn chiều lên) để bs ghi log.
"""
import os
import time
import socket
import threading

from common import protocol as P
from uav import channel
from uav.shaper import MessagePacer, time_scale_for


def _connect(host, port, timeout=30.0):
    """Nối tới bs, thử lại cho tới khi bs sẵn sàng (container khởi động đồng thời).

    Dùng cho kết nối điều khiển (control) và cho mỗi đường ống thiết bị (uav mở một kết nối riêng tới bs cho mỗi iot).
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


class UplinkStats:
    """Byte + thời gian truyền của từng tin nhắn chiều lên trong vòng hiện tại.
    tx_s = thời gian trên kênh R_k (đã ×k nếu nén); wall_s = thời gian máy thực chạy."""

    def __init__(self):
        self.lock = threading.Lock()
        self.items = []

    def add(self, mtype, nbytes, tx_s, wall_s):
        """Ghi một tin nhắn đã chuyển tiếp xong (luồng _pump gọi)."""
        with self.lock:
            self.items.append({"type": mtype, "bytes": nbytes, "tx_s": tx_s, "wall_s": wall_s})

    def take(self):
        """Lấy và xóa toàn bộ số liệu đã ghi (luồng nhịp vòng gọi cuối vòng để gửi UAV_STATS cho bs)."""
        with self.lock:
            items, self.items = self.items, []
        return items


def _pump(src, dst, shaper, stats, stop, chunk):
    """Chuyển tiếp từng tin nhắn từ src sang dst cho tới khi một đầu đóng.
    Có shaper: thân tin nhắn đi theo khối `chunk` byte; khối kết thúc ở byte thứ n gửi lúc t0 + n/R_k (uav/shaper.py).

    Mỗi đường ống thiết bị có 2 luồng chạy hàm này: chiều lên iot->bs (shaper = bộ định nhịp, bị bóp tốc độ) và chiều
    xuống bs->iot (shaper = None, chuyển nguyên tin nhắn). Với chiều lên: đọc header + khối đầu -> biết loại tin nhắn
    -> gửi từng khối theo lịch của shaper -> ghi (loại, byte, thời gian trên kênh, thời gian thực) vào stats.
    Một đầu đóng (hết vòng hoặc lỗi) thì đặt `stop` và đóng cả hai socket để các luồng khác cũng dừng.
    """
    try:
        while not stop.is_set():
            header, n = P.read_header(src)
            first = P.recv_exactly(src, min(n, chunk))
            if shaper is None:
                dst.sendall(header + first + P.recv_exactly(src, n - len(first)))
                continue
            mtype = P.peek_type(first)
            t0 = time.monotonic()
            done = len(first)
            piece = header + first
            sent = 0
            while True:
                sent += len(piece)
                shaper.wait(t0, sent)
                if done >= n:
                    # ghi số liệu TRƯỚC khối cuối: bs chỉ gửi ROUND_DONE sau khi nhận đủ, nên số liệu luôn kịp vào vòng
                    wall = time.monotonic() - t0
                    stats.add(mtype, 4 + n, wall * shaper.scale, wall)   # ×k: quy về thời gian thật trên kênh
                    dst.sendall(piece)
                    break
                dst.sendall(piece)
                piece = P.recv_exactly(src, min(chunk, n - done))
                done += len(piece)
    except (ConnectionError, OSError):
        pass
    finally:
        stop.set()
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def run_uav(bs_host, bs_port, listen_host, iot_ports, rounds, config, channel_fn=None):
    """channel_fn(config, t) -> (g1, g2, R1, R2); mặc định uav.channel.channel_round (test có thể thay).

    Vòng đời của uav:
      1. nối kết nối điều khiển (control) tới bs, gửi HELLO{role: uav}
      2. với mỗi iot: chờ iot nối vào cổng riêng (7001, 7002), mở kết nối tương ứng tới bs, chạy 2 luồng _pump
      3. mỗi vòng t: sinh kênh -> tính k nén -> đặt R_k cho 2 bộ định nhịp -> CHANNEL_REPORT cho bs (bs dùng R_k để
         chọn p) -> chờ ROUND_DONE -> gửi UAV_STATS (số liệu truyền thật của vòng) cho bs ghi log
    uav không giải mã nội dung (chỉ đọc loại tin nhắn), không có khóa HE hay K.
    """
    channel_fn = channel_fn or channel.channel_round
    ch = config["channel"]
    chunk = int(ch.get("shaper_chunk", 16384))

    control = _connect(bs_host, bs_port)
    P.send_msg(control, P.HELLO, {"role": "uav"})
    print(f"[uav] nối control tới bs {bs_host}:{bs_port}", flush=True)

    shapers = {}
    stats = {}
    stop = threading.Event()
    threads = []

    # Thiết lập 2 pipe thiết bị: iot_k <-> uav <-> bs
    for k, port in enumerate(iot_ports, start=1):
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((listen_host, port))
        lsock.listen(1)
        print(f"[uav] chờ iot{k} ở {listen_host}:{port}", flush=True)
        iot_sock, _ = lsock.accept()
        iot_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        lsock.close()

        bs_pipe = _connect(bs_host, bs_port)
        shapers[k] = MessagePacer(rate=None)
        stats[k] = UplinkStats()
        # chiều lên (iot -> bs): bóp tốc độ; chiều xuống (bs -> iot): tự do
        t_up = threading.Thread(target=_pump, args=(iot_sock, bs_pipe, shapers[k], stats[k], stop, chunk),
                                daemon=True)
        t_dn = threading.Thread(target=_pump, args=(bs_pipe, iot_sock, None, None, stop, chunk), daemon=True)
        t_up.start(); t_dn.start()
        threads += [t_up, t_dn]
        print(f"[uav] pipe iot{k} đã mở", flush=True)

    # Nhịp vòng
    try:
        for t in range(1, rounds + 1):
            g1, g2, R1, R2 = channel_fn(config, t)
            mode = ch.get("mode", "uav_noma_pc")
            for k in (1, 2):
                stats[k].take()              # bỏ tin nhắn trước vòng (HELLO) khỏi số liệu
            ks = time_scale_for(ch, R1, R2)   # nén thời gian kênh (mặc định 1 = không nén)
            shapers[1].set_rate(R1, ks)      # R_k cố định trong vòng (mục 4.2), áp cho mọi tin nhắn chiều lên
            shapers[2].set_rate(R2, ks)
            P.send_msg(control, P.CHANNEL_REPORT,
                       {"t": t, "g1": g1, "g2": g2, "R1": R1, "R2": R2, "mode": mode, "time_scale": ks,
                        "warning": channel.OMA_WARNING if mode == "uav_oma" else None})
            msg, _ = P.recv_msg(control)   # chờ bs báo xong vòng
            if msg.get("type") != P.ROUND_DONE:
                raise RuntimeError(f"[uav] mong ROUND_DONE, nhận {msg.get('type')}")
            P.send_msg(control, P.UAV_STATS, {"t": t, "dev1": stats[1].take(), "dev2": stats[2].take()})
            print(f"[uav] vòng {t} xong (R1={R1:.0f} R2={R2:.0f} B/s{f", NÉN k={ks:.1f}" if ks != 1 else ""})", flush=True)
    finally:
        stop.set()
        try:
            control.close()
        except OSError:
            pass
    print("[uav] kết thúc", flush=True)


def main():
    """Điểm vào của container uav: đọc biến môi trường (BS_HOST, BS_PORT, LISTEN_HOST, IOT_PORTS, CONFIG) và
    config.yaml, rồi chạy run_uav với số vòng `rounds`."""
    import yaml
    bs_host = os.environ.get("BS_HOST", "bs")
    bs_port = int(os.environ.get("BS_PORT", "6000"))
    listen_host = os.environ.get("LISTEN_HOST", "0.0.0.0")
    iot_ports = [int(p) for p in os.environ.get("IOT_PORTS", "7001,7002").split(",")]
    with open(os.environ.get("CONFIG", "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_uav(bs_host, bs_port, listen_host, iot_ports, int(config["rounds"]), config)


if __name__ == "__main__":
    main()
