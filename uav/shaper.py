# -*- coding: utf-8 -*-
"""Bóp tốc độ chiều lên (CLAUDE.md mục 4.1): mỗi tin nhắn chiều lên đi đúng R_k byte/s.

Định nhịp NEO THEO ĐẦU TIN NHẮN: byte thứ n của tin nhắn rời uav không sớm hơn t0 + n/R_k (t0 = lúc uav nhận khối đầu).
  - Không có "gửi miễn phí" sau lúc rảnh (khác token bucket): mọi tin nhắn mất đúng bytes/R_k.
  - Khi máy giả lập quá tải (luồng uav ngủ quá giờ, bs đọc chậm làm ứ bộ đệm), phần sau được gửi BÙ ngay để tổng thời gian
    vẫn = bytes/R_k. Gửi bù là đúng vì gói đã serialize sẵn ở iot: mọi khoảng dừng là sai lệch của máy giả lập, không
    phải của kênh (kênh thật truyền liên tục ở R_k). Token bucket cũ bỏ mất token khi trễ -> chậm vĩnh viễn (bước 8 vòng 14:
    0,879·R_k).
R_k đặt lại mỗi vòng theo CHANNEL_REPORT (cố định trong vòng, mục 4.2); chiều xuống không giới hạn.

NÉN THỜI GIAN KÊNH (config `channel.time_scale`, mặc định off): uav cho dữ liệu đi ở R_k·k, rồi báo thời gian truyền ×k
(= thời gian thật trên kênh R_k). Hai thiết bị dùng CÙNG k trong vòng (giữ tỉ lệ tốc độ NOMA). Hợp lệ vì thời gian truyền tỉ lệ
tuyến tính với tốc độ (bước 4: lệch < 1% từ 0,089 đến 1,8 Mbit/s). Dùng cho kênh rất chậm (direct_noma ~0,001 Mbit/s).
  off     -> k = 1 (thời gian thật)
  <số>    -> k cố định (≥ 1)
  auto    -> k = max(1, sàn / min(R_1, R_2)), sàn = `time_scale_floor_bps` (mặc định 1 Mbit/s): chỉ nén vòng kênh xấu.
"""
import time
import threading


def time_scale_for(ch, R1, R2):
    """Hệ số nén k của vòng (R byte/s). ch = config['channel'].

    k = 1: dữ liệu đi đúng tốc độ kênh R_k (thời gian thật). k > 1: dữ liệu đi nhanh gấp k lần, thời gian truyền báo
    về được nhân lại ×k nên số liệu trong log vẫn là thời gian trên kênh thật. "auto" chỉ nén khi kênh chậm hơn sàn.
    """
    ts = ch.get("time_scale", "off")
    if ts is None or ts is False or str(ts).lower() in ("off", "1", "1.0", "none"):   # YAML đọc `off` thành False
        return 1.0
    if str(ts).lower() == "auto":
        floor = float(ch.get("time_scale_floor_bps", 1e6)) / 8.0
        return max(1.0, floor / min(R1, R2))
    k = float(ts)
    if k < 1:
        raise ValueError(f"channel.time_scale phải ≥ 1 (đang {ts})")
    return k


class MessagePacer:
    """Bộ định nhịp của MỘT chiều lên (mỗi iot một bộ): giữ tốc độ R_k và hệ số nén k của vòng hiện tại.

    Luồng chuyển tiếp (_pump) gọi wait() trước mỗi khối; luồng nhịp vòng (run_uav) gọi set_rate() đầu vòng — khóa
    `lock` giữ hai giá trị nhất quán giữa hai luồng.
    """

    def __init__(self, rate=None):
        """rate = None: chưa giới hạn (giai đoạn thiết lập HE trước vòng 1 đi tự do)."""
        self.lock = threading.Lock()
        self.rate = None
        self.scale = 1.0
        self.set_rate(rate)

    def set_rate(self, rate, scale=1.0):
        """rate byte/s (tốc độ kênh R_k); None = không giới hạn. scale = k: dữ liệu thực đi ở rate·k."""
        with self.lock:
            self.rate = None if rate is None else float(rate)
            self.scale = float(scale)

    def wait(self, t0, sent):
        """Chặn tới thời điểm t0 + sent/(R·k) (sent = số byte của tin nhắn tính cả khối sắp gửi).

        t0 = lúc tin nhắn bắt đầu đi qua uav. Nếu đã trễ hơn lịch (máy chậm) thì không ngủ — khối được gửi bù ngay,
        tổng thời gian tin nhắn vẫn = bytes/(R·k). Không giới hạn (rate None) thì trả ngay.
        """
        with self.lock:
            rate, scale = self.rate, self.scale
        if rate is None:
            return
        delay = t0 + sent / (rate * scale) - time.monotonic()
        if delay > 0:
            time.sleep(delay)
