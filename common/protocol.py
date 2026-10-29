# -*- coding: utf-8 -*-
"""Giao thức khung nhị phân length-prefixed, thân msgpack (CLAUDE.md mục 6).

Mọi tin nhắn gồm: 4 byte độ dài (big-endian) + thân msgpack.
Mảng số được đóng gói dưới dạng bytes thô (float32) để log byte thật sát thực tế.
Module này dùng chung cho iot, uav, bs — KHÔNG phụ thuộc Docker nên test được tại chỗ.
"""
import struct
import msgpack
import numpy as np

# --- Loại tin nhắn (mục 6) ---
HELLO          = "HELLO"           # bắt tay: {role, device_id}
HE_CONTEXT     = "HE_CONTEXT"      # bs -> iot: {cc} CryptoContext đã serialize (thiết lập, mục 5.1)
HE_PK1         = "HE_PK1"          # iot1 -> bs -> iot2: {pk} khóa công khai của iot1 (mục 5.2)
HE_PK          = "HE_PK"           # iot2 -> bs -> iot1, iot2: {pk} khóa công khai CHUNG
CHANNEL_REPORT = "CHANNEL_REPORT"  # uav -> bs: {t, g1, g2}
ROUND_START    = "ROUND_START"     # bs -> iot: {t, p}
UPLOAD         = "UPLOAD"          # iot -> bs: {ciphertexts, shuffled}
CSUM           = "CSUM"            # bs -> iot: {ciphertexts}
PARTIAL_DEC    = "PARTIAL_DEC"     # iot -> bs: {delta}
AGGREGATE      = "AGGREGATE"       # bs -> iot: {he_part, shuffled_sum}
METRICS        = "METRICS"         # iot1 -> bs: {t, accuracy, loss}
ROUND_DONE     = "ROUND_DONE"      # bs -> uav: {t}  (đồng bộ nhịp vòng)
UAV_STATS      = "UAV_STATS"       # uav -> bs: {t, dev1, dev2} byte + thời gian truyền thật từng tin nhắn chiều lên

_LEN = struct.Struct(">I")


class ProtocolError(RuntimeError):
    """Sai giao thức / cấu hình hoặc quá thời gian chờ — dừng lượt chạy với thông báo rõ (không dùng assert:
    assert bị tắt khi chạy python -O)."""


def expect(cond, msg):
    """Kiểm tra điều kiện của giao thức; sai thì dừng lượt chạy bằng ProtocolError(msg).

    Dùng thay `assert` ở mọi chỗ bs/iot/uav kiểm tra tin nhắn nhận được (đúng loại, đúng vòng t, đủ trường...),
    vì `assert` bị bỏ qua khi chạy `python -O` (test_robustness chạy với -O để kiểm tra điều này).
    """
    if not cond:
        raise ProtocolError(msg)


def recv_exactly(sock, n):
    """Đọc đúng n byte, lỗi nếu đối phương đóng kết nối giữa chừng.

    TCP có thể trả dữ liệu thành nhiều mẩu nhỏ, nên phải lặp `recv` đến khi đủ n byte. Nếu `recv` trả b"" (đối
    phương đã đóng) thì ném ConnectionError — iot dựa vào lỗi này để biết bs đã chạy xong mọi vòng (client.py).
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("kết nối đóng khi đang đọc khung")
        buf.extend(chunk)
    return bytes(buf)


def read_frame(sock):
    """Đọc nguyên một khung (header + thân) mà KHÔNG giải mã.

    Dùng cho relay của uav: chuyển tiếp nguyên khung, giữ ranh giới tin nhắn
    để bước 4 có thể bóp tốc độ theo từng khung.
    """
    header = recv_exactly(sock, 4)
    (n,) = _LEN.unpack(header)
    body = recv_exactly(sock, n)
    return header + body


def read_header(sock):
    """Đọc 4 byte độ dài; trả về (header, độ dài thân).

    uav dùng hàm này để biết trước kích thước tin nhắn rồi chuyển tiếp thân theo từng khối (uav/relay.py `_pump`),
    nhờ đó bóp tốc độ được ngay trong lúc tin nhắn đang đi qua, không phải chờ nhận đủ cả tin nhắn.
    """
    header = recv_exactly(sock, 4)
    return header, _LEN.unpack(header)[0]


def peek_type(body_prefix):
    """Loại tin nhắn từ phần đầu thân (send_msg đặt 'type' là khóa ĐẦU TIÊN); None nếu không đọc được.

    uav chỉ có khối đầu tiên của thân (vài KB) khi bắt đầu chuyển tiếp; hàm này giải mã đúng cặp khóa-giá trị đầu
    tiên của map msgpack để lấy "type" (UPLOAD, PARTIAL_DEC, ...) mà không cần giải mã cả thân (có thể vài MB).
    Loại tin nhắn dùng để ghi số liệu truyền theo từng loại (UAV_STATS).
    """
    try:
        up = msgpack.Unpacker(raw=False)
        up.feed(body_prefix)
        up.read_map_header()
        if up.unpack() == "type":
            return up.unpack()
    except Exception:   # noqa: BLE001 — phần đầu quá ngắn hoặc không phải map
        pass
    return None


def send_msg(sock, mtype, obj=None):
    """Đóng gói và gửi một tin nhắn. Trả về số byte THẬT đã gửi (mục 9).

    Đầu vào: socket đã kết nối, loại tin nhắn (hằng số ở đầu file), dict nội dung (giá trị là số, chuỗi, bytes,
    list, dict — mảng numpy phải đóng gói trước bằng pack_array). Khung = 4 byte độ dài big-endian + thân msgpack.
    Số byte trả về được bs ghi vào log (record["bytes"]) — đây là số byte thật trên dây, không ước lượng.
    """
    d = {"type": mtype}            # khóa đầu tiên: uav đọc được loại tin nhắn mà không giải mã cả thân
    d.update(obj or {})
    d["type"] = mtype
    body = msgpack.packb(d, use_bin_type=True)
    frame = _LEN.pack(len(body)) + body
    sock.sendall(frame)
    return len(frame)


def recv_msg(sock):
    """Đọc và giải mã một tin nhắn. Trả về (dict, số_byte_thật).

    Chặn (block) đến khi nhận đủ khung. Thời gian chờ do socket quyết định: bs đặt timeout theo từng vòng
    (max(600 s, 5 × T_pred)) để một container bị treo không làm cả lượt chạy treo mãi.
    """
    header = recv_exactly(sock, 4)
    (n,) = _LEN.unpack(header)
    body = recv_exactly(sock, n)
    return msgpack.unpackb(body, raw=False), 4 + n


# --- Đóng/mở gói mảng số (bytes thô trong thân msgpack); mặc định float32 ---
def pack_array(a, dtype=np.float32):
    """Mảng numpy -> dict {shape, dtype, data(bytes thô)} để nhét vào thân msgpack.

    Mặc định float32 (4 byte/giá trị) như thiết bị thật gửi cập nhật mô hình; lưu kèm dtype để bên nhận dựng lại
    đúng kiểu (vd. AGGREGATE gửi float64 khi cần giữ độ chính xác của tổng).
    """
    a = np.ascontiguousarray(a, dtype=dtype)
    return {"shape": list(a.shape), "dtype": a.dtype.str, "data": a.tobytes()}


def unpack_array(d):
    """Ngược của pack_array: dict -> mảng numpy (chỉ đọc, dùng chung bộ nhớ với bytes nhận được)."""
    return np.frombuffer(d["data"], dtype=np.dtype(d.get("dtype", "<f4"))).reshape(tuple(d["shape"]))


def pack_list(arrs):
    """Đóng gói danh sách mảng (mỗi phần tử qua pack_array)."""
    return [pack_array(a) for a in arrs]


def unpack_list(ds):
    """Mở gói danh sách mảng (mỗi phần tử qua unpack_array)."""
    return [unpack_array(d) for d in ds]
