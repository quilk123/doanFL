# -*- coding: utf-8 -*-
"""HE đa khóa giữa các container (CLAUDE.md mục 5) — OpenFHE multiparty CKKS, mọi thứ đi trên dây dưới dạng bytes
(`Serialize(..., BINARY)`).

Thiết lập một lần:
  1. bs tạo CryptoContext (tham số config `he`, như reference_sim MKCKKS), gửi bytes cho 2 iot (HE_CONTEXT).
  2. iot1 KeyGen -> pk1 (HE_PK1, qua bs) -> iot2 MultipartyKeyGen(pk1) -> pk chung (HE_PK) -> bs phát cho cả hai.
     Khóa bí mật KHÔNG rời thiết bị (iot/he.py); bs chỉ có context + khóa công khai.
Mỗi vòng: bs cộng đồng cấu (EvalAdd), rồi ghép 2 phần giải mã (MultipartyDecryptFusion) -> tổng phần HE.
Một phần giải mã đơn lẻ KHÔNG đủ để giải (Fusion báo lỗi / sai số lớn).
"""
import numpy as np

from common.mkckks import make_context


def ser(obj):
    """Đối tượng OpenFHE (context, khóa công khai, ciphertext) -> bytes BINARY để gửi trong tin nhắn.

    Mọi thứ HE đi trên dây dưới dạng bytes này; số byte của nó chính là kích thước thật được log và bị bóp tốc độ.
    """
    from openfhe import Serialize, BINARY
    return Serialize(obj, BINARY)


def _first(r):
    """Chuẩn hóa kết quả Deserialize: một số bản openfhe-python trả (đối tượng, ok), số khác trả đối tượng."""
    return r[0] if isinstance(r, tuple) else r


def load_context(cc_bytes):
    """Context từ bytes của bs. KHÔNG gọi ReleaseAllContexts: test chạy bs/iot chung một tiến trình;
    OpenFHE trả lại context có sẵn nếu trùng tham số."""
    from openfhe import DeserializeCryptoContextString, BINARY
    return _first(DeserializeCryptoContextString(cc_bytes, BINARY))


def load_pk(b):
    """bytes -> khóa công khai (iot2 nhận pk1 của iot1; cả hai nhận pk chung trong HE_PK)."""
    from openfhe import DeserializePublicKeyString, BINARY
    return _first(DeserializePublicKeyString(b, BINARY))


def load_ct(b):
    """bytes -> ciphertext (bs đọc ciphertext trong UPLOAD/PARTIAL_DEC; iot đọc CSUM để giải mã từng phần)."""
    from openfhe import DeserializeCiphertextString, BINARY
    return _first(DeserializeCiphertextString(b, BINARY))


class BsHE:
    """Vai trò HE của bs: tạo context, cộng đồng cấu, ghép phần giải mã. Không có khóa bí mật.

    bs KHÔNG giải mã được riêng cập nhật của từng iot: nó chỉ cộng 2 ciphertext rồi gửi tổng (CSUM) cho 2 iot giải
    mã từng phần; ghép 2 phần chỉ ra TỔNG Δ1 + Δ2 của phần HE — đúng thứ FedAvg cần, không hơn.
    """

    def __init__(self, config):
        """Tạo CryptoContext theo config `he` và giữ sẵn bản bytes để gửi cho 2 iot trong HE_CONTEXT."""
        self.slots = int(config["he"]["slots"])
        self.cc = make_context(config)
        self.cc_bytes = ser(self.cc)

    def add(self, cts1, cts2):
        """Cộng đồng cấu từng cặp ciphertext (bytes) -> danh sách bytes."""
        return [ser(self.cc.EvalAdd(load_ct(a), load_ct(b))) for a, b in zip(cts1, cts2)]

    def fuse(self, shares1, shares2, n):
        """Ghép phần giải mã của iot1 (lead) và iot2 -> n giá trị thực (tổng phần HE)."""
        out = []
        for a, b in zip(shares1, shares2):
            pt = self.cc.MultipartyDecryptFusion([load_ct(a), load_ct(b)])
            pt.SetLength(self.slots)
            out.extend(pt.GetRealPackedValue())
        return np.array(out[:n], dtype=np.float64)
