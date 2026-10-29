# -*- coding: utf-8 -*-
"""CKKS đa bên của OpenFHE (threshold 2-of-2) dùng như xMK-CKKS (logic reference_sim/common.py `MKCKKS`).

Tham số lấy từ config.yaml `he` (ring 8192, 4096 slot, độ sâu 0, scale 50, firstMod 60, HEStd_128_classic).
Ở bước 3 lớp này dùng để ĐO chi phí trong một tiến trình (tools/bench_costs.py).
Bước 5 tách vai trò ra các container (bs tạo context, mỗi iot giữ khóa bí mật của mình).
"""
import math
import os
import tempfile
import numpy as np


def n_ct(n_values, slots):
    """Số ciphertext cần để mã hóa n_values số thực: mỗi ciphertext chứa `slots` (= 4096) giá trị.

    Ví dụ p = 0,5 => 101.157 giá trị => 25 ciphertext. Dùng ở bs/policy.py để dự báo kích thước UPLOAD và thời
    gian mã hóa/truyền theo p, và ở log (decision.n_ciphertext).
    """
    return math.ceil(n_values / slots) if n_values > 0 else 0


def make_context(config):
    """CryptoContext CKKS theo config (chỉ phép cộng nên độ sâu 0, một modulus).

    Hệ thống chỉ CỘNG ciphertext (bs cộng cập nhật của 2 iot), không nhân, nên độ sâu nhân = 0 là đủ — giữ
    ciphertext nhỏ nhất có thể. Tham số (config `he`): ring 8192 => 4096 slot; scale 2^50 (độ chính xác ~1e-7
    sau giải mã, bước 5 đo được 5e-7); firstMod 60 bit; an toàn 128 bit chuẩn HE. Bật MULTIPARTY để có khóa đa bên.
    Chỉ bs gọi hàm này (common/crypto_setup.py BsHE); iot nhận context dạng bytes và dựng lại.
    """
    from openfhe import (CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature, SecurityLevel, FIXEDMANUAL)
    h = config["he"]
    p = CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(0)
    p.SetScalingModSize(int(h["scale_bits"]))
    p.SetFirstModSize(int(h["first_mod_bits"]))
    p.SetRingDim(int(h["ring_dim"]))
    p.SetBatchSize(int(h["slots"]))
    p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    p.SetScalingTechnique(FIXEDMANUAL)
    # ⚠️ GIỚI HẠN (đã quyết, giống reference_sim): KHÔNG đặt SetMultipartyMode => OpenFHE dùng chế độ nhiễu mặc định
    # FIXED_NOISE_MULTIPARTY cho phần giải mã từng bên (nhiễu nhỏ). NOISE_FLOODING_MULTIPARTY an toàn hơn trước bs tò mò
    # (che khóa bí mật trong δ_i) nhưng cần modulus lớn hơn (đổi tham số CKKS, ciphertext to hơn) — không dùng.
    cc = GenCryptoContext(p)
    for f in [PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE,
              PKESchemeFeature.MULTIPARTY]:
        cc.Enable(f)
    return cc


def ser_size(obj):
    """Số byte khi Serialize (BINARY) một đối tượng OpenFHE.

    Dùng ở tools/bench_costs.py (bước 3) để đo kích thước một ciphertext/khóa — bảng chi phí costs/ dùng số này để
    dự báo thời gian truyền (bs/policy.py).
    """
    from openfhe import SerializeToFile, BINARY
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "o.bin")
        SerializeToFile(fn, obj, BINARY)
        return os.path.getsize(fn)


class MKCKKS:
    """Hai bên trong cùng tiến trình: mỗi bên một khóa bí mật, khóa công khai chung; giải mã cần cả hai phần.

    Chỉ dùng để ĐO chi phí (tools/bench_costs.py) và trong test — hệ thống thật tách vai trò ra container:
    bs = common/crypto_setup.py BsHE, iot = iot/he.py DeviceHE.
    """

    def __init__(self, config, n_parties=2):
        """Tạo context và khóa đa bên: bên 0 KeyGen, mỗi bên sau MultipartyKeyGen nối tiếp trên khóa công khai của
        bên trước; khóa công khai của bên cuối là khóa CHUNG (mã hóa bằng nó thì phải đủ mọi bên mới giải được)."""
        self.slots = int(config["he"]["slots"])
        self.cc = cc = make_context(config)
        self.keys = [cc.KeyGen()]
        for _ in range(1, n_parties):
            self.keys.append(cc.MultipartyKeyGen(self.keys[-1].publicKey))
        self.pk = self.keys[-1].publicKey

    def encrypt(self, x):
        """Mã hóa vector x bằng khóa công khai chung, cắt thành từng đoạn `slots` giá trị -> danh sách ciphertext."""
        cc = self.cc
        S = self.slots
        return [cc.Encrypt(self.pk, cc.MakeCKKSPackedPlaintext(list(x[i:i + S])))
                for i in range(0, len(x), S)]

    def add(self, a, b):
        """Cộng đồng cấu từng cặp ciphertext: Enc(x) + Enc(y) = Enc(x + y), không cần khóa bí mật."""
        return [self.cc.EvalAdd(x, y) for x, y in zip(a, b)]

    def partial_decrypt(self, cts, party):
        """Giải mã TỪNG PHẦN bằng khóa bí mật của một bên (bên 0 = Lead, các bên khác = Main).
        Một phần giải mã đơn lẻ không lộ bản rõ; phải ghép (fuse) đủ phần của mọi bên."""
        f = self.cc.MultipartyDecryptLead if party == 0 else self.cc.MultipartyDecryptMain
        return f(cts, self.keys[party].secretKey)

    def fuse(self, shares, n):
        """Ghép phần giải mã của mọi bên (MultipartyDecryptFusion) -> n giá trị thực đầu tiên (bỏ slot đệm)."""
        out = []
        for j in range(len(shares[0])):
            pt = self.cc.MultipartyDecryptFusion([s[j] for s in shares])
            pt.SetLength(self.slots)
            out.extend(pt.GetRealPackedValue())
        return np.array(out[:n])
