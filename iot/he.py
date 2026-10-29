# -*- coding: utf-8 -*-
"""HE phía thiết bị — OpenFHE multiparty CKKS (CLAUDE.md mục 5, 6).

Khóa bí mật sinh và nằm TẠI THIẾT BỊ, không bao giờ serialize / gửi đi.
iot1 là "lead": KeyGen trước, giải mã bằng MultipartyDecryptLead; iot2 MultipartyKeyGen(pk1), MultipartyDecryptMain.
Mọi đầu vào/đầu ra trên dây là bytes (common.crypto_setup).
"""
from common.crypto_setup import load_context, load_pk, load_ct, ser


class DeviceHE:
    """Vai trò HE của một iot: giữ khóa bí mật của mình, mã hóa phần HE của Δ, giải mã từng phần tổng ciphertext."""

    def __init__(self, cc_bytes, slots, lead):
        """Dựng lại CryptoContext từ bytes bs gửi (HE_CONTEXT). lead=True với iot1 (sinh khóa trước, giải mã Lead)."""
        self.cc = load_context(cc_bytes)   # đúng context bs gửi (mục 5.1)
        self.slots = int(slots)
        self.lead = bool(lead)
        self._sk = None                    # khóa bí mật — chỉ ở đây
        self.pk = None                     # khóa công khai chung

    def keygen_lead(self):
        """iot1: KeyGen -> trả pk1 (bytes).

        Khóa bí mật sk1 giữ lại trong đối tượng; chỉ pk1 được gửi đi (qua bs tới iot2). pk1 chưa phải khóa chung.
        """
        if not self.lead:
            raise RuntimeError("chỉ iot1 (lead) gọi keygen_lead")
        kp = self.cc.KeyGen()
        self._sk = kp.secretKey
        return ser(kp.publicKey)

    def keygen_main(self, pk1_bytes):
        """iot2: MultipartyKeyGen(pk1) -> trả pk chung (bytes).

        Sinh khóa bí mật sk2 của iot2 và khóa công khai chung ứng với (sk1 + sk2): mã hóa bằng khóa chung thì phải có
        phần giải mã của CẢ HAI thiết bị mới giải được — bs, uav, hay một thiết bị đơn lẻ đều không giải được.
        """
        if self.lead:
            raise RuntimeError("iot1 (lead) không gọi keygen_main")
        kp = self.cc.MultipartyKeyGen(load_pk(pk1_bytes))
        self._sk = kp.secretKey
        self.pk = kp.publicKey
        return ser(kp.publicKey)

    def set_joint_pk(self, pk_bytes):
        """iot1 nhận khóa công khai chung (HE_PK do bs phát lại) để mã hóa giống iot2."""
        self.pk = load_pk(pk_bytes)

    def encrypt(self, values):
        """Mã hóa bằng pk chung theo khối `slots` -> danh sách ciphertext (bytes); ceil(m/slots) khối.

        Gọi ở iot/protect.py với phần HE của Δ (m = round(p·M) giá trị đầu theo thứ tự ưu tiên). Mỗi ciphertext
        ~130 KB (đo ở bước 8: p = 0,5 => 25 ciphertext, UPLOAD 3,7 MB/thiết bị) nên kích thước UPLOAD tăng gần tuyến
        tính theo p — đây là cái giá truyền mà policy adaptive cân đối.
        """
        cc, S = self.cc, self.slots
        return [ser(cc.Encrypt(self.pk, cc.MakeCKKSPackedPlaintext([float(v) for v in values[i:i + S]])))
                for i in range(0, len(values), S)]

    def partial_decrypt(self, csum):
        """Giải mã từng phần tổng ciphertext (bytes) bằng khóa bí mật của mình -> danh sách δ_i (bytes).

        csum là Enc(Δ1 + Δ2) bs đã cộng (CSUM) — thiết bị chỉ giải mã TỔNG, không bao giờ nhận ciphertext riêng của
        thiết bị kia. δ_i gửi lại bs (PARTIAL_DEC); bs ghép δ_1, δ_2 ra tổng phần HE. Danh sách rỗng khi p = 0.
        """
        if not csum:
            return []
        f = self.cc.MultipartyDecryptLead if self.lead else self.cc.MultipartyDecryptMain
        return [ser(c) for c in f([load_ct(b) for b in csum], self._sk)]
