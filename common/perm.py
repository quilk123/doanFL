# -*- coding: utf-8 -*-
"""Hoán vị bí mật π_t (CLAUDE.md mục 2, 5.3, 6): seed = HKDF-SHA256(K, "perm" ‖ t).

K = secrets/perm_seed (32 byte) chỉ có ở iot1, iot2 — và ở bs CHỈ với policy fas10/fas50 (mô phỏng FAS gốc, mục 8).
"""
import hmac
import hashlib
import numpy as np


def hkdf_sha256(key, info, length=32, salt=b""):
    """HKDF (RFC 5869) với HMAC-SHA256: rút một khóa con `length` byte từ khóa gốc `key` và nhãn `info`.

    Bước 1 (extract): PRK = HMAC(salt, key). Bước 2 (expand): nối các khối HMAC(PRK, khối_trước ‖ info ‖ i).
    Dùng để mỗi vòng t có một seed xáo riêng từ cùng một K, mà biết seed vòng này không suy ra được K hay vòng khác.
    """
    prk = hmac.new(salt or b"\0" * 32, key, hashlib.sha256).digest()
    out, block = b"", b""
    for i in range((length + 31) // 32):
        block = hmac.new(prk, block + info + bytes([i + 1]), hashlib.sha256).digest()
        out += block
    return out[:length]


def perm_for(K, t, n):
    """π_t trên n vị trí. Seed 256 bit = HKDF(K, "perm" ‖ t) -> bộ sinh PCG64 của numpy (không phải PRP mật mã;
    bs không bao giờ thấy π, chỉ thấy vector đã xáo).

    Hai iot cùng có K nên cùng tính ra một π_t: bs cộng 2 vector đã xáo theo CÙNG vị trí => tổng vẫn đúng, iot giải
    xáo tổng bằng unshuffle. π_t đổi mỗi vòng (t nằm trong info HKDF). Gọi ở iot/protect.py (xáo) và iot/client.py
    (giải xáo); với fas10/fas50 bs cũng gọi (bs có K, mô phỏng FAS gốc).
    """
    seed = hkdf_sha256(K, b"perm" + int(t).to_bytes(8, "big"))
    return np.random.default_rng(int.from_bytes(seed, "big")).permutation(n)


def unshuffle(shuffled, pi):
    """Giải xáo: out[π] = vector đã xáo.

    Xáo ở protect.py là shuffled = x[π]; vậy đặt lại từng phần tử về vị trí π[i] cho ra x. pi = None (không xáo,
    vd. policy no_he/full_he) thì trả nguyên vector.
    """
    if pi is None:
        return shuffled
    out = np.empty_like(shuffled)
    out[pi] = shuffled
    return out
