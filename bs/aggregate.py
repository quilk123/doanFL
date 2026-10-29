# -*- coding: utf-8 -*-
"""Tổng hợp tại bs (CLAUDE.md mục 6 bước 4, 6).

- Phần HE: cộng đồng cấu + ghép phần giải mã — common.crypto_setup.BsHE (OpenFHE).
- shuffled_sum: CỘNG MÙ theo vị trí hai vector đã xáo (đúng thiết kế: π(a)+π(b)=π(a+b)).
bs KHÔNG có khóa bí mật HE và KHÔNG có khóa xáo (trừ policy fas).
"""
import numpy as np


def shuffled_sum(vectors):
    """Cộng theo vị trí các vector đã xáo (cộng mù).

    Hai iot xáo bằng cùng π_t nên phần tử thứ i của hai vector ứng với cùng một tham số: cộng theo vị trí ra
    π(Δ1 + Δ2) mà bs không biết phần tử nào là tham số nào. Kết quả gửi về iot trong AGGREGATE để iot giải xáo.
    """
    acc = np.zeros_like(vectors[0])
    for v in vectors:
        acc = acc + v
    return acc
