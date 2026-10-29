# -*- coding: utf-8 -*-
"""Bảo vệ cập nhật Δ trước khi gửi (CLAUDE.md mục 2, 6 bước 3, mục 8).

Phương án chính: `fixed10` — p = 0,1 cố định, phần rõ thêm nhiễu λ·RMS rồi xáo π_t; K chỉ ở 2 thiết bị (bs cộng mù).
`fas10`/`fas50`: cùng cách bảo vệ nhưng bs CÓ K (mô phỏng FAS gốc). `adaptive`: hướng phát triển (clipping + nhiễu theo C).

1. Clipping toàn bộ Δ về ‖Δ‖ ≤ C (chỉ `adaptive`), TRƯỚC khi chia.
1b. Trọng số FedAvg: Δ <- (n_k / (n_1 + n_2)) · Δ (SAU clipping, TRƯỚC khi chia) — bs chỉ CỘNG nên tổng = Σ n_k/n · Δ_k.
2. Chia theo priority_order: m = round(p·M) phần tử đầu -> mã hóa CKKS bằng pk chung (iot/he.py).
3. Phần còn lại: cộng nhiễu Gauss rồi XÁO VỊ TRÍ bằng π_t.
   - nhiễu: `adaptive` N(0, (λ·w_k·C/√M)²); `fas*`, `fixed10` N(0, (λ·RMS(phần rõ))²) như FAS (phần rõ đã ×w_k nên nhiễu
     tự co theo). w_k nhân cả vào nhiễu adaptive để mô hình tổng giữ nguyên như khi chia đôi tại thiết bị (IID: w = ½).
     Nguồn ngẫu nhiên RIÊNG của thiết bị
     (entropy hệ điều hành) — KHÔNG lấy từ K, cũng KHÔNG từ seed trong config (bs có config, sẽ sinh lại được nhiễu).
   - π_t: seed = HKDF-SHA256(K, "perm" ‖ t), K = secrets/perm_seed (32 byte, chỉ 2 thiết bị có) => hai thiết bị cùng π_t,
     bs cộng mù theo vị trí: π(a) + π(b) = π(a + b); thiết bị giải xáo tổng.
Hai thiết bị dùng cùng p nên cùng tập chỉ số (idx_he, idx_plain).
"""
import time
import numpy as np
from common import model
from common.perm import hkdf_sha256, perm_for, unshuffle   # noqa: F401 (dùng lại ở client, test)

# mục 8: policy -> (clipping, kiểu nhiễu, xáo vị trí). Ai giữ khóa xáo K không nằm ở đây mà ở cấu hình triển khai:
# fixed10/adaptive — chỉ iot1, iot2 (docker-compose.yml); fas10/fas50 — thêm bs (docker-compose.fas.yml).
PROFILES = {
    "no_he":    dict(clip=False, noise=None,   shuffle=False),
    "full_he":  dict(clip=False, noise=None,   shuffle=False),
    "fixed10":  dict(clip=False, noise="rms",  shuffle=True),    # PHƯƠNG ÁN CHÍNH (FAS p = 0,1, K chỉ ở thiết bị)
    "fas10":    dict(clip=False, noise="rms",  shuffle=True),
    "fas50":    dict(clip=False, noise="rms",  shuffle=True),
    # [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
    # Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
    #   D(p) <= R_min * (T_max - T_comp)
    # với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
    # Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
    # Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
    "adaptive": dict(clip=True,  noise="clip", shuffle=True),
}


def split_indices(p, config):
    """Trả (idx_he, idx_plain) theo priority_order và p.

    idx_he = m = round(p·M) chỉ số đầu của thứ tự ưu tiên (sẽ mã hóa HE); idx_plain = phần còn lại (gửi rõ đã bảo
    vệ). Chỉ phụ thuộc p và kích thước mô hình => 2 iot (cùng p do bs gửi) có cùng tập chỉ số, bs cộng được theo vị trí.
    p = 0: không mã hóa gì (no_he); p = 1: mã hóa toàn bộ (full_he).
    """
    order = np.asarray(model.priority_order(config))
    M = len(order)
    m = int(round(p * M))
    return order[:m], order[m:]


def protect(dhe, delta, p, config, perm_seed, t, device_id, policy, noise_rng, weight=1.0):
    """Trả về (ciphertexts bytes, phần rõ đã nhiễu + xáo (float32), idx_he, idx_plain, info).
    info: enc_s, pi (None nếu không xáo), sd nhiễu, norm trước clipping, delta_clipped (để test; KHÔNG gửi đi).

    Đầu vào: dhe (DeviceHE có khóa chung), delta = Δ sau huấn luyện (iot/train.py), p và policy do bs gửi trong
    ROUND_START, perm_seed = K, t = số vòng (đổi π_t mỗi vòng), noise_rng = nguồn nhiễu riêng của thiết bị,
    weight = trọng số FedAvg n_k / (n_1 + n_2) của thiết bị (bs gửi số ảnh trong ROUND_START).
    Trình tự (đúng thứ tự, mỗi bước chỉ làm nếu PROFILES[policy] bật):
      1. clipping: Δ <- Δ · min(1, C/‖Δ‖)  (giới hạn độ lớn cập nhật => nhiễu cố định đủ che)
      1b. FedAvg: Δ <- weight · Δ  (luôn làm) — bs chỉ cộng nên tổng nhận được đã là trung bình có trọng số
      2. chia theo thứ tự ưu tiên -> mã hóa phần HE
      3. phần rõ: cộng nhiễu Gauss, rồi xáo vị trí bằng π_t
    pi trong info được giữ lại trong client.py để giải xáo tổng ở cuối vòng.
    """
    # fixedNN (fixed20, fixed30, …): cùng cách bảo vệ như fixed10, chỉ khác p (bs/policy.fixed_p)
    prof = PROFILES.get(policy) or (PROFILES["fixed10"] if policy.startswith("fixed") else PROFILES[policy])
    M = len(delta)
    C = float(config["dp"]["clip"])
    lam = float(config["dp"]["lam"])

    # 1. clipping (chỉ adaptive) — trên TOÀN BỘ Δ, trước khi chia
    norm = float(np.linalg.norm(delta))
    if prof["clip"] and norm > 0:
        # [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
        # Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
        #   D(p) <= R_min * (T_max - T_comp)
        # với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
        # Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
        # Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
        delta = delta * min(1.0, C / norm)

    # 1b. trọng số FedAvg n_k / (n_1 + n_2): sau clipping, trước khi tách phần HE / phần rõ
    delta = delta * weight

    # 2. chia theo thứ tự ưu tiên; mã hóa phần HE (đo thời gian để quy đổi Pi 4)
    idx_he, idx_plain = split_indices(p, config)
    t0 = time.monotonic()
    ciphertexts = dhe.encrypt(delta[idx_he]) if len(idx_he) else []
    enc_s = time.monotonic() - t0

    # 3. phần rõ: nhiễu (adaptive: sd = λ·w·C/√M; fas*/fixed10: sd = λ·RMS(phần rõ đã ×w)) rồi xáo π_t
    x = delta[idx_plain].copy()
    sd = 0.0
    if prof["noise"] and lam > 0 and len(x):
        sd = lam * (weight * C / np.sqrt(M) if prof["noise"] == "clip" else float(np.sqrt(np.mean(x ** 2))))
        x += noise_rng.normal(0.0, sd, len(x))
    pi = None
    if prof["shuffle"] and len(x):
        pi = perm_for(perm_seed, t, len(x))
        x = x[pi]                                   # bs chỉ thấy x[π]
    info = dict(enc_s=enc_s, pi=pi, sd=sd, norm=norm, delta_clipped=delta, weight=weight)   # delta_clipped đã ×weight
    return ciphertexts, x.astype(np.float32), idx_he, idx_plain, info
