# -*- coding: utf-8 -*-
"""Mô hình kênh: 2 thiết bị IoT -> UAV đứng yên (AF, độ lợi biến đổi) -> trạm gốc, đường lên NOMA + SIC
(logic reference_sim/channel.py, tham số từ config.yaml `channel`).

Suy hao:
  - Không đối đất (thiết bị-UAV, UAV-BS): Al-Hourani (2014), đô thị a=9,61, b=0,16, η_LoS=1 dB, η_NLoS=20 dB.
  - Mặt đất trực tiếp (thiết bị-BS, không UAV): 3GPP TR 38.901 UMa NLoS + bóng râm 6 dB.
Fading nhỏ: Rician K=5 dB cho liên kết không đối đất, Rayleigh cho mặt đất; quasi-static trong một vòng FL,
độc lập giữa các vòng (seed = (config.seed, t)).
"""
import numpy as np

C = 3e8
MODES = ("uav_noma_pc", "uav_noma", "uav_oma", "direct_noma")


class Params:
    """Tham số kênh đọc từ config (mặc định = reference_sim/channel.py Params).

    fc tần số sóng mang (Hz); B băng thông (Hz); P_dev/P_uav công suất phát thiết bị/UAV (dBm); NF hệ số nhiễu máy
    thu (dB); h_* độ cao (m); bs/uav/devs tọa độ mặt phẳng (m); a, b, η: tham số Al-Hourani môi trường đô thị;
    K_dB hệ số Rician; shadow_dB bóng râm mặt đất; los_mode "mean" (P_LoS trung bình) hay "random" (bốc LoS/NLoS).
    """

    def __init__(self, config):
        c = config["channel"]
        self.fc = float(c.get("fc", 2e9))
        self.B = float(c.get("B", 1e6))
        self.P_dev_dBm = float(c.get("P_dev_dBm", 23))
        self.P_uav_dBm = float(c.get("P_uav_dBm", 30))
        self.NF_dB = float(c.get("NF_dB", 7))
        self.h_bs = float(c.get("h_bs", 25))
        self.h_uav = float(c.get("h_uav", 100))
        self.h_dev = float(c.get("h_dev", 1.5))
        self.bs = np.array(c.get("bs", [0, 0]), float)
        self.uav = np.array(c.get("uav", [700, 0]), float)
        self.devs = np.array(c.get("devices", [[900, 0], [1200, 0]]), float)
        a2g = c.get("a2g", {})
        self.a = float(a2g.get("a", 9.61))
        self.b = float(a2g.get("b", 0.16))
        self.eta_los = float(a2g.get("eta_los_dB", 1.0))
        self.eta_nlos = float(a2g.get("eta_nlos_dB", 20.0))
        self.K_dB = float(c.get("K_dB", 5.0))
        self.shadow_dB = float(c.get("shadow_dB", 6.0))
        self.los_mode = c.get("los_mode", "mean")


def dbm2w(x):
    """Đổi công suất dBm -> W."""
    return 10 ** ((x - 30) / 10)


def noise_w(P):
    """Công suất nhiễu nhiệt máy thu (W): −174 dBm/Hz + 10·log10(B) + NF."""
    return dbm2w(-174 + 10 * np.log10(P.B) + P.NF_dB)


def pl_a2g_db(P, r, dh):
    """Suy hao không đối đất Al-Hourani (dB) giữa hai điểm cách nhau r (ngang) và dh (cao).

    Góc ngẩng θ càng lớn thì xác suất LoS P_LoS(θ) = 1/(1 + a·exp(−b(θ − a))) càng cao. Suy hao = suy hao không gian
    tự do (FSPL) + suy hao thêm trung bình: η_LoS·P_LoS + η_NLoS·(1 − P_LoS). Dùng cho thiết bị–UAV và UAV–bs.
    """
    d = np.sqrt(r ** 2 + dh ** 2)
    th = np.degrees(np.arctan2(dh, r))
    plos = 1 / (1 + P.a * np.exp(-P.b * (th - P.a)))
    fspl = 20 * np.log10(4 * np.pi * P.fc * d / C)
    return fspl + P.eta_los * plos + P.eta_nlos * (1 - plos)


def pl_ground_db(P, r):
    """Suy hao mặt đất trực tiếp thiết bị–bs (dB), 3GPP TR 38.901 UMa NLoS (khoảng cách 3 chiều d3).

    Chỉ dùng cho chế độ direct_noma (không UAV); suy hao lớn hơn nhiều so với không đối đất => tốc độ thấp.
    """
    d3 = np.sqrt(r ** 2 + (P.h_bs - P.h_dev) ** 2)
    return 13.54 + 39.08 * np.log10(d3) + 20 * np.log10(P.fc / 1e9) - 0.6 * (P.h_dev - 1.5)


def rician_pow(rng, K_dB, size):
    """Độ lợi công suất fading Rician |h|² (trung bình 1): thành phần LoS cố định + thành phần tán xạ Gauss phức.

    K = tỉ lệ công suất LoS / tán xạ (K = 5 dB: có LoS khá mạnh, hợp liên kết không đối đất).
    """
    K = 10 ** (K_dB / 10)
    los = np.sqrt(K / (K + 1))
    s = np.sqrt(1 / (2 * (K + 1)))
    h = los + s * (rng.standard_normal(size) + 1j * rng.standard_normal(size))
    return np.abs(h) ** 2


def rayleigh_pow(rng, size):
    """Độ lợi công suất fading Rayleigh |h|² ~ Exp(1) (không có LoS) — liên kết mặt đất, hoặc NLoS khi los_mode random."""
    return rng.exponential(1.0, size)


def draw(P, rng, n=1, devs=None):
    """Sinh n lần kênh. Trả về dict các SINR (tuyến tính) cho các chế độ, mỗi mảng shape (n, 2).
    Giữ đúng thứ tự rút số ngẫu nhiên của reference_sim để cùng rng ra cùng γ.

    Ký hiệu: L1 suy hao thiết bị–UAV, L2 suy hao UAV–bs, Ld suy hao thiết bị–bs; h, g, hd fading tương ứng;
    a = công suất mỗi thiết bị nhận được tại UAV. UAV khuếch đại-chuyển tiếp (AF) với độ lợi G2 = P_uav/(tổng nhận
    + nhiễu), nên nhiễu tại UAV cũng bị khuếch đại sang bs (số hạng beta·s2 trong mẫu số SINR).
    Các chế độ trả về:
      uav_noma    : 2 thiết bị phát đồng thời, bs giải SIC — thiết bị mạnh giải trước (chịu nhiễu thiết bị kia)
      uav_noma_pc : như trên + điều khiển công suất: thử 61 hệ số giảm công suất của thiết bị mạnh và cả 2 thứ tự
                    giải SIC, chọn cặp làm min(R1, R2) lớn nhất (công bằng giữa 2 thiết bị) — chế độ chính bước 8
      uav_oma     : mỗi thiết bị phát riêng (không nhiễu lẫn nhau)
      direct_noma : không UAV, gửi thẳng tới bs, NOMA + SIC
    test_step4 kiểm γ trùng từng bit với reference_sim.
    """
    devs = P.devs if devs is None else devs
    s2 = noise_w(P)
    Pd = dbm2w(P.P_dev_dBm)
    Pu = dbm2w(P.P_uav_dBm)
    L1 = np.array([10 ** (-pl_a2g_db(P, np.linalg.norm(d - P.uav), P.h_uav - P.h_dev) / 10) for d in devs])
    L2 = 10 ** (-pl_a2g_db(P, np.linalg.norm(P.uav - P.bs), P.h_uav - P.h_bs) / 10)
    Ld = np.array([10 ** (-pl_ground_db(P, np.linalg.norm(d - P.bs)) / 10) for d in devs])
    h = rician_pow(rng, P.K_dB, (n, 2))
    g = rician_pow(rng, P.K_dB, (n, 1))
    if P.los_mode == "random":
        # Biến thể kiểm chứng: trạng thái LoS/NLoS của từng liên kết không đối đất bốc ngẫu nhiên với xác suất P_LoS(θ);
        # LoS: FSPL + η_LoS và Rician; NLoS: FSPL + η_NLoS và Rayleigh.
        def link(r, dh, size):
            d = np.sqrt(r ** 2 + dh ** 2)
            th = np.degrees(np.arctan2(dh, r))
            plos = 1 / (1 + P.a * np.exp(-P.b * (th - P.a)))
            fspl = 20 * np.log10(4 * np.pi * P.fc * d / C)
            los = rng.random(size) < plos
            pl = fspl + np.where(los, P.eta_los, P.eta_nlos)
            fad = np.where(los, rician_pow(rng, P.K_dB, size), rayleigh_pow(rng, size))
            return 10 ** (-pl / 10) * fad
        hl = np.stack([link(np.linalg.norm(d - P.uav), P.h_uav - P.h_dev, n) for d in devs], 1)
        gl = link(np.linalg.norm(P.uav - P.bs), P.h_uav - P.h_bs, n)[:, None]
        L1 = np.ones(2)
        L2 = 1.0
        h, g = hl, gl                                  # suy hao + fading đã gộp vào h, g
    hd = rayleigh_pow(rng, (n, 2)) * 10 ** (rng.normal(0, P.shadow_dB, (n, 2)) / 10)
    a = Pd * L1 * h                                    # công suất nhận tại UAV từ mỗi thiết bị
    out = {}
    # --- UAV AF + NOMA (hai thiết bị phát đồng thời) ---
    G2 = Pu / (a.sum(1, keepdims=True) + s2)
    beta = G2 * L2 * g
    strong = a.argmax(1)
    gam = np.zeros((n, 2))
    for k in range(2):
        a_k, a_o = a[:, k:k + 1], a[:, 1 - k:2 - k]
        first = (strong == k)[:, None]                 # được giải trước -> chịu nhiễu của thiết bị kia
        # ⚠️ CHÚ Ý: SIC LÝ TƯỞNG — thiết bị giải sau không còn nhiễu dư (ξ = 0), không lan truyền lỗi (giống reference_sim)
        interf = np.where(first, beta * a_o, 0.0)
        gam[:, k:k + 1] = beta * a_k / (interf + beta * s2 + s2)
    out["uav_noma"] = gam
    # --- UAV AF + NOMA có điều khiển công suất: thiết bị mạnh giảm công suất theo hệ số alpha,
    #     chọn alpha và thứ tự giải SIC sao cho min(R1, R2) lớn nhất ---
    best = np.full((n, 2), 0.0)
    best_min = np.full(n, -1.0)
    for al in np.logspace(-3, 0, 61):
        aa = a.copy()
        aa[np.arange(n), strong] *= al
        G2a = Pu / (aa.sum(1, keepdims=True) + s2)
        be = (G2a * L2 * g)[:, 0]
        for first in (0, 1):
            o_ = 1 - first
            gf = be * aa[:, first] / (be * aa[:, o_] + be * s2 + s2)
            # ⚠️ CHÚ Ý: SIC LÝ TƯỞNG — mẫu số không có nhiễu dư của thiết bị giải trước (ξ = 0)
            go = be * aa[:, o_] / (be * s2 + s2)
            cand = np.zeros((n, 2))
            cand[:, first] = gf
            cand[:, o_] = go
            mn = cand.min(1)
            upd = mn > best_min
            best[upd] = cand[upd]
            best_min[upd] = mn[upd]
    out["uav_noma_pc"] = best
    # --- UAV AF + OMA (TDMA, mỗi thiết bị phát riêng, toàn băng) ---
    G2o = Pu / (a + s2)
    bo = G2o * L2 * g
    out["uav_oma"] = bo * a / (bo * s2 + s2)
    # --- Trực tiếp, không UAV, NOMA ---
    c = Pd * Ld * hd
    strong_d = c.argmax(1)
    gd = np.zeros((n, 2))
    for k in range(2):
        first = (strong_d == k)
        # ⚠️ CHÚ Ý: SIC LÝ TƯỞNG (direct_noma)
        gd[:, k] = c[:, k] / (np.where(first, c[:, 1 - k], 0.0) + s2)
    out["direct_noma"] = gd
    out["direct_oma"] = c / s2
    return out


def rates(P, gam, mode):
    """Tốc độ (bit/s). Qua UAV có hệ số 1/2 vì hai chặng bán song công.

    Shannon: R = B·log2(1 + γ). Qua UAV, dữ liệu phải đi 2 chặng (thiết bị->UAV rồi UAV->bs) chia đôi thời gian
    nên tốc độ hiệu dụng còn một nửa.
    """
    f = 0.5 if mode.startswith("uav") else 1.0
    return P.B * f * np.log2(1 + gam)


RED = "\033[1;31m"
RESET = "\033[0m"
# ⚠️ CHÚ Ý: OMA (TDMA) nghĩa là hai thiết bị phát LẦN LƯỢT (thời gian = tổng), nhưng relay của testbed vẫn cho
# phát SONG SONG (thời gian = max) => thời gian truyền uav_oma trong testbed SAI (lạc quan). So sánh NOMA/OMA lấy từ
# reference_sim (channel_exp.py E1–E3, tx_time cộng dồn). Bước 8 không dùng uav_oma.
OMA_WARNING = ("CHÚ Ý: uav_oma trong testbed cho 2 thiết bị phát SONG SONG, không phải TDMA — thời gian truyền SAI; "
               "chỉ tham khảo, so sánh NOMA/OMA lấy từ reference_sim")
_warned = False


def channel_round(config, t):
    """Trả (g1, g2, R1, R2) cho vòng t theo `channel.mode`. g là SINR (tuyến tính), R là byte/s.

    uav gọi hàm này đầu mỗi vòng (uav/relay.py). Kênh sinh với seed (config.seed, t): cùng seed thì mọi policy gặp
    CÙNG chuỗi kênh qua 15 vòng — nhờ vậy so sánh policy công bằng (vd. vòng 10 kênh xấu với mọi policy).
    Kênh cố định trong cả vòng (quasi-static). R chia 8 để đổi bit/s -> byte/s cho bộ bóp tốc độ.
    """
    global _warned
    P = Params(config)
    mode = config["channel"].get("mode", "uav_noma_pc")
    if mode not in MODES:
        raise ValueError(f"channel.mode không hợp lệ: {mode} (chọn {', '.join(MODES)})")
    if mode == "uav_oma" and not _warned:
        print(f"{RED}[uav] ⚠️  {OMA_WARNING}{RESET}", flush=True)
        _warned = True
    rng = np.random.default_rng([int(config.get("seed", 0)), int(t)])
    gam = draw(P, rng, 1)[mode][0]
    R = rates(P, gam, mode) / 8.0
    return float(gam[0]), float(gam[1]), float(R[0]), float(R[1])
