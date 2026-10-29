# -*- coding: utf-8 -*-
"""Chính sách chọn p (CLAUDE.md mục 8; logic reference_sim/integrate.py `round_time`, `choose_p`).

PHƯƠNG ÁN CHÍNH: `fixed10` — FAS p = 0,1 cố định mọi vòng, khóa xáo π_t CHỈ ở iot1/iot2 (bs cộng mù), không đọc SINR để
chọn p. So sánh với `no_he` (p = 0), `full_he` (p = 1), `fas10`/`fas50` (FAS gốc: bs giữ khóa xáo).
`adaptive` (chọn p theo SINR) là HƯỚNG PHÁT TRIỂN — giữ lại, chỉ chạy khi policy == 'adaptive'.

  T(p) = T_train + T_tx(D_up(p)) [+ T_enc(p) + T_pdec(p) + T_tx(D_share(p)) nếu có HE]   (dự báo, ghi log mọi policy)
  T_tx: tốc độ từng thiết bị R_k = B·f·log2(1+γ_k) (f = ½ qua UAV); NOMA phát đồng thời -> max, OMA -> tổng.
  adaptive: p lớn nhất trong lưới {p_min, p_min+p_step, …, 1} có T(p) ≤ T_max − δ; không có -> p_min.
  Cờ trễ hạn (dự báo và đo) vẫn so với T_max.
Bảng chi phí: costs/ ĐO TRÊN MÁY NÀY (bước 3, tools/bench_costs.py; T đã ×fidelity quy về Pi 4).

T_max (config `t_max`): số giây, hoặc `auto` = T_train_Pi4 + `t_max_margin_s`
  (T_train_Pi4 = costs/cost_fixed.json T_train_epoch_s; margin 26 s = khoảng reference dành cho mã hóa + truyền:
   50 − 23,8 ≈ 26 s). Máy này: 36,9 + 26 ≈ 62,9 s.
δ (config `safety_margin_s`): biên an toàn vì round_time (như reference) dự báo thấp hơn đo thật 0,4–1,1 s
  (bỏ qua bs cộng/ghép, cập nhật, phần đầu gói); δ = phân vị 95% của lệch, làm tròn lên 0,5 s (tools/calib_margin.py).
"""
import os
import csv
import json
import numpy as np

COSTS = os.environ.get("COSTS", "costs")
# p cố định theo policy. fixed10 (chính) và fas10 cùng p = 0,1; khác ở chỗ ai giữ khóa xáo (fixed10: chỉ thiết bị;
# fas10: cả bs — docker-compose.fas.yml).
_FIXED = {"no_he": 0.0, "full_he": 1.0, "fixed10": 0.1, "fas10": 0.1, "fas50": 0.5}


def fixed_p(policy):
    """p cố định của policy. Ngoài bảng _FIXED: `fixedNN` = FAS p = NN% (bội 10, 10–90), khóa xáo chỉ ở thiết bị —
    như fixed10 với tỉ lệ khác (vd. fixed20, fixed30). Không phải policy p cố định -> KeyError."""
    if policy in _FIXED:
        return _FIXED[policy]
    if policy.startswith("fixed") and policy[5:].isdigit() and int(policy[5:]) % 10 == 0 and 10 <= int(policy[5:]) <= 90:
        return int(policy[5:]) / 100.0
    raise KeyError(f"policy không hợp lệ: {policy} (no_he, full_he, fixed10…fixed90, fas10, fas50, adaptive)")


class CostModel:
    """Bảng chi phí đo ở bước 3 (thang Pi 4): với mỗi p = 0; 0,1; …; 1 có kích thước UPLOAD (D_up_bytes), kích
    thước phần giải mã (D_share_bytes), thời gian mã hóa (T_enc_s) và giải mã từng phần (T_pdec_s); cộng thời gian
    huấn luyện 1 epoch T_train (cost_fixed.json). bs dùng bảng này để DỰ BÁO thời gian vòng trước khi chọn p."""

    def __init__(self, costs_dir=COSTS):
        with open(os.path.join(costs_dir, "cost_table.csv"), newline="") as f:
            self.rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        self.fixed = json.load(open(os.path.join(costs_dir, "cost_fixed.json"), encoding="utf-8"))
        self.T_train = float(self.fixed["T_train_epoch_s"])
        # thời gian huấn luyện RIÊNG từng thiết bị theo phân hoạch (tools/bench_costs.py --train-only); không có => T_train
        p = os.path.join(costs_dir, "train_time.json")
        self.train_time = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

    def row(self, p):
        """Dòng chi phí ứng với p (làm tròn về bội 0,1)."""
        return self.rows[int(round(p * 10))]   # bảng theo p = 0; 0,1; …; 1 như reference

    def train_times(self, split):
        """(T_train iot1, T_train iot2) — s, thang Pi 4 — cho phân hoạch `split`.

        non-IID: hai thiết bị khác số ảnh (21.619 / 28.381) nên huấn luyện khác thời gian (đo riêng, train_time.json).
        IID (hoặc chưa đo riêng): cả hai dùng T_train của bảng chính (25.000 ảnh) — giống integrate.round_time.
        """
        d = self.train_time.get(split)
        if d:
            return float(d["1"]["T_train_s"]), float(d["2"]["T_train_s"])
        return self.T_train, self.T_train


_cm = None


def cost_model(costs_dir=None):
    """CostModel dùng chung (đọc file một lần); truyền costs_dir để đọc lại từ thư mục khác (test)."""
    global _cm
    if _cm is None or costs_dir is not None:
        _cm = CostModel(costs_dir or COSTS)
    return _cm


def resolve_t_max(config, cm=None):
    """T_max (s, thang Pi 4). `auto` = T_train_Pi4 + t_max_margin_s.

    T_max là hạn chót của một vòng FL: vòng đo được (thang Pi 4) > T_max thì bị đánh dấu trễ hạn (late).
    Máy này: 36,93 + 26 = 62,93 s.
    """
    t = config.get("t_max", 50)
    if str(t).lower() == "auto":
        cm = cm or cost_model()
        return cm.T_train + float(config.get("t_max_margin_s", 26))
    return float(t)


def rates(config, gam):
    """Tốc độ (bit/s) từ SINR như reference_sim/channel.rates."""
    mode = config["channel"].get("mode", "uav_noma_pc")
    f = 0.5 if mode.startswith("uav") else 1.0
    return float(config["channel"]["B"]) * f * np.log2(1 + np.asarray(gam, float))


def tx_time(config, gam, bits):
    """Thời gian để CẢ 2 thiết bị gửi xong `bits` mỗi bên. gam: (..., 2).

    NOMA: 2 thiết bị phát đồng thời => xong khi thiết bị chậm hơn xong (max). OMA: lần lượt => cộng (sum).
    """
    t = bits / rates(config, gam)
    mode = config["channel"].get("mode", "uav_noma_pc")
    return t.max(-1) if "noma" in mode else t.sum(-1)


# [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
# Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
#   D(p) <= R_min * (T_max - T_comp)
# với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
# Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
# Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
# (Cài đặt: T(p) = round_time — gồm cả mã hóa, giải mã từng phần, truyền PARTIAL_DEC — so với T_max − δ. Với policy p cố
#  định, round_time chỉ dùng để GHI dự báo T_pred vào log, không ảnh hưởng p.)
def round_time(config, p, gam, he=True, cm=None):
    """DỰ BÁO thời gian một vòng (s, thang Pi 4) nếu mã hóa tỉ lệ p với kênh gam — như integrate.round_time.

    T = max_k(T_train_k + truyền UPLOAD_k) [+ mã hóa + giải mã từng phần + truyền PARTIAL_DEC nếu có HE và p > 0]
    (NOMA; OMA: max_k T_train_k + tổng truyền). T_train_k riêng từng thiết bị (CostModel.train_times): non-IID thiết
    bị nhiều ảnh hơn học lâu hơn. Khi T_train_1 = T_train_2 (IID) công thức trùng ĐÚNG TỪNG BIT integrate.round_time
    (T_train + max_k truyền_k).
    Bỏ qua thời gian bs cộng/ghép, cập nhật mô hình và phần đầu gói => dự báo thấp hơn đo thật 0,4–1,1 s; vì vậy
    adaptive chọn p với ngưỡng T_max − δ. he=False cho no_he (không có pha HE).
    """
    cm = cm or cost_model()
    r = cm.row(p)
    Tk = np.array(cm.train_times(config.get("devices", {}).get("split", "iid")))
    t_up = r["D_up_bytes"] * 8 / rates(config, gam)          # (..., 2): thời gian truyền UPLOAD từng thiết bị
    if "noma" in config["channel"].get("mode", "uav_noma_pc"):
        t = (Tk + t_up).max(-1)                               # phát đồng thời: thiết bị xong muộn nhất quyết định
    else:
        t = Tk.max() + t_up.sum(-1)                           # OMA: phát lần lượt sau khi cả hai học xong
    if he and p > 0:
        t = t + r["T_enc_s"] + r["T_pdec_s"] + tx_time(config, gam, r["D_share_bytes"] * 8)
    return t


# [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
# Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
#   D(p) <= R_min * (T_max - T_comp)
# với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
# Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
# Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
def grid(config):
    """Lưới p adaptive được chọn: {p_min, p_min + p_step, …, 1} = {0,2; 0,3; …; 1} với config mặc định."""
    p_min = float(config.get("p_min", 0.2))
    step = float(config.get("p_step", 0.1))
    return np.round(np.arange(p_min, 1.0001, step), 1)


# [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
# Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
#   D(p) <= R_min * (T_max - T_comp)
# với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
# Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
# Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
def adaptive_p(config, gam, t_max, cm=None):
    """gam: (n, 2). Trả (p (n,), dự báo trễ hạn (n,)) — như integrate.choose_p.

    Thử p tăng dần trên lưới, giữ p LỚN NHẤT (bảo vệ nhiều nhất) mà vòng vẫn kịp `t_max` (ở đây là T_max − δ).
    Không p nào kịp (kênh quá xấu) => p_min và dự báo trễ hạn. Tính được cho n kênh cùng lúc (test, quét t_max).
    """
    gam = np.atleast_2d(gam)
    n = gam.shape[0]
    p = np.full(n, float(config.get("p_min", 0.2)))
    late = np.ones(n, bool)
    for q in grid(config):                     # tăng dần; giữ p lớn nhất còn kịp hạn
        ok = round_time(config, q, gam, cm=cm) <= t_max
        p[ok] = q
        late[ok] = False
    return p, late


# [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
# Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
#   D(p) <= R_min * (T_max - T_comp)
# với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
# Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
# Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
# (t_select = T_max − δ: ngưỡng CHỌN p của adaptive; các policy khác chỉ ghi giá trị này vào log.)
def t_select(config, cm=None):
    """Ngưỡng dùng để CHỌN p: T_max − δ (safety_margin_s)."""
    return resolve_t_max(config, cm) - float(config.get("safety_margin_s", 0.0))


def decide(config, g1, g2, t):
    """Quyết định của bs cho vòng t: dict(p, T_pred (thang Pi 4), late_pred (so với T_max), t_max, t_select).

    Gọi ở bs/server.py ngay sau CHANNEL_REPORT (bs biết SINR g1, g2 của vòng). adaptive: p theo kênh; các policy
    khác: p cố định (_FIXED). p được gửi tới 2 iot trong ROUND_START. T_pred ghi log để so với thời gian đo thật.
    """
    policy = config.get("policy", "adaptive")
    cm = cost_model()
    t_max = resolve_t_max(config, cm)
    t_sel = t_select(config, cm)
    gam = np.array([[g1, g2]])
    if policy == "adaptive":
        # [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
        # Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
        #   D(p) <= R_min * (T_max - T_comp)
        # với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
        # Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
        # Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
        p, late = adaptive_p(config, gam, t_sel, cm)
        p = float(p[0])
    else:
        # p cố định theo policy (fixed10 — phương án chính —, fas10, fas50, no_he, full_he): KHÔNG đọc SINR để chọn p;
        # kênh vẫn mô phỏng (uav bóp tốc độ) để đo thời gian truyền thật.
        p = fixed_p(policy)
    T = float(round_time(config, p, gam, he=(policy != "no_he"), cm=cm)[0])
    return {"p": p, "T_pred": T, "late_pred": bool(T > t_max), "t_max": t_max, "t_select": t_sel}


# [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
# Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
#   D(p) <= R_min * (T_max - T_comp)
# với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
# Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
# Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
def choose_p(config, g1, g2, t):
    """Chỉ lấy p của decide (giao diện giống integrate.choose_p, dùng trong test so lịch p với reference_sim)."""
    return decide(config, g1, g2, t)["p"]
