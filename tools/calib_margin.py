# -*- coding: utf-8 -*-
# [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
# Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
#   D(p) <= R_min * (T_max - T_comp)
# với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
# Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
# Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
"""Hiệu chỉnh biên an toàn δ (config `safety_margin_s`) cho adaptive.

δ = phân vị 95% của (thời gian vòng ĐO trên testbed, thang Pi 4 − thời gian DỰ BÁO của round_time), làm tròn LÊN bội 0,5 s.
Số đo: các vòng testbed thật ở bước 7 (experiments/KET_QUA_CU/tmax_sweep_adaptive/measured/*/round_*.json, tools/run_tmax_sweep.sh).
Nguyên nhân lệch: round_time (như reference_sim) bỏ qua thời gian bs cộng đồng cấu / ghép phần giải mã, cập nhật mô hình,
và phần đầu gói (msgpack, khung, chênh Serialize chuỗi vs tệp) — xem experiments/KET_QUA_CU/tmax_sweep_adaptive/pred_error.csv.
  python tools/calib_margin.py
"""
import os
import glob
import json
import math

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "experiments", "KET_QUA_CU", "tmax_sweep_adaptive", "measured", "*", "round_*.json")


def pred_errors():
    rows = []
    for fn in sorted(glob.glob(SRC)):
        r = json.load(open(fn, encoding="utf-8"))
        tm = r["time"]
        rows.append(dict(T_max=r["t_max"], t=r["t"], p=r["decision"]["p"], T_pred=r["decision"]["T_pred_s"],
                         T_pi4=tm["round_pi4_s"], err=tm["round_pi4_s"] - r["decision"]["T_pred_s"],
                         bs_add=tm["he_add_s"], bs_fuse=tm["he_fuse_s"], update=1.4 * tm["dev1"].get("update_s", 0.0)))
    return rows


def margin(errs, q=95, step=0.5):
    return math.ceil(np.percentile(errs, q) / step) * step


def main():
    rows = pred_errors()
    assert rows, f"chưa có số đo: {SRC}"
    errs = np.array([r["err"] for r in rows])
    d = margin(errs)
    with open(os.path.join(ROOT, "experiments", "tmax_sweep", "pred_error.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    bs = np.array([r["bs_add"] + r["bs_fuse"] + r["update"] for r in rows])
    print(f"[margin] {len(errs)} vòng: lệch đo − dự báo {errs.min():.2f}…{errs.max():.2f} s, trung bình {errs.mean():.2f} s, "
          f"phân vị 95% {np.percentile(errs, 95):.3f} s -> safety_margin_s = {d}")
    print(f"[margin] trong đó bs cộng + ghép + cập nhật (×1,4): trung bình {bs.mean():.2f} s; phần còn lại (gói, khác bảng chi phí) "
          f"{(errs - bs).mean():.2f} s")


if __name__ == "__main__":
    main()
