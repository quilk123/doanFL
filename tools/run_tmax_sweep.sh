#!/bin/sh
# [HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]
# Cơ chế chọn tỷ lệ mã hóa p theo SINR từng vòng: chọn p lớn nhất sao cho
#   D(p) <= R_min * (T_max - T_comp)
# với R_min là tốc độ của thiết bị yếu hơn sau SIC qua UAV AF.
# Phương án chính của đồ án là FAS p cố định (policy 'fixed10').
# Giữ lại để so sánh và cho mục "Hướng phát triển"; chỉ chạy khi policy == 'adaptive'.
# Thí nghiệm quét T_max trên testbed THẬT (4 container): adaptive, uav_noma_pc, mỗi T_max chạy ROUNDS vòng với CÙNG chuỗi kênh
# (seed cố định) => khác biệt chỉ do T_max. Chạy từ thư mục testbed:  sh tools/run_tmax_sweep.sh [ROUNDS] [T_max ...]
# Sau đó: python tools/tmax_sweep.py summarize
set -eu
ROUNDS=${1:-3}
shift 2>/dev/null || true
TLIST=${*:-"50 60 70 80 90"}
BASE=experiments/KET_QUA_CU/tmax_sweep_adaptive/measured
for T in $TLIST; do
  D="$BASE/tmax_$T"
  mkdir -p "$D"
  sed -e "s/^rounds: .*/rounds: $ROUNDS/" -e "s/^t_max: [^ ]*/t_max: $T/" -e "s/^policy: [^ ]*/policy: adaptive/" \
      -e "s/^  mode: [^ ]*/  mode: uav_noma_pc/" config.yaml > "$D/config.yaml"
  cat > "$D/override.yml" <<EOF
services:
  bs:   {volumes: ["./$D/config.yaml:/app/config.yaml:ro"]}
  uav:  {volumes: ["./$D/config.yaml:/app/config.yaml:ro"]}
  iot1: {volumes: ["./$D/config.yaml:/app/config.yaml:ro"]}
  iot2: {volumes: ["./$D/config.yaml:/app/config.yaml:ro"]}
EOF
  echo "[sweep] T_max = $T s, $ROUNDS vòng"
  rm -f results/round_*.json results/log.jsonl results/setup.json
  docker compose -f docker-compose.yml -f "$D/override.yml" up --abort-on-container-exit --exit-code-from bs 2>&1 \
    | grep -E "\[bs\] (policy|vòng)" || true
  docker compose -f docker-compose.yml -f "$D/override.yml" down >/dev/null 2>&1
  cp results/round_*.json results/setup.json "$D/"
done
echo "[sweep] xong: $BASE"
