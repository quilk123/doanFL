#!/bin/sh
# Chạy nối tiếp sau khi `run_step8.sh uav_noma_pc iid/noniid` (đang chạy) xong — không đụng lượt đang đo:
#   1. chờ không còn run_step8.sh; chạy nốt uav_noma_pc noniid nếu chưa đủ
#   2. build lại image (nén thời gian kênh, timeout giao thức)
#   3. direct_noma iid × 5 policy × 3 vòng, nén auto (TIME_SCALE=auto)
#   4. chạy lại toàn bộ test -> experiments/regression/<thời điểm>/<test>.log + summary.txt
# Chạy từ thư mục testbed: sh tools/run_after_step8.sh
set -u
TB_W=$(cygpath -w "$PWD" 2>/dev/null || echo "$PWD")
idle=0
while [ $idle -lt 6 ]; do                       # 6 lần × 10 s liên tiếp không thấy run_step8 => đã xong
  # chỉ xét tiến trình có lệnh ĐÚNG là `sh tools/run_step8.sh` (cột 6–7), không khớp nhầm dòng lệnh cha
  if ps -ef | awk '$6 == "sh" && $7 ~ /run_step8\.sh$/ {f = 1} END {exit !f}'; then idle=0; else idle=$((idle + 1)); fi
  sleep 10
done
echo "[sau8] $(date +%H:%M:%S) uav_noma_pc xong; chạy nốt noniid nếu thiếu"
sh tools/run_step8.sh uav_noma_pc noniid 15

echo "[sau8] $(date +%H:%M:%S) build lại image"
docker compose build -q 2>&1 | grep -i error
echo "[sau8] $(date +%H:%M:%S) direct_noma, nén auto"
TIME_SCALE=auto sh tools/run_step8.sh direct_noma iid 3

R="experiments/regression/$(date +%Y%m%d_%H%M)"
mkdir -p "$R"
run() {   # run <tên> <docker args...> -- python args
  name=$1; shift
  if MSYS_NO_PATHCONV=1 docker run --rm -e NTHREADS=2 -e OMP_NUM_THREADS=2 -e PYTHONUNBUFFERED=1 \
       -v "$TB_W:/tb" -w /tb "$@" > "$R/$name.log" 2>&1; then st=ĐẠT; else st=LỖI; fi
  echo "$name: $st — $(grep '\[TEST\] ĐẠT\|Error\|assert' "$R/$name.log" | tail -1 | cut -c1-160)" | tee -a "$R/summary.txt"
}
echo "[sau8] $(date +%H:%M:%S) chạy lại test -> $R"
run test_round --cpus 6 testbed-tools python tests/test_round.py
run test_step2_fl --cpus 6 testbed-tools python tests/test_step2_fl.py
run test_step3_pi4 --cpus 1.6 --memory 4g --memory-swap 4g testbed-tools python tests/test_step3_pi4.py
run test_step4_channel --cpus 6 testbed-tools python tests/test_step4_channel.py
run test_step5_he --cpus 6 testbed-tools python tests/test_step5_he.py
run test_step6_protect --cpus 6 testbed-tools python tests/test_step6_protect.py
run test_step7_policy --cpus 6 testbed-tools python tests/test_step7_policy.py
run test_robustness --cpus 2 testbed-tools python -O tests/test_robustness.py
if sh tests/check_secrets_docker.sh > "$R/check_secrets_docker.log" 2>&1; then st=ĐẠT; else st=LỖI; fi
echo "check_secrets_docker: $st" | tee -a "$R/summary.txt"
echo "[sau8] $(date +%H:%M:%S) XONG"
