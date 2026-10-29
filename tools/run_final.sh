#!/bin/sh
# Chạy nối tiếp sau tools/run_after_step8.sh (bước 8 + chạy lại test) — phần tổng hợp cuối:
#   1. chờ không còn run_after_step8.sh / run_step8.sh
#   2. so với reference_sim mọi cấu hình bước 8 (3 container song song) + report
#   3. bảng/hình bước 8, đánh giá tấn công trên mọi capture
# Log: experiments/final_<thời điểm>.log. Chạy từ thư mục testbed: REF_SIM=<reference_sim> sh tools/run_final.sh
set -u
TB_W=$(cygpath -w "$PWD" 2>/dev/null || echo "$PWD")
REF_W=$(cygpath -w "$REF_SIM" 2>/dev/null || echo "$REF_SIM")
idle=0
while [ $idle -lt 6 ]; do                       # 6 lần × 10 s liên tiếp không thấy => đã xong
  # chỉ xét tiến trình có lệnh ĐÚNG là `sh tools/run_...` (cột 6–7 của ps -ef); grep cả dòng sẽ khớp nhầm dòng lệnh
  # cha `bash -c "... sh tools/run_after_step8.sh; sh tools/run_final.sh"` và chờ mãi
  if ps -ef | awk '$6 == "sh" && $7 ~ /run_(after_)?step8\.sh$/ {f = 1} END {exit !f}'; then idle=0; else idle=$((idle + 1)); fi
  sleep 10
done
dk() { MSYS_NO_PATHCONV=1 docker run --rm -e PYTHONUNBUFFERED=1 -v "$TB_W:/tb" -w /tb "$@"; }

echo "[cuối] $(date +%H:%M:%S) so với reference_sim"
set -- experiments/step8/*/
n=0
for g in 0 1 2; do                              # chia cấu hình thành 3 nhóm, mỗi nhóm một container
  grp=""; i=0
  for d in "$@"; do [ $((i % 3)) -eq $g ] && grp="$grp $d"; i=$((i + 1)); done
  [ -n "$grp" ] && dk --cpus 2 -e NTHREADS=2 -e OMP_NUM_THREADS=2 -v "$REF_W:/ref:ro" testbed-tools \
      python tools/ref_compare.py run $grp > "experiments/ref_compare_g$g.log" 2>&1 &
done
wait
dk testbed-tools python tools/ref_compare.py report

echo "[cuối] $(date +%H:%M:%S) bảng/hình bước 8"
dk testbed-tools python tools/step8_report.py
echo "[cuối] $(date +%H:%M:%S) đánh giá tấn công"
dk --cpus 1 --memory 6g -e NTHREADS=1 testbed-tools python tools/attack_eval.py
echo "[cuối] $(date +%H:%M:%S) XONG"
