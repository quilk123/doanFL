#!/bin/sh
# CHẠY TIẾP bước 8 non-IID sau khi bs dùng T_train RIÊNG từng thiết bị (costs/train_time.json) — xem GHI_CHU_CHAY_LAI.md.
# Kết quả non-IID cũ (T_train chung) đã ở experiments/KET_QUA_CU/step8_noniid_T_chung/. Cấu hình đã đủ vòng thì bỏ qua.
# Chạy từ thư mục testbed (Git Bash): sh tools/redo_noniid.sh   (~2 giờ)
set -u
TB_W=$(cygpath -w "$PWD")
REF_W=$(cygpath -w "/c/Users/ADMIN/Downloads/testbed_starter/testbed_starter/reference_sim")
dk() { MSYS_NO_PATHCONV=1 docker run --rm -e PYTHONUNBUFFERED=1 -e PYTHONIOENCODING=utf-8 -v "$TB_W:/tb" -w /tb "$@"; }

echo "[redo] $(date +%H:%M:%S) build lại image"
docker compose build -q 2>&1 | grep -i error


echo "[redo] $(date +%H:%M:%S) bước 8 non-IID"
sh tools/run_step8.sh uav_noma_pc noniid 15

echo "[redo] $(date +%H:%M:%S) so reference_sim (adaptive non-IID: lịch p mới)"
dk --cpus 2 -e NTHREADS=2 -e OMP_NUM_THREADS=2 -v "$REF_W:/ref:ro" testbed-tools \
  python tools/ref_compare.py run experiments/step8/uav_noma_pc_noniid_adaptive > experiments/ref_compare/redo_noniid.log 2>&1
dk testbed-tools python tools/ref_compare.py report
echo "[redo] $(date +%H:%M:%S) bảng bước 8 + tấn công"
dk testbed-tools python tools/step8_report.py
dk --cpus 4 --memory 6g -e NTHREADS=4 testbed-tools python tools/attack_eval.py > experiments/attack/attack_eval.log 2>&1
sh tools/run_plots.sh | grep -E "XONG|error"

echo "[redo] $(date +%H:%M:%S) chia lại IID + chạy lại toàn bộ test"
dk testbed-tools python tools/prep_data.py --split-only 2>&1 | grep "\[prep\]"
R="experiments/regression/$(date +%Y%m%d_%H%M)"
mkdir -p "$R"
run() {
  name=$1; shift
  if MSYS_NO_PATHCONV=1 docker run --rm -e NTHREADS=2 -e OMP_NUM_THREADS=2 -e PYTHONUNBUFFERED=1 \
       -v "$TB_W:/tb" -w /tb "$@" > "$R/$name.log" 2>&1; then st=ĐẠT; else st=LỖI; fi
  echo "$name: $st — $(grep '\[TEST\] ĐẠT\|Error\|assert' "$R/$name.log" | tail -1 | cut -c1-160)" | tee -a "$R/summary.txt"
}
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
echo "[redo] $(date +%H:%M:%S) XONG"
