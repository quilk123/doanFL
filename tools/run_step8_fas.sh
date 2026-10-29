#!/bin/sh
# Bước 8 (đổi hướng FAS, 28/10/2026): IID, seed 0, 15 vòng
#   uav_noma_pc : no_he fixed10 full_he fas10
#   direct_noma : no_he fixed10 full_he   (nén thời gian kênh TIME_SCALE=auto)
# 3 tập train/val/test; rồi bảng bước 8, tấn công (góc nhìn bs / kẻ nghe lén), hình so sánh kiểu E18/E20,
# hình MATLAB, chạy lại toàn bộ test. Kết quả bước 8 cũ (adaptive là đề xuất) -> experiments/KET_QUA_CU/step8_v1_adaptive/.
# Chạy từ thư mục testbed (Git Bash): sh tools/run_step8_fas.sh   (~4 giờ; cấu hình đã đủ vòng thì bỏ qua)
set -u
TB_W=$(cygpath -w "$PWD")
REF_W=$(cygpath -w "/c/Users/ADMIN/Downloads/testbed_starter/testbed_starter/reference_sim")
dk() { MSYS_NO_PATHCONV=1 docker run --rm -e PYTHONUNBUFFERED=1 -e PYTHONIOENCODING=utf-8 -v "$TB_W:/tb" -w /tb "$@"; }

# 3 TẬP (train/val/test, 28/10/2026): kết quả bản FAS 2 tập -> KET_QUA_CU (bản adaptive đã ở step8_v1_adaptive)
OLD=experiments/KET_QUA_CU/step8_v2_fas_2tap
if [ ! -d "$OLD" ]; then
  mkdir -p "$OLD"
  for x in experiments/step8/* experiments/ref_compare experiments/attack experiments/fas experiments/regression/20261028_1526; do
    [ -e "$x" ] && mv "$x" "$OLD/"
  done
  echo "[fas] kết quả bước 8 bản FAS 2 tập -> $OLD"
fi
mkdir -p experiments/step8 experiments/attack experiments/fas

echo "[fas] $(date +%H:%M:%S) build image"
docker compose build -q 2>&1 | grep -i error
echo "[fas] $(date +%H:%M:%S) uav_noma_pc IID"
sh tools/run_step8.sh uav_noma_pc iid 15 no_he fixed10 full_he fas10
echo "[fas] $(date +%H:%M:%S) direct_noma IID (nén auto)"
TIME_SCALE=auto sh tools/run_step8.sh direct_noma iid 15 no_he fixed10 full_he

# (không so reference_sim: testbed là nguồn kết quả báo cáo; từ khi tách 3 tập, dữ liệu huấn luyện khác reference_sim)

echo "[fas] $(date +%H:%M:%S) bảng bước 8, tấn công, hình so sánh"
dk testbed-tools python tools/step8_report.py
dk --cpus 4 --memory 6g -e NTHREADS=4 testbed-tools python tools/attack_eval.py > experiments/attack/attack_eval.log 2>&1
dk testbed-tools python tools/fas_report.py
sh tools/run_plots.sh | grep -E "XONG|error"

echo "[fas] $(date +%H:%M:%S) chạy lại toàn bộ test"
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
run test_fedavg_fixed10 --cpus 6 testbed-tools python tests/test_fedavg_fixed10.py
run test_robustness --cpus 2 testbed-tools python -O tests/test_robustness.py
if sh tests/check_secrets_docker.sh > "$R/check_secrets_docker.log" 2>&1; then st=ĐẠT; else st=LỖI; fi
echo "check_secrets_docker: $st" | tee -a "$R/summary.txt"
echo "[fas] $(date +%H:%M:%S) XONG"
