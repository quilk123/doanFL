#!/bin/sh
# Đánh giá tấn công (tools/attack_eval.py) trên dữ liệu bs nhận vòng 1 — góc nhìn trạm gốc và kẻ nghe lén.
# Chạy từ thư mục testbed (Git Bash; PowerShell dùng .\attack.bat cùng tham số):
#   sh tools/attack.sh                          # mọi cấu hình bước 8 -> experiments/attack/ (KẾT QUẢ CHÍNH, ghi đè)
#   sh tools/attack.sh fixed10 fixed20 fixed30  # các policy/tỉ lệ đã chạy ở bước 8 (mọi kênh)
#                                               #   -> experiments/attack/rieng_<thời điểm>/
#   sh tools/attack.sh --demo                   # lần demo gần nhất -> experiments/demo/<thời điểm>/attack/
#   sh tools/attack.sh experiments/step8/uav_noma_pc_iid_fixed20 ...   # thư mục bất kỳ (có capture/)
# Mất ~5–15 phút (tính gradient từng ảnh cho 1.000 ảnh thành viên + 1.000 không thành viên mỗi thiết bị).
# In bảng tóm tắt: % cập nhật bên tấn công khôi phục được (thấp là tốt), AUC suy luận thành viên (0,5 = không đoán được).
set -u
TB_W=$(cygpath -w "$PWD" 2>/dev/null || echo "$PWD")
OUT=""
if [ $# -eq 0 ]; then
  DIRS=""                                                # attack_eval tự lấy mọi cấu hình bước 8
elif [ "$1" = "--demo" ]; then
  last=$(ls -d experiments/demo/2*/ 2>/dev/null | sort | tail -1)
  [ -n "$last" ] || { echo "[attack] chưa có lần demo nào"; exit 1; }
  DIRS=$(ls -d "$last"*/ | grep -v "/fig/\|/attack/")
  OUT="${last}attack"
else
  DIRS=""
  for a in "$@"; do
    if [ -d "$a" ]; then DIRS="$DIRS $a"; continue; fi
    m=$(ls -d experiments/step8/*_"$a" experiments/step8/*_"$a"_ts-* 2>/dev/null)
    [ -n "$m" ] || { echo "[attack] chưa có kết quả bước 8 cho '$a' (chạy: sh tools/run_step8.sh uav_noma_pc iid 15 $a)"; exit 1; }
    DIRS="$DIRS $m"
  done
  OUT="experiments/attack/rieng_$(date +%Y%m%d_%H%M%S)"
fi
echo "[attack] cấu hình:${DIRS:- mọi cấu hình bước 8}"
MSYS_NO_PATHCONV=1 docker run --rm --cpus 4 --memory 6g -e NTHREADS=4 -e PYTHONIOENCODING=utf-8 -e PYTHONUNBUFFERED=1 \
  -v "$TB_W:/tb" -w /tb testbed-tools python tools/attack_eval.py $DIRS ${OUT:+--out "$OUT"} \
  | grep -v "^\[attack\] {\|gradient từng ảnh"
