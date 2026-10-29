#!/bin/sh
# Demo chạy thật trên 4 container (iot1, iot2, uav, bs) — KHÔNG đụng số liệu bước 8.
#   sh tools/demo.sh [số vòng FL] [policy ...]       vd. sh tools/demo.sh 3 no_he fixed10 full_he
# Số epoch mỗi client học trước khi gửi: EPOCHS=3 sh tools/demo.sh 3 adaptive  (tự đo lại thời gian huấn luyện một lần)
# Mặc định: 3 vòng, fixed10 (phương án chính), kênh uav_noma_pc, IID, thời gian thật (không nén).
# Demo chạy theo thời gian kênh thật, không nén thời gian truyền.
# Có / không UAV: MODE=uav_noma_pc (có UAV) | MODE=direct_noma (không UAV — kênh rất chậm)
# Kết quả: experiments/demo/<thời điểm>/<policy>/ + bảng chi tiết in cuối + hình experiments/demo/<thời điểm>/fig/ (tự mở).
set -eu
ROUNDS=${1:-3}; [ $# -gt 0 ] && shift
POLICIES=${*:-fixed10}
# Demo mặc định chạy theo thời gian kênh thật; không kế thừa TIME_SCALE từ shell.
TS=off
if ! docker info >/dev/null 2>&1; then echo "[demo] Docker Desktop chưa chạy — mở Docker Desktop rồi chạy lại"; exit 1; fi
if docker ps --format '{{.Names}}' | grep -q "^testbed-"; then
  echo "[demo] đang có container testbed chạy (thí nghiệm khác?) — dừng lại để không làm hỏng phép đo"; exit 1
fi
TB_W=$(cygpath -w "$PWD" 2>/dev/null || echo "$PWD")
dk() { MSYS_NO_PATHCONV=1 docker run --rm -e PYTHONIOENCODING=utf-8 -v "$TB_W:/tb" -w /tb "$@"; }
SPLIT=$(sed -n 's/^  split: *\([a-z]*\).*/\1/p' config.yaml)
EP_CFG=$(sed -n 's/^  local_epochs: *\([0-9]*\).*/\1/p' config.yaml)
EPOCHS=${EPOCHS:-$EP_CFG}            # số epoch mỗi client học trước khi gửi (mặc định theo config.yaml)
MODE=${MODE:-$(sed -n 's/^  mode: *\([a-z_]*\).*/\1/p' config.yaml)}   # kênh: uav_noma_pc (có UAV) | direct_noma (không UAV)
case $MODE in uav_noma_pc|direct_noma|uav_noma) ;; *) echo "[demo] MODE không hợp lệ: $MODE"; exit 1;; esac
if [ "$MODE" = direct_noma ] && [ "$TS" = off ]; then
  echo "[demo] ⚠️  không UAV mà không nén kênh: kênh có vòng chỉ ~0,001 Mbit/s => một vòng có thể mất HÀNG GIỜ (Ctrl+C để dừng)"
fi

# dữ liệu đã chia phải khớp `split` trong config (nếu không: thiết bị học sai dữ liệu mà không báo lỗi)
if ! grep -q "\"split\": \"$SPLIT\"" data/train/client1/meta.json 2>/dev/null; then
  echo "[demo] dữ liệu đang chia khác config ($SPLIT) — chia lại"
  dk testbed-tools python tools/prep_data.py --split-only 2>&1 | grep "\[prep\]"
fi

# đổi số epoch => bs phải biết thời gian huấn luyện MỚI (dự báo vòng, chọn p, t_max, thang Pi 4):
# đo lại một lần trong container giới hạn như Pi 4, lưu bộ chi phí riêng costs_ep<N>_<split>/ (dùng lại lần sau)
CDIR=""
if [ "$EPOCHS" != "$EP_CFG" ]; then
  CDIR="costs_ep${EPOCHS}_${SPLIT}"
  if [ ! -f "$CDIR/train_time.json" ]; then
    echo "[demo] $(date +%H:%M:%S) $EPOCHS epoch/vòng: đo lại thời gian huấn luyện từng client (~$((EPOCHS * 4)) phút, một lần)"
    mkdir -p "$CDIR"
    cp costs/cost_table.csv costs/cost_fixed.json "$CDIR/"
    sed -e "s/^  local_epochs: .*/  local_epochs: $EPOCHS/" config.yaml > "$CDIR/config_bench.yaml"
    dk --cpus 1.6 --memory 4g --memory-swap 4g -e NTHREADS=2 -e OMP_NUM_THREADS=2 testbed-tools \
      python tools/bench_costs.py --train-only --train-reps 3 --config "$CDIR/config_bench.yaml" --out "$CDIR" \
      | grep "T_train ="
    # T_train của bảng chính (dùng cho t_max auto) = trung bình 2 client với số epoch mới
    dk testbed-tools python -c "
import json; d='$CDIR'; t=json.load(open(d+'/train_time.json'))['$SPLIT']; f=json.load(open(d+'/cost_fixed.json'))
f['T_train_epoch_s']=(t['1']['T_train_s']+t['2']['T_train_s'])/2; f['local_epochs']=$EPOCHS
json.dump(f,open(d+'/cost_fixed.json','w'),indent=2,ensure_ascii=False); print('[demo] T_train =', round(f['T_train_epoch_s'],2), 's (Pi 4)')"
  else
    echo "[demo] $EPOCHS epoch/vòng: dùng thời gian huấn luyện đã đo ($CDIR/)"
  fi
fi

OUT="experiments/demo/$(date +%Y%m%d_%H%M%S)"
for POL in $POLICIES; do
  D="$OUT/$POL"; mkdir -p "$D"
  sed -e "s/^rounds: .*/rounds: $ROUNDS/" -e "s/^policy: [^ ]*/policy: $POL/" \
      -e "s/^  time_scale: [^ ]*/  time_scale: $TS/" -e "s/^  local_epochs: .*/  local_epochs: $EPOCHS/" \
      -e "s/^  mode: [^ ]*/  mode: $MODE/" \
      config.yaml > "$D/config.yaml"
  O="$D/override.yml"
  printf 'services:\n' > "$O"
  for s in uav iot1 iot2; do printf '  %s: {volumes: ["./%s/config.yaml:/app/config.yaml:ro"]}\n' $s "$D" >> "$O"; done
  if [ -n "$CDIR" ]; then   # bs dùng bộ chi phí của số epoch mới
    printf '  bs: {volumes: ["./%s/config.yaml:/app/config.yaml:ro", "./%s:/app/costs:ro"]}\n' "$D" "$CDIR" >> "$O"
  else
    printf '  bs: {volumes: ["./%s/config.yaml:/app/config.yaml:ro"]}\n' "$D" >> "$O"
  fi
  F="-f docker-compose.yml"
  case $POL in fas*) F="$F -f docker-compose.fas.yml";; esac
  echo "[demo] $(date +%H:%M:%S) bắt đầu $POL ($ROUNDS vòng FL, $EPOCHS epoch/vòng ở client, kênh $MODE, $SPLIT, nén kênh: $TS)"
  rm -f results/round_*.json results/log.jsonl results/setup.json capture/upload_*.npz
  # log đầy đủ 4 container -> $D/chay.log (màn hình in cùng nội dung, bỏ dòng kỹ thuật của docker)
  docker compose $F -f "$O" up --abort-on-container-exit --exit-code-from bs 2>&1 | tee "$D/chay.log" \
    | grep --line-buffered -E "\[(bs|uav|iot[12])\]|Error|error:" || true
  docker compose $F -f "$O" down >/dev/null 2>&1
  cp results/round_*.json results/setup.json "$D/" 2>/dev/null || echo "[demo] $POL: không có kết quả"
  mkdir -p "$D/capture" && cp capture/upload_0001.npz "$D/capture/" 2>/dev/null || true   # dữ liệu bs nhận vòng 1 (tools/attack.sh)
  echo "[demo] $(date +%H:%M:%S) xong $POL"
done

echo; echo "=== SỐ LIỆU CHI TIẾT ($OUT) ==="
sh tools/show.sh --rounds "$OUT"/*/

# tự vẽ hình của lần demo này (Octave trong Docker, ~30 s); run_plots.sh tự mở thư mục hình (DEMO_PLOT=0 để bỏ qua)
if [ "${DEMO_PLOT:-1}" != 0 ]; then
  echo; echo "=== VẼ HÌNH ($OUT/fig) ==="
  sh tools/run_plots.sh "$OUT" | grep -E "\[plot\] [a-z_]+\.png|XONG|error"
fi
