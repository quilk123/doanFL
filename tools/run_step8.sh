#!/bin/sh
# Bước 8: chạy thí nghiệm trên 4 container thật. sh tools/run_step8.sh <mode> <split> <rounds> [policy ...]
#   vd. sh tools/run_step8.sh uav_noma_pc iid 15 ; TIME_SCALE=auto sh tools/run_step8.sh direct_noma iid 3
# TIME_SCALE (nén kênh): off | <k> | auto -> thư mục có hậu tố _ts-<giá trị>.
# Kết quả: experiments/step8/<mode>_<split>_<policy>[_ts-..]/. Đủ vòng thì bỏ qua. fas*: thêm docker-compose.fas.yml.
set -eu
MODE=$1; SPLIT=$2; ROUNDS=$3; shift 3
POLICIES=${*:-"no_he fixed10 full_he"}   # mặc định: bộ chính (FAS p cố định 10%); tỉ lệ khác: fixed20 … fixed90; thêm fas10 fas50 adaptive nếu cần
TS=${TIME_SCALE:-off}
SUF=""; [ "$TS" != off ] && SUF="_ts-$TS"
TB_W=$(cygpath -w "$PWD" 2>/dev/null || echo "$PWD")
for POL in $POLICIES; do
  D="experiments/step8/${MODE}_${SPLIT}_${POL}$SUF"
  if [ -f "$(printf "%s/round_%04d.json" "$D" "$ROUNDS")" ]; then echo "[step8] $D: đủ vòng, bỏ qua"; continue; fi
  mkdir -p "$D"
  sed -e "s/^rounds: .*/rounds: $ROUNDS/" -e "s/^policy: [^ ]*/policy: $POL/" \
      -e "s/^  mode: [^ ]*/  mode: $MODE/" -e "s/^  split: [^ ]*/  split: $SPLIT/" \
      -e "s/^  time_scale: [^ ]*/  time_scale: $TS/" config.yaml > "$D/config.yaml"
  if ! grep -q "\"split\": \"$SPLIT\"" data/train/client1/meta.json 2>/dev/null; then
    echo "[step8] chia lại dữ liệu: $SPLIT"
    MSYS_NO_PATHCONV=1 docker run --rm -v "$TB_W:/tb" -w /tb testbed-tools \
      python tools/prep_data.py --split-only --config "$D/config.yaml" 2>&1 | grep "\[prep\]"
  fi
  O="$D/override.yml"
  printf 'services:\n' > "$O"
  for s in bs uav iot1 iot2; do printf '  %s: {volumes: ["./%s/config.yaml:/app/config.yaml:ro"]}\n' $s "$D" >> "$O"; done
  F="-f docker-compose.yml"
  case $POL in fas*) F="$F -f docker-compose.fas.yml";; esac
  echo "[step8] $(date +%H:%M:%S) bắt đầu $D"
  rm -f results/round_*.json results/log.jsonl results/setup.json capture/upload_*.npz
  # log ĐẦY ĐỦ 4 container (iot1, iot2, uav, bs — từ thiết lập HE, huấn luyện từng vòng, ...) -> $D/chay.log;
  # màn hình chỉ in dòng tóm tắt của bs
  docker compose $F -f "$O" up --abort-on-container-exit --exit-code-from bs 2>&1 | tee "$D/chay.log" \
    | grep --line-buffered -E "\[bs\] (policy|vòng)|Error|error:" || true
  docker compose $F -f "$O" down >/dev/null 2>&1
  cp results/round_*.json results/setup.json "$D/"
  mkdir -p "$D/capture" && cp capture/upload_*.npz "$D/capture/"
  echo "[step8] $(date +%H:%M:%S) xong $D"
done
