#!/bin/sh
# Test bước 6 (phần container thật): bs, uav KHÔNG đọc được secrets/perm_seed; iot1, iot2 đọc được (32 byte).
# Chạy trên máy chủ từ thư mục testbed: sh tests/check_secrets_docker.sh
set -u
fail=0
for svc in bs uav; do
  out=$(docker compose run --rm --no-deps -T --entrypoint sh "$svc" -c \
    'for f in /run/secrets/perm_seed /app/secrets/perm_seed; do [ -e "$f" ] && echo "CO $f"; done; \
     find / -xdev -name perm_seed 2>/dev/null; echo XONG' 2>&1 | tr -d '\r')
  if ! echo "$out" | grep -q "XONG"; then
    echo "[TEST] LỖI: không chạy được lệnh kiểm tra trong $svc: $out"; fail=1
  elif echo "$out" | grep -q "perm_seed"; then
    echo "[TEST] LỖI: $svc thấy perm_seed: $out"; fail=1
  else
    echo "[TEST] $svc: không có perm_seed"
  fi
done
for svc in iot1 iot2; do
  n=$(docker compose run --rm --no-deps -T --entrypoint sh "$svc" -c 'wc -c < /run/secrets/perm_seed' 2>/dev/null | tr -d '\r ')
  if [ "$n" = "32" ]; then
    echo "[TEST] $svc: đọc được K (32 byte)"
  else
    echo "[TEST] LỖI: $svc không đọc được K (được: '$n')"; fail=1
  fi
done
[ $fail -eq 0 ] && echo "[TEST] ĐẠT: chỉ iot1, iot2 có K" || { echo "[TEST] KHÔNG ĐẠT"; exit 1; }
