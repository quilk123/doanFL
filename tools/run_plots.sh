#!/bin/sh
# Xuất CSV (tools/export_matlab.py) rồi vẽ bằng GNU Octave trong Docker (matlab/plot_results.m — cùng mã chạy được trên MATLAB).
# Bộ vẽ Qt qua xvfb (gnuplot làm mất dấu tiếng Việt; gọi xvfb-run trực tiếp không qua sh thì treo).
# Chạy từ thư mục testbed:
#   sh tools/run_plots.sh                                  # số liệu bước 8 -> experiments/matlab/fig/*.png
#   sh tools/run_plots.sh experiments/demo/<thời điểm>     # một lần demo   -> experiments/demo/<thời điểm>/fig/*.png
#   sh tools/run_plots.sh experiments/demo/<A> experiments/demo/<B> ...   # GỘP nhiều lần demo
#   sh tools/run_plots.sh --last 2                         # GỘP 2 lần demo gần nhất (vd. có UAV + không UAV)
#     -> experiments/demo/gop_<thời điểm>/fig/*.png (+ nguon.txt ghi các lần đã gộp), tự mở thư mục hình
set -u
TB_W=$(cygpath -w "$PWD" 2>/dev/null || echo "$PWD")
if [ "${1:-}" = "--last" ]; then
  set -- $(ls -d experiments/demo/2*/ 2>/dev/null | sort | tail -n "${2:-2}")
  [ $# -ge 1 ] || { echo "[vẽ] chưa có lần demo nào trong experiments/demo/"; exit 1; }
fi
OPEN=""
if [ $# -eq 0 ]; then
  CALL="plot_results"
elif [ $# -eq 1 ]; then
  RUN=${1%/}                                              # vd. experiments/demo/20261027_204230
  CALL="plot_results('demo', '${RUN#experiments/}/', '/tb/$RUN/fig')"
  OPEN="$RUN/fig"
else
  G="experiments/demo/gop_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$G"
  LIST=""
  for d in "$@"; do
    d=${d%/}; LIST="$LIST'${d#experiments/}/', "
    echo "$d" >> "$G/nguon.txt"
  done
  CALL="plot_results('demo', {${LIST%, }}, '/tb/$G/fig')"
  OPEN="$G/fig"
  echo "[vẽ] gộp $# lần demo:"; sed 's/^/        /' "$G/nguon.txt"
fi
echo "[vẽ] $(date +%H:%M:%S) xuất CSV"
sh tools/export_matlab.sh
echo "[vẽ] $(date +%H:%M:%S) vẽ bằng Octave: $CALL"
MSYS_NO_PATHCONV=1 docker run --rm --cpus "${PLOT_CPUS:-2}" -v "$TB_W:/tb" -w /tb/matlab \
  gnuoctave/octave:9.2.0 sh -c "xvfb-run -a octave --no-gui --quiet --eval \"graphics_toolkit('qt'); $CALL\"" 2>&1 \
  | grep -v "XDG_RUNTIME\|legend: 'best'"
echo "[vẽ] $(date +%H:%M:%S) XONG"
if [ -n "$OPEN" ]; then
  echo "[vẽ] hình: $OPEN"
  explorer "$(cygpath -w "$OPEN" 2>/dev/null || echo "$OPEN")" 2>/dev/null || true
fi
