#!/bin/sh
# Xem số liệu chi tiết (máy không cần python): chạy tools/show_runs.py trong image testbed-tools.
#   sh tools/show.sh                                         # tóm tắt mọi cấu hình bước 8 + so sánh có/không UAV
#   sh tools/show.sh --rounds experiments/step8/uav_noma_pc_iid_adaptive    # thêm bảng từng vòng
TB_W=$(cygpath -w "$PWD" 2>/dev/null || echo "$PWD")
MSYS_NO_PATHCONV=1 docker run --rm -e PYTHONIOENCODING=utf-8 -v "$TB_W:/tb" -w /tb testbed-tools python tools/show_runs.py "$@"
