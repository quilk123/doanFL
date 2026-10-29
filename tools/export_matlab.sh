#!/bin/sh
# Xuất CSV cho MATLAB (máy không cần python): chạy tools/export_matlab.py trong image testbed-tools.
TB_W=$(cygpath -w "$PWD" 2>/dev/null || echo "$PWD")
MSYS_NO_PATHCONV=1 docker run --rm -e PYTHONIOENCODING=utf-8 -v "$TB_W:/tb" -w /tb testbed-tools python tools/export_matlab.py "$@"
