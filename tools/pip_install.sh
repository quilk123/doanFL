#!/bin/sh
# pip install dùng trong Dockerfile. Nếu máy chủ có phần mềm quét TLS (vd. Avast Web Shield) thì truyền CA gốc của nó
# qua build secret "extra_ca" (certs/extra_ca.pem): chỉ dùng lúc build, KHÔNG nằm trong image.
set -e
if [ -s /run/secrets/extra_ca ]; then
  cat /etc/ssl/certs/ca-certificates.crt /run/secrets/extra_ca > /tmp/ca.pem
  export PIP_CERT=/tmp/ca.pem
fi
pip install --no-cache-dir "$@"
rm -f /tmp/ca.pem
