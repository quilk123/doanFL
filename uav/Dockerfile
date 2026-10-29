# Bước 1 — uav (CLAUDE.md mục 4, 11). Image theo quy ước: python:3.12-slim.
FROM python:3.12-slim
WORKDIR /app
COPY tools/pip_install.sh /tmp/pip_install.sh
COPY uav/requirements.txt /app/uav/requirements.txt
RUN --mount=type=secret,id=extra_ca sh /tmp/pip_install.sh -r /app/uav/requirements.txt
COPY common /app/common
COPY uav /app/uav
COPY config.yaml /app/config.yaml
CMD ["python", "-m", "uav.relay"]
