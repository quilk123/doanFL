# iot (CLAUDE.md mục 4, 11). Image theo quy ước: python:3.12-slim. Bước 2: PyTorch CPU + trọng số SqueezeNet ImageNet.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*   # OpenFHE cần OpenMP
COPY tools/pip_install.sh /tmp/pip_install.sh
COPY iot/requirements.txt /app/iot/requirements.txt
RUN --mount=type=secret,id=extra_ca sh /tmp/pip_install.sh -r /app/iot/requirements.txt
COPY weights/squeezenet1_1_imagenet.pth /app/weights/squeezenet1_1_imagenet.pth
COPY common /app/common
COPY iot /app/iot
COPY config.yaml /app/config.yaml
# torch + OpenFHE (OpenMP): khớp ~1,6 lõi; để mặc định OpenMP mở 16 luồng dưới quota => chậm tới 40×
ENV NTHREADS=2 OMP_NUM_THREADS=2
CMD ["python", "-m", "iot.client"]
