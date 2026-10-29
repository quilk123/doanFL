# bs (CLAUDE.md mục 4, 11). Image theo quy ước: python:3.12-slim.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*   # OpenFHE cần OpenMP
COPY tools/pip_install.sh /tmp/pip_install.sh
COPY bs/requirements.txt /app/bs/requirements.txt
RUN --mount=type=secret,id=extra_ca sh /tmp/pip_install.sh -r /app/bs/requirements.txt
COPY common /app/common
COPY bs /app/bs
COPY costs /app/costs
COPY config.yaml /app/config.yaml
# khớp cpus 1.0; để mặc định OpenMP mở 16 luồng dưới quota 1 lõi => ghép HE chậm ~40×
ENV OMP_NUM_THREADS=1
CMD ["python", "-m", "bs.server"]
