# Image công cụ chạy trên máy chủ (không phải container testbed): chuẩn bị dữ liệu, chạy reference_sim để so sánh, vẽ hình.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*   # OpenFHE cần OpenMP
COPY tools/pip_install.sh /tmp/pip_install.sh
COPY iot/requirements.txt /tmp/iot-requirements.txt
RUN --mount=type=secret,id=extra_ca sh /tmp/pip_install.sh -r /tmp/iot-requirements.txt onnx onnxruntime pillow pandas matplotlib
