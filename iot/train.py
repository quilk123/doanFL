# -*- coding: utf-8 -*-
"""Huấn luyện cục bộ THẬT (bước 2; logic reference_sim/fl.py `local_train`).

Mỗi vòng: dựng mô hình từ w, fine-tune fire9 + classifier `local_epochs` epoch trên đặc trưng thân SqueezeNet đã trích
(data/train/client{k}/X.npy, fp16, mmap — tập TRAIN), SGD momentum mới mỗi vòng như reference_sim. Trả Δ = w_mới − w (float64).
"""
import os
import numpy as np
import torch
import torch.nn as nn

from common import model

DATA_DIR = os.environ.get("DATA_DIR", "data")
torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))

_data = {}


def load_client(device_id):
    """Đặc trưng + nhãn của riêng thiết bị (client_k chỉ mount vào iot_k).

    X: đặc trưng thân SqueezeNet (fp16, đọc kiểu mmap — không nạp hết vào RAM, hợp giới hạn 4 GB như Pi 4);
    y: nhãn lớp. Phân hoạch IID/non-IID do tools/prep_data.py tạo theo config `devices`. Nạp một lần, giữ trong _data.
    """
    if device_id not in _data:
        d = os.path.join(DATA_DIR, "train", f"client{device_id}")   # TẬP TRAIN của thiết bị (data/train/clientK)
        _data[device_id] = (np.load(os.path.join(d, "X.npy"), mmap_mode="r"),
                            np.load(os.path.join(d, "y.npy")))
    return _data[device_id]


def _batches(X, y, bs, rng):
    """Chia dữ liệu thành các lô `bs` ảnh theo thứ tự ngẫu nhiên (xáo lại mỗi epoch) -> (tensor đặc trưng, nhãn).

    Chỉ số trong mỗi lô được sắp tăng dần để đọc mmap tuần tự (nhanh hơn trên đĩa); đặc trưng fp16 đổi sang float32.
    """
    idx = rng.permutation(len(y))
    for i in range(0, len(idx), bs):
        b = np.sort(idx[i:i + bs])
        yield torch.from_numpy(np.asarray(X[b], np.float32)), torch.from_numpy(y[b])


def local_train(w, config, device_id, t, log=None):
    """Trả Δ (float64) sau huấn luyện cục bộ từ w.

    Đầu vào: w = mô hình toàn cục hiện tại (vector 202.314 số), device_id (chọn dữ liệu riêng), t = số vòng.
    Các bước:
      1. dựng mô hình fire9 + classifier, nạp w (set_vec)
      2. SGD (lr, momentum, weight_decay trong config `train`) — bộ tối ưu tạo MỚI mỗi vòng như reference_sim
      3. `local_epochs` epoch (mặc định 1 = ~391 bước với 25.000 ảnh, lô 64); opt.step() là nơi trọng số thay đổi
      4. lấy trọng số sau huấn luyện (get_vec) và trả Δ = w_mới − w
    Thứ tự lô và khởi tạo ngẫu nhiên của torch tái lập theo (seed, thiết bị, vòng) — chạy lại cho cùng Δ.
    Gọi ở iot/client.py ngay sau ROUND_START; Δ đi tiếp vào iot/protect.py trước khi gửi.
    log: hàm in log (client.py truyền vào) — in tiến độ mỗi 25% epoch: số lô, loss và độ chính xác trung bình trên các lô
    đã học. Không truyền (bench, test) thì không in. Thống kê epoch cuối giữ ở LAST_STATS.
    """
    tr = config.get("train", {})
    X, y = load_client(device_id)
    # nguồn ngẫu nhiên riêng từng thiết bị, tái lập theo (seed, thiết bị, vòng)
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng([seed, device_id, t])
    torch.manual_seed(seed * 1_000_003 + device_id * 1009 + t)

    net = model.build_head(config)
    model.set_vec(net, w)          # bắt đầu từ mô hình toàn cục w
    opt = torch.optim.SGD(net.parameters(), lr=float(tr.get("lr", 0.01)),
                          momentum=float(tr.get("momentum", 0.9)),
                          weight_decay=float(tr.get("weight_decay", 5e-4)))
    lossf = nn.CrossEntropyLoss()
    net.train()
    E = int(tr.get("local_epochs", 1))
    bsz = int(tr.get("batch", 64))
    n_batch = (len(y) + bsz - 1) // bsz
    marks = {max(1, round(n_batch * q)) for q in (0.25, 0.5, 0.75, 1.0)}   # các mốc in tiến độ
    if log:
        log(f"vòng {t}: bắt đầu huấn luyện {E} epoch trên {len(y)} ảnh của thiết bị ({n_batch} lô × {bsz} ảnh, "
            f"lr {float(tr.get('lr', 0.01))})")
    for e in range(1, E + 1):
        s_loss = s_ok = s_n = 0
        for i, (xb, yb) in enumerate(_batches(X, y, bsz, rng), 1):
            opt.zero_grad()                    # xóa gradient của lô trước
            out = net(xb)
            loss = lossf(out, yb)
            loss.backward()                    # lan truyền ngược: gradient của loss theo trọng số
            opt.step()                         # cập nhật trọng số trong `net`
            s_loss += loss.item() * len(yb); s_ok += (out.argmax(1) == yb).sum().item(); s_n += len(yb)
            if log and i in marks:
                log(f"vòng {t}: epoch {e}/{E} — {100 * i // n_batch}% ({i}/{n_batch} lô): "
                    f"loss TB {s_loss / s_n:.4f}, acc trên lô đã học {s_ok / s_n:.4f}")
        LAST_STATS.update(epoch=e, loss=s_loss / s_n, acc=s_ok / s_n, n=s_n)
    return model.get_vec(net) - w  #trọng số sau khi huấn luyện model


LAST_STATS = {}   # thống kê epoch cuối của lần huấn luyện gần nhất (loss, acc trên lô đã học)
