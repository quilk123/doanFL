# -*- coding: utf-8 -*-
"""Đánh giá THẬT mô hình toàn cục (bước 2; logic reference_sim/fl.py `evaluate`) trên 2 tập, chỉ iot1 có:
  data/val  — VALIDATION (5.000 ảnh tách từ train gốc): theo dõi mỗi vòng (metrics.val_accuracy, val_loss)
  data/test — TEST (10.000 ảnh test gốc CIFAR-10): CHỈ để báo cáo (metrics.accuracy, loss)
Testbed không chọn gì theo tập test (số vòng cố định, siêu tham số cố định); nếu cần chọn/tinh chỉnh thì dùng validation."""
import os
import numpy as np
import torch
import torch.nn as nn

from common import model

DATA_DIR = os.environ.get("DATA_DIR", "data")

_sets = {}
_net = None


def _load(name):
    """Đặc trưng + nhãn của một tập (val | test), nạp một lần (mmap)."""
    if name not in _sets:
        d = os.path.join(DATA_DIR, name)
        _sets[name] = (np.load(os.path.join(d, "X.npy"), mmap_mode="r"), np.load(os.path.join(d, "y.npy")))
    return _sets[name]


@torch.no_grad()
def _score(net, X, y):
    """(accuracy, loss trung bình) trên một tập, duyệt theo lô 1.000 ảnh."""
    correct = 0
    loss = 0.0
    for i in range(0, len(y), 1000):
        xb = torch.from_numpy(np.asarray(X[i:i + 1000], np.float32))
        yb = torch.from_numpy(y[i:i + 1000])
        out = net(xb)
        loss += nn.functional.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return correct / len(y), loss / len(y)


@torch.no_grad()
def evaluate(w, config, t):
    """Trả dict {accuracy, loss (TEST), val_accuracy, val_loss (VALIDATION)} của mô hình w.

    Gọi ở iot1 cuối mỗi vòng, sau khi cập nhật w toàn cục (client.py) — gửi bs trong METRICS và ghi log.
    `accuracy` giữ nghĩa cũ (tập test) để mọi bảng/hình cũ vẫn đọc đúng. `t` hiện không dùng trong tính toán.
    """
    global _net
    if _net is None:
        _net = model.build_head(config)
    model.set_vec(_net, w)
    _net.eval()
    acc, loss = _score(_net, *_load("test"))
    val_acc, val_loss = _score(_net, *_load("val"))
    return {"accuracy": acc, "loss": loss, "val_accuracy": val_acc, "val_loss": val_loss}


@torch.no_grad()
def evaluate_train(w, config, device_id):
    """Trả {train_accuracy, train_loss}: mô hình chung w đo trên TẬP TRAIN của chính thiết bị (data/train/clientK).

    CẢ 2 iot gọi mỗi vòng (mỗi thiết bị chỉ có dữ liệu của mình) — cho biết mô hình chung khớp dữ liệu từng thiết bị
    đến đâu (non-IID: hai số khác nhau rõ). Là ảnh đã học nên cao hơn validation/test; không dùng để báo cáo độ tổng quát.
    """
    global _net
    if _net is None:
        _net = model.build_head(config)
    from iot import train
    model.set_vec(_net, w)
    _net.eval()
    acc, loss = _score(_net, *train.load_client(device_id))
    return {"train_accuracy": acc, "train_loss": loss}
