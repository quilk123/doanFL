# -*- coding: utf-8 -*-
"""Mô hình phần huấn luyện: fire9 + classifier của SqueezeNet 1.1 (logic reference_sim/common.py, 202.314 tham số).

Chỉ iot nhập torch (bs, uav không có mô hình). w0 = trọng số ImageNet + lớp conv cuối mới (10 lớp) khởi tạo theo
`seed` trong config, nên 2 iot TỰ DỰNG ra w0 giống hệt nhau (CLAUDE.md mục 5.4).
Trọng số: biến môi trường WEIGHTS, mặc định weights/squeezenet1_1_imagenet.pth (tạo bằng tools/prep_data.py).
"""
import os
import numpy as np

NUM_PARAMS = 202314   # fire9 + classifier (reference_sim/common.py)
NUM_CLASSES = 10
WEIGHTS = os.environ.get("WEIGHTS", os.path.join("weights", "squeezenet1_1_imagenet.pth"))

_order = None


def num_params(config=None):
    """Số tham số của phần mô hình được huấn luyện/truyền (202.314 = độ dài vector cập nhật Δ mỗi vòng)."""
    return NUM_PARAMS


def build_head(config=None, pretrained=True):
    """fire9 (features[12]) + classifier; conv cuối thay bằng 10 lớp, khởi tạo N(0, 0.01) theo seed.

    Đây là phần mô hình ĐƯỢC HUẤN LUYỆN và truyền giữa các bên. Phần thân SqueezeNet (features[0..11]) đóng băng
    và đã được chạy trước một lần (tools/prep_data.py) -> đặc trưng lưu sẵn trong data/, nên iot chỉ cần chạy phần
    đầu này mỗi vòng (nhẹ, hợp với Pi 4).
      pretrained=True : nạp trọng số ImageNet (fire9 dùng luôn trọng số đã học) — dùng khi tạo w0 và khi huấn luyện.
      pretrained=False: chỉ cần hình dạng tham số (priority_order) — không đọc file trọng số.
    Lớp phân loại cuối (512 -> 10) sinh ngẫu nhiên bằng generator RIÊNG theo `seed` => 2 iot tự dựng ra cùng w0.
    Kiểm tra số tham số = NUM_PARAMS để phát hiện sai phiên bản torchvision/trọng số.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import squeezenet1_1
    net = squeezenet1_1(weights=None)
    if pretrained:
        net.load_state_dict(torch.load(WEIGHTS))
    net.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
    seed = int((config or {}).get("seed", 0))
    g = torch.Generator().manual_seed(seed)   # generator riêng: không phụ thuộc trạng thái ngẫu nhiên toàn cục
    with torch.no_grad():
        net.classifier[1].weight.normal_(0, 0.01, generator=g)
        net.classifier[1].bias.zero_()
    head = nn.Sequential(net.features[12], net.classifier, nn.Flatten())
    n = sum(p.numel() for p in head.parameters())
    if n != NUM_PARAMS:
        raise RuntimeError(f"mô hình có {n} tham số, mong {NUM_PARAMS}")
    return head


def get_vec(model):
    """Trọng số của mô hình -> một vector phẳng float64 (202.314 số), nối các tensor theo thứ tự khai báo.

    Mọi phép toán FL (Δ = w_mới − w, FedAvg, clipping, mã hóa) làm trên vector phẳng này. Gọi ở iot/train.py
    (lấy trọng số SAU huấn luyện) và init_vec (lấy w0).
    """
    import torch
    return torch.cat([p.detach().flatten() for p in model.parameters()]).numpy().astype(np.float64)


def set_vec(model, v):
    """Ngược của get_vec: chép vector phẳng v vào các tensor tham số của mô hình (cùng thứ tự, ép về float32).

    Gọi trước khi huấn luyện (nạp w toàn cục hiện tại) và trước khi đánh giá độ chính xác (iot/evaluate.py).
    """
    import torch
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(np.ascontiguousarray(v[i:i + n])).view_as(p).float())
            i += n


def init_vec(config):
    """w0 dạng vector phẳng float64 — mô hình toàn cục ở vòng 1.

    Mỗi iot tự gọi khi khởi động (iot/client.py); vì cùng trọng số ImageNet và cùng seed nên kết quả giống hệt
    nhau ở 2 iot, không cần gửi w0 qua mạng. bs không có mô hình.
    """
    return get_vec(build_head(config))


def priority_order(config=None):
    """Thứ tự ưu tiên mã hóa cố định: classifier trước, rồi fire9 từ lớp cuối ngược lên (mục 3).
    Trả về mảng chỉ số trên vector phẳng; chỉ phụ thuộc kích thước tham số nên tính một lần.

    iot/protect.py lấy p·n chỉ số ĐẦU của thứ tự này để mã hóa HE (lớp gần đầu ra được che trước), phần còn lại
    gửi dạng rõ đã bảo vệ (clipping + nhiễu + xáo). Thứ tự lấy y nguyên reference_sim/common.py `priority_order`
    — là một phần thiết kế, không tự đổi.
    """
    global _order
    if _order is None:
        sizes = [p.numel() for p in build_head(config, pretrained=False).parameters()]
        starts = np.cumsum([0] + sizes[:-1])
        blocks = [np.arange(s, s + n) for s, n in zip(starts, sizes)]
        _order = np.concatenate(blocks[::-1])   # tham số khai báo sau (gần đầu ra) được ưu tiên
    return _order
