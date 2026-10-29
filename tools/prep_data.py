# -*- coding: utf-8 -*-
"""Chuẩn bị dữ liệu cho testbed (bước 2, CLAUDE.md mục 3, 4, 11) — chạy MỘT LẦN trên máy chủ, không trong container.

1. Chuyển trọng số SqueezeNet 1.1 ImageNet ONNX -> state_dict PyTorch (logic reference_sim/convert_weights.py)
   -> weights/squeezenet1_1_imagenet.pth (iot dùng để tự dựng w0).
2. Trích đặc trưng thân đóng băng features[0:12] (ảnh 112×112, fp16; logic reference_sim/data_prep.py)
   -> data/full/{train,test}_{X,y}.npy  (đặt tên như reference_sim/cache để chạy được fl.py tham chiếu).
3. Chia 3 TẬP (28/10/2026): từ 50.000 ảnh train gốc tách VALIDATION (`devices.val_size`, mặc định 5.000 = 500 ảnh/lớp,
   chọn theo seed), 45.000 ảnh còn lại chia cho 2 thiết bị theo config (devices.split, dirichlet_alpha, seed; cùng
   split_clients của fl.py); TEST = 10.000 ảnh test gốc của CIFAR-10.
   -> data/train/client1/, data/train/client2/ (TRAIN — mỗi thiết bị chỉ thấy phần của mình)
      data/val/  (VALIDATION — theo dõi mỗi vòng)        data/test/ (TEST — chỉ để báo cáo)
   Xem data/DOC_TRUOC.md.

Ví dụ:
  python tools/prep_data.py --cifar <đường dẫn>/CIFAR-10-images --onnx <đường dẫn>/squeezenet1.1-7.onnx
  python tools/prep_data.py --split-only            # chỉ chia lại (đổi iid <-> noniid trong config.yaml)
"""
import os
import sys
import glob
import json
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
SIZE = 112
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def convert_weights(onnx_path, out_path):
    """ONNX -> state_dict PyTorch, kiểm tra sai khác đầu ra với onnxruntime."""
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import numpy_helper
    from torchvision.models import squeezenet1_1

    m = onnx.load(onnx_path)
    W = [numpy_helper.to_array(t) for t in m.graph.initializer if t.name.startswith('squeezenet0_conv')]
    net = squeezenet1_1(weights=None).eval()
    with torch.no_grad():
        for p, w in zip(net.parameters(), W):   # cùng thứ tự: conv0, (squeeze, expand1x1, expand3x3) x 8, classifier
            assert tuple(p.shape) == w.shape
            p.copy_(torch.from_numpy(w.copy()))
    s = ort.InferenceSession(onnx_path)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    diff = np.abs(s.run(None, {s.get_inputs()[0].name: x})[0].reshape(1, -1)
                  - net(torch.from_numpy(x)).detach().numpy()).max()
    print(f'[prep] sai khác tối đa ONNX vs PyTorch: {diff:.2e}', flush=True)
    assert diff < 1e-4, 'chuyển trọng số sai'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(net.state_dict(), out_path)
    print(f'[prep] đã lưu {out_path}', flush=True)


def _to_tensor(paths):
    import torch
    from PIL import Image
    arr = np.stack([np.asarray(Image.open(p).convert('RGB').resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
                    for p in paths])
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())


def extract(cifar_dir, weights_path, out_dir, split):
    """Trích đặc trưng thân SqueezeNet (features[0:12]) cho một tập, lưu fp16."""
    import torch
    from torchvision.models import squeezenet1_1
    net = squeezenet1_1(weights=None)
    net.load_state_dict(torch.load(weights_path))
    trunk = net.features[:12].eval()

    paths, labels = [], []
    for c, name in enumerate(CLASSES):
        ps = sorted(glob.glob(os.path.join(cifar_dir, split, name, '*.jpg')))
        paths += ps
        labels += [c] * len(ps)
    assert paths, f'không thấy ảnh trong {cifar_dir}/{split}'
    feats = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), 256):
            feats.append(trunk(_to_tensor(paths[i:i + 256])).half().numpy())
            if i % 5120 == 0:
                print(f'[prep] {split} {i}/{len(paths)} {time.time() - t0:.0f}s', flush=True)
    X = np.concatenate(feats)
    y = np.array(labels, np.int64)
    np.save(os.path.join(out_dir, f'{split}_X.npy'), X)
    np.save(os.path.join(out_dir, f'{split}_y.npy'), y)
    print(f'[prep] {split} {X.shape} {time.time() - t0:.0f}s', flush=True)


def split_clients(y, n=2, mode='iid', alpha=0.5, seed=0):
    """Giống hệt reference_sim/fl.py để hai bên có cùng phân hoạch."""
    rng = np.random.default_rng(seed)
    if mode == 'iid':
        idx = rng.permutation(len(y))
        return np.array_split(idx, n)
    parts = [[] for _ in range(n)]
    for c in range(10):
        ic = rng.permutation(np.where(y == c)[0])
        w = rng.dirichlet([alpha] * n)
        cuts = (np.cumsum(w) * len(ic)).astype(int)[:-1]
        for k, s in enumerate(np.split(ic, cuts)):
            parts[k] += s.tolist()
    return [np.array(sorted(p)) for p in parts]


def train_val_split(y, config):
    """Tách 3 tập trên 50.000 ảnh train gốc: trả (chỉ số validation, [chỉ số thiết bị 1, chỉ số thiết bị 2]).

    Validation: `devices.val_size` ảnh (mặc định 5.000), ĐỀU các lớp (val_size/10 ảnh/lớp), chọn ngẫu nhiên theo seed —
    không đổi khi đổi iid <-> noniid. Phần còn lại (45.000) chia cho 2 thiết bị bằng split_clients.
    Dùng chung cho prep_data (ghi dữ liệu) và tools/attack_eval.py (dựng lại đúng ảnh thành viên của thiết bị)."""
    dev = config.get('devices', {})
    seed = int(config.get('seed', 0))
    per = int(dev.get('val_size', 5000)) // 10
    rng = np.random.default_rng([seed, 31])                 # nguồn riêng cho validation, không đụng split_clients
    val = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], per, replace=False) for c in range(10)]))
    pool = np.setdiff1d(np.arange(len(y)), val)
    parts = split_clients(y[pool], 2, dev.get('split', 'iid'), float(dev.get('dirichlet_alpha', 0.5)), seed)
    return val, [np.sort(pool[p]) for p in parts]


def write_splits(config, full_dir, data_dir):
    """Ghi 3 tập: data/train/client{1,2} (mỗi client chỉ mount vào iot của nó), data/val, data/test."""
    import shutil
    X = np.load(os.path.join(full_dir, 'train_X.npy'), mmap_mode='r')
    y = np.load(os.path.join(full_dir, 'train_y.npy'))
    dev = config.get('devices', {})
    mode = dev.get('split', 'iid')
    seed = int(config.get('seed', 0))
    for old in ('client1', 'client2'):                      # bố cục cũ (2 tập) — đã thay bằng data/train/
        if os.path.isdir(os.path.join(data_dir, old)):
            shutil.rmtree(os.path.join(data_dir, old))
    val, parts = train_val_split(y, config)
    for k, idx in enumerate(parts, 1):
        d = os.path.join(data_dir, 'train', f'client{k}')
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, 'X.npy'), np.asarray(X[idx]))
        np.save(os.path.join(d, 'y.npy'), y[idx])
        hist = np.bincount(y[idx], minlength=10).tolist()
        json.dump({'tap': 'train', 'split': mode, 'seed': seed, 'n': len(idx), 'val_size': len(val), 'class_hist': hist},
                  open(os.path.join(d, 'meta.json'), 'w'), indent=1)
        print(f'[prep] train/client{k}: {len(idx)} mẫu, phân bố lớp {hist}', flush=True)
    d = os.path.join(data_dir, 'val')
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, 'X.npy'), np.asarray(X[val]))
    np.save(os.path.join(d, 'y.npy'), y[val])
    json.dump({'tap': 'validation', 'seed': seed, 'n': len(val), 'class_hist': np.bincount(y[val], minlength=10).tolist()},
              open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    print(f'[prep] val: {len(val)} mẫu ({len(val) // 10}/lớp)', flush=True)
    d = os.path.join(data_dir, 'test')
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, 'X.npy'), np.load(os.path.join(full_dir, 'test_X.npy')))
    np.save(os.path.join(d, 'y.npy'), np.load(os.path.join(full_dir, 'test_y.npy')))
    n_test = len(np.load(os.path.join(d, "y.npy")))
    print(f'[prep] test: {n_test} mẫu', flush=True)
    with open(os.path.join(data_dir, 'DOC_TRUOC.md'), 'w', encoding='utf-8') as f:
        f.write(f"""# Dữ liệu testbed — 3 tập (tạo bởi tools/prep_data.py, seed {seed}, chia {mode})

Ảnh gốc CIFAR-10 -> đặc trưng thân SqueezeNet (512×6×6, float16) ở `full/`. Mỗi thư mục: X.npy (đặc trưng), y.npy (nhãn 0–9).

| Thư mục | Tập | Số ảnh | Ai dùng | Dùng để |
|---|---|---|---|---|
| `train/client1/` | TRAIN | {len(parts[0])} | chỉ iot1 | huấn luyện cục bộ mỗi vòng |
| `train/client2/` | TRAIN | {len(parts[1])} | chỉ iot2 | huấn luyện cục bộ mỗi vòng |
| `val/` | VALIDATION | {len(val)} ({len(val) // 10}/lớp, tách từ train gốc) | iot1 | theo dõi mỗi vòng (metrics.val_accuracy) |
| `test/` | TEST | {n_test} (test gốc CIFAR-10) | iot1 | CHỈ để báo cáo (metrics.accuracy) |
| `full/` | — | 50.000 train + 10.000 test | công cụ (tách lại tập, đánh giá tấn công) | không mount vào container |

Ba tập không trùng ảnh nào. Chia lại (đổi iid/noniid hoặc seed trong config.yaml):
`docker run --rm -v "${{PWD}}:/tb" -w /tb testbed-tools python tools/prep_data.py --split-only`
""")


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument('--cifar', help='thư mục CIFAR-10-images (có train/, test/)')
    ap.add_argument('--onnx', help='squeezenet1.1-7.onnx')
    ap.add_argument('--config', default=os.path.join(ROOT, 'config.yaml'))
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--split-only', action='store_true', help='chỉ chia lại client từ data/full')
    args = ap.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    weights = os.path.join(ROOT, 'weights', 'squeezenet1_1_imagenet.pth')
    data_dir = os.path.join(ROOT, 'data')
    full_dir = os.path.join(data_dir, 'full')
    if not args.split_only:
        import torch
        torch.set_num_threads(args.threads)
        if not os.path.isfile(weights):
            if not args.onnx:
                sys.exit('thiếu --onnx để tạo weights/squeezenet1_1_imagenet.pth')
            convert_weights(args.onnx, weights)
        if not args.cifar:
            sys.exit('thiếu --cifar')
        os.makedirs(full_dir, exist_ok=True)
        extract(args.cifar, weights, full_dir, 'test')
        extract(args.cifar, weights, full_dir, 'train')
    write_splits(config, full_dir, data_dir)


if __name__ == '__main__':
    main()
