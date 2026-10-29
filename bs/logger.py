# -*- coding: utf-8 -*-
"""Ghi log mỗi vòng một bản ghi JSON tại bs (CLAUDE.md mục 9), và một bản ghi thiết lập HE (setup.json)."""
import os
import json

# pha TÍNH TOÁN của thiết bị (nhân fidelity để quy về Pi 4); pha truyền không nhân
DEVICE_COMPUTE = ("train_s", "enc_s", "pdec_s", "update_s")


class Logger:
    """Ghi kết quả của bs: results/round_XXXX.json (mỗi vòng một file), results/log.jsonl (mọi vòng, mỗi dòng một
    bản ghi) và results/setup.json (thiết lập HE). fidelity = hệ số quy thời gian tính toán máy này về Pi 4 (1,4)."""

    def __init__(self, results_dir, fidelity=1.4):
        self.dir = results_dir
        self.fidelity = fidelity
        os.makedirs(results_dir, exist_ok=True)
        self.jsonl = os.path.join(results_dir, "log.jsonl")

    def write_setup(self, record):
        """Ghi setup.json (thời gian và số byte của giai đoạn thiết lập HE trước vòng 1)."""
        path = os.path.join(self.dir, "setup.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return path

    def write(self, record):
        """Ghi bản ghi của một vòng; trước đó thêm thời gian các pha tính toán quy về Pi 4 (× fidelity).

        Chỉ pha tính toán (huấn luyện, mã hóa, giải mã, cập nhật; tính toán của bs) được nhân; pha truyền giữ nguyên
        vì đã đi đúng tốc độ kênh. Trả về đường dẫn round_XXXX.json.
        """
        # Thời gian hiệu chỉnh Pi 4: pha tính toán × fidelity (mục 9)
        tm = record.setdefault("time", {})
        if tm.get("compute_s") is not None:
            tm["compute_pi4_s"] = tm["compute_s"] * self.fidelity
        for dev in ("dev1", "dev2"):
            if isinstance(tm.get(dev), dict):
                tm[f"{dev}_pi4"] = {k: v * self.fidelity for k, v in tm[dev].items() if k in DEVICE_COMPUTE}

        t = record["t"]
        path = os.path.join(self.dir, f"round_{t:04d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
