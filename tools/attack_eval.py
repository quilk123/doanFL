# -*- coding: utf-8 -*-
"""Đánh giá tấn công trên ĐÚNG dữ liệu bs nhận (CLAUDE.md mục 9) — capture/ của các lượt chạy bước 8.

Vòng 1 (mô hình toàn cục = w0, bs biết w0 từ config => thuận lợi nhất cho kẻ tấn công). Với mỗi cấu hình và mỗi thiết bị:
  A. Suy luận thành viên (như reference_sim/member.py E11): điểm cos(−∇L(x; w0)[vis], quan sát[vis]); 1.000 ảnh thành viên
     (dữ liệu thiết bị, phân hoạch theo config của lượt chạy) + 1.000 không thành viên (tập test, CÙNG PHÂN BỐ LỚP với
     thành viên của thiết bị đó — bắt buộc với non-IID, nếu không AUC đo chênh lệch lớp). AUC 0,5 = không rò rỉ.
     Góc nhìn của bs theo policy: no_he — Δ đúng thứ tự; fas10/50 — bs có K, giải xáo (bs_view); adaptive — vector ĐÃ XÁO:
     bs thử (i) so khớp thẳng, (ii) so sau khi sắp xếp (như leak2 E16, không cần vị trí), (iii) thứ tự đoán ở B;
     báo AUC tốt nhất (cận trên). Kèm "nếu bs có K" (tách tác dụng của xáo và của nhiễu). full_he: không có phần rõ.
     Thêm (mạnh hơn cos, vì cos không phát hiện được thành viên từ cập nhật 1 epoch / 25.000 ảnh): suy luận thành viên theo
     MỨC GIẢM LOSS — bs dựng w0 + Δ̂ từ cái nó thấy, điểm = L(x; w0) − L(x; w0 + Δ̂).
     Độ khôi phục cập nhật (recovery_corr): tương quan ước lượng giữa Δ̂ của bs và Δ sạch của thiết bị (gồm xáo + nhiễu).
  B. Gỡ xáo (adaptive, fas* khi bs KHÔNG có K): bs ước lượng hướng cập nhật từ dữ liệu công khai (−trung bình gradient trên
     ảnh test tại w0), xếp vị trí theo thứ hạng; đo tỷ lệ vị trí đúng (ngẫu nhiên: 1/n), tương quan với Δ thật, khớp dấu.
  C. Hai thiết bị dùng CHUNG π_t: bs ghép được cặp (Δ1_j, Δ2_j) cùng vị trí — đo tương quan cặp (π riêng => ~0).
  HAI GÓC NHÌN (cột *_bs và *_eav): trạm gốc (bs) và kẻ nghe lén (thấy mọi gói chiều lên, KHÔNG BAO GIỜ có K).
     no_he: cả hai thấy Δ đúng thứ tự. fixed10/adaptive: cả hai chỉ thấy vector đã xáo (K chỉ ở thiết bị).
     fas10/fas50 (FAS gốc): bs có K -> thấy phần rõ đúng thứ tự (chỉ còn nhiễu); kẻ nghe lén chỉ thấy vector đã xáo.
     full_he: chỉ ciphertext. recovery_corr = góc nhìn bs (giữ tên cũ), recovery_eav = kẻ nghe lén; auc_bs / auc_eav.
Chấm điểm cần π_t thật: người đánh giá tính từ K (secrets/perm_seed) — đặc quyền của người đánh giá, bs không có.
  python tools/attack_eval.py [thư mục ...] [--out thư mục]   (mặc định: mọi cấu hình bước 8 có capture/upload_0001.npz)
  sh tools/attack.sh ...  /  .\attack.bat ...   (máy không có python; tỉ lệ khác, lần demo — xem tools/attack.sh)
Kết quả: experiments/attack/attack_round1.{json,csv}
"""
import os
import sys
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.func import functional_call, vmap, grad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
from common import model                           # noqa: E402
from common.perm import perm_for, unshuffle        # noqa: E402
from prep_data import split_clients, train_val_split   # noqa: E402

torch.set_num_threads(int(os.environ.get("NTHREADS", "1")))
N_MEM = N_NON = 1000
OUT = os.path.join(ROOT, "experiments", "attack")


def auc(score_pos, score_neg):
    """AUC = P(điểm thành viên > điểm không thành viên) (Mann–Whitney, xử lý hòa)."""
    import pandas as pd
    s = np.r_[score_pos, score_neg]
    r = pd.Series(s).rank(method="average").to_numpy()   # hạng trung bình cho giá trị trùng
    n1, n0 = len(score_pos), len(score_neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per_sample_grads(head, X, y, idx):
    names = [n for n, _ in head.named_parameters()]
    prm = {n: p.detach() for n, p in head.named_parameters()}
    f = vmap(grad(lambda pr, z, t: F.cross_entropy(functional_call(head, pr, (z[None],)), t[None])), in_dims=(None, 0, 0))
    out = []
    for i in range(0, len(idx), 100):
        b = np.sort(idx[i:i + 100])
        g = f(prm, torch.from_numpy(np.asarray(X[b], np.float32)), torch.from_numpy(y[b]))
        out.append(torch.cat([g[n].flatten(1) for n in names], 1))
    return torch.cat(out)


def cos_scores(G, obs):
    """cos(−g_i, obs) cho mọi dòng g_i."""
    return F.cosine_similarity(-G, obs[None], dim=1).numpy()


def sorted_scores(G, obs):
    """cos(sort(−g_i), sort(obs)) — không cần biết vị trí."""
    so = torch.sort(obs).values
    return F.cosine_similarity(torch.sort(-G, dim=1).values, so[None], dim=1).numpy()


@torch.no_grad()
def per_sample_loss(net, X, y, idx):
    b = np.sort(idx)
    out = net(torch.from_numpy(np.asarray(X[b], np.float32)))
    return F.cross_entropy(out, torch.from_numpy(y[b]), reduction="none").numpy()


def lossdrop_auc(head, w0, vis, delta_vis, data):
    """Suy luận thành viên dựa trên mức giảm loss: bs dựng mô hình cục bộ ước lượng w0 + Δ̂ (Δ̂ chỉ trên vị trí bs thấy),
    điểm = L(x; w0) − L(x; w0 + Δ̂). Thành viên thường giảm loss nhiều hơn."""
    (Xm, ym, mem, L0m), (Xn, yn, non, L0n) = data
    w = w0.copy()
    w[vis] += delta_vis
    net = model.build_head(None, pretrained=False).eval()
    model.set_vec(net, w)
    return auc(L0m - per_sample_loss(net, Xm, ym, mem), L0n - per_sample_loss(net, Xn, yn, non))


def rank_match(obs, proxy):
    """Đoán thứ tự: giá trị lớn thứ r của quan sát đặt vào vị trí có proxy lớn thứ r. Trả (vector đoán, hoán vị đoán)."""
    pos = np.argsort(proxy)            # vị trí theo proxy tăng dần
    val = np.argsort(obs)              # chỉ số quan sát theo giá trị tăng dần
    guess = np.empty_like(obs)
    guess[pos] = obs[val]
    pi_hat = np.empty(len(obs), int)   # obs[k] được đoán ở vị trí pi_hat[k]
    pi_hat[val] = pos
    return guess, pi_hat


def main():
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    K = open(os.path.join("secrets", "perm_seed"), "rb").read()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="thư mục kết quả (có capture/upload_0001.npz); bỏ trống = mọi cấu hình bước 8")
    ap.add_argument("--out", default=OUT, help="thư mục ghi kết quả (mặc định experiments/attack — kết quả chính)")
    args = ap.parse_args()
    out = args.out
    dirs = args.dirs or sorted(d for d in glob.glob("experiments/step8/*") if os.path.isfile(os.path.join(d, "capture", "upload_0001.npz")))
    miss = [d for d in dirs if not os.path.isfile(os.path.join(d, "capture", "upload_0001.npz"))]
    if miss:
        sys.exit(f"[attack] thiếu capture/upload_0001.npz (dữ liệu bs nhận): {miss}")
    full = os.path.join("data", "full")
    Xtr = np.load(os.path.join(full, "train_X.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(full, "train_y.npy"))
    Xte = np.load(os.path.join(full, "test_X.npy"), mmap_mode="r")
    yte = np.load(os.path.join(full, "test_y.npy"))
    head = model.build_head(config).eval()       # w0: bs biết (dựng từ config)
    order = np.asarray(model.priority_order(config))
    M = len(order)
    rng = np.random.default_rng(1)
    # dữ liệu công khai của bs để ước lượng hướng cập nhật (gỡ xáo); không thành viên lấy riêng từng thiết bị (bên dưới)
    pub = rng.choice(len(yte), 1000, replace=False)
    proxy_full = -per_sample_grads(head, Xte, yte, pub).mean(0).numpy()   # hướng cập nhật ước lượng từ dữ liệu công khai
    w0 = model.get_vec(head)
    lam = float(config["dp"]["lam"])
    C = float(config["dp"]["clip"])
    Gm_cache = {}
    rows = []
    for d in dirs:
        cfg = yaml.safe_load(open(os.path.join(d, "config.yaml"), encoding="utf-8"))
        rec = json.load(open(os.path.join(d, "round_0001.json"), encoding="utf-8"))
        pol = cfg["policy"]
        split = cfg["devices"]["split"]
        n_he = int(rec["decision"]["n_he"])
        vis = order[n_he:]
        cap = np.load(os.path.join(d, "capture", "upload_0001.npz"))
        if "val_size" in cfg["devices"]:   # 3 tập (từ 28/10/2026): validation tách khỏi train, 45.000 ảnh chia 2 thiết bị
            parts = train_val_split(ytr, cfg)[1]
        else:                              # kết quả cũ (2 tập): 50.000 ảnh chia 2 thiết bị
            parts = split_clients(ytr, 2, split, float(cfg["devices"].get("dirichlet_alpha", 0.5)), int(cfg.get("seed", 0)))
        shuffled = pol.startswith("fixed") or pol in ("fas10", "fas50", "adaptive")
        pi = perm_for(K, 1, len(vis)) if shuffled and len(vis) else None
        true_order = {}
        for k in (1, 2):
            obs = cap[f"shuffled_dev{k}"].astype(np.float64)
            true_order[k] = unshuffle(obs, pi) if pi is not None else obs
        for k in (1, 2):
            r = dict(config=os.path.basename(d), policy=pol, split=split, device=k, p=rec["decision"]["p"], n_visible=len(vis))
            if len(vis) == 0:
                r.update(auc_bs=0.5, recovery_corr=0.0, auc_eav=0.5, recovery_eav=0.0,
                         method="không có phần rõ (toàn bộ đã mã hóa)")
                rows.append(r); print("[attack]", r, flush=True)
                continue
            key = (split, k)
            if key not in Gm_cache:
                mem = np.random.default_rng(10 + k).choice(parts[k - 1], N_MEM, replace=False)
                print(f"[attack] gradient từng ảnh tại w0: {N_MEM} ảnh thành viên thiết bị {k} ({split})", flush=True)
                # không thành viên CÙNG PHÂN BỐ LỚP với thành viên: non-IID lệch lớp mạnh (vd. iot1: 4.298 ảnh lớp 0, 13 ảnh
                # lớp 5); lấy test đều các lớp thì AUC đo chênh lệch lớp chứ không phải rò rỉ thành viên (lần chạy đầu:
                # adaptive non-IID AUC 0,41/0,58 "có ý nghĩa" dù bs khôi phục ≈ 0,6% cập nhật)
                rk = np.random.default_rng(20 + k)
                cnt = np.bincount(ytr[mem], minlength=10)
                pool = np.setdiff1d(np.arange(len(yte)), pub)   # tách khỏi dữ liệu công khai bs dùng để ước lượng
                nonk = np.sort(np.concatenate([rk.choice(pool[yte[pool] == c], cnt[c], replace=False) for c in range(10)]))
                Gm_cache[key] = (per_sample_grads(head, Xtr, ytr, mem), mem, per_sample_loss(head, Xtr, ytr, mem),
                                 per_sample_grads(head, Xte, yte, nonk), nonk, per_sample_loss(head, Xte, yte, nonk))
            Gm, mem, L0m, Gn, non, L0n = Gm_cache[key]
            data = ((Xtr, ytr, mem, L0m), (Xte, yte, non, L0n))
            vt = torch.from_numpy(vis)
            Gmv, Gnv = Gm[:, vt], Gn[:, vt]
            obs = cap[f"shuffled_dev{k}"].astype(np.float64)
            xt = true_order[k]
            # độ khôi phục: tương quan (phần rõ đã nhiễu, Δ sạch) ước từ phương sai: var(x) = var(Δ) + sd²
            # trọng số FedAvg n_k/n đã nhân vào Δ (log có "aggregation" từ 28/10/2026; log cũ: 1)
            wk = float(rec.get("aggregation", {}).get("weights", [1.0, 1.0])[k - 1])
            if pol == "no_he":
                snr_corr = 1.0
            elif pol == "adaptive":
                sd = lam * wk * C / np.sqrt(M) if lam > 0 else 0.0
                snr_corr = float(np.sqrt(max(0.0, 1 - sd ** 2 / xt.var())))
            else:                                   # fixed10/fas: sd = λ·RMS(phần rõ sạch) => corr = 1/sqrt(1+λ²)
                snr_corr = float(1 / np.sqrt(1 + lam ** 2))
            r["noisy_vs_clean_corr"] = snr_corr
            # GÓC NHÌN CÓ THỨ TỰ (có K hoặc không xáo): no_he — mọi bên; fas — bs (có K); fixed10/adaptive — giả định
            xt_t = torch.from_numpy(xt.astype(np.float32))
            r["auc_cos_if_bs_has_K"] = auc(cos_scores(Gmv, xt_t), cos_scores(Gnv, xt_t))
            r["auc_loss_if_bs_has_K"] = lossdrop_auc(head, w0, vis, xt, data)
            view_K = dict(auc_cos=r["auc_cos_if_bs_has_K"], auc_loss=r["auc_loss_if_bs_has_K"], recovery=snr_corr)
            # GÓC NHÌN KHÔNG CÓ K (chỉ thấy vector đã xáo): fixed10/adaptive — bs VÀ kẻ nghe lén; fas — kẻ nghe lén
            view_noK = None
            if shuffled:
                ot = torch.from_numpy(obs.astype(np.float32))
                guess, pi_hat = rank_match(obs, proxy_full[vis])
                gt = torch.from_numpy(guess.astype(np.float32))
                cands = {"cos_thẳng": auc(cos_scores(Gmv, ot), cos_scores(Gnv, ot)),
                         "cos_sắp_xếp": auc(sorted_scores(Gmv, ot), sorted_scores(Gnv, ot)),
                         "cos_đoán_thứ_tự": auc(cos_scores(Gmv, gt), cos_scores(Gnv, gt)),
                         "loss_thẳng": lossdrop_auc(head, w0, vis, obs, data),
                         "loss_đoán_thứ_tự": lossdrop_auc(head, w0, vis, guess, data)}
                r.update({f"auc_{a}": v for a, v in cands.items()})
                view_noK = dict(auc_cos=max((v for a, v in cands.items() if a.startswith("cos")), key=lambda v: abs(v - 0.5)),
                                auc_loss=max((v for a, v in cands.items() if a.startswith("loss")), key=lambda v: abs(v - 0.5)),
                                recovery=float(np.corrcoef(guess, xt)[0, 1]) * snr_corr)
            bs_has_K = (not shuffled) or pol.startswith("fas")      # fas gốc: bs giữ khóa xáo
            vb = view_K if bs_has_K else view_noK
            ve = view_K if not shuffled else view_noK               # kẻ nghe lén KHÔNG BAO GIỜ có K
            r.update(auc_cos=vb["auc_cos"], auc_loss=vb["auc_loss"], recovery_corr=vb["recovery"],
                     method=("đúng thứ tự" if not shuffled else "bs có K (FAS gốc): giải xáo" if bs_has_K
                             else "bs KHÔNG có K: tốt nhất của thẳng / sắp xếp / đoán thứ tự"))
            r["auc_bs"] = max(vb["auc_cos"], vb["auc_loss"], key=lambda v: abs(v - 0.5))
            r["auc_eav"] = max(ve["auc_cos"], ve["auc_loss"], key=lambda v: abs(v - 0.5))
            r["recovery_eav"] = ve["recovery"]
            # khoảng 95% của AUC khi KHÔNG rò rỉ: 0,5 ± 1,96·sqrt((n1+n0+1)/(12·n1·n0)) (Mann–Whitney)
            r["auc_null_halfwidth95"] = 1.96 * np.sqrt((N_MEM + N_NON + 1) / (12 * N_MEM * N_NON))
            r["leak_significant"] = bool(abs(r["auc_bs"] - 0.5) > r["auc_null_halfwidth95"])
            r["leak_significant_eav"] = bool(abs(r["auc_eav"] - 0.5) > r["auc_null_halfwidth95"])
            if shuffled:
                guess, pi_hat = rank_match(obs, proxy_full[vis])
                r.update(deshuffle_exact=float((pi_hat == pi).mean()), deshuffle_random=1.0 / len(vis),
                         deshuffle_corr=float(np.corrcoef(guess, xt)[0, 1]),
                         deshuffle_sign=float((np.sign(guess) == np.sign(xt)).mean()))
            rows.append(r)
            print("[attack]", {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
        if shuffled:
            c_pair = float(np.corrcoef(cap["shuffled_dev1"], cap["shuffled_dev2"])[0, 1])
            other = perm_for(bytes(32), 1, len(vis))   # nếu thiết bị 2 dùng π khác
            c_sep = float(np.corrcoef(cap["shuffled_dev1"], cap["shuffled_dev2"][np.argsort(pi)][other])[0, 1])
            rows.append(dict(config=os.path.basename(d), policy=pol, split=split, device="1&2", pair_corr_same_pi=c_pair,
                             pair_corr_if_separate_pi=c_sep))
            print(f"[attack] {os.path.basename(d)}: tương quan cặp (Δ1_j, Δ2_j) bs thấy được do chung π_t = {c_pair:.3f}; "
                  f"nếu π riêng: {c_sep:.3f}", flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(dict(round=1, n_member=N_MEM, n_nonmember=N_NON, rows=rows),
              open(os.path.join(out, "attack_round1.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "attack_round1.csv"), index=False)
    # bảng tóm tắt dễ đọc: trung bình 2 thiết bị
    d = df[df["device"].astype(str).isin(["1", "2"])].copy()
    s = d.groupby(["config", "policy"], sort=False).agg(p=("p", "first"), tram_goc_khoi_phuc_pct=("recovery_corr", "mean"),
                                                         nghe_len_khoi_phuc_pct=("recovery_eav", "mean"),
                                                         auc_tram_goc=("auc_bs", "mean"), auc_nghe_len=("auc_eav", "mean"))
    s[["tram_goc_khoi_phuc_pct", "nghe_len_khoi_phuc_pct"]] *= 100
    s.reset_index().to_csv(os.path.join(out, "tom_tat.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n=== TÓM TẮT: bên tấn công khôi phục được bao nhiêu % cập nhật (vòng 1; thấp là tốt)"
          " — AUC 0,5 = không đoán được thành viên ===")
    print(s.round(3).to_string())
    print(f"[attack] -> {out}/attack_round1.(json|csv), tom_tat.csv")


if __name__ == "__main__":
    main()
