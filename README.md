# Testbed FL – UAV relay AF – NOMA – HE thích nghi

Hệ thống dựng theo `CLAUDE.md` (thiết kế đồ án Phan Hồng Quí, FPT).
Dựng theo **8 bước** của mục 10, mỗi bước test đạt mới sang bước sau.

## ĐỔI HƯỚNG (28/10/2026): phương án chính là FAS p = 10%, khóa xáo ở thiết bị (`fixed10`)

`fixed10`: mã hóa HE đa khóa 10% tham số (thứ tự ưu tiên), phần còn lại thêm nhiễu λ·RMS rồi xáo π_t — khóa xáo K CHỈ ở
2 thiết bị (bs cộng mù). So với `no_he`, `full_he`, và `fas10` (FAS gốc: bs giữ khóa xáo), có UAV và không UAV.
`adaptive` (chọn p theo SINR) là **hướng phát triển** — giữ code, chú thích `[HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]`.
FedAvg chuyển về thiết bị: Δ_k × n_k/(n_1+n_2) (sau clipping, trước khi tách), bs chỉ cộng, iot w ← w + tổng
(`tests/test_fedavg_fixed10.py`: trùng từng bit công thức cũ; 4 container IID 3 vòng: no_he trùng độ chính xác cũ).
Không đổi CKKS, kênh, C, λ, T_max (62,93 s).

```bash
sh tools/run_step8_fas.sh        # bước 8 mới + reference_sim + tấn công + hình + chạy lại test (~2,5 giờ)
python tools/fas_report.py       # hình experiments/fas/e20_he_vs_fas.png + bảng so_voi_mo_phong.csv (trong testbed-tools)
```
Dữ liệu **3 tập** (`data/DOC_TRUOC.md`): train 45.000 (22.500/thiết bị, `data/train/client{1,2}`), validation 5.000
(`data/val`, 500/lớp, tách từ train gốc), test 10.000 (`data/test`, chỉ để báo cáo). Mỗi vòng: cả 2 iot đo mô hình chung
trên tập train của mình (`train_accuracy`), iot1 đo thêm validation (`val_accuracy`) và test (`accuracy`).
Testbed là nguồn kết quả báo cáo — không so reference_sim (dữ liệu huấn luyện đã khác).

Kết quả (IID, seed 0, 15 vòng, 4 container; độ chính xác = TB 3 vòng cuối; rò rỉ = độ khôi phục cập nhật vòng 1 —
`tools/attack_eval.py`, tách góc nhìn trạm gốc / kẻ nghe lén):

| Phương án | Kênh | Test | Validation | Train iot1 / iot2 | Thời gian vòng (Pi 4) | Trễ hạn | Gửi lên / thiết bị | bs khôi phục | Nghe lén khôi phục |
|---|---|---|---|---|---|---|---|---|---|
| Không HE | Có UAV | 80,5% | 81,3% | 88,3 / 88,5% | 37,2 s | 0/15 | 0,81 MB | 100% | 100% |
| | Không UAV | 80,5% | 81,3% | 88,3 / 88,5% | 1.445 s | 15/15 | 0,81 MB | 100% | 100% |
| **FAS 10%, khóa ở thiết bị** | **Có UAV** | **79,1%** | **80,3%** | **85,7 / 85,7%** | **42,6 s** | **1/15** | **1,39 MB** | **0,7%** | **0,7%** |
| | Không UAV | 79,1% | 80,2% | 85,9 / 85,6% | 3.034 s | 15/15 | 1,39 MB | 0,7% | 0,7% |
| FAS gốc 10%, bs giữ khóa | Có UAV | 79,2% | 80,0% | 85,9 / 85,8% | 42,6 s | 1/15 | 1,39 MB | **31,6%** | 0,7% |
| HE toàn phần | Có UAV | 80,7% | 81,5% | 88,4 / 88,4% | 92,0 s | 15/15 | 6,60 MB | 0% | 0% |
| | Không UAV | 80,5% | 81,3% | 88,3 / 88,3% | 17.340 s | 15/15 | 6,60 MB | 0% | 0% |

- FAS 10% (khóa ở thiết bị) chỉ chậm hơn Không HE 5,5 s/vòng, kịp hạn 14/15 (vòng 10: kênh 0,39 Mbit/s); cả trạm gốc lẫn
  kẻ nghe lén chỉ khôi phục ~0,7% cập nhật. FAS gốc cùng chi phí nhưng trạm gốc khôi phục 31,6% (= 1/√(1+λ²), chỉ còn nhiễu).
  Độ chính xác FAS 10% thấp hơn Không HE ~1,4 điểm (do nhiễu trên 90% phần không mã hóa).
- HE toàn phần gửi gấp 4,7 lần và trễ hạn mọi vòng. Không UAV (tốc độ ~33 lần thấp hơn): mọi phương án trễ hạn.
- Train cao hơn validation ~5–7 điểm: mô hình khớp ảnh đã học hơn ảnh mới; 2 thiết bị gần như nhau (IID).
- Thời gian vòng đo được khớp mô hình dự báo `round_time` trên cùng 15 kênh: lệch < 1 s có UAV, < 0,1% không UAV; trễ hạn
  trùng hoàn toàn (`experiments/fas/so_voi_mo_phong.csv`).
- Suy luận thành viên vòng 1: không có ý nghĩa thống kê ở mọi phương án (AUC 0,49–0,51).
- Rò rỉ trong testbed là độ khôi phục cập nhật, KHÔNG phải tấn công nhận dạng ảnh 1-gradient (không tái hiện được với cập
  nhật 1 epoch). Hình: `experiments/fas/e20_he_vs_fas.png` (có UAV), `e20_co_va_khong_uav.png`, `e20_chi_tiet_6_o.png`.
- Chạy lại toàn bộ test (`experiments/regression/20261029_0029/summary.txt`): **9/10 ĐẠT**. `test_fedavg_fixed10` lần đầu lỗi
  vì test còn mong 25.000 ảnh/thiết bị (đã sửa cho 3 tập, chạy lại ĐẠT). `test_step3_pi4` (5 epoch phải lệch nhau ≤ 10%) chưa
  đạt do máy bận việc nền (Chrome, VS Code): dao động 12,6–34,8%; trung vị 23,5 s khớp số đo T_train riêng 22,0 s (6,7%) —
  phụ thuộc trạng thái máy, không phải mã (lần trước đạt khi máy rảnh). Chạy lại khi đóng bớt ứng dụng.
- **Mọi kết quả CŨ ở `experiments/KET_QUA_CU/`** (xem `DOC_TRUOC.md` trong đó); các mục "Bước 8", "Đánh giá tấn công",
  "Kiểm chứng toàn thiết kế" bên dưới mô tả bản cũ đó.

## Trạng thái: ✅ BƯỚC 1 — ✅ 2 (FL thật) — ✅ 3 (Pi 4 + chi phí) — ✅ 4 (kênh + bóp tốc độ) — ✅ 5 (HE đa khóa) — ✅ 6 (clipping + nhiễu + xáo π_t) — ✅ 7 (policies) — ✅ 8 (thí nghiệm, seed 0)

| Bước | Test | Kết quả |
|---|---|---|
| 1 | `tests/test_round.py`: 3 vòng đủ 7 bước, đủ log | ĐẠT (giờ chạy với huấn luyện thật) |
| 2 | `tests/test_step2_fl.py`: `no_he`, IID, 15 vòng, ≈ 80%, lệch ≤ 1,5 điểm so với reference_sim | ĐẠT: **81,38%** vs reference_sim **81,01%** (lệch 0,37 điểm) |
| 3 | `tests/test_step3_pi4.py`: giới hạn Pi 4 (compose + cgroup), epoch ổn định ±10%, bảng chi phí đo trên máy này (OpenMP = 2 luồng) | ĐẠT: epoch trung vị 27,3 s thô (lệch tối đa 7,4%), T_train = 36,9 s sau ×1,4 |
| 4 | `tests/test_step4_channel.py`: γ trùng reference_sim; 5 MB @ 1 Mbit/s, 2 thiết bị cùng lúc ≈ 40 s; R_k thật `uav_noma_pc`, `direct_noma`; 1 vòng FL; **uav bị dừng 3 s giữa tin nhắn** | ĐẠT: 40,02 s (lệch 0,04%); R_k thật lệch ≤ 0,03%; UPLOAD trong vòng FL lệch 0,01%; dừng 3 s: vẫn 10,00 s (token bucket cũ: 13,10 s) |
| 5 | `tests/test_step5_he.py`: HE đa khóa giữa bs và 2 iot, p = 0,5 và 1 | ĐẠT: sai số tổng 5,2–5,8e-7 (< 1e-6); bs chỉ nhận dữ liệu công khai; 1 phần giải mã không giải được; byte UPLOAD lệch bảng chi phí 0,2% |
| 6 | `tests/test_step6_protect.py` + `tests/check_secrets_docker.sh`: HKDF, π_t, bảo vệ theo policy, λ=0 trùng FedAvg, nhiễu λ=3, K chỉ ở iot | ĐẠT: sai khác phần rõ so với FedAvg **0** (< 1e-9); 2 iot cùng π_t, π_t đổi theo vòng; sd nhiễu 0,01670 vs 0,01667; bs, uav không có K (container thật) |
| 7 | `tests/test_step7_policy.py`: lịch p `adaptive` vs `integrate.choose_p` (cùng bảng chi phí, cùng t_max), policy cố định, log Pi 4, fas50 | ĐẠT: **0/4040** khác biệt p và cờ trễ hạn (uav_noma_pc + direct_noma, ngưỡng chọn t_max − δ = 61,43 s); δ = 1,5 s khớp p95 số đo; fas50: bs giải xáo đúng, mô hình 2 iot giống hệt |
| 8 | 15 cấu hình × 4 container thật (mục "Bước 8" bên dưới) + chạy lại toàn bộ test sau bước 8 (`experiments/KET_QUA_CU/regression_20261027_1823/`) | 165 vòng, 165/165 phép đo tin cậy; adaptive trễ hạn 1/15 (IID và non-IID); khớp reference_sim ≤ 0,5 điểm (trung bình 5 vòng cuối); test **9/9 ĐẠT** |

> [!CAUTION]
> **⚠️ CHÚ Ý 1 — `uav_oma` trong testbed CHẠY SAI (để sau).** OMA (TDMA) = hai thiết bị phát **lần lượt** (thời gian = tổng),
> nhưng relay vẫn cho phát **song song** (thời gian = max) ⇒ thời gian truyền lạc quan. **Không dùng số testbed cho OMA**;
> so sánh NOMA/OMA lấy từ `reference_sim` (`channel_exp.py` E1–E3). Không thuộc bước 8.
> **Khi sửa:** uav cho 2 iot gửi LẦN LƯỢT (TDMA). Hiện khi chọn, uav in cảnh báo đỏ và log vòng có `"warnings"`.
>
> **⚠️ CHÚ Ý 2 — SIC LÝ TƯỞNG** (`uav/channel.py`, giống `reference_sim`): thiết bị giải sau không còn nhiễu dư (ξ = 0),
> không lan truyền lỗi; CSI hoàn hảo (chọn α, thứ tự SIC); tốc độ Shannon (không khoảng cách MCS); kênh cố định trong vòng.
> ⇒ R_k là **cận trên**, thời gian truyền là **cận dưới**, `adaptive` có thể chọn p cao hơn thực tế.
> **Đã quyết: giữ SIC lý tưởng như reference_sim — ghi là GIỚI HẠN trong báo cáo.**
>
> **⚠️ CHÚ Ý 3 — GIỚI HẠN: dùng chế độ nhiễu mặc định của OpenFHE** (`common/mkckks.py` `make_context`, giống
> `reference_sim`): không đặt `SetMultipartyMode` => `FIXED_NOISE_MULTIPARTY` (nhiễu nhỏ trong phần giải mã từng bên δ_i).
> bs nhận δ_1, δ_2 mỗi vòng; về lý thuyết nhiễu nhỏ có thể làm lộ thông tin về khóa bí mật qua nhiều vòng.
> `NOISE_FLOODING_MULTIPARTY` che tốt hơn nhưng cần modulus lớn hơn (đổi tham số CKKS, ciphertext to hơn, đo lại chi phí).
> Ngoài ra bs là bên tạo context (mục 5.1) nên bs chọn chế độ này — chấp nhận theo giả định bs trung thực nhưng tò mò.
> **Đã quyết: giữ mặc định, ghi là GIỚI HẠN trong báo cáo.**

Bước 2: iot huấn luyện fire9 + classifier (202.314 tham số) 1 epoch/vòng trên đặc trưng thân SqueezeNet đã trích;
w0 và `priority_order` khớp từng bit với `reference_sim/common.py`. Kết quả reference_sim dùng để so sánh
(`fl.py --schedule 15×p=0 --lam 0 --he 0`, cùng đặc trưng, cùng phân hoạch) lưu ở `tests/ref/`.
Từ bước 5, HE là OpenFHE thật.

## 1. Chuẩn bị (một lần)

**CA phụ khi build** — nếu máy có phần mềm quét HTTPS (vd. Avast Web Shield) thì pip trong container báo
`CERTIFICATE_VERIFY_FAILED`. Xuất CA gốc của phần mềm đó ra `certs/extra_ca.pem` (chỉ dùng lúc build qua
build secret, không nằm trong image). Máy không có: tạo `certs/extra_ca.pem` rỗng.

**Dữ liệu** (image công cụ `testbed-tools`, chạy ~10 phút):
```bash
git clone https://github.com/YoongiKim/CIFAR-10-images        # 60.000 ảnh
# tải squeezenet1.1-7.onnx (link trong tools/prep_data.py / reference_sim/convert_weights.py)
docker build --secret id=extra_ca,src=certs/extra_ca.pem -t testbed-tools -f tools/Dockerfile .
docker run --rm -v "<testbed>:/tb" -v "<CIFAR-10-images>:/cifar:ro" -v "<thư mục onnx>:/w:ro" -w /tb \
  testbed-tools python tools/prep_data.py --cifar /cifar --onnx /w/squeezenet1.1-7.onnx --threads 12
```
Tạo `weights/squeezenet1_1_imagenet.pth`, `data/full/` (đặc trưng đầy đủ, ~2,2 GB), `data/client1|2/` (theo
`devices.split` trong config), `data/test/`. Đổi iid ↔ noniid: sửa config rồi chạy `prep_data.py --split-only`.

## 1b. Bảng chi phí (bước 3)
```bash
docker run --rm --cpus 1.6 --memory 4g --memory-swap 4g -e NTHREADS=2 -e OMP_NUM_THREADS=2 -v "<testbed>:/tb" -w /tb testbed-tools python tools/bench_costs.py
```
Giới hạn như iot1/iot2 (1,6 lõi, 4 GB, không swap; tương đương `reference_sim/run_pi4.sh`). Kết quả `costs/cost_table.csv`,
`costs/cost_fixed.json` (thời gian thiết bị đã ×`fidelity`; có kèm thời gian thô, CPU máy đo). Đo lại khi đổi máy.

## 1c. Kênh và bóp tốc độ (bước 4)
- `uav/channel.py`: mô hình của reference_sim (Al-Hourani, Rician, AF, NOMA + SIC; 4 chế độ, `los_mode`), tham số trong
  `config.yaml channel`; kênh vòng t sinh với seed `(seed, t)`, cùng rng thì γ trùng reference_sim từng bit.
- `uav/relay.py` + `uav/shaper.py`: chiều lên chuyển tiếp theo khối 16 KB, **định nhịp neo theo đầu tin nhắn**: byte thứ n
  rời uav không sớm hơn t0 + n/R_k. Không "gửi miễn phí" sau lúc rảnh; khi máy giả lập quá tải (luồng uav ngủ quá giờ, bs đọc
  chậm) thì gửi bù để tổng thời gian vẫn = bytes/R_k. (Token bucket cũ bỏ mất token khi trễ: bước 8 lần đầu có vòng chỉ đạt
  0,879·R_k; thử dừng 3 s giữa tin nhắn 10 s: 13,10 s thay vì 10,00 s.) uav ghi byte + thời gian truyền thật từng tin nhắn
  chiều lên, gửi `UAV_STATS` cho bs sau `ROUND_DONE`; log có `uplink.devK[].throughput_Bps`, `ratio_to_R`,
  `time.upload_tx_s`. Tin nhắn nhỏ (METRICS) tỉ lệ vô nghĩa.
- **Cờ tin cậy phép đo:** `measurement_ok` = mọi tin nhắn chiều lên ≥ 64 KB có thông lượng trong R_k·(1 ± `shaping_tol` = 5%);
  sai thì `measurement_issues` liệt kê. `tools/step8_report.py` đếm `bad_rounds` và tính `late_pct_ok` chỉ trên vòng tin cậy
  (log cũ không có cờ thì tính lại từ `uplink`). Trong lúc chạy thí nghiệm nên tránh dùng máy cho việc nặng khác.
- **Nén thời gian kênh (tùy chọn, mặc định TẮT)** — `config.yaml channel.time_scale`:
  `off` (thời gian thật) | `<số k ≥ 1>` (nén cố định) | `auto` (k = max(1, `time_scale_floor_bps` / min(R_1, R_2)), sàn mặc định
  1 Mbit/s: chỉ nén vòng kênh xấu, `uav_noma_pc` ≈ 1,3–1,9 Mbit/s gần như không bị nén). uav cho dữ liệu đi ở R_k·k, báo thời gian
  ×k (= thời gian thật trên kênh R_k) nên `round_pi4_s`, cờ trễ hạn không đổi cách tính; hai thiết bị cùng k. Log: `time_scale`,
  `uplink.*.wall_s` (thời gian máy); `late_wallclock` = null khi k > 1. Dòng lệnh: `TIME_SCALE=auto sh tools/run_step8.sh ...`.
  Hợp lệ vì thời gian truyền tỉ lệ tuyến tính với tốc độ (test bước 4 F: k = 50 và auto k = 1000, lệch ≤ 0,03%). Lưu ý: sai số
  thời gian của máy bị phóng đại ×k — tránh k lớn cho gói nhỏ (auto với sàn 1 Mbit/s an toàn: gói thật chạy ≥ vài giây).
  Dùng cho `direct_noma` (vòng 1: 0,001 Mbit/s => chạy thật ~33 giờ cho 5 policy × 3 vòng).
- bs nhận UPLOAD / PARTIAL_DEC của 2 thiết bị SONG SONG (đọc lần lượt làm ứ bộ đệm TCP, sai thời gian truyền tới 40%).
- `direct_noma` có vòng rất xấu (vd. vòng 1 thiết bị 2 ≈ 0,001 Mbit/s: UPLOAD 809 KB mất ~1,8 giờ) — đúng mô hình, mục 7.

## 1d. HE đa khóa (bước 5)
- Thiết lập một lần: bs tạo CryptoContext (`HE_CONTEXT`, 517 B) -> iot1 `KeyGen` -> `HE_PK1` (qua bs) -> iot2
  `MultipartyKeyGen(pk1)` -> `HE_PK` chung (~132 KB) -> bs phát cho cả hai. Khóa bí mật chỉ nằm trong `iot/he.py DeviceHE`.
  Log: `results/setup.json`.
- Mỗi vòng: iot mã hóa bằng pk chung (ciphertext ~132 KB/4096 giá trị, bytes `Serialize(BINARY)`), bs `EvalAdd` ->
  `CSUM` -> iot1 `MultipartyDecryptLead`, iot2 `MultipartyDecryptMain` -> `PARTIAL_DEC` (~66 KB/khối) -> bs
  `MultipartyDecryptFusion` -> tổng phần HE (float64) trong `AGGREGATE`.
- **OpenMP:** đặt `OMP_NUM_THREADS` = số lõi container (bs 1, iot 2). Để mặc định, OpenFHE mở 16 luồng dưới quota 1 lõi
  => ghép HE tại bs chậm ~40× (3,8 s thay vì 0,1 s).
- Mỗi thiết bị gửi kèm thời gian pha (`train_s`, `enc_s`, `pdec_s`, `update_s`); log có `time.dev1|dev2` và bản `_pi4` (×fidelity).

## 1e. Clipping + nhiễu + xáo π_t (bước 6, `iot/protect.py`)
- Theo policy nhận trong `ROUND_START` (bảng mục 8): `adaptive` clipping ‖Δ‖ ≤ C, nhiễu N(0,(λ·C/√M)²), xáo;
  `fas10/50` nhiễu λ·RMS(phần rõ), xáo; `no_he`, `full_he` không gì. (bs giữ K và giải xáo cho `fas*`: bước 7.)
- π_t: seed = HKDF-SHA256(K, "perm" ‖ t) (RFC 5869, kiểm bằng vector chuẩn) -> `numpy` PCG64 `permutation`. K = `secrets/perm_seed`
  (32 byte) chỉ mount vào iot1, iot2. bs cộng mù theo vị trí; iot giải xáo tổng một lần.
- Nhiễu dùng entropy hệ điều hành của từng thiết bị — KHÔNG từ K, KHÔNG từ `seed` trong config (bs có config, sẽ sinh lại
  được nhiễu và trừ đi). Hệ quả: các lần chạy có nhiễu không tái lập được từng bit.
- Phần rõ truyền float32 (4 byte/tham số như bảng chi phí) => so với FedAvg float64 lệch ~1e-9; so với FedAvg trên cùng
  dữ liệu truyền: 0.

## 1f. Policies và T_max (bước 7, `bs/policy.py`)
- **T_max = T_train_Pi4 + 26 s** (`config.yaml`: `t_max: auto`, `t_max_margin_s: 26`). T_train_Pi4 = `costs/cost_fixed.json`
  `T_train_epoch_s` (đo bước 3, đã ×1,4). Máy này: 36,93 + 26 = **62,93 s**. Lý do: T_train máy này lớn hơn reference (23,8 s);
  giữ nguyên khoảng ≈ 26 s dành cho mã hóa + truyền như reference (T_max 50 = 23,8 + 26,2).
- `adaptive`: `choose_p` y hệt `reference_sim/integrate.choose_p` (lưới p_min..1 bước p_step, `round_time` từ bảng chi phí,
  R_k từ γ), nhưng chọn p với **T_max − δ** (biên an toàn `safety_margin_s` = 1,5 s); cờ trễ hạn vẫn so với T_max.
  Lịch tham chiếu: `tools/ref_choose_p.py --ref-sim <reference_sim>` (integrate.choose_p với T_max − δ) ->
  `tests/ref/choose_p_ref.json` (**chạy lại sau mỗi lần đo lại bảng chi phí** — test kiểm mã băm).
- **Kết quả kiểm chứng mô hình thời gian** (bước 7, 15 vòng testbed thật, `experiments/KET_QUA_CU/tmax_sweep_adaptive/pred_error.csv`):
  thời gian vòng ĐO (thang Pi 4) lớn hơn DỰ BÁO của `round_time` **0,4–1,1 s** (trung bình 0,73 s, p95 1,10 s).
  Nguyên nhân: `round_time` (như reference_sim) bỏ qua **thời gian tổng hợp tại bs** (cộng đồng cấu + ghép phần giải mã) và
  **cập nhật mô hình** (cùng ≈ 0,26 s), và **overhead gói** (khung, msgpack, Serialize chuỗi lớn hơn tệp ~0,2%; ≈ 0,47 s).
  Vì `adaptive` chọn p sát biên, không có δ thì vòng dự báo cách T_max < ~1 s trễ hạn thật (T_max 80: 2/3 vòng).
  => **δ = phân vị 95% của lệch, làm tròn lên bội 0,5 s = 1,5 s** (`tools/calib_margin.py`; test bước 7 kiểm lại).
  Log mỗi vòng: `pred_error_s` = đo − dự báo, `t_select` = T_max − δ.
- Trễ hạn (quyết định 2a): `late = time.round_pi4_s > t_max`, thang Pi 4 cùng `choose_p`:
  max_k(T_train_bảng + 1,4·enc_k + tx UPLOAD_k) + bs cộng + max_k(1,4·pdec_k + tx PARTIAL_DEC_k) + bs ghép + 1,4·update.
  Huấn luyện lấy từ bảng chi phí (quyết định 1a: thiết bị thật chạy một mình). Log thêm `round_pi4_measured_train_s`
  (dùng huấn luyện đo được), `late_wallclock` (thời gian thô), `decision.T_pred_s` (dự báo của mô hình).
- **Sai lệch do giả lập trên một máy** (`sim_artifact` mỗi vòng): 2 iot huấn luyện song song chậm hơn 1 thiết bị một mình
  (bảng chi phí): +33% ở vòng ổn định, tới +90…146% ở vòng 1 (khởi động lạnh). Không tính vào trễ hạn.
- `fas10/fas50`: chạy với `docker compose -f docker-compose.yml -f docker-compose.fas.yml up` — bs có K, giải xáo từng
  thiết bị rồi cộng (như FAS gốc; `capture/` có `bs_view_devK`). Thiếu file này mà đặt policy fas thì bs từ chối chạy.
  (CLAUDE.md nói "profile fas"; dùng file override vì profile của compose không thêm được secret cho một service có sẵn.)

## 2. Chạy test (không cần docker-compose)
```bash
docker run --rm -e NTHREADS=2 -v "<testbed>:/tb" -w /tb testbed-tools python tests/test_round.py      # ~3 phút
docker run --rm -e NTHREADS=2 -v "<testbed>:/tb" -w /tb testbed-tools python tests/test_step2_fl.py   # ~12 phút
docker run --rm --cpus 1.6 --memory 4g --memory-swap 4g -e NTHREADS=2 -e OMP_NUM_THREADS=2 -v "<testbed>:/tb" -w /tb testbed-tools python tests/test_step3_pi4.py  # ~3 phút
docker run --rm -e NTHREADS=2 -v "<testbed>:/tb" -w /tb testbed-tools python tests/test_step4_channel.py   # ~3 phút
docker run --rm -e NTHREADS=2 -v "<testbed>:/tb" -w /tb testbed-tools python tests/test_step5_he.py        # ~5 phút
docker run --rm -e NTHREADS=2 -v "<testbed>:/tb" -w /tb testbed-tools python tests/test_step6_protect.py  # ~5 phút
sh tests/check_secrets_docker.sh                                                                          # container thật
docker run --rm -v "<testbed>:/tb" -w /tb testbed-tools python tests/test_robustness.py                    # vài giây
docker run --rm -e NTHREADS=2 -v "<testbed>:/tb" -w /tb testbed-tools python tests/test_step7_policy.py   # ~6 phút
```
4 thành phần chạy bằng thread trên 127.0.0.1. Log: `results/round_XXXX.json`, `capture/upload_XXXX.npz`.

Test ghi vào `tests/_out/` (KHÔNG đụng `results/`, `capture/` của docker-compose / thí nghiệm bước 8 — trước đây test xóa
và ghi đè vào đó). Độ bền: `python tests/test_robustness.py` (cả `python -O`).

## 2b. Độ bền giao thức
- bs đặt timeout: chờ kết nối / thiết lập HE = `protocol.round_timeout_min_s` (600 s); mỗi vòng =
  max(600 s, `round_timeout_factor` (5) × thời gian vòng dự báo) — kênh chậm tự nới. Một container treo hoặc sai giao thức
  => `ProtocolError` có thông báo rõ, bs thoát, compose dừng cả 4 container, `run_step8.sh` sang cấu hình kế (không treo mãi).
- Kiểm tra giao thức/cấu hình dùng `P.expect` / `raise`, không dùng `assert` (bị tắt khi `python -O`).

## 3. Chạy "như thật" (4 container)
```bash
docker compose up --build
```

## [BẢN CŨ — adaptive, hướng phát triển] Thí nghiệm: quét T_max 50–90 s (sau bước 7) — `experiments/KET_QUA_CU/tmax_sweep_adaptive/`
```bash
docker run --rm -v "<testbed>:/tb" -w /tb testbed-tools python tools/tmax_sweep.py analytic   # mô hình, 1000 mẫu kênh
sh tools/run_tmax_sweep.sh 3 50 60 70 80 90                                                     # 4 container thật, ~25 phút
docker run --rm -v "<testbed>:/tb" -w /tb testbed-tools python tools/tmax_sweep.py summarize  # measured.csv, summary.csv, hình
```
Mô hình (bảng chi phí máy này, `uav_noma_pc`) — `adaptive`: T_max 50 s -> p 20%, trễ hạn 37,7%; 60 s -> 38%, 2,3%;
65 s -> 49%, 1,3%; 70 s -> 60%, 0,7%; 80 s -> 83%, 0,4%; 90 s -> 94%, 0,3%. `fas50` trễ hạn 100% tới 55 s, 29% ở 65 s;
`full_he` ≥ 98% tới 75 s.

Testbed đo thật (`adaptive`, 3 vòng/T_max, cùng chuỗi kênh): p trung bình 0,20 / 0,37 / 0,53 / 0,80 / 0,93 cho T_max 50…90
(khớp đường mô hình). **Thời gian vòng đo được (thang Pi 4) luôn lớn hơn dự báo 0,4–1,1 s** (mô hình bỏ qua bs cộng/ghép,
cập nhật, phần đầu gói). Vì `adaptive` chọn p sát biên, vòng có dự báo cách T_max < ~1 s bị **trễ hạn thật**
(T_max 80: dự báo 79,5 s -> đo 80,3–80,4 s, 2/3 vòng trễ). Thí nghiệm này chạy KHI CHƯA có δ; từ đó `adaptive` chọn p với
T_max − δ (δ = 1,5 s tính từ chính các số đo này — mục 1f).

## [BẢN CŨ] Đánh giá tấn công trên dữ liệu bs nhận (mục 9) — `tools/attack_eval.py`, `experiments/KET_QUA_CU/step8_v1_adaptive/attack/`
```bash
docker run --rm --cpus 1 --memory 6g -e NTHREADS=1 -v "<testbed>:/tb" -w /tb testbed-tools python tools/attack_eval.py
```
Vòng 1 của các lượt bước 8 (`capture/`), mô hình toàn cục = w0 (bs biết từ config — thuận lợi nhất cho kẻ tấn công).
Chấm điểm dùng π_t thật tính từ K (đặc quyền người đánh giá; bs không có K trừ fas). Không thành viên lấy từ tập test
**cùng phân bố lớp** với thành viên của từng thiết bị (lần chạy đầu lấy test đều các lớp: non-IID cho AUC 0,41–0,64 "có ý
nghĩa" kể cả adaptive — vì AUC đo chênh lệch phân bố lớp, không phải rò rỉ; đã sửa). Kết quả 15 cấu hình (IID, non-IID,
có/không UAV — giống nhau theo policy):

| Policy | AUC thành viên tốt nhất (cos / mức giảm loss; khoảng 95% không rò rỉ 0,5 ± 0,025) | bs khôi phục Δ của thiết bị (tương quan) | Gỡ xáo khi không có K |
|---|---|---|---|
| `no_he` | 0,510–0,518 (không ý nghĩa) | **100%** | — |
| `fas10`, `fas50` (bs có K) | 0,486–0,507 | **32%** (chỉ nhiễu λ = 3: 1/√(1+λ²)) | — |
| `adaptive` (bs không có K) | 0,479–0,509 (không ý nghĩa) | **0,2–0,5%** | đúng vị trí 0% (ngẫu nhiên 1/n), tương quan 0,004–0,025, đúng dấu 50% |
| `full_he` | 0,5 | 0% (chưa tính tổng phần HE bs biết) | — |

- **Suy luận thành viên ở vòng 1 không phân biệt được policy nào — kể cả không bảo vệ**: cập nhật 1 epoch / 25.000 ảnh
  trung bình hóa quá mạnh (khớp CLAUDE.md mục 9: tấn công gradient 1 ảnh của reference là trường hợp xấu nhất, không tái hiện
  với giao thức thật). Thước đo phân biệt được là **độ khôi phục cập nhật**: 100% -> 32% (FAS) -> 0,5% (adaptive).
- Xáo π_t không gỡ được bằng thông tin công khai (đoán thứ tự theo hướng cập nhật ước từ dữ liệu công khai).
- **Rò rỉ do 2 thiết bị dùng CHUNG π_t:** bs ghép được cặp (Δ1_j, Δ2_j) cùng vị trí — tương quan cặp 0,032–0,045 IID,
  0,012–0,017 non-IID (π riêng: ≈ 0,002). Nhỏ vì nhiễu λ = 3 che; ghi là giới hạn (thiết kế cần π chung để bs cộng mù).
- Chưa đo: rò rỉ từ **tổng phần HE** mà bs biết (giới hạn mục 2). Từ bản build sau bước 8 `uav_noma_pc`, bs lưu `he_sum` vào
  `capture/` (để phân tích sau; cần thêm Δ thật phía thiết bị để chấm). Chưa đánh giá các vòng sau (bs không có w_t).

## [BẢN CŨ] Kiểm chứng toàn thiết kế với reference_sim — `tools/ref_compare.py`, `experiments/KET_QUA_CU/step8_v1_adaptive/ref_compare/`
Chạy `reference_sim/fl.py` (không sửa) với ĐÚNG lịch p testbed đã dùng (round_*.json bước 8) + cờ bảo vệ theo policy
(no_he/full_he `--lam 0`; fas `--lam λ --perm 1`; adaptive `--clip C --lam λ --perm 1`), cùng đặc trưng/phân hoạch/seed:
```bash
docker run --rm -v "<testbed>:/tb" -v "<reference_sim>:/ref:ro" -w /tb testbed-tools python tools/ref_compare.py run experiments/step8/<cấu hình> ...
docker run --rm -v "<testbed>:/tb" -w /tb testbed-tools python tools/ref_compare.py report
```
Nặng (huấn luyện): chạy khi máy rảnh (sau bước 8).

Kết quả 15 cấu hình bước 8 (`experiments/ref_compare/summary.csv`, `compare.png`): độ chính xác **trung bình 5 vòng cuối
lệch ≤ 0,5 điểm** ở mọi cấu hình `uav_noma_pc` (IID: −0,04…+0,37; non-IID: −0,49…+0,20). Vòng cuối riêng lẻ của non-IID lệch
tới −2 điểm vì non-IID dao động 2,5–5 điểm/vòng (vd. no_he vòng 14 → 15 của testbed: 76,9% → 73,5%); phân hoạch non-IID hai
bên giống hệt (cùng `split_clients`, α = 0,5, seed). `direct_noma` (3 vòng): vòng 3 lệch 0,1–1,1 điểm.

## [BẢN CŨ — adaptive là đề xuất] Bước 8 — thí nghiệm (seed 0) — `experiments/KET_QUA_CU/step8_v1_adaptive/`
```bash
sh tools/run_step8.sh uav_noma_pc iid 15 ; sh tools/run_step8.sh uav_noma_pc noniid 15
TIME_SCALE=auto sh tools/run_step8.sh direct_noma iid 3        # không UAV: kênh ~0,05 Mbit/s => nén thời gian kênh
docker run --rm -v "<testbed>:/tb" -w /tb testbed-tools python tools/step8_report.py   # summary.csv + hình
sh tools/show.sh [--rounds]                                     # xem số liệu chi tiết từng vòng (tốc độ, p, thời gian, trễ)
```
Thang Pi 4, T_max = 62,93 s, δ = 1,5 s. 165 vòng, mọi vòng `measurement_ok` (tốc độ đo lệch R_k ≤ 0,01% có UAV, ≤ 0,05% không UAV).

| Kênh / dữ liệu | Policy | p tb | Thời gian vòng tb (s) | Trễ hạn | Độ chính xác vòng cuối |
|---|---|---|---|---|---|
| Có UAV, IID | no_he | 0 | 41,7 | 0/15 | 81,2% |
| | fas10 | 0,1 | 47,3 | 1/15 | 80,3% |
| | **adaptive** | **0,41** | **61,7** | **1/15** | **81,0%** |
| | fas50 | 0,5 | 69,7 | 8/15 | 79,4% |
| | full_he | 1 | 96,7 | 15/15 | 81,0% |
| Có UAV, non-IID | no_he | 0 | 41,7 | 0/15 | 73,5% (tốt nhất 76,9%) |
| | fas10 | 0,1 | 47,3 | 1/15 | 72,1% |
| | **adaptive** | **0,41** | **61,5** | **1/15** | **72,4%** (tốt nhất 75,4%) |
| | fas50 | 0,5 | 69,2 | 6/15 | 72,2% |
| | full_he | 1 | 96,6 | 15/15 | 73,5% |
| Không UAV, IID (3 vòng) | mọi policy | 0–1 | 1.585–19.000 | 3/3 | 73,9–76,7% |

- `adaptive` mã hóa trung bình 41% cập nhật mà chỉ trễ hạn 1/15 vòng — vòng 10 (kênh 0,39 Mbit/s: p_min = 0,2 vẫn không
  kịp, đúng thiết kế). Không vòng adaptive nào trễ hạn khi dự báo kịp.
- Tốc độ kênh trung vị: **có UAV 1,77 Mbit/s, không UAV 0,054 Mbit/s** (thấp nhất 0,001) — không UAV mọi policy trễ hạn.
- **Lệch dự báo − đo** (150 vòng có UAV): trung bình 0,39 s, p95 0,98 s, lớn nhất 1,66 s (full_he); vòng có HE: p95 1,03 s;
  adaptive: p95 0,83 s, lớn nhất 0,87 s. δ = 1,5 s giữ nguyên (chưa đổi — chờ quyết định). `direct_noma` (nén k tới 685):
  lệch tới 44 s nhưng chỉ ~0,2% thời gian vòng — độ trễ cố định mỗi tin nhắn của máy giả lập bị nhân ×k khi quy đổi.
- Sai lệch do giả lập (`sim_artifact`): 2 iot huấn luyện song song trên một máy chậm thêm 7–99% (trung bình theo cấu hình) so với bảng chi phí; thang
  Pi 4 dùng T_train của bảng (quyết định 1a) nên không ảnh hưởng cờ trễ hạn.

## Demo, xem số liệu, vẽ hình (MATLAB/Octave)
- **Tỉ lệ mã hóa khác:** policy `fixed20` … `fixed90` (FAS p = 20…90%, khóa ở thiết bị), vd.
  `sh tools/run_step8.sh uav_noma_pc iid 15 fixed20 fixed30` hoặc `.\demo.bat 3 fixed10 fixed20`.
- **Đánh giá tấn công** (trạm gốc / kẻ nghe lén): `.\attack.bat fixed10 fixed20` (các tỉ lệ đã chạy ở bước 8 ->
  `experiments/attack/rieng_<thời điểm>/`), `.\attack.bat --demo` (lần demo gần nhất), `.\attack.bat` (mọi cấu hình bước 8,
  kết quả chính) — in bảng `tom_tat.csv`; hoặc `CHAY.bat` -> A.
- **Demo:** bấm đúp `demo.bat` (menu) hoặc `sh tools/demo.sh [số vòng] [policy ...]` — chạy thật 4 container, ghi vào
  `experiments/demo/<thời điểm>/` (không đụng số liệu bước 8), in bảng chi tiết cuối. `TIME_SCALE=auto` để nén kênh.
- **Hình kiểu MATLAB:** `sh tools/run_plots.sh` = `tools/export_matlab.py` (CSV phẳng `experiments/matlab/*.csv`) +
  `matlab/plot_results.m` chạy bằng GNU Octave trong Docker -> `experiments/matlab/fig/*.png`. Cùng file `.m` chạy được trên
  MATLAB (mở `matlab/`, gõ `plot_results`; lưu thêm `.fig`).

## Những gì CHƯA làm
- Seed 1–2 (khoảng tin cậy), `uav_oma` dạng TDMA (CHÚ Ý 1), rò rỉ từ tổng phần HE (cần Δ thật phía thiết bị).

> Quy ước (mục 11): mọi tham số trong `config.yaml`; không tự đổi thiết kế khi chưa hỏi;
> mỗi bước viết test trong `tests/` và báo kết quả trước khi sang bước sau.
