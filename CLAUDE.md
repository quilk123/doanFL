# Testbed FL – UAV relay AF – NOMA – HE chọn lọc thích nghi (đồ án Phan Hồng Quí, FPT)

> **ĐỔI HƯỚNG (28/10/2026, theo người dùng):** PHƯƠNG ÁN CHÍNH là **`fixed10`** — FAS với tỷ lệ mã hóa CỐ ĐỊNH p = 10%,
> khóa xáo π_t CHỈ do 2 thiết bị giữ (bs cộng mù), so với `no_he` và `full_he`, có UAV (`uav_noma_pc`) và không UAV
> (`direct_noma`); kèm `fas10` (bs giữ khóa xáo, FAS gốc) để tách tác dụng của việc ai giữ khóa.
> **`adaptive` (chọn p theo SINR) là HƯỚNG PHÁT TRIỂN**, không thuộc kết quả chính: giữ nguyên code (đã chú thích
> `[HƯỚNG PHÁT TRIỂN – KHÔNG THUỘC KẾT QUẢ CHÍNH]`), chỉ chạy khi policy == 'adaptive'.
> Không đổi tham số CKKS, mô hình kênh, C, λ, T_max.
> FedAvg: mỗi iot nhân Δ (sau clipping, trước khi tách HE/phần rõ) với n_k/(n_1+n_2); bs chỉ cộng; iot: w ← w + tổng.

## 1. Mục tiêu
Dựng testbed giả lập mạng gồm 4 container Docker chạy "như thật" quy trình tích hợp:
2 thiết bị IoT (giới hạn như Raspberry Pi 4) → UAV relay AF đứng yên → trạm gốc (BS).
- Tầng vật lý (kênh UAV, AF, NOMA, SIC) được MÔ HÌNH HÓA: mỗi vòng tính tốc độ R_k rồi áp lên luồng dữ liệu.
- Tầng ứng dụng (FL, HE, DP, xáo trộn, giao thức) chạy THẬT giữa các container.
- Đường xuống (bs → iot) giả định lý tưởng: đi qua uav nhưng KHÔNG bóp tốc độ.
Kết quả testbed dùng để KIỂM CHỨNG bộ mô phỏng `reference_sim/`.

## 2. Thiết kế bảo vệ và mô hình mối đe dọa
Mỗi vòng, cập nhật Δ của mỗi thiết bị được giới hạn độ lớn (clipping) rồi chia làm hai phần theo tỷ lệ p:
- p% đầu: mã hóa HE đa khóa (xMK-CKKS). Không bên nào tự giải mã được; cần phần giải mã δ của cả hai thiết bị.
- Phần còn lại: thêm nhiễu DP rồi XÁO VỊ TRÍ bằng hoán vị bí mật π_t mà CHỈ HAI THIẾT BỊ biết.
  bs "cộng mù" theo vị trí (đúng vì hai thiết bị dùng cùng π_t: π(a) + π(b) = π(a + b)); thiết bị giải xáo tổng.
- ~~p chọn theo SINR từng vòng (đóng góp chính).~~ → phương án chính: p = 10% cố định (`fixed10`); chọn p theo SINR
  (`adaptive`) là hướng phát triển.
Đối tượng tấn công:
- bs và uav: trung thực nhưng tò mò (thụ động). Kẻ nghe lén: thấy mọi gói.
- bs KHÔNG được biết: khóa bí mật HE, π_t, phần HE của từng thiết bị, vị trí các giá trị trong phần rõ.
- bs ĐƯỢC biết (chấp nhận, ghi là giới hạn): TỔNG phần HE sau khi ghép δ (như mọi hệ tổng hợp an toàn), nên suy ra được
  phần mô hình ứng với vùng HE; không suy ra được phần còn lại (đã xáo).
- Với 2 thiết bị, không chống được thiết bị này suy ra cập nhật của thiết bị kia (biết tổng, trừ phần của mình) — ghi là giới hạn.
Khác FAS: trong FAS máy chủ giữ khóa xáo và giải xáo để tổng hợp; ở đây máy chủ không có khóa xáo.

## 3. Tham chiếu
`reference_sim/` là bộ mô phỏng đã chạy và kiểm chứng (một tiến trình, không phải testbed). Dùng lại logic, KHÔNG chép nguyên đường dẫn
(các file dùng `ROOT/../weights`, `ROOT/cache` — phải sửa theo cấu trúc testbed):
- `common.py`: mô hình fire9 + classifier (202.314 tham số), `priority_order` (classifier trước, rồi fire9 từ lớp cuối),
  lớp `MKCKKS` (OpenFHE multiparty CKKS: ring 8192, 4096 slot, depth 0, scale 50, firstMod 60, HEStd_128_classic).
- `channel.py`: Al-Hourani (đô thị a=9,61, b=0,16, η_LoS=1 dB, η_NLoS=20 dB), Rician K=5 dB, AF độ lợi biến đổi, NOMA + SIC;
  các chế độ `uav_noma_pc` (chính), `uav_noma`, `uav_oma`, `direct_noma`; `los_mode='random'` để kiểm chứng.
- `integrate.py`: `round_time`, `choose_p`.
- `fl.py`: huấn luyện, clipping (`--clip`), nhiễu (`--lam`), xáo vị trí với π_t chung (`--perm 1`), FedAvg — đúng thiết kế này.
- `data_prep.py` (cần ảnh CIFAR-10 từ `git clone https://github.com/YoongiKim/CIFAR-10-images`),
  `convert_weights.py` (cần `squeezenet1.1-7.onnx` từ ONNX Model Zoo, link trong file).
- `bench_costs.py`, `run_pi4.sh`, `cost_table.csv`, `cost_fixed.json`: bảng chi phí đo trên máy cloud (phải đo lại).
- `leak2.py`: tấn công nhận dạng ảnh lên từng lớp bảo vệ (kết quả mô phỏng của thiết kế này).

## 4. Kiến trúc
```
testbed/
  docker-compose.yml
  config.yaml
  secrets/ perm_seed      # chỉ mount vào iot1, iot2
  common/  model.py protocol.py crypto_setup.py
  iot/     Dockerfile client.py train.py protect.py he.py evaluate.py
  uav/     Dockerfile relay.py channel.py shaper.py
  bs/      Dockerfile server.py policy.py aggregate.py logger.py
  data/    client1/ client2/ test/         # đặc trưng thân SqueezeNet đã trích (fp16); client_k chỉ mount vào iot_k, test vào iot1
  results/ capture/
```
- iot1, iot2: CÙNG image, khác `DEVICE_ID`; `cpus`, `mem_limit: 4g`, `memswap_limit: 4g`; chỉ nối mạng `chang1`.
- uav: nối `chang1` và `chang2`. bs: chỉ nối `chang2`. Mọi tin nhắn iot ↔ bs đi qua uav.
- Tổng `cpus` của các container phải ≤ số lõi máy; nếu thiếu thì giảm `cpus` của iot và ghi hệ số hiệu chỉnh.

### 4.1 Relay và bóp tốc độ
Mặc định: `relay.py` là proxy TCP tầng ứng dụng. Mỗi iot nối tới uav qua MỘT CỔNG RIÊNG; uav mở kết nối tương ứng tới bs.
Chiều lên của từng thiết bị bị giới hạn bằng token bucket đúng R_k byte/s (cập nhật mỗi vòng); chiều xuống không giới hạn.
(Không lọc `tc` theo IP iot ở phía bs được vì proxy đổi IP nguồn.) Tùy chọn: định tuyến IP thật + `tc htb` trên uav.
Kiểm tra: gửi 5 MB ở 1 Mbit/s mất ≈ 40 s (sai lệch ≤ 5%), làm với cả 2 thiết bị cùng lúc.

### 4.2 Giả định NOMA
Trong một vòng, R_1, R_2 cố định (quasi-static), áp cho cả UPLOAD và PARTIAL_DEC. Thiết bị gửi xong trước không làm thiết bị kia
nhanh lên (đơn giản hóa, bảo thủ) — ghi trong báo cáo.

## 5. Thiết lập ban đầu (một lần, `crypto_setup.py`)
1. bs tạo `CryptoContext` (tham số như `MKCKKS`), serialize, gửi cho 2 iot. Hai iot PHẢI dùng đúng context này.
2. HE đa bên: iot1 `KeyGen()` → gửi pk1 (qua bs) → iot2 `MultipartyKeyGen(pk1)` → pk chung → phát cho cả hai.
   Khóa bí mật không rời thiết bị. iot1 là "lead" khi giải mã (`MultipartyDecryptLead`), iot2 dùng `MultipartyDecryptMain`.
3. Hạt giống hoán vị (CÀI SẴN): một chuỗi bí mật ngẫu nhiên 32 byte K nằm trong file `secrets/perm_seed`, chỉ mount vào
   container iot1 và iot2 (docker-compose `secrets`/volume read-only). KHÔNG mount vào bs, uav; KHÔNG đặt trong config.yaml.
   Giả định: K được phân phối an toàn khi triển khai thiết bị (trao đổi khóa X25519 để ở hướng phát triển).
4. Mô hình khởi tạo w0 (SqueezeNet ImageNet + classifier mới, seed trong config) được 2 iot TỰ DỰNG giống hệt nhau.
   Từ đó mô hình chỉ nằm ở 2 thiết bị.

## 6. Trình tự một vòng (common/protocol.py)
Định dạng tin nhắn: khung nhị phân có độ dài ở đầu (length-prefixed), thân mã hóa msgpack; ciphertext dùng `Serialize` của OpenFHE.
1. uav: sinh kênh vòng t (seed từ config + t) → γ1, γ2 → `CHANNEL_REPORT` {t, γ1, γ2} cho bs → đặt R_k cho chiều lên.
2. bs: chọn p theo policy → `ROUND_START` {t, p} xuống iot. (KHÔNG gửi mô hình: bs không có mô hình đầy đủ.)
3. iot: huấn luyện 1 epoch trên đặc trưng đã lưu → Δ → clipping về C.
   Chia theo `priority_order`; cả hai thiết bị dùng cùng p nên cùng tập chỉ số:
   - m = round(p·M) phần tử đầu: mã hóa CKKS bằng pk chung (ceil(m/4096) ciphertext).
   - phần còn lại: cộng nhiễu N(0, (λ·C/√M)²) bằng nguồn ngẫu nhiên RIÊNG của từng thiết bị (KHÔNG lấy từ K);
     rồi xáo vị trí bằng π_t, seed = HKDF(K, "perm" ‖ t). Hai thiết bị BẮT BUỘC ra cùng π_t.
   → `UPLOAD` {ciphertexts, vector đã xáo} lên (bị bóp tốc độ).
4. bs: chờ đủ 2 UPLOAD → cộng đồng cấu phần HE; cộng theo vị trí hai vector đã xáo → `CSUM` {tổng ciphertext} xuống.
5. iot: giải mã từng phần → `PARTIAL_DEC` {δ_i} lên (bị bóp tốc độ).
6. bs: ghép δ_1, δ_2 → tổng phần HE → `AGGREGATE` {tổng phần HE, tổng đã xáo} xuống.
7. iot: giải xáo tổng bằng π_t⁻¹ (một lần), ghép với tổng HE, w ← w + tổng/2. Hai iot giữ mô hình giống hệt nhau.
   iot1 đánh giá trên tập test, gửi `METRICS` {t, accuracy, loss} cho bs. Mỗi container gửi mốc thời gian pha của mình.
Không bỏ thiết bị nào khỏi vòng: kênh xấu thì vòng chậm, ghi là trễ hạn (giải mã đa bên cần cả hai thiết bị).

## 7. Chế độ kênh (config.yaml `channel.mode`)
- `uav_noma_pc` (chính): AF + NOMA có điều khiển công suất, R_k = B·½·log2(1+γ_k).
- `direct_noma` (không UAV): 3GPP TR 38.901 UMa NLoS + bóng râm 6 dB, Rayleigh, NOMA + SIC, R_k = B·log2(1+γ_k).
  uav chỉ chuyển tiếp như router.
- `uav_noma`, `uav_oma`: so sánh phụ. `los_mode: mean | random`.
Không UAV rất chậm (≈ 0,09–0,25 Mbit/s; HE toàn phần ~26 phút/vòng): chạy 3 vòng hoặc chỉ đo thời gian truyền.

## 8. Policies (bs/policy.py)
| policy | p | clipping | nhiễu | xáo vị trí | ai biết khóa xáo |
|---|---|---|---|---|---|
| `no_he` | 0 | không | không | không | — |
| `full_he` | 1 | không | — | — | — |
| **`fixed10` (PHƯƠNG ÁN CHÍNH)** | 0,1 cố định, không đọc SINR | không | λ·RMS(phần rõ) | có | CHỈ 2 thiết bị (bs cộng mù) |
| `fas10`, `fas50` | 0,1 / 0,5 cố định | không | λ·RMS(phần rõ) | có | thiết bị VÀ bs (bs giải xáo từng thiết bị rồi cộng, như FAS gốc) |
| `adaptive` (HƯỚNG PHÁT TRIỂN) | `choose_p` theo γ, p_min, T_max | có (C) | λ·C/√M (×n_k/n) | có | CHỈ 2 thiết bị (bs cộng mù) |
Riêng fas: chạy với profile docker-compose `fas`, mount thêm `secrets/perm_seed` vào bs để bs có K (như FAS gốc: máy chủ giữ khóa
xáo, giải xáo rồi cộng; FAS gốc xáo bit, ở đây dùng xáo vị trí làm xấp xỉ — tương đương về việc bs đọc được phần rõ đã nhiễu).
Vẫn để mô hình ở thiết bị cho giao thức thống nhất. Các policy khác: bs KHÔNG có K.
`choose_p` dùng bảng chi phí ĐO TRÊN MÁY NÀY (bước 3) và γ từ CHANNEL_REPORT.

## 9. Đo và ghi log (bs/logger.py, mỗi vòng một bản ghi JSON)
- Kênh: γ1, γ2, R_1, R_2. Quyết định: p, số ciphertext.
- Byte THẬT của từng tin nhắn (ROUND_START, UPLOAD, CSUM, PARTIAL_DEC, AGGREGATE, METRICS).
- Thời gian thật từng pha (time.monotonic): huấn luyện, mã hóa, truyền UPLOAD, chờ đồng bộ, tổng hợp, CSUM,
  giải mã từng phần, truyền PARTIAL_DEC, ghép, AGGREGATE, cập nhật; thời gian vòng đầu-cuối tại bs.
- Thông lượng thực tế = byte / thời gian truyền, so với R_k. Trễ hạn: vòng > T_max.
- Báo thời gian thô và thời gian hiệu chỉnh Pi 4 (pha tính toán × `fidelity`, mặc định 1,4).
- `capture/`: lưu nguyên UPLOAD bs nhận được để phân tích tấn công offline. Tấn công "gradient 1 ảnh" trong reference_sim là
  trường hợp xấu nhất, không tái hiện được với giao thức thật (cập nhật là 1 epoch); testbed kiểm tra bs/uav không biết vị trí
  phần rõ và chạy suy luận thành viên trên đúng dữ liệu bs nhận.

## 10. Thứ tự dựng (mỗi bước có test đạt mới sang bước sau)
1. Khung rỗng: protocol + server/client/relay gửi mảng ngẫu nhiên, đủ 7 bước của mục 6. Test: 3 vòng không lỗi, đủ log.
2. FL thật chưa bảo vệ (`no_he`). Test: IID, 15 vòng, độ chính xác ≈ 80% (±1,5 điểm so với reference_sim).
3. Giới hạn Pi 4 + đo lại bảng chi phí trên máy này. Test: thời gian huấn luyện 1 epoch ổn định (±10%).
4. Kênh + bóp tốc độ, cả `uav_noma_pc` và `direct_noma`. Test: mục 4.1.
5. HE đa khóa giữa các container (mục 5). Test: sai số tổng sau giải mã < 1e-6; bs không có khóa bí mật.
6. Clipping + nhiễu + xáo vị trí. Test: λ = 0 thì mô hình sau giải xáo trùng FedAvg thường (sai khác < 1e-9);
   hai iot sinh cùng π_t; π_t đổi giữa các vòng; container bs, uav không đọc được `secrets/perm_seed`.
7. Policies (mục 8). Test: lịch p của `adaptive` trùng `integrate.choose_p` với cùng γ và bảng chi phí.
8. Thí nghiệm: {uav_noma_pc, direct_noma} × {no_he, full_he, fas10, fas50, adaptive}; IID và non-IID (Dirichlet α=0,5);
   seed 0 trước, thêm seed 1–2 nếu đủ thời gian. Xuất bảng và hình như reference_sim.

## 11. Quy ước
- Image `python:3.12-slim` (wheel OpenFHE chỉ có cho cp312, Linux x86_64). PyTorch CPU:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Thân SqueezeNet đóng băng: iot huấn luyện trên đặc trưng đã trích. Chi phí trích một lần đo riêng.
- Mọi tham số trong `config.yaml`. Comment và log bằng tiếng Việt.
- Không dùng xáo bit. Không để bs giữ khóa xáo (trừ policy fas10/fas50 để mô phỏng FAS).
- KHÔNG tự đổi thiết kế (thứ tự ưu tiên, tham số CKKS, mô hình kênh, cách tính nhiễu, mô hình đe dọa) khi chưa hỏi người dùng.
- Mỗi bước: viết test nhỏ trong `tests/`, chạy và báo kết quả trước khi sang bước sau.
