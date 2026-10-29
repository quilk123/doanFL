# Ghi chú: chạy tiếp

## Hiện tại (28/10/2026): bước 8 theo hướng FAS — `sh tools/run_step8_fas.sh`

Đồ án đổi hướng: phương án chính là **`fixed10`** (FAS p = 10%, khóa xáo chỉ ở thiết bị); `adaptive` là hướng phát triển.
Đã xong và commit: chú thích phần thích ứng, FedAvg tại thiết bị (n_k/n), policy `fixed10` mặc định.

`tools/run_step8_fas.sh` (~4 giờ; cấu hình đã đủ vòng thì bỏ qua — chạy lại được nếu bị dừng):
1. kết quả bước 8 cũ -> `experiments/KET_QUA_CU/step8_v1_adaptive/` (một lần)
2. IID, 15 vòng, seed 0: có UAV `no_he fixed10 full_he fas10`; không UAV (nén kênh) `no_he fixed10 full_he`
3. so reference_sim, bảng bước 8, tấn công (góc nhìn bs / kẻ nghe lén), `tools/fas_report.py`
   (`experiments/fas/e20_he_vs_fas.png`, `so_voi_mo_phong.csv`), hình MATLAB, chạy lại toàn bộ test.

Nếu bị dừng giữa chừng: cắm sạc, mở Docker Desktop, Git Bash trong `testbed`: `sh tools/run_step8_fas.sh`
— hoặc nhắn Claude **"chạy tiếp"**.

## Không còn cần: chạy lại bước 8 non-IID (`tools/redo_noniid.sh`)

Việc dừng ngày 27/10 (non-IID với T_train riêng từng thiết bị). Phương án mới chỉ chạy IID nên bỏ; kết quả non-IID cũ ở
`experiments/KET_QUA_CU/step8_noniid_T_chung/` và `experiments/KET_QUA_CU/step8_v1_adaptive/uav_noma_pc_noniid_no_he/`.
