@echo off
rem Xem số liệu chi tiết (tốc độ có/không UAV, p, thời gian, trễ hạn, độ chính xác) trong PowerShell/cmd:
rem   .\show.bat                                           -> tóm tắt mọi cấu hình bước 8
rem   .\show.bat --rounds                                  -> thêm bảng từng vòng
rem   .\show.bat --rounds experiments/step8/uav_noma_pc_iid_adaptive   -> từng vòng của 1 cấu hình
chcp 65001 >nul
cd /d "%~dp0"
"C:\Program Files\Git\bin\bash.exe" tools/show.sh %*
