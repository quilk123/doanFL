@echo off
rem Đánh giá tấn công (trạm gốc / kẻ nghe lén) — PowerShell/cmd: .\attack.bat [tham số như tools/attack.sh]
rem   .\attack.bat                         -> mọi cấu hình bước 8 (kết quả chính, experiments\attack)
rem   .\attack.bat fixed10 fixed20 fixed30 -> các tỉ lệ đã chạy ở bước 8 (experiments\attack\rieng_<thời điểm>)
rem   .\attack.bat --demo                  -> lần demo gần nhất (experiments\demo\<thời điểm>\attack)
chcp 65001 >nul
cd /d "%~dp0"
"C:\Program Files\Git\bin\bash.exe" tools/attack.sh %*
