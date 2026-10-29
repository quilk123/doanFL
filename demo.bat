@echo off
rem Demo testbed: bấm đúp file này. Gọi tools/demo.sh qua Git Bash (chạy thật trên 4 container).
chcp 65001 >nul
cd /d "%~dp0"
set "BASH=C:\Program Files\Git\bin\bash.exe"
if not exist "%BASH%" (
  echo Không tìm thấy Git Bash: %BASH%
  pause
  exit /b 1
)
rem Có tham số => chạy thẳng, không hiện menu:  .\demo.bat <số vòng> <policy ...>   (vd. .\demo.bat 3 no_he fixed10 full_he)
rem Demo chạy không nén thời gian kênh.
if not "%~1"=="" (
  "%BASH%" tools/demo.sh %*
  exit /b
)
echo ==============================================================
echo   DEMO FL - UAV relay AF - NOMA - FAS p = 10%% (fixed10)
echo ==============================================================
echo   1. Demo fixed10 (FAS p = 10%%), 3 vòng, có UAV, thời gian thật   (~5 phút)
echo   2. So sánh no_he / fixed10 / full_he, 3 vòng, CÓ UAV            (~15 phút)
echo   3. So sánh no_he / fixed10 / full_he, 3 vòng, KHÔNG UAV (thời gian kênh thật; có thể mất nhiều giờ)
echo   4. Mở kết quả: hình bước 8, so sánh reference_sim, thư mục demo
echo   5. Xem số liệu chi tiết bước 8 (tốc độ có/không UAV, p, thời gian, trễ hạn)
echo ==============================================================
choice /c 12345 /n /m "Chọn 1-5: "
set "C=%errorlevel%"
if "%C%"=="1" "%BASH%" -c "MODE=uav_noma_pc sh tools/demo.sh 3 fixed10"
if "%C%"=="2" "%BASH%" -c "MODE=uav_noma_pc sh tools/demo.sh 3 no_he fixed10 full_he"
if "%C%"=="3" "%BASH%" -c "MODE=direct_noma sh tools/demo.sh 3 no_he fixed10 full_he"
if "%C%"=="4" (
  for %%f in (experiments\step8\acc_vs_round.png experiments\step8\acc_vs_time.png experiments\step8\p_schedule.png experiments\ref_compare\compare.png) do if exist "%%f" start "" "%%f"
  if exist experiments\demo start "" explorer "experiments\demo"
)
if "%C%"=="5" "%BASH%" -c "sh tools/show.sh --rounds"
echo.
pause
