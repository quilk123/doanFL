@echo off
rem MỘT CỬA cho mọi việc thường dùng: bấm đúp file này rồi chọn số. (Các .bat/.sh khác là công cụ chi tiết, không cần nhớ.)
chcp 65001 >nul
cd /d "%~dp0"
set "BASH=C:\Program Files\Git\bin\bash.exe"
if not exist "%BASH%" (
  echo Không tìm thấy Git Bash: %BASH% — cài Git for Windows rồi chạy lại.
  pause
  exit /b 1
)
docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop CHƯA CHẠY — mở Docker Desktop, đợi biểu tượng xanh, rồi bấm đúp lại file này.
  pause
  exit /b 1
)

:menu
cls
echo ================================================================
echo   TESTBED FL - UAV - NOMA - FAS p = 10%% (fixed10)   (thư mục: testbed)
echo ================================================================
echo   CHẠY THẬT (4 container, xong tự in bảng + vẽ hình + mở hình)
echo     1. Demo nhanh: fixed10 (FAS p = 10%%), 3 vòng, có UAV      (~5 phút)
echo     2. So sánh no_he / fixed10 / full_he, 3 vòng, CÓ UAV       (~15 phút)
echo     K. So sánh no_he / fixed10 / full_he, 3 vòng, KHÔNG UAV    (thời gian kênh thật; có thể mất nhiều giờ)
echo     8. TÙY CHỈNH: số vòng FL, số epoch client, policy, có/không UAV (không nén kênh)
echo.
echo   XEM KẾT QUẢ CÓ SẴN (không chạy lại gì)
echo     3. Mở hình kết quả bước 8
echo     4. Xem bảng số liệu bước 8
echo     5. Mở thư mục các lần demo
echo     9. VẼ GỘP các lần demo gần nhất vào cùng bộ hình (vd. có UAV + không UAV)
echo     A. ĐÁNH GIÁ TẤN CÔNG (trạm gốc / kẻ nghe lén): lần demo gần nhất hoặc các tỉ lệ đã chạy (~5–15 phút)
echo.
echo   KHÁC
echo     6. Vẽ lại hình bước 8 (chỉ khi số liệu bước 8 thay đổi; demo tự vẽ hình riêng)
echo     7. Dừng mọi container đang chạy
echo     0. Thoát
echo ================================================================
choice /c 1234567890KA /n /m "Chọn số (hoặc K, A): "
set "C=%errorlevel%"
if "%C%"=="8" goto custom
if "%C%"=="9" goto gop
if "%C%"=="10" exit /b 0
if "%C%"=="12" goto attack
if "%C%"=="1" "%BASH%" -c "MODE=uav_noma_pc sh tools/demo.sh 3 fixed10"
if "%C%"=="2" "%BASH%" -c "MODE=uav_noma_pc sh tools/demo.sh 3 no_he fixed10 full_he"
if "%C%"=="11" "%BASH%" -c "MODE=direct_noma sh tools/demo.sh 3 no_he fixed10 full_he"
if "%C%"=="3" start "" explorer "experiments\matlab\fig"
if "%C%"=="4" "%BASH%" tools/show.sh
if "%C%"=="5" if exist experiments\demo (start "" explorer "experiments\demo") else (echo Chưa có lần demo nào.)
if "%C%"=="6" "%BASH%" tools/run_plots.sh && start "" explorer "experiments\matlab\fig"
if "%C%"=="7" docker compose down
echo.
pause
goto menu

:attack
echo.
echo   Enter = lần demo gần nhất ^| gõ tên các tỉ lệ đã chạy ở bước 8, vd. fixed10 fixed20 ^| all = mọi cấu hình bước 8
set "AT="
set /p AT="Đánh giá tấn công cho: "
if "%AT%"=="" ("%BASH%" tools/attack.sh --demo) else if /i "%AT%"=="all" ("%BASH%" tools/attack.sh) else ("%BASH%" tools/attack.sh %AT%)
echo.
pause
goto menu

:gop
echo.
set "NG=2"
set /p NG="Gộp bao nhiêu lần demo GẦN NHẤT? [2]: "
"%BASH%" tools/run_plots.sh --last %NG%
echo.
pause
goto menu

:custom
echo.
echo ---------------- TÙY CHỈNH (Enter = giữ giá trị trong ngoặc) ----------------
set "R=3"
set /p R="Số vòng FL (bs gom và cập nhật mô hình bao nhiêu lần) [3]: "
set "E=1"
set /p E="Số epoch mỗi client tự học trước khi gửi về bs [1]: "
set "PO=fixed10"
echo   Policy: no_he, fixed10 (chính), fixed20, fixed30 … fixed90 (FAS tỉ lệ khác, khóa ở thiết bị), full_he, fas10, fas50, adaptive
set /p PO="Policy, cách nhau dấu cách, vd. fixed10 fixed20 fixed30 [fixed10]: "
set "U=c"
set /p U="Có UAV không? (c = có UAV, k = KHÔNG UAV: thiết bị gửi thẳng tới bs) [c]: "
set "MODE=uav_noma_pc"
if /i "%U%"=="k" set "MODE=direct_noma"
if /i "%U%"=="k" echo   (không UAV: kênh rất chậm; chạy theo thời gian kênh thật nên có thể mất nhiều giờ)
set "TS=off"
echo.
echo  => %R% vòng FL, %E% epoch/vòng ở client, kênh %MODE%, policy: %PO%, không nén kênh
if not "%E%"=="1" echo  => Đổi số epoch: lần đầu sẽ tự đo lại thời gian huấn luyện (~%E%x4 phút), các lần sau dùng lại.
echo.
"%BASH%" -c "EPOCHS=%E% MODE=%MODE% TIME_SCALE=%TS% sh tools/demo.sh %R% %PO%"
echo.
pause
goto menu

