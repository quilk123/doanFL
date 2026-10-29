@echo off
rem Vẽ hình kiểu MATLAB (Octave trong Docker): bấm đúp, hoặc trong PowerShell/cmd: .\plots.bat
rem   .\plots.bat                                   -> hình bước 8: experiments\matlab\fig
rem   .\plots.bat experiments/demo/<thời điểm>       -> hình một lần demo: experiments\demo\<thời điểm>\fig
rem   .\plots.bat --last 2                           -> GỘP 2 lần demo gần nhất: experiments\demo\gop_<thời điểm>\fig
rem   .\plots.bat experiments/demo/<A> experiments/demo/<B>   -> GỘP các lần demo chỉ định
chcp 65001 >nul
cd /d "%~dp0"
set "SH=C:\Program Files\Git\bin\sh.exe"
if not exist "%SH%" (
  echo Không tìm thấy Git Bash: %SH%
  pause
  exit /b 1
)
"%SH%" tools/run_plots.sh %*
rem có tham số: tools/run_plots.sh tự mở thư mục hình
if "%~1"=="" start "" explorer "experiments\matlab\fig"
echo.
pause
