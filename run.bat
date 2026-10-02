@echo off
rem Curvelens launcher: first run sets up a private .venv and installs the
rem libraries (needs internet), later runs start the program straight away.
rem Extra arguments are passed on to app.py, e.g.  run.bat samples\INR18650P28A-V1-80093.pdf
setlocal
cd /d "%~dp0"
title Curvelens

set "VENV=%~dp0.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "VPYW=%VENV%\Scripts\pythonw.exe"

if exist "%VPY%" if exist "%VENV%\.deps_ok" goto :run

rem ---- 1. find Python 3.11 or newer ----
set "BASEPY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 set "BASEPY=py -3"
if defined BASEPY goto :havepy
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 set "BASEPY=python"
if defined BASEPY goto :havepy

echo Python 3.11 veya daha yenisi bulunamadi.
where winget >nul 2>nul
if errorlevel 1 goto :nopython
echo Python winget ile kuruluyor, lutfen bekleyin...
winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :nopython
echo.
echo Python kuruldu. Bu pencereyi kapatip run.bat dosyasini tekrar calistirin.
pause
exit /b 0

:nopython
echo.
echo Lutfen https://www.python.org/downloads/ adresinden Python 3.11+ kurun
echo ("Add python.exe to PATH" kutusunu isaretleyin), sonra run.bat dosyasini tekrar calistirin.
pause
exit /b 1

rem ---- 2. virtual environment + libraries (first run only) ----
:havepy
if not exist "%VPY%" (
    echo Sanal ortam olusturuluyor...
    %BASEPY% -m venv "%VENV%"
    if errorlevel 1 goto :fail
)
echo Kutuphaneler indiriliyor (ilk calistirmada bir kez, birkac dakika surebilir)...
"%VPY%" -m pip install -r "%~dp0requirements-run.txt"
if errorlevel 1 goto :fail
echo ok> "%VENV%\.deps_ok"

rem ---- 3. start ----
:run
if "%~1"=="" (
    start "" "%VPYW%" "%~dp0app.py"
    exit /b 0
)
"%VPY%" "%~dp0app.py" %*
if errorlevel 1 pause
exit /b %errorlevel%

:fail
echo.
echo Kurulum basarisiz oldu. Internet baglantinizi kontrol edip tekrar deneyin.
pause
exit /b 1
