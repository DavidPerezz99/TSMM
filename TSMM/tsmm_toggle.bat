@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  TSMM Toggle — Start or stop all TSMM services
REM  Usage:  tsmm_toggle.bat on
REM          tsmm_toggle.bat off
REM ============================================================

set TSMM_DIR=C:\Users\USUARIO\Documents\TSMM\TSMM
cd /d "%TSMM_DIR%"

if /i "%1"=="on" goto :on
if /i "%1"=="off" goto :off

echo Usage: %0 {on^|off}
echo   on  - Start all TSMM services (listeners + parity enforcer)
echo   off - Stop all TSMM services and clean up state
exit /b 1

:: ============================================================
:: OFF — Kill everything
:: ============================================================
:off
echo [TSMM] ========== SHUTTING DOWN ==========

REM --- Kill by window title (most reliable for started windows) ---
echo [TSMM] Killing FTMO listener...
taskkill //F //FI "WINDOWTITLE eq TSMM-FTMO-Listener*" //T >nul 2>&1

echo [TSMM] Killing endpoint service...
taskkill //F //FI "WINDOWTITLE eq TSMM-Endpoint*" //T >nul 2>&1

echo [TSMM] Killing dashboard...
taskkill //F //FI "WINDOWTITLE eq TSMM-Dashboard*" //T >nul 2>&1

REM --- Kill any Python running from TSMM directory (detached processes) ---
REM Uses PowerShell because wmic may not be available
echo [TSMM] Checking for detached TSMM processes...
powershell.exe -NoProfile -Command ^
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe' AND CommandLine like '%%TSMM%%'\" ^| Select-Object -ExpandProperty ProcessId ^| ForEach-Object { Write-Host $_ }" ^
    > "%TEMP%\tsmm_pids.txt" 2>nul
for /f "tokens=*" %%p in ('type "%TEMP%\tsmm_pids.txt" 2^>nul') do (
    if not "%%p"=="" (
        echo [TSMM] Killing detached TSMM Python PID %%p...
        taskkill //F //PID %%p //T >nul 2>&1
    )
)
del /f /q "%TEMP%\tsmm_pids.txt" 2>nul

REM --- Kill MT5 terminals (they get auto-launched by mt5.initialize) ---
echo [TSMM] Killing MT5 terminals...
taskkill //F //IM "terminal64.exe" //T >nul 2>&1

REM --- Clean up stale runtime files ---
echo [TSMM] Cleaning up runtime state files...
del /f /q reports\runtime\local_signal_endpoint_service.pid    2>nul
del /f /q reports\runtime\trading_job.pid                      2>nul
del /f /q reports\runtime\deployment_pipeline_stop.flag        2>nul
del /f /q reports\runtime\agent_channel_enabled.flag           2>nul

echo [TSMM] ========== ALL SERVICES STOPPED ==========
echo.
echo To restart later, run:  tsmm_toggle.bat on
exit /b 0


:: ============================================================
:: ON — Start everything
:: ============================================================
:on
echo [TSMM] ========== STARTING UP ==========

REM --- Read FTMO password from registry ---
set MT5_FTMO_LOGIN=531158622
for /f "tokens=2*" %%A in (
    'reg query "HKCU\Environment" /v MT5_FTMO_PASSWORD 2^>nul ^| findstr MT5_FTMO_PASSWORD'
) do set MT5_FTMO_PASSWORD=%%B

REM --- 1) FTMO Telegram Listener ---
echo [TSMM 1/1] Starting FTMO Telegram listener...
start "TSMM-FTMO-Listener" "C:\Users\USUARIO\AppData\Local\Programs\Python\Python311\python.exe" scripts\telegram_command_listener.py --trading-config config\trading_agent.yaml
if %ERRORLEVEL% neq 0 (
    echo [TSMM] WARNING: FTMO listener may not have started.
)

echo.
echo [TSMM] ========== SERVICES LAUNCHED ==========
echo.
echo   - FTMO listener (TSMM-FTMO-Listener)
echo.
echo Send a Telegram command to start trading:
echo   /tsmm trading start --submission-mode programmed
echo.
echo Or start the endpoint manually:
echo   .venv\Scripts\python.exe scripts\local_signal_endpoint_service.py
echo.
exit /b 0
