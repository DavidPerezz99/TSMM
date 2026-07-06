@echo off
REM Start TSMM Telegram listener (FTMO-only trading account)

cd /d C:\Users\USUARIO\Documents\TSMM\TSMM
set MT5_FTMO_LOGIN=531158622
REM FTMO password was set via setx - read it
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v MT5_FTMO_PASSWORD 2^>nul ^| findstr MT5_FTMO_PASSWORD') do set MT5_FTMO_PASSWORD=%%B

echo Starting TSMM FTMO Telegram listener
start "TSMM-FTMO-Listener" "C:\Users\USUARIO\AppData\Local\Programs\Python\Python311\python.exe" scripts\telegram_command_listener.py --trading-config config\trading_agent.yaml
