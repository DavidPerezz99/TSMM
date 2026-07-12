"""
Agent Validation Dashboard

Run:
    py -3.11 apps/validation_dashboard.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
from datetime import datetime, timedelta

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

dash_mod = __import__("dash", fromlist=["Dash", "dcc", "html", "Input", "Output", "State"])
Dash = dash_mod.Dash
dcc = dash_mod.dcc
html = dash_mod.html
Input = dash_mod.Input
Output = dash_mod.Output
State = dash_mod.State

from utils.agent_validation import run_agent_validation_days, run_agent_backtest_advanced

BASE_DIR = str(PROJECT_ROOT)
TRADING_CFG_PATH = os.environ.get("TRADING_CONFIG_PATH", os.path.join(BASE_DIR, "config", "trading_agent.yaml"))
MAIN_CFG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
VALIDATION_ROOT = os.path.join(BASE_DIR, "reports", "agent_validation")
STATUS_PATH = os.path.join(VALIDATION_ROOT, "status.json")
MAX_LOG_LINES = 300
STALE_RUN_MINUTES = 20


def _load_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _write_status(payload: dict):
    os.makedirs(VALIDATION_ROOT, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_status() -> dict:
    if not os.path.exists(STATUS_PATH):
        return {}
    try:
        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _append_status_log(message: str):
    payload = _read_status()
    logs = payload.get("logs") or []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs.append(f"[{ts}] {message}")
    payload["logs"] = logs[-MAX_LOG_LINES:]
    payload["heartbeat_at"] = ts
    _write_status(payload)


def _parse_dt(v: str | None) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.strptime(str(v), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _status_is_stale(st: dict, stale_minutes: int = STALE_RUN_MINUTES) -> bool:
    if not bool(st.get("running", False)):
        return False
    now = datetime.now()
    hb = _parse_dt(st.get("heartbeat_at"))
    started = _parse_dt(st.get("started_at"))
    ref = hb or started
    if ref is None:
        return True
    return (now - ref) > timedelta(minutes=max(int(stale_minutes), 1))


def _mark_status_stale(st: dict, reason: str = "Stale validation run state detected and reset."):
    payload = dict(st or {})
    logs = payload.get("logs") or []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs.append(f"[{ts}] {reason}")
    payload.update(
        {
            "running": False,
            "done": False,
            "phase": "failed",
            "error": reason,
            "logs": logs[-MAX_LOG_LINES:],
            "heartbeat_at": ts,
            "finished_at": ts,
        }
    )
    _write_status(payload)


def _run_validation_async(master_path: str, n_days: int, mode: str, start_date: str, end_date: str):
    trading_cfg = _load_yaml(TRADING_CFG_PATH)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(VALIDATION_ROOT, f"run_{ts}")

    _write_status({
        "running": True,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "heartbeat_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
        "current": 0,
        "total": max(int(n_days), 1),
        "day": "",
        "mode": mode,
        "start_date": start_date,
        "end_date": end_date,
        "phase": "initializing",
        "logs": [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation queued.",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Mode={str(mode).lower()} | n_days={max(int(n_days), 1)} | start={start_date or '-'} | end={end_date or '-'}",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Run directory: {run_dir}",
        ],
        "done": False,
    })
    _append_status_log("Validation worker started.")

    def _progress(curr: int, total: int, day: str):
        payload = _read_status()
        logs = payload.get("logs") or []
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing day {curr}/{total}: {day}")
        _write_status({
            "running": True,
            "started_at": payload.get("started_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "heartbeat_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir,
            "current": int(curr),
            "total": int(total),
            "day": day,
            "mode": mode,
            "start_date": start_date,
            "end_date": end_date,
            "phase": "running",
            "logs": logs[-MAX_LOG_LINES:],
            "done": False,
        })

    try:
        _append_status_log("Validation run started.")
        if str(mode).lower() == "advanced":
            _append_status_log("Using advanced technical backtest runner.")
            result = run_agent_backtest_advanced(
                master_table_path=master_path,
                output_dir=run_dir,
                trading_cfg=trading_cfg,
                n_days=max(int(n_days), 1),
                start_date=(start_date or None),
                end_date=(end_date or None),
                config_root=os.path.join(BASE_DIR, "config"),
                progress_cb=_progress,
            )
        else:
            _append_status_log("Using simple daily replay runner.")
            result = run_agent_validation_days(
                master_table_path=master_path,
                n_days=max(int(n_days), 1),
                output_dir=run_dir,
                trading_cfg=trading_cfg,
                progress_cb=_progress,
            )

        payload = _read_status()
        logs = payload.get("logs") or []
        if bool(result.get("ok", False)):
            logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation completed successfully.")
        else:
            logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation finished with error: {result.get('error')}")
        payload.update(
            {
                "running": False,
                "done": bool(result.get("ok", False)),
                "error": result.get("error"),
                "summary_path": result.get("summary_path"),
                "phase": "completed" if bool(result.get("ok", False)) else "failed",
                "logs": logs[-MAX_LOG_LINES:],
                "heartbeat_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        _write_status(payload)
    except Exception as e:
        payload = _read_status()
        logs = payload.get("logs") or []
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unhandled validation error: {e}")
        payload.update(
            {
                "running": False,
                "done": False,
                "error": f"Unhandled error: {e}",
                "phase": "failed",
                "logs": logs[-MAX_LOG_LINES:],
                "heartbeat_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        _write_status(payload)


cfg_main = _load_yaml(MAIN_CFG_PATH)
cfg_trading = _load_yaml(TRADING_CFG_PATH)
dashboard_cfg = (cfg_trading.get("dashboard") or {})
refresh_cfg = (cfg_main.get("data_refresh") or {})

app = Dash(__name__, assets_folder=str(PROJECT_ROOT / "assets"))
app.title = "TSMM Agent Validation"

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("TSMM Agent Validation Dashboard", className="tsmm-title"),
                html.P("Replay past trading sessions using minute master table data.", className="tsmm-subtitle"),
            ],
            style={"marginBottom": "10px"},
        ),
        html.Div(
            [
                html.Div([html.Span("Master table path (CSV or SQLite)", className="tsmm-label"), html.Span("*", className="tsmm-mark")]),
                dcc.Input(
                    id="master-path",
                    type="text",
                    value=dashboard_cfg.get("master_table_path") or refresh_cfg.get("raw_data_path", ""),
                    className="tsmm-input",
                    style={"width": "100%"},
                ),
                html.Div([html.Span("Past trading days to test", className="tsmm-label"), html.Span("*", className="tsmm-mark")]),
                dcc.Input(id="n-days", type="number", value=10, min=1, step=1, className="tsmm-input"),
                html.Div([html.Span("Mode", className="tsmm-label"), html.Span("*", className="tsmm-mark")]),
                dcc.Dropdown(
                    id="validation-mode",
                    options=[
                        {"label": "Simple daily replay", "value": "simple"},
                        {"label": "Advanced technical backtest", "value": "advanced"},
                    ],
                    value="advanced",
                    clearable=False,
                ),
                html.Div(html.Span("Optional explicit date range (YYYY-MM-DD or timestamp)", className="tsmm-label")),
                html.Div(
                    [
                        dcc.Input(id="start-date", type="text", placeholder="Start date/time", className="tsmm-input", style={"width": "100%"}),
                        dcc.Input(id="end-date", type="text", placeholder="End date/time", className="tsmm-input", style={"width": "100%"}),
                    ],
                    className="tsmm-grid-2",
                ),
                html.Div(
                    [
                        html.Button("Run Validation", id="run-validation", n_clicks=0, className="tsmm-btn tsmm-btn-primary"),
                        html.Div(id="run-status", className="tsmm-status-info"),
                    ],
                    className="tsmm-toolbar",
                    style={"marginTop": "8px"},
                ),
                html.Progress(id="progress", value=0, max=1, style={"width": "100%", "marginTop": "10px", "height": "14px"}),
                html.Div(id="progress-label", className="tsmm-status-info", style={"marginTop": "6px"}),
            ],
            className="tsmm-card",
            style={"padding": "12px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Overall Performance"),
                        html.Pre(id="overall-box", className="tsmm-pretty-pre", style={"maxHeight": "300px", "overflowY": "auto"}),
                    ],
                    className="tsmm-card",
                    style={"padding": "12px"},
                ),
                html.Div(
                    [
                        html.H3("Sessions"),
                        html.Pre(id="sessions-box", className="tsmm-pretty-pre", style={"maxHeight": "320px", "overflowY": "auto"}),
                    ],
                    className="tsmm-card",
                    style={"padding": "12px"},
                ),
            ],
            className="tsmm-grid-2",
            style={"marginTop": "12px"},
        ),
        html.Div(
            [
                html.H3("Validation Log"),
                html.Pre(
                    id="validation-log-box",
                    className="tsmm-pretty-pre",
                    style={"maxHeight": "260px", "overflowY": "auto", "whiteSpace": "pre-wrap"},
                ),
            ],
            className="tsmm-card",
            style={"padding": "12px", "marginTop": "12px"},
        ),
        dcc.Interval(id="poll", interval=1500, n_intervals=0),
    ],
    className="tsmm-shell",
    style={"padding": "16px"},
)


@app.callback(
    Output("run-status", "children"),
    Input("run-validation", "n_clicks"),
    State("master-path", "value"),
    State("n-days", "value"),
    State("validation-mode", "value"),
    State("start-date", "value"),
    State("end-date", "value"),
    prevent_initial_call=True,
)
def start_run(_, master_path, n_days, mode, start_date, end_date):
    st = _read_status()
    if _status_is_stale(st):
        _mark_status_stale(st, reason="Stale validation run was reset before starting a new run.")
        st = _read_status()
    if st.get("running"):
        return "Validation already running."

    t = threading.Thread(
        target=_run_validation_async,
        args=(master_path or "", int(n_days or 1), str(mode or "simple"), str(start_date or "").strip(), str(end_date or "").strip()),
        daemon=True,
    )
    t.start()
    return f"Validation started in {str(mode or 'simple').lower()} mode."


@app.callback(
    Output("progress", "value"),
    Output("progress", "max"),
    Output("progress-label", "children"),
    Output("overall-box", "children"),
    Output("sessions-box", "children"),
    Output("validation-log-box", "children"),
    Output("run-status", "children", allow_duplicate=True),
    Input("poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_status(_):
    st = _read_status()
    if _status_is_stale(st):
        _mark_status_stale(st)
        st = _read_status()
    current = int(st.get("current", 0) or 0)
    total = int(st.get("total", 1) or 1)
    phase = str(st.get("phase") or "idle")
    logs = st.get("logs") or []

    status_msg = (
        f"Running [{st.get('mode', 'simple')}]: {current}/{total} day(s). Current day: {st.get('day', '')}"
        if st.get("running")
        else ("Completed" if st.get("done") else (st.get("error") or "Idle"))
    )

    if st.get("running"):
        pct = (current / max(total, 1)) * 100.0
        progress_msg = f"{pct:.1f}% complete | phase={phase} | day {current}/{total}"
    elif st.get("done"):
        progress_msg = f"100.0% complete | phase={phase}"
    elif st.get("error"):
        progress_msg = f"Stopped | phase={phase}"
    else:
        progress_msg = "Idle"

    overall_txt = "No results yet."
    sessions_txt = "No sessions yet."
    logs_txt = "\n".join(logs[-150:]) if logs else "No logs yet."

    summary_path = st.get("summary_path")
    if summary_path and os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            overall_txt = json.dumps(summary.get("overall", {}), indent=2)
            sessions_txt = json.dumps(summary.get("sessions", []), indent=2)
        except Exception as e:
            overall_txt = f"Error reading summary: {e}"
            logs_txt += f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Summary read error: {e}"

    return current, max(total, 1), progress_msg, overall_txt, sessions_txt, logs_txt, status_msg


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8052)
