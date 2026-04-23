"""
Agent Validation Dashboard

Run:
    py -3.11 validation_dashboard.py
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime

import yaml

dash_mod = __import__("dash", fromlist=["Dash", "dcc", "html", "Input", "Output", "State"])
Dash = dash_mod.Dash
dcc = dash_mod.dcc
html = dash_mod.html
Input = dash_mod.Input
Output = dash_mod.Output
State = dash_mod.State

from utils.agent_validation import run_agent_validation_days

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADING_CFG_PATH = os.path.join(BASE_DIR, "config", "trading_agent.yaml")
MAIN_CFG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
VALIDATION_ROOT = os.path.join(BASE_DIR, "reports", "agent_validation")
STATUS_PATH = os.path.join(VALIDATION_ROOT, "status.json")


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


def _run_validation_async(master_path: str, n_days: int):
    trading_cfg = _load_yaml(TRADING_CFG_PATH)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(VALIDATION_ROOT, f"run_{ts}")

    _write_status({
        "running": True,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
        "current": 0,
        "total": max(int(n_days), 1),
        "day": "",
        "done": False,
    })

    def _progress(curr: int, total: int, day: str):
        _write_status({
            "running": True,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir,
            "current": int(curr),
            "total": int(total),
            "day": day,
            "done": False,
        })

    result = run_agent_validation_days(
        master_table_path=master_path,
        n_days=max(int(n_days), 1),
        output_dir=run_dir,
        trading_cfg=trading_cfg,
        progress_cb=_progress,
    )

    payload = _read_status()
    payload.update(
        {
            "running": False,
            "done": bool(result.get("ok", False)),
            "error": result.get("error"),
            "summary_path": result.get("summary_path"),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    _write_status(payload)


cfg_main = _load_yaml(MAIN_CFG_PATH)
cfg_trading = _load_yaml(TRADING_CFG_PATH)
dashboard_cfg = (cfg_trading.get("dashboard") or {})
refresh_cfg = (cfg_main.get("data_refresh") or {})

app = Dash(__name__)
app.title = "TSMM Agent Validation"

app.layout = html.Div(
    [
        html.H2("TSMM Agent Validation Dashboard"),
        html.P("Replay past trading sessions using minute master table data."),
        html.Div(
            [
                html.Label("Master table CSV path"),
                dcc.Input(
                    id="master-path",
                    type="text",
                    value=dashboard_cfg.get("master_table_path") or refresh_cfg.get("raw_data_path", ""),
                    style={"width": "100%"},
                ),
                html.Br(),
                html.Label("Past trading days to test"),
                dcc.Input(id="n-days", type="number", value=10, min=1, step=1),
                html.Br(),
                html.Button("Run Validation", id="run-validation", n_clicks=0),
                html.Div(id="run-status", style={"marginTop": "8px"}),
                html.Progress(id="progress", value=0, max=1, style={"width": "100%", "marginTop": "8px"}),
            ],
            style={"padding": "12px", "border": "1px solid #ddd", "borderRadius": "8px"},
        ),
        html.H3("Overall Performance"),
        html.Pre(id="overall-box", style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "6px"}),
        html.H3("Sessions"),
        html.Pre(id="sessions-box", style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "6px", "maxHeight": "320px", "overflowY": "auto"}),
        dcc.Interval(id="poll", interval=1500, n_intervals=0),
    ],
    style={"padding": "12px", "fontFamily": "Segoe UI, Arial"},
)


@app.callback(
    Output("run-status", "children"),
    Input("run-validation", "n_clicks"),
    State("master-path", "value"),
    State("n-days", "value"),
    prevent_initial_call=True,
)
def start_run(_, master_path, n_days):
    st = _read_status()
    if st.get("running"):
        return "Validation already running."

    t = threading.Thread(target=_run_validation_async, args=(master_path or "", int(n_days or 1)), daemon=True)
    t.start()
    return "Validation started."


@app.callback(
    Output("progress", "value"),
    Output("progress", "max"),
    Output("overall-box", "children"),
    Output("sessions-box", "children"),
    Output("run-status", "children", allow_duplicate=True),
    Input("poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_status(_):
    st = _read_status()
    current = int(st.get("current", 0) or 0)
    total = int(st.get("total", 1) or 1)

    status_msg = (
        f"Running: {current}/{total} day(s). Current day: {st.get('day', '')}"
        if st.get("running")
        else ("Completed" if st.get("done") else (st.get("error") or "Idle"))
    )

    overall_txt = "No results yet."
    sessions_txt = "No sessions yet."

    summary_path = st.get("summary_path")
    if summary_path and os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            overall_txt = json.dumps(summary.get("overall", {}), indent=2)
            sessions_txt = json.dumps(summary.get("sessions", []), indent=2)
        except Exception as e:
            overall_txt = f"Error reading summary: {e}"

    return current, max(total, 1), overall_txt, sessions_txt, status_msg


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8052)
