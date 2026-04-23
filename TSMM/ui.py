"""
TSMM Config UI (Dash)

Provides editors for:
- config/config.yaml
- config/trading_agent.yaml
- config/sweep_definition.yaml
And Mode B interrupt/resume controls.

Run:
    py -3.11 ui.py
"""

from __future__ import annotations

import os
from datetime import datetime
import yaml

from utils.live_data import update_fx_master_table_file

dash_mod = __import__("dash", fromlist=["Dash", "dcc", "html", "Input", "Output", "State", "ctx"])
Dash = dash_mod.Dash
dcc = dash_mod.dcc
html = dash_mod.html
Input = dash_mod.Input
Output = dash_mod.Output
State = dash_mod.State


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_MAIN = os.path.join(BASE_DIR, "config", "config.yaml")
CFG_TRADING = os.path.join(BASE_DIR, "config", "trading_agent.yaml")
CFG_SWEEP = os.path.join(BASE_DIR, "config", "sweep_definition.yaml")
INTERRUPT_FLAG = os.path.join(BASE_DIR, "reports", "runtime", "mode_b_interrupt.flag")


def _load_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"# Error reading {path}: {e}"


def _write_text(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Saved {path} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"Error writing {path}: {e}"


app = Dash(__name__)
app.title = "TSMM UI"

_cfg_main = _load_yaml(CFG_MAIN)
_cfg_trading = _load_yaml(CFG_TRADING)
_data_refresh_cfg = (_cfg_main.get("data_refresh") or {})
_dash_cfg = (_cfg_trading.get("dashboard") or {})

app.layout = html.Div(
    [
        html.H2("TSMM Configuration UI"),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Main Config",
                    children=[
                        dcc.Textarea(id="main-text", value=_read_text(CFG_MAIN), style={"width": "100%", "height": "520px"}),
                        html.Button("Save Main Config", id="save-main", n_clicks=0),
                        html.Div(id="save-main-status"),
                    ],
                ),
                dcc.Tab(
                    label="Trading Agent Config",
                    children=[
                        dcc.Textarea(id="trading-text", value=_read_text(CFG_TRADING), style={"width": "100%", "height": "520px"}),
                        html.Button("Save Trading Config", id="save-trading", n_clicks=0),
                        html.Div(id="save-trading-status"),
                    ],
                ),
                dcc.Tab(
                    label="Sweep Definition",
                    children=[
                        dcc.Textarea(id="sweep-text", value=_read_text(CFG_SWEEP), style={"width": "100%", "height": "520px"}),
                        html.Button("Save Sweep Config", id="save-sweep", n_clicks=0),
                        html.Div(id="save-sweep-status"),
                    ],
                ),
                dcc.Tab(
                    label="Data Refresh",
                    children=[
                        html.H4("Master Table Update"),
                        html.Label("Master table CSV path"),
                        dcc.Input(
                            id="master-path",
                            type="text",
                            value=_dash_cfg.get("master_table_path") or _data_refresh_cfg.get("raw_data_path", ""),
                            style={"width": "100%"},
                        ),
                        html.Br(),
                        html.Label("Tiingo symbol"),
                        dcc.Input(id="tiingo-symbol", type="text", value=_data_refresh_cfg.get("symbol", "xauusd"), style={"width": "100%"}),
                        html.Br(),
                        html.Label("Tiingo rate"),
                        dcc.Input(id="tiingo-rate", type="text", value=_data_refresh_cfg.get("rate", "1min"), style={"width": "100%"}),
                        html.Br(),
                        html.Label("Token env var"),
                        dcc.Input(
                            id="token-env",
                            type="text",
                            value=_dash_cfg.get("tiingo_token_env", _data_refresh_cfg.get("token_env", "TIINGO_API_TOKEN")),
                            style={"width": "100%"},
                        ),
                        html.Br(),
                        html.Button("Update Master Now", id="update-master-btn", n_clicks=0),
                        html.Div(id="update-master-ui-status", style={"marginTop": "10px"}),
                    ],
                ),
                dcc.Tab(
                    label="Mode B Control",
                    children=[
                        html.Button("Interrupt Mode B", id="stop-b", n_clicks=0),
                        html.Button("Resume Mode B", id="resume-b", n_clicks=0, style={"marginLeft": "8px"}),
                        html.Div(id="mode-b-status", style={"marginTop": "10px"}),
                        html.P("Open dashboard separately with: py -3.11 dashboard.py"),
                    ],
                ),
            ]
        ),
    ],
    style={"padding": "12px"},
)


@app.callback(Output("save-main-status", "children"), Input("save-main", "n_clicks"), State("main-text", "value"), prevent_initial_call=True)
def save_main(_, val):
    return _write_text(CFG_MAIN, val or "")


@app.callback(Output("save-trading-status", "children"), Input("save-trading", "n_clicks"), State("trading-text", "value"), prevent_initial_call=True)
def save_trading(_, val):
    return _write_text(CFG_TRADING, val or "")


@app.callback(Output("save-sweep-status", "children"), Input("save-sweep", "n_clicks"), State("sweep-text", "value"), prevent_initial_call=True)
def save_sweep(_, val):
    return _write_text(CFG_SWEEP, val or "")


@app.callback(Output("mode-b-status", "children"), Input("stop-b", "n_clicks"), Input("resume-b", "n_clicks"))
def control_mode_b(_, __):
    os.makedirs(os.path.dirname(INTERRUPT_FLAG), exist_ok=True)
    if dash_mod.ctx.triggered_id == "stop-b":
        with open(INTERRUPT_FLAG, "w", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    elif dash_mod.ctx.triggered_id == "resume-b":
        if os.path.exists(INTERRUPT_FLAG):
            os.remove(INTERRUPT_FLAG)

    return f"Mode B state: {'INTERRUPTED' if os.path.exists(INTERRUPT_FLAG) else 'RUNNING/AVAILABLE'}"


@app.callback(
    Output("update-master-ui-status", "children"),
    Input("update-master-btn", "n_clicks"),
    State("master-path", "value"),
    State("tiingo-symbol", "value"),
    State("tiingo-rate", "value"),
    State("token-env", "value"),
    prevent_initial_call=True,
)
def update_master_ui(_, master_path, symbol, rate, token_env):
    token_key = (token_env or "TIINGO_API_TOKEN").strip()
    token = os.environ.get(token_key, "")
    if not token:
        return f"Missing token in env var: {token_key}"

    result = update_fx_master_table_file(
        master_table_path=master_path or "",
        rate=(rate or "1min").strip(),
        symbol=(symbol or "xauusd").strip().lower(),
        token=token,
    )
    if not bool(result.get("updated", False)):
        return f"Update failed: {result.get('error', 'unknown error')}"
    return f"Master updated: +{int(result.get('new_rows', 0))} rows | latest={result.get('latest_date', 'N/A')}"


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8051)
