"""
TSMM Config UI (Dash)

Provides editors for:
- config/config.yaml
- config/trading_agent.yaml
- config/sweep_definition.yaml
And Mode B interrupt/resume controls.

Run:
    py -3.11 apps/ui.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from datetime import datetime
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.live_data import update_fx_master_table_file, update_fx_master_table_db, resolve_tiingo_token_candidates
from utils.runtime_scope import resolve_runtime_file

dash_mod = __import__("dash", fromlist=["Dash", "dcc", "html", "Input", "Output", "State", "ctx"])
Dash = dash_mod.Dash
dcc = dash_mod.dcc
html = dash_mod.html
Input = dash_mod.Input
Output = dash_mod.Output
State = dash_mod.State


BASE_DIR = str(PROJECT_ROOT)
CFG_MAIN = os.path.join(BASE_DIR, "config", "config.yaml")
CFG_TRADING = os.environ.get("TRADING_CONFIG_PATH", os.path.join(BASE_DIR, "config", "trading_agent.yaml"))
CFG_SWEEP = os.path.join(BASE_DIR, "config", "sweep_definition.yaml")
CFG_LLM = os.path.join(BASE_DIR, "config", "llm_providers.yaml")


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


def _parse_token_env_input(token_env: str) -> list[str]:
    raw = str(token_env or "").strip()
    if not raw:
        return ["TIINGO_API_TOKEN"]

    parts = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
    if not parts:
        return ["TIINGO_API_TOKEN"]
    return list(dict.fromkeys(parts))


def _token_env_input_value(dash_cfg: dict, refresh_cfg: dict) -> str:
    envs_cfg = (dash_cfg or {}).get("tiingo_token_envs")
    envs: list[str]
    if isinstance(envs_cfg, str):
        envs = _parse_token_env_input(envs_cfg)
    elif isinstance(envs_cfg, (list, tuple, set)):
        envs = [str(x).strip() for x in envs_cfg if str(x).strip()]
    else:
        envs = [str((dash_cfg or {}).get("tiingo_token_env", (refresh_cfg or {}).get("token_env", "TIINGO_API_TOKEN"))).strip() or "TIINGO_API_TOKEN"]
    return ", ".join(list(dict.fromkeys(envs)))


app = Dash(__name__, assets_folder=str(PROJECT_ROOT / "assets"))
app.title = "TSMM UI"

_cfg_main = _load_yaml(CFG_MAIN)
_cfg_trading = _load_yaml(CFG_TRADING)
_data_refresh_cfg = (_cfg_main.get("data_refresh") or {})
_dash_cfg = (_cfg_trading.get("dashboard") or {})
INTERRUPT_FLAG = str(
    resolve_runtime_file(
        configured_path=((_cfg_trading.get("mode_b") or {}).get("interrupt_flag_path")),
        fallback_name="mode_b_interrupt.flag",
        trading_cfg=_cfg_trading,
        base_dir=BASE_DIR,
    )
)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("TSMM Configuration UI", className="tsmm-title"),
                html.Div("Professional config controls for forecasting, trading, and refresh operations.", className="tsmm-subtitle"),
            ],
            style={"marginBottom": "12px"},
        ),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Main Config",
                    children=[
                        html.Div(
                            [
                                dcc.Textarea(id="main-text", value=_read_text(CFG_MAIN), className="tsmm-input", style={"width": "100%", "height": "520px"}),
                                html.Div(
                                    [
                                        html.Button("Save Main Config", id="save-main", n_clicks=0, className="tsmm-btn tsmm-btn-primary"),
                                        html.Div(id="save-main-status", className="tsmm-status-info"),
                                    ],
                                    className="tsmm-toolbar",
                                    style={"marginTop": "10px"},
                                ),
                            ],
                            className="tsmm-card",
                            style={"padding": "12px"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Trading Agent Config",
                    children=[
                        html.Div(
                            [
                                dcc.Textarea(id="trading-text", value=_read_text(CFG_TRADING), className="tsmm-input", style={"width": "100%", "height": "520px"}),
                                html.Div(
                                    [
                                        html.Button("Save Trading Config", id="save-trading", n_clicks=0, className="tsmm-btn tsmm-btn-primary"),
                                        html.Div(id="save-trading-status", className="tsmm-status-info"),
                                    ],
                                    className="tsmm-toolbar",
                                    style={"marginTop": "10px"},
                                ),
                            ],
                            className="tsmm-card",
                            style={"padding": "12px"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Sweep Definition",
                    children=[
                        html.Div(
                            [
                                dcc.Textarea(id="sweep-text", value=_read_text(CFG_SWEEP), className="tsmm-input", style={"width": "100%", "height": "520px"}),
                                html.Div(
                                    [
                                        html.Button("Save Sweep Config", id="save-sweep", n_clicks=0, className="tsmm-btn tsmm-btn-primary"),
                                        html.Div(id="save-sweep-status", className="tsmm-status-info"),
                                    ],
                                    className="tsmm-toolbar",
                                    style={"marginTop": "10px"},
                                ),
                            ],
                            className="tsmm-card",
                            style={"padding": "12px"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="LLM Providers",
                    children=[
                        html.Div(
                            [
                                html.P("Configure API providers and secrets for Agent A/B LLM assistance."),
                                dcc.Textarea(id="llm-text", value=_read_text(CFG_LLM), className="tsmm-input", style={"width": "100%", "height": "520px"}),
                                html.Div(
                                    [
                                        html.Button("Save LLM Providers Config", id="save-llm", n_clicks=0, className="tsmm-btn tsmm-btn-primary"),
                                        html.Div(id="save-llm-status", className="tsmm-status-info"),
                                    ],
                                    className="tsmm-toolbar",
                                    style={"marginTop": "10px"},
                                ),
                            ],
                            className="tsmm-card",
                            style={"padding": "12px"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Data Refresh",
                    children=[
                        html.Div(
                            [
                                html.H4("Master Table Update"),
                                html.P("Works with CSV and SQLite. Use SQLite path for SQL-first flow.", className="tsmm-status-info"),
                                html.Div([html.Span("Master table path", className="tsmm-label"), html.Span("*", className="tsmm-mark")]),
                                dcc.Input(
                                    id="master-path",
                                    type="text",
                                    value=_dash_cfg.get("master_table_path") or _data_refresh_cfg.get("raw_data_path", ""),
                                    className="tsmm-input",
                                    style={"width": "100%"},
                                ),
                                html.Div([html.Span("Tiingo symbol", className="tsmm-label"), html.Span("*", className="tsmm-mark")]),
                                dcc.Input(id="tiingo-symbol", type="text", value=_data_refresh_cfg.get("symbol", "xauusd"), className="tsmm-input", style={"width": "100%"}),
                                html.Div([html.Span("Tiingo rate", className="tsmm-label"), html.Span("*", className="tsmm-mark")]),
                                dcc.Input(id="tiingo-rate", type="text", value=_data_refresh_cfg.get("rate", "1min"), className="tsmm-input", style={"width": "100%"}),
                                html.Div([html.Span("Token env var(s)", className="tsmm-label"), html.Span("*", className="tsmm-mark")]),
                                dcc.Input(
                                    id="token-env",
                                    type="text",
                                    value=_token_env_input_value(_dash_cfg, _data_refresh_cfg),
                                    className="tsmm-input",
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    [
                                        html.Button("Update Master Now", id="update-master-btn", n_clicks=0, className="tsmm-btn tsmm-btn-primary"),
                                        html.Div(id="update-master-ui-status", className="tsmm-status-info"),
                                    ],
                                    className="tsmm-toolbar",
                                    style={"marginTop": "10px"},
                                ),
                            ],
                            className="tsmm-card",
                            style={"padding": "12px"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Mode B Control",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Button("Interrupt Mode B", id="stop-b", n_clicks=0, className="tsmm-btn tsmm-btn-secondary"),
                                        html.Button("Resume Mode B", id="resume-b", n_clicks=0, className="tsmm-btn tsmm-btn-primary"),
                                    ],
                                    className="tsmm-toolbar",
                                ),
                                html.Div(id="mode-b-status", className="tsmm-status-info", style={"marginTop": "10px"}),
                                html.P("Open dashboard separately with: py -3.11 apps/dashboard.py"),
                            ],
                            className="tsmm-card",
                            style={"padding": "12px"},
                        ),
                    ],
                ),
            ]
        ),
    ],
    className="tsmm-shell",
    style={"padding": "16px"},
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


@app.callback(Output("save-llm-status", "children"), Input("save-llm", "n_clicks"), State("llm-text", "value"), prevent_initial_call=True)
def save_llm(_, val):
    return _write_text(CFG_LLM, val or "")


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
    token_envs = _parse_token_env_input(token_env)
    token_key = token_envs[0]
    token = os.environ.get(token_key, "")
    token_candidates = resolve_tiingo_token_candidates(
        token_env=token_key,
        token_envs=token_envs,
        token=token,
    )
    if not token_candidates:
        return f"Missing token in configured env vars: {', '.join(token_envs)}"

    rotation_state_path = str((_dash_cfg.get("tiingo_token_rotation_state_path") or "")).strip() or None

    p = str(master_path or "").strip()
    if p.lower().endswith(".db") or p.lower().endswith(".sqlite"):
        result = update_fx_master_table_db(
            db_path=p,
            rate=(rate or "1min").strip(),
            symbol=(symbol or "xauusd").strip().lower(),
            token=token,
            token_env=token_key,
            token_envs=token_envs,
            token_rotation_state_path=rotation_state_path,
        )
    else:
        result = update_fx_master_table_file(
            master_table_path=p,
            rate=(rate or "1min").strip(),
            symbol=(symbol or "xauusd").strip().lower(),
            token=token,
            token_env=token_key,
            token_envs=token_envs,
            token_rotation_state_path=rotation_state_path,
        )
    if not bool(result.get("updated", False)):
        return f"Update failed: {result.get('error', 'unknown error')}"
    used_env = str(result.get("used_token_env") or token_key)
    rotated = bool(result.get("token_rotated", False))
    return (
        f"Master updated: +{int(result.get('new_rows', 0))} rows | "
        f"latest={result.get('latest_date', 'N/A')} | token_env={used_env} | rotated={'yes' if rotated else 'no'}"
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8051)
