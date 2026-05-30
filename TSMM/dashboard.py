"""
TSMM Live Dashboard

Run:
    py -3.11 dashboard.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from utils.live_data import read_csv_tail, update_fx_master_table_file, update_fx_master_table_db
from utils.market_db import query_ohlc
from utils.agent_channel import read_channel_messages, set_channel_enabled, is_channel_enabled
from utils.runtime_scope import resolve_runtime_dir, resolve_runtime_file

dash_mod = __import__("dash", fromlist=["Dash", "dcc", "html", "Input", "Output", "State", "ctx"])
plotly_go = __import__("plotly.graph_objects", fromlist=["Figure", "Candlestick", "Scatter", "Heatmap", "Surface"])

Dash = dash_mod.Dash
dcc = dash_mod.dcc
html = dash_mod.html
Input = dash_mod.Input
Output = dash_mod.Output
State = dash_mod.State
go = plotly_go


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADING_CFG_PATH = os.environ.get("TRADING_CONFIG_PATH", os.path.join(BASE_DIR, "config", "trading_agent.yaml"))


def _load_trading_cfg() -> dict:
    try:
        import yaml

        with open(TRADING_CFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _runtime_root() -> str:
    return str(resolve_runtime_dir(base_dir=BASE_DIR, trading_cfg=_load_trading_cfg()))


def _default_state_path() -> str:
    return os.path.join(_runtime_root(), "agent_state_latest.json")


def _default_trading_job_state_path() -> str:
    return os.path.join(_runtime_root(), "trading_job_state.json")


def _default_startup_status_path() -> str:
    return os.path.join(_runtime_root(), "startup_sync_status.json")


def _load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_startup_sync_status(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_data(path: str, latest_records: int = 50000) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])

    # SQL-first option: if a SQLite DB path is provided, read from table queries.
    if str(path).lower().endswith(".db") or str(path).lower().endswith(".sqlite"):
        return query_ohlc(path, timeframe_minutes=1, latest_records=max(int(latest_records or 1), 1))

    df = read_csv_tail(path, max(int(latest_records or 1), 1))
    if "DATE" not in df.columns:
        return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE")
    return df


def _plot_fx_data(data: pd.DataFrame, plot_type: str = "candlestick", interval_minutes: int = 1, template: str = "plotly_white") -> go.Figure:
    if data.empty:
        fig = go.Figure()
        fig.update_layout(title="No data", template=template)
        return fig

    filtered_data = data.copy()
    if plot_type == "candlestick":
        res = (
            filtered_data.set_index("DATE")
            .resample(f"{interval_minutes}T")
            .agg({"OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last"})
            .dropna()
            .reset_index()
        )
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=res["DATE"],
                    open=res["OPEN"],
                    high=res["HIGH"],
                    low=res["LOW"],
                    close=res["CLOSE"],
                    name="Candlesticks",
                )
            ]
        )
        fig.update_layout(title=f"{interval_minutes}-Minute Candlestick Chart")
        fig.update_layout(xaxis_rangeslider_visible=True)
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=filtered_data["DATE"],
                    y=filtered_data["CLOSE"],
                    mode="lines",
                    name="Close Price",
                )
            ]
        )
        fig.update_layout(title="Curve Plot")

    fig.update_layout(template=template, hovermode="x unified", xaxis_title="Date", yaxis_title="Price")
    return fig


def _build_heatmaps(df: pd.DataFrame, lookback_minutes: int = 240, n_paths: int = 800, bins: int = 60, template: str = "plotly_white"):
    if df.empty:
        return go.Figure(), go.Figure()

    end_ts = df["DATE"].max()
    start_ts = end_ts - timedelta(minutes=int(lookback_minutes))
    w = df[df["DATE"] >= start_ts].copy()
    if len(w) < 20:
        return go.Figure(), go.Figure()

    closes = w["CLOSE"].astype(float).values
    rets = np.diff(closes)
    if len(rets) < 5:
        return go.Figure(), go.Figure()

    horizon = min(60, max(20, int(len(rets) * 0.15)))
    mu = float(np.mean(rets))
    sigma = float(np.std(rets, ddof=1))
    sigma = max(sigma, 1e-6)
    anchor = float(closes[-1])

    shocks = np.random.normal(mu, sigma, size=(n_paths, horizon))
    paths = anchor + np.cumsum(shocks, axis=1)

    pmin, pmax = float(np.min(paths)), float(np.max(paths))
    if pmin == pmax:
        pmin -= 1.0
        pmax += 1.0
    edges = np.linspace(pmin, pmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    density = np.zeros((horizon, bins), dtype=float)
    for t in range(horizon):
        hist, _ = np.histogram(paths[:, t], bins=edges, density=True)
        density[t, :] = hist

    heat2d = go.Figure(
        data=[
            go.Heatmap(
                z=density.T,
                x=np.arange(1, horizon + 1),
                y=centers,
                colorscale="Viridis",
                colorbar=dict(title="Density"),
            )
        ]
    )
    heat2d.update_layout(title="2D Time-Price Probability Heatmap", xaxis_title="Horizon Step", yaxis_title="Price", template=template)

    X, Y = np.meshgrid(np.arange(1, horizon + 1), centers)
    heat3d = go.Figure(data=[go.Surface(x=X, y=Y, z=density.T, colorscale="Viridis")])
    heat3d.update_layout(
        title="3D Time-Price Density Surface",
        scene=dict(xaxis_title="Horizon Step", yaxis_title="Price", zaxis_title="Density"),
        template=template,
    )
    return heat2d, heat3d


def _pdf_fig(series: np.ndarray, title: str, x_title: str, template: str = "plotly_white") -> go.Figure:
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 5:
        return go.Figure(layout=dict(title=f"{title} (insufficient data)", template=template))

    hist = np.histogram(s, bins=50, density=True)
    centers = 0.5 * (hist[1][:-1] + hist[1][1:])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers, y=hist[0], name="Empirical PDF", opacity=0.55))
    fig.update_layout(title=title, template=template, xaxis_title=x_title, yaxis_title="Density")
    return fig


def _update_master_from_tiingo(master_table_path: str, symbol: str, rate: str, token_env: str) -> str:
    if not master_table_path:
        return "Missing Master table CSV path."

    token_key = (token_env or "TIINGO_API_TOKEN").strip()
    token = os.environ.get(token_key, "")
    if not token:
        return f"Missing Tiingo token in env var: {token_key}"

    try:
        if str(master_table_path).lower().endswith(".db") or str(master_table_path).lower().endswith(".sqlite"):
            result = update_fx_master_table_db(
                db_path=master_table_path,
                rate=(rate or "1min").strip(),
                symbol=(symbol or "xauusd").strip().lower(),
                token=token,
            )
        else:
            result = update_fx_master_table_file(
                master_table_path=master_table_path,
                rate=(rate or "1min").strip(),
                symbol=(symbol or "xauusd").strip().lower(),
                token=token,
            )
        if not bool(result.get("updated", False)):
            return f"Update failed: {result.get('error', 'unknown error')}"
        return (
            f"Master updated: +{int(result.get('new_rows', 0))} rows, "
            f"latest={result.get('latest_date', 'N/A')}"
        )
    except Exception as e:
        return f"Update failed: {e}"


cfg = _load_trading_cfg()
dashboard_cfg = (cfg.get("dashboard") or {})

app = Dash(__name__)
app.title = "TSMM Live Dashboard"

_controls_grid_style = {
    "display": "grid",
    "gridTemplateColumns": "repeat(4, minmax(260px, 1fr))",
    "gap": "10px",
    "marginBottom": "10px",
}
_control_cell_style = {
    "padding": "10px",
    "border": "1px solid #dbe4f3",
    "borderRadius": "10px",
    "background": "linear-gradient(180deg, #ffffff 0%, #f8fbff 100%)",
    "boxShadow": "0 2px 10px rgba(16, 24, 40, 0.06)",
}
_label_style = {"fontSize": "12px", "marginBottom": "4px", "color": "#495057"}

_theme = {
    "bg": "#f3f6fb",
    "surface": "#ffffff",
    "surface_alt": "#f8fbff",
    "text": "#1f2937",
    "muted": "#64748b",
    "primary": "#3454d1",
    "success": "#1f9d74",
    "border": "#dbe4f3",
    "shadow": "0 8px 24px rgba(15, 23, 42, 0.08)",
}

_theme_dark = {
    "bg": "#0b1220",
    "surface": "#111827",
    "surface_alt": "#1f2937",
    "text": "#e5e7eb",
    "muted": "#94a3b8",
    "primary": "#4f7cff",
    "success": "#34d399",
    "border": "#263246",
    "shadow": "0 8px 24px rgba(2, 6, 23, 0.55)",
}

_panel_card_style = {
    "backgroundColor": _theme["surface"],
    "border": f"1px solid {_theme['border']}",
    "borderRadius": "12px",
    "padding": "14px",
    "boxShadow": _theme["shadow"],
}

_chart_card_style = {
    "backgroundColor": _theme["surface"],
    "border": f"1px solid {_theme['border']}",
    "borderRadius": "12px",
    "padding": "8px",
    "boxShadow": "0 4px 14px rgba(15, 23, 42, 0.07)",
}

_button_style = {
    "backgroundColor": _theme["primary"],
    "color": "white",
    "border": "none",
    "padding": "9px 14px",
    "borderRadius": "8px",
    "fontWeight": "600",
    "cursor": "pointer",
    "transition": "transform 0.1s ease, filter 0.16s ease",
}

_button_secondary_style = {
    "backgroundColor": _theme["surface_alt"],
    "color": _theme["text"],
    "border": f"1px solid {_theme['border']}",
    "padding": "9px 14px",
    "borderRadius": "8px",
    "fontWeight": "600",
    "cursor": "pointer",
    "transition": "transform 0.1s ease, filter 0.16s ease",
}


def _mode_theme(mode: str) -> dict:
    return _theme_dark if str(mode or "light").lower() == "dark" else _theme


def _dynamic_styles(mode: str):
    t = _mode_theme(mode)
    root_style = {
        "padding": "14px",
        "backgroundColor": t["bg"],
        "minHeight": "100vh",
        "fontFamily": "Manrope, Space Grotesk, Segoe UI, sans-serif",
    }
    main_panel_style = {
        "padding": "14px",
        "backgroundColor": t["surface"],
        "border": f"1px solid {t['border']}",
        "borderRadius": "12px",
        "boxShadow": t["shadow"],
    }
    chart_card_style = {
        "backgroundColor": t["surface"],
        "border": f"1px solid {t['border']}",
        "borderRadius": "12px",
        "padding": "8px",
        "boxShadow": "0 4px 14px rgba(15, 23, 42, 0.07)",
    }
    signal_style = {
        "padding": "8px",
        "backgroundColor": t["surface_alt"],
        "border": f"1px solid {t['border']}",
        "borderRadius": "10px",
        "color": t["text"],
    }
    title_style = {"margin": "0", "color": t["text"], "fontWeight": "700"}
    subtitle_style = {"color": t["muted"], "marginTop": "4px"}
    return t, root_style, main_panel_style, chart_card_style, signal_style, title_style, subtitle_style


def _control_cell(label: str, component):
    return html.Div([html.Div(label, style=_label_style), component], style=_control_cell_style)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("TSMM Trading Dashboard", id="dashboard-title", style={"margin": "0", "color": _theme["text"], "fontWeight": "700"}),
                html.Div("Live forecasting, risk signals, and probability analytics", id="dashboard-subtitle", style={"color": _theme["muted"], "marginTop": "4px"}),
            ],
            style={"marginBottom": "14px"},
        ),
        html.Div(
            [
                html.Div(
                    "This is the minute-level master table to read/update (millions of rows supported).",
                    style={"fontSize": "12px", "color": _theme["muted"], "marginBottom": "8px"},
                ),
                html.Div(
                    [
                        _control_cell(
                            "Master table CSV path (required)",
                            dcc.Input(
                                id="master-table-path",
                                type="text",
                                value=dashboard_cfg.get("master_table_path", dashboard_cfg.get("raw_data_path", "")),
                                placeholder="Example: data/xauusd/xauusd_1m_master.csv",
                                style={"width": "100%"},
                            ),
                        ),
                        _control_cell(
                            "Latest records to load",
                            dcc.Input(
                                id="latest-records",
                                type="number",
                                value=int(dashboard_cfg.get("latest_records", 50000)),
                                min=500,
                                step=500,
                                style={"width": "100%"},
                            ),
                        ),
                        _control_cell(
                            "Tiingo symbol",
                            dcc.Input(id="tiingo-symbol", type="text", value=dashboard_cfg.get("tiingo_symbol", "xauusd"), style={"width": "100%"}),
                        ),
                        _control_cell(
                            "Tiingo rate",
                            dcc.Input(id="tiingo-rate", type="text", value=dashboard_cfg.get("tiingo_rate", "1min"), style={"width": "100%"}),
                        ),
                        _control_cell(
                            "Token env var",
                            dcc.Input(id="tiingo-token-env", type="text", value=dashboard_cfg.get("tiingo_token_env", "TIINGO_API_TOKEN"), style={"width": "100%"}),
                        ),
                        _control_cell(
                            "Auto update master",
                            dcc.Dropdown(
                                id="auto-update-enabled",
                                options=[
                                    {"label": "On", "value": "on"},
                                    {"label": "Off", "value": "off"},
                                ],
                                value="on" if bool(dashboard_cfg.get("auto_update_master", True)) else "off",
                                clearable=False,
                            ),
                        ),
                        _control_cell(
                            "Auto update every N ticks",
                            dcc.Input(
                                id="auto-update-every",
                                type="number",
                                value=int(dashboard_cfg.get("auto_update_every_ticks", 3)),
                                min=1,
                                step=1,
                                style={"width": "100%"},
                            ),
                        ),
                        _control_cell(
                            "Live lookback minutes",
                            dcc.Input(id="live-mins", type="number", value=dashboard_cfg.get("live_past_minutes", 1440), min=30, step=30, style={"width": "100%"}),
                        ),
                        _control_cell(
                            "Grouping minutes",
                            dcc.Input(id="group-mins", type="number", value=dashboard_cfg.get("group_minutes", 420), min=1, step=1, style={"width": "100%"}),
                        ),
                        _control_cell(
                            "Plot type",
                            dcc.Dropdown(id="plot-type", options=[{"label": "Candlestick", "value": "candlestick"}, {"label": "Curve", "value": "curve"}], value="candlestick"),
                        ),
                        _control_cell(
                            "Heatmap lookback minutes",
                            dcc.Input(id="heat-mins", type="number", value=dashboard_cfg.get("heatmap_minutes", 480), min=60, step=10, style={"width": "100%"}),
                        ),
                        _control_cell(
                            "PDF lookback minutes",
                            dcc.Input(id="pdf-mins", type="number", value=dashboard_cfg.get("pdf_minutes", 360), min=60, step=10, style={"width": "100%"}),
                        ),
                        _control_cell(
                            "State JSON path",
                            dcc.Input(id="state-path", type="text", value=dashboard_cfg.get("state_path", _default_state_path()), style={"width": "100%"}),
                        ),
                        _control_cell(
                            "Trading job state path",
                            dcc.Input(
                                id="trading-job-state-path",
                                type="text",
                                value=((cfg.get("trading_job") or {}).get("state_path") or _default_trading_job_state_path()),
                                style={"width": "100%"},
                            ),
                        ),
                        _control_cell(
                            "Theme",
                            dcc.Dropdown(
                                id="theme-mode",
                                options=[
                                    {"label": "Light", "value": "light"},
                                    {"label": "Dark", "value": "dark"},
                                ],
                                value=str(dashboard_cfg.get("theme", "light")).lower(),
                                clearable=False,
                            ),
                        ),
                    ],
                    style=_controls_grid_style,
                ),
                html.Div(
                    [
                        html.Button("Update Master from Tiingo", id="btn-update-master", n_clicks=0, style=_button_style),
                        html.Button("Refresh Dashboard", id="btn-refresh-dashboard", n_clicks=0, style={**_button_style, "marginLeft": "8px"}),
                        html.Button("Interrupt Mode B", id="btn-stop-mode-b", n_clicks=0, style={**_button_secondary_style, "marginLeft": "8px"}),
                        html.Button("Resume Mode B", id="btn-resume-mode-b", n_clicks=0, style={**_button_secondary_style, "marginLeft": "8px"}),
                        html.Button("Enable Agent Channel", id="btn-enable-channel", n_clicks=0, style={**_button_secondary_style, "marginLeft": "8px"}),
                        html.Button("Disable Agent Channel", id="btn-disable-channel", n_clicks=0, style={**_button_secondary_style, "marginLeft": "8px"}),
                    ],
                    style={"marginBottom": "8px"},
                ),
                html.Div(id="update-master-status", style={"marginTop": "4px", "fontSize": "12px", "color": _theme["success"]}),
                html.Div(id="auto-update-status", style={"marginTop": "4px", "fontSize": "12px", "color": _theme["muted"]}),
                html.Div(id="mode-b-control-status", style={"marginTop": "4px", "fontSize": "12px", "color": _theme["muted"]}),
                html.Div(id="agent-channel-status", style={"marginTop": "4px", "fontSize": "12px", "color": _theme["muted"]}),
                html.Div(id="agent-channel-panel", style={"marginTop": "8px", **_panel_card_style}),
                html.Div(id="startup-sync-panel", style={"marginTop": "8px", **_panel_card_style}),
                html.Div(
                    [
                        html.Div(id="agent-a-panel", style={**_panel_card_style}),
                        html.Div(id="agent-b-panel", style={**_panel_card_style}),
                    ],
                    style={"marginTop": "8px", "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
                ),
                html.Div(dcc.Graph(id="live-plot", config={"displaylogo": False}), id="live-plot-card", style={"marginTop": "10px", **_chart_card_style}),
                html.Div(
                    [
                        html.Div(dcc.Graph(id="heat2d", config={"displaylogo": False}), id="heat2d-card", style=_chart_card_style),
                        html.Div(dcc.Graph(id="heat3d", config={"displaylogo": False}), id="heat3d-card", style=_chart_card_style),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))", "gap": "10px"},
                ),
                html.Div(
                    [
                        html.Div(dcc.Graph(id="pdf-price", config={"displaylogo": False}), id="pdf-price-card", style=_chart_card_style),
                        html.Div(dcc.Graph(id="pdf-return", config={"displaylogo": False}), id="pdf-return-card", style=_chart_card_style),
                        html.Div(dcc.Graph(id="pdf-vol", config={"displaylogo": False}), id="pdf-vol-card", style=_chart_card_style),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "10px"},
                ),
            ],
            id="main-panel",
            style={"padding": "14px", **_panel_card_style},
        ),
        dcc.Interval(id="tick", interval=int(dashboard_cfg.get("refresh_seconds", 20)) * 1000, n_intervals=0),
    ],
    id="app-root",
    style={
        "padding": "14px",
        "backgroundColor": _theme["bg"],
        "minHeight": "100vh",
        "fontFamily": "Manrope, Space Grotesk, Segoe UI, sans-serif",
    },
)


@app.callback(
    Output("agent-a-panel", "children"),
    Output("agent-b-panel", "children"),
    Output("live-plot", "figure"),
    Output("heat2d", "figure"),
    Output("heat3d", "figure"),
    Output("pdf-price", "figure"),
    Output("pdf-return", "figure"),
    Output("pdf-vol", "figure"),
    Output("auto-update-status", "children"),
    Output("startup-sync-panel", "children"),
    Output("app-root", "style"),
    Output("main-panel", "style"),
    Output("live-plot-card", "style"),
    Output("heat2d-card", "style"),
    Output("heat3d-card", "style"),
    Output("pdf-price-card", "style"),
    Output("pdf-return-card", "style"),
    Output("pdf-vol-card", "style"),
    Output("dashboard-title", "style"),
    Output("dashboard-subtitle", "style"),
    Input("tick", "n_intervals"),
    Input("btn-refresh-dashboard", "n_clicks"),
    Input("master-table-path", "value"),
    Input("latest-records", "value"),
    Input("state-path", "value"),
    Input("trading-job-state-path", "value"),
    Input("live-mins", "value"),
    Input("group-mins", "value"),
    Input("plot-type", "value"),
    Input("heat-mins", "value"),
    Input("pdf-mins", "value"),
    Input("auto-update-enabled", "value"),
    Input("auto-update-every", "value"),
    Input("tiingo-symbol", "value"),
    Input("tiingo-rate", "value"),
    Input("tiingo-token-env", "value"),
    Input("theme-mode", "value"),
)
def refresh(n_intervals, _manual_refresh_clicks, master_table_path, latest_records, state_path, trading_job_state_path, live_mins, group_mins, plot_type, heat_mins, pdf_mins, auto_update_enabled, auto_update_every, symbol, rate, token_env, theme_mode):
    auto_status = ""
    _, root_style, main_panel_style, chart_card_style, signal_style, title_style, subtitle_style = _dynamic_styles(theme_mode)
    template = "plotly_dark" if str(theme_mode or "light").lower() == "dark" else "plotly_white"
    startup_status_path = str(dashboard_cfg.get("startup_status_path", _default_startup_status_path()))
    startup_status = _load_startup_sync_status(startup_status_path)

    result = (startup_status.get("result") or {}) if isinstance(startup_status, dict) else {}
    attempts = result.get("attempts") if isinstance(result, dict) else []
    attempts_count = len(attempts) if isinstance(attempts, list) else 0
    latest_aligned_ts = (
        result.get("latest_date")
        if isinstance(result, dict)
        else startup_status.get("latest_date")
    )
    weekend_relaxed = bool(result.get("is_weekend", False)) if isinstance(result, dict) else False
    startup_ok = bool(startup_status.get("ok", False)) if isinstance(startup_status, dict) else False
    startup_ts = startup_status.get("timestamp", "N/A") if isinstance(startup_status, dict) else "N/A"
    startup_reason = startup_status.get("reason") if isinstance(startup_status, dict) else None
    startup_panel = html.Div(
        [
            html.H4("Startup Sync Summary"),
            html.P(f"Status: {'ALIGNED' if startup_ok else 'NOT ALIGNED / SKIPPED'}"),
            html.P(f"Pull attempts: {attempts_count}"),
            html.P(f"Latest aligned timestamp: {latest_aligned_ts or 'N/A'}"),
            html.P(f"Weekend relaxation applied: {'Yes' if weekend_relaxed else 'No'}"),
            html.P(f"Last startup sync write: {startup_ts}"),
            html.P(f"Reason: {startup_reason}") if startup_reason else html.Div(),
        ],
        style=signal_style,
    )

    if not master_table_path:
        empty = go.Figure(layout=dict(title="Set Master table CSV path to start dashboard", template="plotly_white"))
        panel_a = html.Div(
            [
                html.H4("Agent A Signal"),
                html.P("Missing required input: Master table CSV path."),
            ],
            style=signal_style,
        )
        panel_b = html.Div(
            [
                html.H4("Agent B Logs / Signals"),
                html.P("Missing required input: Master table CSV path."),
            ],
            style=signal_style,
        )
        return panel_a, panel_b, empty, empty, empty, empty, empty, empty, auto_status, startup_panel, root_style, main_panel_style, chart_card_style, chart_card_style, chart_card_style, chart_card_style, chart_card_style, chart_card_style, title_style, subtitle_style

    enabled_auto = str(auto_update_enabled or "on").lower() == "on"
    every_n = max(int(auto_update_every or 1), 1)
    if enabled_auto and (int(n_intervals or 0) % every_n == 0):
        auto_status = _update_master_from_tiingo(master_table_path, symbol, rate, token_env)
    elif enabled_auto:
        auto_status = f"Auto update enabled (next run every {every_n} tick(s))."
    else:
        auto_status = "Auto update disabled."

    df = _load_data(master_table_path, latest_records=max(int(latest_records or 50000), 500))
    state = _load_state(state_path)
    tj_state = _load_state(trading_job_state_path or _default_trading_job_state_path())

    if not df.empty:
        end_ts = df["DATE"].max()
        start_ts = end_ts - timedelta(minutes=int(live_mins or 1440))
        live_df = df[df["DATE"] >= start_ts].copy()
    else:
        live_df = df

    # Agent A panel
    mode = state.get("mode", "N/A")
    plan = (tj_state.get("plan") or state.get("plan") or {})
    mode_b = (tj_state.get("mode_b") or state.get("mode_b") or {})
    positions = (tj_state.get("open_positions") or state.get("open_positions") or [])

    panel_a = html.Div(
        [
            html.H4("Agent A Signal"),
            html.P(f"Mode: {tj_state.get('mode', mode)} | Stage: {tj_state.get('stage', 'N/A')} | Status: {tj_state.get('status', 'N/A')}") ,
            html.P(f"Decision: {plan.get('decision', 'N/A')} | Model: {plan.get('model', 'N/A')}"),
            html.P(f"Success prob: {plan.get('success_probability', 'N/A')} | Confidence: {plan.get('confidence', 'N/A')} | CM acc: {plan.get('cm_accuracy', 'N/A')}"),
            html.P(f"Input fooling risk: {plan.get('input_fooling_risk', 'N/A')} | Score: {plan.get('signal_score', 'N/A')}"),
            html.P(f"Entry: {plan.get('entry', 'N/A')} | SL: {plan.get('stop_loss', 'N/A')} | TP: {plan.get('take_profit', 'N/A')}"),
            html.P(f"Rationale: {plan.get('rationale', 'N/A')}"),
            html.Pre("\n".join([str(x) for x in (plan.get("risk_notes") or [])[:6]]) or "No risk notes", style={"whiteSpace": "pre-wrap", "fontSize": "12px"}),
        ],
        style=signal_style,
    )

    # Agent B logs/signals/reasoning panel
    tf_signals = ((mode_b.get("timeframe_signals") or {}) if isinstance(mode_b, dict) else {})
    consensus = tf_signals.get("consensus", mode_b.get("consensus", "N/A") if isinstance(mode_b, dict) else "N/A")
    consensus_score = tf_signals.get("consensus_score", mode_b.get("consensus_score", "N/A") if isinstance(mode_b, dict) else "N/A")
    tf_map = tf_signals.get("timeframes", {}) if isinstance(tf_signals, dict) else {}
    tf_lines = []
    if isinstance(tf_map, dict):
        for tf, sig in sorted(tf_map.items()):
            if isinstance(sig, dict):
                tf_lines.append(
                    f"{tf}: signal={sig.get('signal', 'N/A')} conf={sig.get('confidence', 'N/A')} status={sig.get('status_code', sig.get('error', 'N/A'))}"
                )

    assistance = tj_state.get("agent_b_assistance", []) if isinstance(tj_state, dict) else []
    assist_lines = []
    for item in (assistance[-5:] if isinstance(assistance, list) else []):
        if isinstance(item, dict):
            txt = str(item.get("text") or item.get("error") or "")
            if len(txt) > 260:
                txt = txt[:260] + "..."
            assist_lines.append(
                f"[{item.get('timestamp', 'N/A')}] ok={item.get('ok', False)} provider={item.get('provider', 'N/A')} :: {txt}"
            )

    approval_channels = ((cfg.get("agent") or {}).get("approval_channels") or ["popup", "terminal"])
    panel_b = html.Div(
        [
            html.H4("Agent B Logs / Signals / Reasoning"),
            html.P(f"Consensus: {consensus} | Consensus score: {consensus_score} | Open positions shown: {len(positions)}"),
            html.P(f"Closed reason: {tj_state.get('closed_reason', 'N/A')} | Last tick: {tj_state.get('last_mode_b_tick', 'N/A')}") ,
            html.P("Timeframe signals:"),
            html.Pre("\n".join(tf_lines) or "No timeframe signal snapshots yet.", style={"whiteSpace": "pre-wrap", "fontSize": "12px", "maxHeight": "130px", "overflowY": "auto"}),
            html.P("Reasoning/log snippets (latest 5):"),
            html.Pre("\n".join(assist_lines) or "No Agent B reasoning logs yet.", style={"whiteSpace": "pre-wrap", "fontSize": "12px", "maxHeight": "130px", "overflowY": "auto"}),
            html.P(f"Approvals are handled in a separate popup/terminal flow. Configured channels: {approval_channels}"),
        ],
        style=signal_style,
    )

    live_fig = _plot_fx_data(live_df, plot_type=plot_type or "candlestick", interval_minutes=max(int(group_mins or 1), 1), template=template)

    heat2d, heat3d = _build_heatmaps(live_df, lookback_minutes=int(heat_mins or 480), template=template)

    if not live_df.empty:
        end_ts = live_df["DATE"].max()
        start_pdf = end_ts - timedelta(minutes=int(pdf_mins or 360))
        pdf_df = live_df[live_df["DATE"] >= start_pdf].copy()
    else:
        pdf_df = live_df

    close_vals = pdf_df["CLOSE"].astype(float).values if "CLOSE" in pdf_df.columns else np.array([])
    ret_vals = np.diff(close_vals) if close_vals.size > 1 else np.array([])
    vol_vals = pd.Series(ret_vals).rolling(30).std().dropna().values if ret_vals.size > 30 else np.array([])

    pdf_price = _pdf_fig(close_vals, "Price PDF (latest window)", "Price", template=template)
    pdf_return = _pdf_fig(ret_vals, "Return PDF (latest window)", "Return", template=template)
    pdf_vol = _pdf_fig(vol_vals, "Volatility PDF (latest window)", "Volatility", template=template)

    return panel_a, panel_b, live_fig, heat2d, heat3d, pdf_price, pdf_return, pdf_vol, auto_status, startup_panel, root_style, main_panel_style, chart_card_style, chart_card_style, chart_card_style, chart_card_style, chart_card_style, chart_card_style, title_style, subtitle_style


@app.callback(
    Output("update-master-status", "children"),
    Input("btn-update-master", "n_clicks"),
    State("master-table-path", "value"),
    State("tiingo-symbol", "value"),
    State("tiingo-rate", "value"),
    State("tiingo-token-env", "value"),
    prevent_initial_call=True,
)
def update_master_table(_, master_table_path, symbol, rate, token_env):
    return _update_master_from_tiingo(master_table_path, symbol, rate, token_env)


@app.callback(
    Output("mode-b-control-status", "children"),
    Input("btn-stop-mode-b", "n_clicks"),
    Input("btn-resume-mode-b", "n_clicks"),
    State("state-path", "value"),
)
def mode_b_controls(stop_clicks, resume_clicks, state_path):
    del stop_clicks, resume_clicks
    state_path = state_path or _default_state_path()
    flag_path = str(
        resolve_runtime_file(
            configured_path=((cfg.get("mode_b") or {}).get("interrupt_flag_path")),
            fallback_name="mode_b_interrupt.flag",
            trading_cfg=cfg,
            base_dir=BASE_DIR,
        )
    )
    os.makedirs(os.path.dirname(flag_path), exist_ok=True)

    trig = dash_mod.ctx.triggered_id
    if trig == "btn-stop-mode-b":
        with open(flag_path, "w", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    elif trig == "btn-resume-mode-b":
        if os.path.exists(flag_path):
            os.remove(flag_path)

    status = "INTERRUPTED" if os.path.exists(flag_path) else "RUNNING/AVAILABLE"
    return f"Mode B control status: {status} (flag: {flag_path})"


@app.callback(
    Output("agent-channel-status", "children"),
    Input("btn-enable-channel", "n_clicks"),
    Input("btn-disable-channel", "n_clicks"),
    State("state-path", "value"),
)
def channel_controls(enable_clicks, disable_clicks, state_path):
    del enable_clicks, disable_clicks
    state_path = state_path or _default_state_path()
    output_dir = os.path.join(BASE_DIR, "reports")
    trig = dash_mod.ctx.triggered_id
    if trig == "btn-enable-channel":
        set_channel_enabled(output_dir=output_dir, trading_cfg=cfg, enabled=True)
    elif trig == "btn-disable-channel":
        set_channel_enabled(output_dir=output_dir, trading_cfg=cfg, enabled=False)

    enabled = is_channel_enabled(output_dir=output_dir, trading_cfg=cfg)
    return f"Agent channel: {'ENABLED' if enabled else 'DISABLED'}"


@app.callback(
    Output("agent-channel-panel", "children"),
    Input("tick", "n_intervals"),
    State("state-path", "value"),
)
def render_agent_channel(_n, state_path):
    state_path = state_path or _default_state_path()
    output_dir = os.path.join(BASE_DIR, "reports")
    msgs = read_channel_messages(output_dir=output_dir, trading_cfg=cfg, max_lines=80)

    if not msgs:
        return html.Div(
            [
                html.H4("Agent Live Channel"),
                html.P("No channel messages yet. Channel wakes on user trigger or emergency."),
            ]
        )

    items = []
    for m in msgs[-25:]:
        ts = m.get("timestamp_utc", "N/A")
        ch = m.get("channel", "agent")
        kind = m.get("kind", "info")
        emergency = bool(m.get("emergency", False))
        msg = str(m.get("message", ""))
        dl = m.get("approval_deadline_utc")
        head = f"[{ts}] {ch}/{kind}{' [EMERGENCY]' if emergency else ''}"
        if dl:
            head += f" | deadline={dl}"
        items.append(
            html.Div(
                [
                    html.Div(head, style={"fontWeight": "700", "fontSize": "12px"}),
                    html.Div(msg, style={"fontSize": "12px", "whiteSpace": "pre-wrap"}),
                ],
                style={"padding": "8px", "border": "1px solid #dbe4f3", "borderRadius": "8px", "marginBottom": "6px", "backgroundColor": "#f8fbff"},
            )
        )

    return html.Div(
        [
            html.H4("Agent Live Channel"),
            html.Div(items, style={"maxHeight": "280px", "overflowY": "auto"}),
        ]
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
