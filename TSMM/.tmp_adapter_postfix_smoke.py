import json
import os
from datetime import datetime, timezone

import yaml

from utils.investing_agent import MT5Adapter, load_trading_config


def y(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_one(label, cfg_path, force_path=None):
    cfg = load_trading_config(cfg_path)
    broker = (cfg.get("broker") or {}).get("mt5") or {}
    ex = cfg.get("execution") or {}
    symbol = str(ex.get("symbol") or "XAUUSD")
    vol = float(ex.get("default_volume") or 0.01)

    if force_path:
        os.environ["MT5_TERMINAL_PATH"] = force_path

    out = {"label": label, "connect": {}, "account": {}, "pending": {}, "market": {}}
    a = MT5Adapter(broker)
    try:
        ok, msg = a.connect()
        out["connect"] = {"ok": ok, "message": msg}
        if not ok:
            return out

        mt5 = a._mt5
        ai = mt5.account_info(); ti = mt5.terminal_info()
        out["account"] = {
            "login": int(getattr(ai, "login", 0) or 0) if ai else 0,
            "server": str(getattr(ai, "server", "") or "") if ai else "",
            "trade_allowed_account": bool(getattr(ai, "trade_allowed", False)) if ai else False,
            "trade_allowed_terminal": bool(getattr(ti, "trade_allowed", False)) if ti else False,
        }

        mt5.symbol_select(symbol, True)
        t = mt5.symbol_info_tick(symbol)
        ask = float(getattr(t, "ask", 0.0) or 0.0)

        # Pending GTC + cancel
        p = a.place_programmed_order(symbol=symbol, side="buy", volume=vol, entry=round(ask-40.0,2), stop_loss=round(ask-60.0,2), take_profit=round(ask-20.0,2), expiration_utc=None)
        out["pending"]["place"] = p
        if p.get("ok"):
            out["pending"]["cancel"] = a.cancel_pending_order(int(p.get("order_ticket") or 0))

        # Market + modify + close
        m = a.place_market_order(symbol=symbol, side="buy", volume=vol, stop_loss=round(ask-20.0,2), take_profit=round(ask+20.0,2))
        out["market"]["place"] = m
        ticket = int(((m.get("position") or {}).get("ticket") or 0))
        if ticket <= 0 and int(m.get("order_ticket") or 0) > 0:
            f = a.find_position_by_order(int(m.get("order_ticket") or 0))
            out["market"]["find_position"] = f
            ticket = int(((f.get("position") or {}).get("ticket") or 0))
        if ticket > 0:
            out["market"]["modify"] = a.modify_position_risk(ticket=ticket, stop_loss=round(ask-18.0,2), take_profit=round(ask+22.0,2))
            out["market"]["close"] = a.close_position_by_ticket(ticket=ticket)
        else:
            out["market"]["modify"] = {"ok": False, "message": "no ticket"}
            out["market"]["close"] = {"ok": False, "message": "no ticket"}

    finally:
        try:
            a.shutdown()
        except Exception:
            pass
    return out


if __name__ == "__main__":
    pipe = y("config/agent_pipeline.yaml")
    pepper_path = str(((pipe.get("trading") or {}).get("mt5_terminal_path") or "")).strip()
    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "results": [
            run_one("pepperstone", "config/trading_agent.yaml", force_path=pepper_path if pepper_path else None),
            run_one("ftmo", "config/trading_agent_ftmo.yaml", force_path=None),
        ],
    }
    print(json.dumps(out, indent=2, default=str))
