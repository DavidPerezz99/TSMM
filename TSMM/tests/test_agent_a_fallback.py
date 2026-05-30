from utils.trading_job import _realign_plan_risk_levels


def test_realign_plan_risk_levels_hold_to_buy():
    plan = {
        "decision": "hold",
        "entry": 100.0,
        "stop_loss": 101.0,
        "take_profit": 98.0,
    }

    out = _realign_plan_risk_levels(plan, "buy")

    assert out["stop_loss"] == 99.0
    assert out["take_profit"] == 102.0


def test_realign_plan_risk_levels_hold_to_sell():
    plan = {
        "decision": "hold",
        "entry": 100.0,
        "stop_loss": 99.0,
        "take_profit": 102.0,
    }

    out = _realign_plan_risk_levels(plan, "sell")

    assert out["stop_loss"] == 101.0
    assert out["take_profit"] == 98.0
