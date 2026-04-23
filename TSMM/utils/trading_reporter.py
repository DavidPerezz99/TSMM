"""
Trading plan report generator.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List

from fpdf import FPDF


class TradingPlanPDF(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self.title_text = title
        self.set_auto_page_break(auto=True, margin=12)
        self.add_page()

    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 8, self.title_text, 0, 1, "C")
        self.set_font("Arial", "", 9)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
        self.ln(2)

    def section(self, title: str):
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, title, 0, 1, "L")
        self.set_font("Arial", "", 10)

    def kv(self, k: str, v: Any):
        self.multi_cell(0, 6, f"{k}: {v}")


def generate_trading_plan_report(
    output_path: str,
    target_col: str,
    mode: str,
    plan: Dict[str, Any],
    backtest: Dict[str, Any],
    warnings: List[str],
    heatmaps: Dict[str, Any] | None = None,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pdf = TradingPlanPDF(title=f"Trading Session Plan - {target_col}")

    pdf.section("Plan Summary")
    pdf.kv("Mode", mode)
    pdf.kv("Model", plan.get("model", "N/A"))
    pdf.kv("Decision", plan.get("decision", "hold"))
    pdf.kv("Rationale", plan.get("rationale", "N/A"))
    pdf.kv("Entry", plan.get("entry"))
    pdf.kv("Stop Loss", plan.get("stop_loss"))
    pdf.kv("Take Profit", plan.get("take_profit"))
    pdf.kv("Volume", plan.get("volume"))
    pdf.kv("Confidence", plan.get("confidence"))
    pdf.kv("Estimated Signal Success Probability", plan.get("success_probability"))

    pdf.ln(2)
    pdf.section("Risk and Reversal Notes")
    for note in (plan.get("risk_notes") or []):
        pdf.multi_cell(0, 6, f"- {note}")

    if backtest and backtest.get("enabled"):
        pdf.ln(2)
        pdf.section("Backtest Summary (Validation Window)")
        keys = [
            "model_name", "n_trades", "win_rate", "total_return_pct",
            "max_drawdown_pct", "avg_trade_return_pct"
        ]
        for k in keys:
            if k in backtest:
                pdf.kv(k, backtest.get(k))

    if heatmaps and heatmaps.get("enabled"):
        pdf.ln(2)
        pdf.section("Probability Concentration Maps")
        pdf.kv("Simulation paths", heatmaps.get("n_paths"))
        pdf.kv("Residual std used", heatmaps.get("residual_std"))

        p2d = heatmaps.get("heatmap_2d_path")
        if p2d and os.path.exists(p2d):
            pdf.ln(2)
            y = pdf.get_y()
            if y > 220:
                pdf.add_page()
                y = pdf.get_y()
            pdf.image(p2d, x=10, y=y, w=190, h=70)
            pdf.set_y(y + 75)

        p3d = heatmaps.get("heatmap_3d_path")
        if p3d and os.path.exists(p3d):
            y = pdf.get_y()
            if y > 220:
                pdf.add_page()
                y = pdf.get_y()
            pdf.image(p3d, x=10, y=y, w=190, h=70)
            pdf.set_y(y + 75)

    if warnings:
        pdf.ln(2)
        pdf.section("Warnings")
        for w in warnings:
            pdf.multi_cell(0, 6, f"- {w}")

    pdf.output(output_path)
    return output_path
