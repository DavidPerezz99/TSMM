"""
Live data update loop for TSMM.

Usage:
    py -3.11 scripts/live_data_loop.py --config config/config.yaml --every-seconds 60
"""

from __future__ import annotations

import argparse
import time
import yaml
from pathlib import Path

from utils.live_data import refresh_dataset_from_tiingo
from utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--every-seconds", type=int, default=60)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    log_dir = cfg.get("log_dir", "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(Path(log_dir) / "live_data_loop.log"))

    refresh_cfg = cfg.get("data_refresh", {}) or {}
    if not refresh_cfg.get("enabled", False):
        logger.warning("data_refresh.enabled=false in config. Exiting.")
        return

    logger.info("Starting live data refresh loop every %s seconds", args.every_seconds)
    while True:
        try:
            result = refresh_dataset_from_tiingo(refresh_cfg, output_path=cfg.get("data_path"), logger=logger)
            logger.info("refresh_result=%s", result)
        except Exception as e:
            logger.exception("Refresh loop error: %s", e)
        time.sleep(max(args.every_seconds, 10))


if __name__ == "__main__":
    main()
