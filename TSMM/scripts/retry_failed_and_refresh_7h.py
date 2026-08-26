from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports" / "runtime" / "deployment_pipeline_last.json"
FINAL_7H_CFG = ROOT / "config" / "high7hResults" / "nbeats" / "top1_07000.yaml"
MODEL_DIR = ROOT / "model_files"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _dedup_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in paths:
        p = str(raw or "").strip()
        if not p:
            continue
        try:
            key = str(Path(p).resolve())
        except Exception:
            key = str(Path(p))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _dedup_model_files() -> int:
    pat_art = re.compile(
        r"^(?P<model>[A-Za-z0-9_]+)_artifacts_(?P<target>high|low|open|close)_(?P<timeframe>[A-Za-z0-9]+)_(?P<ts>\d{8}_\d{6})\.joblib$",
        flags=re.IGNORECASE,
    )
    pat_mod = re.compile(
        r"^(?P<model>[A-Za-z0-9_]+)_(?P<target>high|low|open|close)_(?P<timeframe>[A-Za-z0-9]+)_(?P<ts>\d{8}_\d{6})\.joblib$",
        flags=re.IGNORECASE,
    )

    groups: Dict[str, List[Path]] = {}
    for p in MODEL_DIR.glob("*.joblib"):
        m_art = pat_art.match(p.name)
        if m_art:
            key = f"{m_art.group('model').lower()}|{m_art.group('target').lower()}|{m_art.group('timeframe').lower()}|art"
            groups.setdefault(key, []).append(p)
            continue
        m_mod = pat_mod.match(p.name)
        if m_mod:
            key = f"{m_mod.group('model').lower()}|{m_mod.group('target').lower()}|{m_mod.group('timeframe').lower()}|mod"
            groups.setdefault(key, []).append(p)

    deleted = 0
    ts_pat = re.compile(r"(\d{8}_\d{6})")
    for items in groups.values():
        items_sorted = sorted(
            items,
            key=lambda x: ts_pat.search(x.name).group(1) if ts_pat.search(x.name) else "",
            reverse=True,
        )
        for old in items_sorted[1:]:
            try:
                old.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                pass
    return deleted


def main() -> int:
    if not SUMMARY_PATH.exists():
        print(f"missing_summary={SUMMARY_PATH}")
        return 1

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    retrain = summary.get("retrain") or []
    failed_cfgs = [str(x.get("config_path") or "") for x in retrain if not bool(x.get("ok", False))]
    failed_cfgs = [x for x in failed_cfgs if x]

    queue = list(failed_cfgs)
    queue.append(str(FINAL_7H_CFG))
    queue = _dedup_paths(queue)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "reports" / "runtime" / f"manual_retrain_retry_{ts}.log"

    py = sys.executable
    print(f"log_path={log_path}")
    print(f"retry_count={len(queue)}")

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_now()}] retry_count={len(queue)}\n")
        for cfg in queue:
            log.write(f"[{_now()}] START cfg={cfg}\n")
            log.flush()
            env = os.environ.copy()
            env["CONFIG_PATH"] = cfg
            proc = subprocess.run(
                [py, "app.py", "forecast"],
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log.write(f"[{_now()}] END rc={int(proc.returncode)} cfg={cfg}\n")
            log.flush()

        deleted = _dedup_model_files()
        log.write(f"[{_now()}] dedup_deleted={deleted}\n")
        log.flush()

    print(f"done_log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
