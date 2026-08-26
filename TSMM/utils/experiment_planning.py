"""Resource estimates and deadline helpers for forecasting experiments."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import math
from typing import Any, Dict, Iterable, List


GIB = float(1024 ** 3)


def selected_models(config: Dict[str, Any]) -> List[str]:
    models = config.get("models_to_run") or {}
    return [
        str(value).strip().lower()
        for value in list(models.get("univariate") or [])
        + list(models.get("multivariate") or [])
        if str(value).strip()
    ]


def _nbeats_parameter_count(config: Dict[str, Any]) -> int:
    n_steps = max(int(config.get("n_steps", 1) or 1), 1)
    input_features = list(config.get("input_features") or [config.get("target_col", "target")])
    target_features = list(config.get("target_features") or [config.get("target_col", "target")])
    m_steps = max(int(config.get("m_steps", 1) or 1), 1)
    input_size = n_steps * max(len(input_features), 1)
    forecast_size = m_steps * max(len(target_features), 1)
    nbeats = config.get("nbeats") or {}
    model_type = str(nbeats.get("model_type", "interpretable")).strip().lower()
    hidden_default = max(int(nbeats.get("hidden_size", 256) or 256), 1)

    if model_type == "blackbox":
        blackbox = nbeats.get("blackbox_config") or {}
        num_blocks = max(int(blackbox.get("num_blocks", 4) or 4), 1) * 2
        num_layers = max(int(blackbox.get("num_layers", 4) or 4), 1)
        # Input projection, hidden layers, and generic backcast/forecast head.
        per_block = (
            input_size * hidden_default
            + max(num_layers - 1, 0) * hidden_default * hidden_default
            + hidden_default * (input_size + forecast_size)
        )
        return int(num_blocks * per_block)

    stacks = list(nbeats.get("stacks_config") or [])
    if not stacks:
        stacks = [
            {"type": "trend", "num_blocks": 2, "hidden_size": hidden_default, "degree": 4},
            {
                "type": "seasonality",
                "num_blocks": 2,
                "hidden_size": hidden_default,
                "num_harmonics": 8,
            },
        ]
    total = 0
    for stack in stacks:
        hidden = max(int(stack.get("hidden_size", hidden_default) or hidden_default), 1)
        blocks = max(int(stack.get("num_blocks", 1) or 1), 1)
        if str(stack.get("type", "trend")).lower() == "seasonality":
            theta = 2 * max(int(stack.get("num_harmonics", 6) or 6), 1)
        else:
            theta = max(int(stack.get("degree", 3) or 3), 1) + 1
        total += blocks * (input_size * hidden + hidden * hidden + hidden * theta)
    return int(total)


def estimate_experiment_memory(config: Dict[str, Any], records: int | None = None) -> Dict[str, Any]:
    """Return a conservative peak-RAM estimate for one model experiment.

    The training implementations materialize overlapping sequences, scaled
    copies, train/test views, and (for N-BEATS) float32 tensors. The estimate is
    intentionally conservative and is used as a planning guard, not as a claim
    about exact allocator behavior.
    """
    cfg = deepcopy(config)
    row_count = max(int(records if records is not None else cfg.get("records", 0) or 0), 1)
    n_steps = max(int(cfg.get("n_steps", 1) or 1), 1)
    m_steps = max(int(cfg.get("m_steps", 1) or 1), 1)
    input_count = max(len(list(cfg.get("input_features") or [cfg.get("target_col", "target")])), 1)
    target_count = max(len(list(cfg.get("target_features") or [cfg.get("target_col", "target")])), 1)
    sequence_count = max(row_count - n_steps - m_steps + 1, 1)
    models = selected_models(cfg) or ["unknown"]

    # SQLite/pandas ingestion buffers coexist briefly with the normalized OHLC
    # frame. Materialized SQL caches avoid resampling the full minute history,
    # but they do not eliminate these per-experiment objects.
    sql_ingestion_bytes = row_count * 6 * 32

    # pandas input + engineered features + transient preprocessing copies.
    rolling_count = len(list(cfg.get("rolling_windows") or [7, 30, 60]))
    estimated_columns = max(12 + rolling_count * 6, input_count + target_count + 6)
    dataframe_bytes = row_count * estimated_columns * 24

    input_elements = sequence_count * n_steps * input_count
    target_elements = sequence_count * m_steps * target_count
    sequence_elements = input_elements + target_elements

    # Array/list/scaler/LAPACK or tensor coexistence at peak.
    bytes_per_sequence_element = 28
    if "nbeats" in models:
        bytes_per_sequence_element = 32
    elif "svr" in models:
        bytes_per_sequence_element = 30
    sequence_bytes = sequence_elements * bytes_per_sequence_element

    # Held-out arrays, inverse-scaled predictions, recursive-horizon work, and
    # evaluator metric buffers can overlap the training arrays near peak use.
    evaluation_bytes = int(
        sequence_count * max(target_count, 1) * max(m_steps, 1) * 8 * 8
        + row_count * max(input_count + target_count, 1) * 8 * 2
    )

    model_bytes = 0
    if "nbeats" in models:
        # weights + gradients + Adam moments + allocator/workspace allowance
        model_bytes = _nbeats_parameter_count(cfg) * 24
    elif "svr" in models:
        svr = cfg.get("svr") or {}
        model_bytes = int(float(svr.get("cache_size", 200) or 200) * 1024 ** 2 * target_count)
    else:
        # LinearRegression/SVD workspace scales with the flattened design matrix.
        model_bytes = int(input_elements * 8)

    fixed_runtime_bytes = int(1.5 * GIB)
    subtotal = (
        fixed_runtime_bytes
        + sql_ingestion_bytes
        + dataframe_bytes
        + sequence_bytes
        + evaluation_bytes
        + model_bytes
    )
    # Covers allocator fragmentation, BLAS/PyTorch workspaces, and objects not
    # directly visible in the analytical shape calculation.
    estimated_bytes = int(subtotal * 1.25)
    return {
        "records": row_count,
        "models": models,
        "n_steps": n_steps,
        "input_features": input_count,
        "target_features": target_count,
        "estimated_peak_gb": round(estimated_bytes / GIB, 2),
        "estimated_peak_bytes": estimated_bytes,
        "components_gb": {
            "runtime": round(fixed_runtime_bytes / GIB, 2),
            "sql_ingestion": round(sql_ingestion_bytes / GIB, 2),
            "dataframe": round(dataframe_bytes / GIB, 2),
            "sequences": round(sequence_bytes / GIB, 2),
            "evaluation": round(evaluation_bytes / GIB, 2),
            "model_and_workspace": round(model_bytes / GIB, 2),
        },
    }


def estimate_max_records(config: Dict[str, Any], ram_limit_gb: float, upper_bound: int = 10_000_000) -> int:
    """Binary-search the largest record count under the planning RAM limit."""
    limit_bytes = float(ram_limit_gb) * GIB
    low, high = 1, max(int(upper_bound), 1)
    while low < high:
        mid = (low + high + 1) // 2
        estimate = estimate_experiment_memory(config, records=mid)
        if int(estimate["estimated_peak_bytes"]) <= limit_bytes:
            low = mid
        else:
            high = mid - 1
    return int(low)


def duration_signature(config: Dict[str, Any]) -> str:
    models = selected_models(config) or ["unknown"]
    model = models[0]
    input_count = max(len(list(config.get("input_features") or ["target"])), 1)
    target_count = max(len(list(config.get("target_features") or ["target"])), 1)
    parts = [
        model,
        f"n{int(config.get('n_steps', 1) or 1)}",
        f"in{input_count}",
        f"out{target_count}",
    ]
    if model == "nbeats":
        nbeats = config.get("nbeats") or {}
        parts.extend(
            [
                str(nbeats.get("model_type", "interpretable")),
                f"h{int(nbeats.get('hidden_size', 256) or 256)}",
                f"e{int(nbeats.get('epochs', 100) or 100)}",
                f"p{_nbeats_parameter_count(config)}",
            ]
        )
    return ":".join(parts)


def memory_shape_signature(config: Dict[str, Any]) -> str:
    """Identify configurations with the same analytical RAM growth curve."""
    models = selected_models(config) or ["unknown"]
    nbeats = config.get("nbeats") or {}
    svr = config.get("svr") or {}
    return ":".join(
        [
            ",".join(models),
            f"n{int(config.get('n_steps', 1) or 1)}",
            f"m{int(config.get('m_steps', 1) or 1)}",
            f"in{max(len(list(config.get('input_features') or ['target'])), 1)}",
            f"out{max(len(list(config.get('target_features') or ['target'])), 1)}",
            f"roll{len(list(config.get('rolling_windows') or [7, 30, 60]))}",
            f"type{str(nbeats.get('model_type', ''))}",
            f"hidden{int(nbeats.get('hidden_size', 0) or 0)}",
            f"params{_nbeats_parameter_count(config) if 'nbeats' in models else 0}",
            f"svrcache{float(svr.get('cache_size', 0) or 0)}",
        ]
    )


def estimate_experiment_duration_seconds(
    config: Dict[str, Any],
    records: int | None = None,
    cpu_threads: int = 6,
) -> float:
    """Estimate CPU wall time from local Ryzen 5 calibration benchmarks.

    This estimate deliberately errs high. Runtime history for an exact model
    signature supersedes it after successful session experiments complete.
    """
    row_count = max(int(records if records is not None else config.get("records", 0) or 0), 1)
    n_steps = max(int(config.get("n_steps", 1) or 1), 1)
    input_count = max(len(list(config.get("input_features") or ["target"])), 1)
    models = selected_models(config) or ["unknown"]
    model = models[0]
    thread_factor = 6.0 / max(int(cpu_threads or 1), 1)

    if model == "nbeats":
        nbeats = config.get("nbeats") or {}
        epochs = max(int(nbeats.get("epochs", 100) or 100), 1)
        parameters = max(_nbeats_parameter_count(config), 1)
        calibrated_rows = 940.0
        calibrated_parameters = 474_624.0
        seconds_per_epoch = (
            2.0
            * (row_count / calibrated_rows)
            * math.sqrt(parameters / calibrated_parameters)
            * thread_factor
        )
        return float(20.0 + epochs * seconds_per_epoch)

    flattened_features = n_steps * input_count
    if model == "ulr":
        calibrated_work = 6_000.0 * (96 * 5) ** 2
        work = row_count * flattened_features ** 2
        return float(5.0 + 8.0 * (work / calibrated_work) * thread_factor)
    if model == "svr":
        # RBF SVR trends super-linearly in sample count; this is principally a
        # refusal/ordering estimate rather than a precise completion forecast.
        return float(15.0 + 60.0 * (row_count / 5_000.0) ** 2 * thread_factor)
    return float(10.0 + row_count * flattened_features / 1_000_000.0)


def estimate_max_records_for_duration(
    config: Dict[str, Any],
    max_duration_minutes: float,
    cpu_threads: int = 6,
    upper_bound: int = 10_000_000,
) -> int:
    limit_seconds = float(max_duration_minutes) * 60.0
    low, high = 1, max(int(upper_bound), 1)
    while low < high:
        mid = (low + high + 1) // 2
        estimate = estimate_experiment_duration_seconds(
            config,
            records=mid,
            cpu_threads=cpu_threads,
        )
        if estimate <= limit_seconds:
            low = mid
        else:
            high = mid - 1
    return int(low)


def next_local_deadline(now: datetime, deadline_hhmm: str) -> datetime:
    """Resolve the next local wall-clock deadline after ``now``."""
    raw = str(deadline_hhmm or "05:00").strip()
    try:
        hour_text, minute_text = raw.split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
    except Exception as exc:
        raise ValueError(f"Invalid deadline_local '{deadline_hhmm}'; expected HH:MM") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid deadline_local '{deadline_hhmm}'; expected HH:MM")
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def worst_case_memory(configurations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    estimates = [estimate_experiment_memory(cfg) for cfg in configurations]
    if not estimates:
        return {}
    return max(estimates, key=lambda item: float(item["estimated_peak_gb"]))
