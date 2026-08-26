"""
Hyperparameter Search Module - Bulk Search Mode

This module provides functionality for running bulk hyperparameter searches
using a sweep definition file. It generates experiment configurations that
are meaningful per model type, avoiding redundant parameter combinations.

Key Features:
- Smart experiment generation per model type
- Model-specific parameter sweeps
- Parallel execution of experiments
- Metrics tracking for each run
"""

#!/usr/bin/env python

import argparse
import asyncio
from copy import deepcopy
from collections import Counter
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
import yaml
import numpy as np
from utils.cache_management import clear_cache


RESERVED_SWEEP_KEYS = {
    'smart_generation',
    'models_to_run',
    'input_target_sets',
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def yaml_dump(data, path):
    """Dump data to YAML file."""
    with open(path, "w") as fp:
        yaml.safe_dump(data, fp)


def yaml_load(path):
    """Load data from YAML file."""
    with open(path) as fp:
        return yaml.safe_load(fp)


def parse_range(expr):
    """Accepts strings like 'range(10, 100, 10)' or 'linspace(0,1,5)'."""
    if expr.startswith("range"):
        args = eval(expr.replace("range", ""))
        return list(range(*args))
    if expr.startswith("linspace"):
        args = eval(expr.replace("linspace", ""))
        return np.linspace(*args).tolist()
    raise ValueError(f"Unsupported range expression: {expr}")


def deep_merge_dict(base, override):
    """
    Recursively merge two dictionaries.
    """
    for key, value in override.items():
        if key in base:
            if isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge_dict(base[key], value)
            else:
                base[key] = value
        else:
            base[key] = value
    return base


def set_nested_value(d, key_path, value):
    """Set a value in a nested dictionary using dot notation."""
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_nested_value(d, key_path):
    """Get a value from a nested dictionary using dot notation."""
    keys = key_path.split('.')
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


# -----------------------------------------------------------------------------
# Smart Experiment Generation - Fixed Version
# -----------------------------------------------------------------------------

def extract_model_specific_params(sweep_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract model-specific parameters from sweep configuration.
    
    For nested parameters (like nbeats.stacks_config), we treat them as
    complete units rather than expanding each leaf independently.
    
    Parameters:
    -----------
    sweep_cfg : dict
        Sweep definition configuration
    
    Returns:
    --------
    dict
        Dictionary mapping model names to their specific parameters
    """
    model_params = {}
    
    # Define model prefixes to look for
    model_prefixes = {
        'nbeats': ['nbeats'],
        'svr': ['svr'],
        'xgboost': ['xgboost'],
        'prophet': ['prophet'],
        'lstm': ['lstm'],
        'ulr': [],  # ULR uses global params
        'mlr': [],  # MLR uses global params
        'sarimax': []  # SARIMAX uses global params
    }
    
    # Group parameters by model
    for param_key, param_value in sweep_cfg.items():
        if param_key in RESERVED_SWEEP_KEYS:
            continue
        if not isinstance(param_value, list):
            continue
            
        # Check which model this parameter belongs to
        for model_name, prefixes in model_prefixes.items():
            if any(param_key.startswith(prefix) or param_key == prefix for prefix in prefixes):
                if model_name not in model_params:
                    model_params[model_name] = {}
                model_params[model_name][param_key] = param_value
    
    return model_params


def get_global_params(sweep_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract global (non-model-specific) parameters from sweep configuration.
    
    Parameters:
    -----------
    sweep_cfg : dict
        Sweep definition configuration
    
    Returns:
    --------
    dict
        Dictionary of global parameters
    """
    global_params = {}
    model_prefixes = ['nbeats.', 'svr.', 'xgboost.', 'prophet.', 'lstm.']
    
    for param_key, param_value in sweep_cfg.items():
        if param_key in RESERVED_SWEEP_KEYS:
            continue
        if not isinstance(param_value, list):
            continue
            
        # Check if this is NOT a model-specific parameter
        if not any(param_key.startswith(prefix) for prefix in model_prefixes):
            global_params[param_key] = param_value
    
    return global_params


def build_model_config_variants(model_name: str, model_params: Dict[str, List]) -> List[Dict]:
    """
    Build valid configuration variants for a specific model.
    
    For models with nested configs (like N-BEATS), this ensures that
    related nested parameters are combined correctly.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model_params : dict
        Dictionary of parameter names to their possible values
    
    Returns:
    --------
    list
        List of valid configuration dictionaries
    """
    if not model_params:
        return [{}]  # No model-specific params, return single empty config
    
    # Special handling for N-BEATS due to its complex nested structure
    if model_name == 'nbeats':
        return build_nbeats_variants(model_params)
    
    # For other models, simple cartesian product of flat parameters
    param_keys = list(model_params.keys())
    param_values = [model_params[k] for k in param_keys]
    
    variants = []
    for combo in itertools.product(*param_values):
        variant = {}
        for key, value in zip(param_keys, combo):
            set_nested_value(variant, key, value)
        variants.append(variant)
    
    return variants


def build_nbeats_variants(nbeats_params: Dict[str, List]) -> List[Dict]:
    """
    Build valid N-BEATS configuration variants.
    
    N-BEATS has two modes:
    1. Interpretable: uses stacks_config with trend and seasonality blocks
    2. Blackbox: uses blackbox_config
    
    These are mutually exclusive, so we need to handle them separately.
    """
    def _expand_dict_choice_variants(raw_choices: List[Any]) -> List[Dict[str, Any]]:
        """Expand dict-valued sweep choices whose leaf values may still be lists."""
        variants: List[Dict[str, Any]] = []
        for choice in raw_choices:
            if not isinstance(choice, dict):
                variants.append(choice)
                continue
            keys = list(choice.keys())
            value_options = [
                value if isinstance(value, list) else [value]
                for value in (choice[key] for key in keys)
            ]
            for combo in itertools.product(*value_options):
                variants.append({key: value for key, value in zip(keys, combo)})
        return variants

    def _expand_stacks_config_variants(raw_choices: List[Any]) -> List[List[Dict[str, Any]]]:
        """Expand stack templates whose per-stack fields still contain sweep lists."""
        variants: List[List[Dict[str, Any]]] = []
        for choice in raw_choices:
            if not isinstance(choice, list):
                variants.append(choice)
                continue

            per_stack_variants: List[List[Dict[str, Any]]] = []
            for stack in choice:
                if not isinstance(stack, dict):
                    per_stack_variants.append([stack])
                    continue
                stack_keys = list(stack.keys())
                stack_value_options = [
                    value if isinstance(value, list) else [value]
                    for value in (stack[key] for key in stack_keys)
                ]
                expanded_stack_variants = []
                for combo in itertools.product(*stack_value_options):
                    expanded_stack_variants.append(
                        {key: value for key, value in zip(stack_keys, combo)}
                    )
                per_stack_variants.append(expanded_stack_variants)

            for stack_combo in itertools.product(*per_stack_variants):
                variants.append([deepcopy(item) for item in stack_combo])
        return variants

    # Separate parameters by category
    flat_params = {}  # e.g., model_type, hidden_size, epochs
    stacks_params = {}  # e.g., stacks_config.0.type, stacks_config.0.num_blocks
    blackbox_params = {}  # e.g., blackbox_config.num_blocks, blackbox_config.num_layers
    
    for key, values in nbeats_params.items():
        if key == 'nbeats.stacks_config':
            stacks_params[key] = _expand_stacks_config_variants(values)
        elif 'stacks_config' in key:
            stacks_params[key] = values
        elif key == 'nbeats.blackbox_config':
            blackbox_params[key] = _expand_dict_choice_variants(values)
        elif 'blackbox_config' in key:
            blackbox_params[key] = values
        else:
            flat_params[key] = values
    
    variants = []
    
    # Build combinations of flat parameters
    flat_keys = list(flat_params.keys())
    flat_values = [flat_params[k] for k in flat_keys]

    def _apply_interpretable_stack_hidden_size(variant: Dict[str, Any]) -> Dict[str, Any]:
        nbeats_cfg = variant.get("nbeats") or {}
        if str(nbeats_cfg.get("model_type", "interpretable")).strip().lower() != "interpretable":
            return variant
        hidden_size = nbeats_cfg.get("hidden_size")
        stacks_config = nbeats_cfg.get("stacks_config")
        if hidden_size is None or not isinstance(stacks_config, list):
            return variant

        normalized_stacks = []
        for stack in stacks_config:
            if isinstance(stack, dict):
                stack_copy = dict(stack)
                stack_copy.setdefault("hidden_size", hidden_size)
                normalized_stacks.append(stack_copy)
            else:
                normalized_stacks.append(stack)

        nbeats_copy = dict(nbeats_cfg)
        nbeats_copy["stacks_config"] = normalized_stacks
        variant_copy = dict(variant)
        variant_copy["nbeats"] = nbeats_copy
        return variant_copy

    for flat_combo in itertools.product(*flat_values):
        base_variant = {}
        for key, value in zip(flat_keys, flat_combo):
            set_nested_value(base_variant, key, value)
        
        # Determine which mode to use based on model_type
        model_type = base_variant.get('nbeats', {}).get('model_type', 'interpretable')
        
        if model_type == 'interpretable' and stacks_params:
            # Build stacks_config variants
            stacks_keys = list(stacks_params.keys())
            stacks_values = [stacks_params[k] for k in stacks_keys]
            
            for stacks_combo in itertools.product(*stacks_values):
                variant = deepcopy(base_variant)
                for key, value in zip(stacks_keys, stacks_combo):
                    set_nested_value(variant, key, value)
                variants.append(_apply_interpretable_stack_hidden_size(variant))
        
        elif model_type == 'blackbox' and blackbox_params:
            # Build blackbox_config variants
            blackbox_keys = list(blackbox_params.keys())
            blackbox_values = [blackbox_params[k] for k in blackbox_keys]
            
            for blackbox_combo in itertools.product(*blackbox_values):
                variant = deepcopy(base_variant)
                for key, value in zip(blackbox_keys, blackbox_combo):
                    set_nested_value(variant, key, value)
                variants.append(variant)
        else:
            # No nested params or unknown mode
            variants.append(_apply_interpretable_stack_hidden_size(base_variant))
    
    return variants


def generate_smart_experiments(
    base_cfg: Dict[str, Any],
    sweep_cfg: Dict[str, Any],
    verbose: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Generate experiment configurations intelligently per model.
    
    Each model only varies its own specific parameters, avoiding redundant
    experiment configurations.
    
    Parameters:
    -----------
    base_cfg : dict
        Base configuration
    sweep_cfg : dict
        Sweep definition configuration
    
    Yields:
    -------
    dict
        Experiment configuration
    """
    # Get models to run from sweep config
    models_to_run = sweep_cfg.get('models_to_run', {
        'univariate': ['ulr', 'svr', 'nbeats', 'xgboost', 'prophet', 'sarimax', 'lstm'],
        'multivariate': ['mlr', 'sarimax', 'lstm']
    })
    
    # Get model-specific parameters
    model_params = extract_model_specific_params(sweep_cfg)
    
    # Get global parameters
    global_params = get_global_params(sweep_cfg)
    
    # Get input_target_sets if specified
    special_sets = sweep_cfg.get("input_target_sets", None)
    
    # Generate global parameter combinations
    global_keys = list(global_params.keys())
    global_values = [global_params[k] for k in global_keys]
    
    if global_keys:
        global_combinations = list(itertools.product(*global_values))
    else:
        global_combinations = [()]
    
    # Generate experiments for each model
    experiment_count = 0
    all_models = list(dict.fromkeys(
        models_to_run.get('univariate', []) + models_to_run.get('multivariate', [])
    ))
    
    for model_name in all_models:
        # Get parameters specific to this model
        model_specific = model_params.get(model_name, {})
        
        # Build valid model configuration variants
        model_variants = build_model_config_variants(model_name, model_specific)
        
        if verbose:
            print(f"  {model_name}: {len(model_variants)} model variants × {len(global_combinations)} global combinations")
        
        # Combine global and model-specific parameters
        for global_combo in global_combinations:
            for model_variant in model_variants:
                experiment = deepcopy(base_cfg)
                
                # Add global parameters
                for key, value in zip(global_keys, global_combo):
                    set_nested_value(experiment, key, value)
                
                # Add model-specific parameters (deep merge)
                deep_merge_dict(experiment, model_variant)
                
                # Add models_to_run to only run this specific model
                experiment['models_to_run'] = {
                    'univariate': [model_name] if model_name in models_to_run.get('univariate', []) else [],
                    'multivariate': [model_name] if model_name in models_to_run.get('multivariate', []) else []
                }
                
                # Add input_target_sets if specified
                if special_sets:
                    for rec in special_sets:
                        exp_with_set = deepcopy(experiment)
                        exp_with_set.update(rec)
                        experiment_count += 1
                        yield exp_with_set
                else:
                    experiment_count += 1
                    yield experiment
    
    if verbose:
        print(f"\nTotal experiments generated: {experiment_count}")


# -----------------------------------------------------------------------------
# Legacy Factorial Expansion (for comparison)
# -----------------------------------------------------------------------------

def expand_nested_config(config_dict):
    """
    Expand nested configuration dictionaries into flat parameter combinations.
    WARNING: This can generate a very large number of combinations!
    """
    flat_params = {}
    
    def collect_params(d, prefix=''):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                has_list_values = any(isinstance(v, list) for v in value.values())
                has_nested_dict = any(isinstance(v, dict) for v in value.values())
                
                if has_list_values and not has_nested_dict:
                    flat_params[full_key] = value
                else:
                    collect_params(value, full_key)
            elif isinstance(value, list):
                flat_params[full_key] = value
            else:
                flat_params[full_key] = [value]
    
    collect_params(config_dict)
    
    keys = list(flat_params.keys())
    values = [flat_params[key] for key in keys]
    
    for combination in itertools.product(*values):
        result = {}
        for key, value in zip(keys, combination):
            set_nested_value(result, key, value)
        yield result


def generate_factorial_experiments(
    base_cfg: Dict[str, Any],
    sweep_cfg: Dict[str, Any]
) -> Iterator[Dict[str, Any]]:
    """
    Legacy factorial expansion - generates ALL combinations.
    WARNING: This can create thousands of experiments!
    """
    special_sets = sweep_cfg.get("input_target_sets", None)
    
    # Build parameter grid (excluding special keys)
    grid = {}
    for k, v in sweep_cfg.items():
        if k in ['smart_generation', 'models_to_run', 'input_target_sets']:
            continue
        grid[k] = parse_range(v) if isinstance(v, str) else v

    keys, values = zip(*grid.items()) if grid else ([], [])
    
    for combo in itertools.product(*values):
        patch = dict(zip(keys, combo))
        
        if special_sets:
            for rec in special_sets:
                experiment = deepcopy(base_cfg)
                deep_merge_dict(experiment, patch)
                experiment.update(rec)
                experiment['models_to_run'] = deepcopy(sweep_cfg.get('models_to_run') or {})
                yield experiment
        else:
            experiment = deepcopy(base_cfg)
            deep_merge_dict(experiment, patch)
            experiment['models_to_run'] = deepcopy(sweep_cfg.get('models_to_run') or {})
            yield experiment


def experiment_fingerprint(config: Dict[str, Any]) -> str:
    """Return a stable identity for an effective experiment configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unique_experiments(experiments: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Yield effective configurations once, preserving generation order."""
    seen = set()
    for experiment in experiments:
        snapshot = deepcopy(experiment)
        fingerprint = experiment_fingerprint(snapshot)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield snapshot


def build_sweep_plan(base_cfg: Dict[str, Any], sweep_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate exact raw and unique counts without writing configuration files."""
    if bool(sweep_cfg.get('smart_generation', True)):
        generated = generate_smart_experiments(base_cfg, sweep_cfg, verbose=False)
        generation_mode = 'smart'
    else:
        generated = generate_factorial_experiments(base_cfg, sweep_cfg)
        generation_mode = 'legacy'

    raw_count = 0
    unique_by_hash: Dict[str, Dict[str, Any]] = {}
    per_model = Counter()
    for experiment in generated:
        raw_count += 1
        snapshot = deepcopy(experiment)
        fingerprint = experiment_fingerprint(snapshot)
        if fingerprint in unique_by_hash:
            continue
        unique_by_hash[fingerprint] = snapshot
        models = snapshot.get('models_to_run') or {}
        names = list(models.get('univariate') or []) + list(models.get('multivariate') or [])
        per_model[','.join(names) if names else 'default_models'] += 1

    return {
        'generation_mode': generation_mode,
        'raw_generated': int(raw_count),
        'unique_experiments': int(len(unique_by_hash)),
        'duplicates_removed': int(raw_count - len(unique_by_hash)),
        'per_model': dict(sorted(per_model.items())),
    }


# -----------------------------------------------------------------------------
# Bulk Search Engine
# -----------------------------------------------------------------------------

class BulkSearchEngine:
    """Engine for running bulk hyperparameter searches."""
    
    def __init__(
        self,
        base_cfg,
        sweep_cfg,
        out_dir,
        sem,
        summary_dir=None,
        restart=False,
        experiment_timeout_sec=None,
        max_experiments=None,
        worthy_r2_threshold=0.6,
    ):
        self.base_cfg = base_cfg
        self.sweep_cfg = sweep_cfg
        self.out_dir = Path(out_dir)
        self.summary_dir = Path(summary_dir) if summary_dir else self.out_dir
        self.sem = sem
        self.restart = bool(restart)
        self.experiment_timeout_sec = experiment_timeout_sec
        self.max_experiments = int(max_experiments or 0)
        self.worthy_r2_threshold = float(worthy_r2_threshold)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.failure_log_dir = self.out_dir / "failed_logs"
        self.failure_log_dir.mkdir(parents=True, exist_ok=True)
        self.worthy_artifact_dir = self.out_dir / "worthy_artifacts"
        self.summary_dirs = [self.summary_dir]
        if self.out_dir.resolve() != self.summary_dir.resolve():
            self.summary_dirs.append(self.out_dir)
        self.jobs = []
        clear_cache()

    def _parse_cfg_index(self, cfg_path: Path) -> int:
        """Extract numeric index from cfg filename for stable ordering."""
        match = re.match(r"cfg_(\d+)", cfg_path.stem)
        return int(match.group(1)) if match else 10**12

    def _sorted_cfg_paths(self, cfg_paths: List[Path]) -> List[Path]:
        """Sort config files by their generated numeric index."""
        return sorted(cfg_paths, key=lambda p: (self._parse_cfg_index(p), p.name))

    def _existing_cfg_paths(self) -> List[Path]:
        """Return existing generated config files, ordered by run index."""
        return self._sorted_cfg_paths(list(self.out_dir.glob("cfg_*.yaml")))

    def _latest_summary_info(self, cfg_stem: str) -> Optional[Dict[str, Any]]:
        """Load latest summary artifact for a config stem across known summary dirs."""
        candidates = []
        for directory in self.summary_dirs:
            if not directory.exists():
                continue
            candidates.extend(directory.glob(f"{cfg_stem}__summary.json"))
            candidates.extend(directory.glob(f"{cfg_stem}__*__summary.json"))

        if not candidates:
            return None

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        status = None
        try:
            with latest.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            status = payload.get("status")
        except Exception:
            status = None

        return {
            "path": latest,
            "status": status,
            "is_failed_suffix": latest.name.endswith("__failed__summary.json"),
            "is_no_metrics_suffix": latest.name.endswith("__no_metrics__summary.json")
        }

    def _is_cfg_complete(self, cfg_path: Path) -> bool:
        """A config is complete only when its latest summary is SUCCESS."""
        info = self._latest_summary_info(cfg_path.stem)
        if info is None:
            return False
        if info["is_failed_suffix"] or info["is_no_metrics_suffix"]:
            return False
        return info.get("status") == "SUCCESS"

    def _find_resume_start_idx(self, cfg_paths: List[Path]) -> Optional[int]:
        """Find first config that needs execution, rerunning latest incomplete if needed."""
        if not cfg_paths:
            return None

        if self.restart:
            print("Restart requested: running all experiments from the beginning")
            return 0

        for idx, cfg_path in enumerate(cfg_paths):
            if not self._is_cfg_complete(cfg_path):
                latest = self._latest_summary_info(cfg_path.stem)
                if latest is None:
                    print(f"Resume: first pending experiment is {cfg_path.name} (no summary found)")
                else:
                    print(
                        "Resume: first incomplete experiment is "
                        f"{cfg_path.name} (latest status={latest.get('status')}, file={latest['path'].name})"
                    )
                return idx

        return None

    def _write_failure_log(self, cfg_path, cmd, returncode, stdout_text, stderr_text, stage="run"):
        """Write a per-experiment failure log artifact for post-mortem debugging."""
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": stage,
            "config": str(cfg_path),
            "command": cmd,
            "returncode": returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }
        out_file = self.failure_log_dir / f"{Path(cfg_path).stem}__failed__log.json"
        with out_file.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    def _write_failed_summary(self, cfg_path, stage, error_text):
        """Write fallback failed summary when search_mode cannot finish and persist one."""
        summary_file = self.summary_dir / f"{Path(cfg_path).stem}__failed__summary.json"
        payload = {
            "config_path": str(Path(cfg_path).resolve()),
            "status": "FAILED",
            "metric": {
                "run_error": {
                    "stage": stage,
                    "error": error_text
                }
            },
            "wall_time": datetime.utcnow().timestamp()
        }
        with summary_file.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    def expand_grid(self):
        """Yield {param: value} dictionaries for every combination."""
        # Check if smart generation is enabled (default: True)
        use_smart = self.sweep_cfg.get('smart_generation', True)
        
        if use_smart:
            print("Using SMART generation (recommended)")
            generated = generate_smart_experiments(self.base_cfg, self.sweep_cfg)
        else:
            print("WARNING: Using LEGACY factorial expansion (may generate many experiments)")
            generated = generate_factorial_experiments(self.base_cfg, self.sweep_cfg)
        return unique_experiments(generated)

    def materialize_configs(self):
        """Generate and save experiment configuration files, or reuse existing ones."""
        existing = self._existing_cfg_paths()
        if existing:
            if self.max_experiments and len(existing) > self.max_experiments:
                raise RuntimeError(
                    f"Existing session contains {len(existing)} configs, above "
                    f"max_experiments={self.max_experiments}."
                )
            print(f"Found {len(existing)} existing config files in {self.out_dir}; reusing for resume")
            return existing

        experiments = list(self.expand_grid())
        if self.max_experiments and len(experiments) > self.max_experiments:
            raise RuntimeError(
                f"Sweep has {len(experiments)} unique experiments, above "
                f"max_experiments={self.max_experiments}; no config files were written."
            )

        cfg_paths = []
        for idx, experiment in enumerate(experiments, 1):
            cfg = deepcopy(experiment)
            cfg_name = f"cfg_{idx:05d}.yaml"
            cfg_path = self.out_dir / cfg_name
            yaml_dump(cfg, cfg_path)
            cfg_paths.append(cfg_path)
        return cfg_paths

    async def _run_one(self, cfg_path):
        """Run a single experiment."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        
        cmd = [
            sys.executable,
            "search_mode.py",
            "--config",
            str(cfg_path),
            "--summary-dir",
            str(self.summary_dir),
            "--bulk-search",
            "--worthy-artifact-dir",
            str(self.worthy_artifact_dir),
            "--worthy-r2-threshold",
            str(self.worthy_r2_threshold),
        ]
        async with self.sem:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
            except Exception as exc:
                self._write_failure_log(
                    cfg_path=cfg_path,
                    cmd=cmd,
                    returncode=None,
                    stdout_text="",
                    stderr_text=str(exc),
                    stage="spawn"
                )
                print(f"[{cfg_path.name}] finished -> FAIL")
                print(f"Spawn failed: {exc}", file=sys.stderr)
                return

            try:
                if self.experiment_timeout_sec:
                    out, err = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=float(self.experiment_timeout_sec)
                    )
                else:
                    out, err = await proc.communicate()
            except asyncio.TimeoutError:
                proc.kill()
                out, err = await proc.communicate()
                out_text = out.decode(errors="replace") if out else ""
                err_text = err.decode(errors="replace") if err else ""
                timeout_msg = (
                    f"Experiment exceeded timeout of {self.experiment_timeout_sec} seconds "
                    "and was terminated."
                )
                self._write_failure_log(
                    cfg_path=cfg_path,
                    cmd=cmd,
                    returncode=None,
                    stdout_text=out_text,
                    stderr_text=f"{timeout_msg}\n{err_text}".strip(),
                    stage="timeout"
                )
                self._write_failed_summary(
                    cfg_path=cfg_path,
                    stage="timeout",
                    error_text=timeout_msg
                )
                print(f"[{cfg_path.name}] finished -> FAIL (TIMEOUT)")
                return

            out_text = out.decode(errors="replace") if out else ""
            err_text = err.decode(errors="replace") if err else ""
            if proc.returncode == 0:
                summary_info = self._latest_summary_info(cfg_path.stem) or {}
                summary_status = str(summary_info.get("status") or "").strip().upper()
                display_status = summary_status or "NO_SUMMARY"
            else:
                display_status = "FAIL"

            print(f"[{cfg_path.name}] finished -> {display_status}")
            if proc.returncode != 0:
                self._write_failure_log(
                    cfg_path=cfg_path,
                    cmd=cmd,
                    returncode=proc.returncode,
                    stdout_text=out_text,
                    stderr_text=err_text,
                    stage="run"
                )
                if err_text:
                    print(err_text[:1000], file=sys.stderr)
                elif out_text:
                    print(out_text[:1000], file=sys.stderr)

    async def launch_all(self):
        """Launch all experiments, resuming from first incomplete config by default."""
        cfg_paths = self._sorted_cfg_paths(self.materialize_configs())
        start_idx = self._find_resume_start_idx(cfg_paths)

        if start_idx is None:
            print("All experiments already completed successfully. Nothing to run.")
            return

        pending_cfg_paths = cfg_paths[start_idx:]
        print(f"\nLaunching {len(pending_cfg_paths)} experiments (of {len(cfg_paths)} total)...")
        print("-" * 60)
        tasks = [asyncio.create_task(self._run_one(p)) for p in pending_cfg_paths]
        await asyncio.gather(*tasks, return_exceptions=True)


def _find_latest_execution_dir(root_dir: Path) -> Optional[Path]:
    """Find latest execution directory containing generated cfg files."""
    if not root_dir.exists():
        return None

    if any(root_dir.glob("cfg_*.yaml")):
        return root_dir

    candidates = []
    for child in root_dir.iterdir():
        if child.is_dir() and any(child.glob("cfg_*.yaml")):
            candidates.append(child)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


# -----------------------------------------------------------------------------
# Smart Search Engine (Rerun Top Configs)
# -----------------------------------------------------------------------------

class SmartSearchEngine:
    """Engine for rerunning top configurations from previous experiments."""
    
    def __init__(self, archive_dir, top_n, sem):
        self.archive_dir = Path(archive_dir)
        self.top_n = top_n
        self.sem = sem
        clear_cache()

    def _best_summaries(self):
        """Find the best experiment summaries based on metrics."""
        summaries = list(self.archive_dir.glob("*__summary.json"))
        if not summaries:
            raise SystemExit("[smart_search] No bulk experiment summaries found.")
        
        scored = []
        for s in summaries:
            with open(s) as fp:
                data = json.load(fp)
            
            # Try to get the best metric across all models
            metric = None
            if "metric" in data:
                metrics = data["metric"]
                if isinstance(metrics, dict):
                    # Get MAPE or RMSE from any model
                    for model_metrics in metrics.values():
                        if isinstance(model_metrics, dict):
                            metric = model_metrics.get("MAPE") or model_metrics.get("RMSE")
                            if metric is not None:
                                break
            
            if metric is not None:
                scored.append((metric, Path(data["config_path"])))
      
        return [p for _, p in sorted(scored)[:self.top_n]]

    async def _run_again(self, cfg_path):
        """Rerun a configuration."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        
        cmd = [sys.executable, "search_mode.py", "--config", str(cfg_path)]
        async with self.sem:
            proc = await asyncio.create_subprocess_exec(
                *cmd, 
                stdout=asyncio.subprocess.DEVNULL, 
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            _, err = await proc.communicate()
            if proc.returncode != 0:
                print(f"[rerun {cfg_path.name}] failed")
                print(err.decode()[:300], file=sys.stderr)

    async def launch_top(self):
        """Launch top configurations."""
        cfgs = self._best_summaries()
        print(f"Rerunning top {len(cfgs)} configurations...")
        tasks = [asyncio.create_task(self._run_again(p)) for p in cfgs]
        await asyncio.gather(*tasks)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main_cli():
    """Command-line interface for hyperparameter search."""
    p = argparse.ArgumentParser(
        description="Bulk Hyperparameter Search for Time Series Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview exact counts without generating files
    python hypersearch.py plan --base-config config_templates/univariate.yaml --param-grid config/sweeps/sweep_30m_high_return.yaml

    # Run bulk search with smart experiment generation (creates isolated execution folder)
    python hypersearch.py bulk_search --base-config config/config.yaml --param-grid config/sweep_definition.yaml --output-dir experiments --max-parallel 4

    # Resume the latest bulk execution folder
    python hypersearch.py resume_bulk --configs-root experiments --max-parallel 4
  
  # Run bulk search with legacy factorial approach (NOT recommended - may create thousands of experiments)
  python hypersearch.py bulk_search --base-config config/config.yaml --param-grid config/sweep_definition.yaml --output-dir experiments --legacy
  
  # Rerun top configurations
  python hypersearch.py smart_search --from-experiments experiments --top-n 50 --max-parallel 2
        """
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # read-only exact-count preflight
    sp_plan = sub.add_parser("plan", help="Preview exact raw and unique experiment counts")
    sp_plan.add_argument("--base-config", required=True, help="Path to base configuration file")
    sp_plan.add_argument("--param-grid", required=True, help="Path to sweep definition file")
    sp_plan.add_argument("--legacy", action="store_true", help="Plan with legacy factorial expansion")
    sp_plan.add_argument("--json", action="store_true", help="Print the plan as JSON")

    # bulk
    sp = sub.add_parser("bulk_search", help="Run bulk hyperparameter search")
    sp.add_argument("--base-config", required=True, help="Path to base configuration file")
    sp.add_argument("--param-grid", required=True, help="Path to sweep definition file")
    sp.add_argument("--output-dir", default="generated_cfgs", help="Root directory for isolated execution folders")
    sp.add_argument("--session-name", default=None, help="Optional execution folder name inside --output-dir")
    sp.add_argument("--summary-dir", default=None, help="Optional summary directory (defaults to the execution folder)")
    sp.add_argument("--max-parallel", type=int, default=4, help="Maximum parallel experiments")
    sp.add_argument("--experiment-timeout-sec", type=int, default=0, help="Kill a single experiment if it runs longer than this (0 disables timeout)")
    sp.add_argument("--max-experiments", type=int, default=0, help="Refuse to run above this exact unique count")
    sp.add_argument("--legacy", action="store_true", help="Use legacy factorial expansion (WARNING: may generate many experiments)")
    sp.add_argument("--restart", action="store_true", help="Ignore resume and run all experiments from the beginning")
    sp.add_argument("--worthy-r2-threshold", type=float, default=0.6, help="Save a reproducible model bundle only for R2 scores strictly above this gate")

    # dedicated resume for latest/incomplete bulk execution
    sp_resume = sub.add_parser("resume_bulk", help="Resume the latest bulk hyperparameter execution")
    sp_resume.add_argument("--configs-root", default="generated_cfgs", help="Root directory containing execution folders")
    sp_resume.add_argument("--config-dir", default=None, help="Explicit execution config directory to resume")
    sp_resume.add_argument("--summaries-root", default=None, help="Optional summaries root when different from configs root")
    sp_resume.add_argument("--summary-dir", default=None, help="Explicit summary directory to use")
    sp_resume.add_argument("--max-parallel", type=int, default=4, help="Maximum parallel experiments")
    sp_resume.add_argument("--experiment-timeout-sec", type=int, default=0, help="Kill a single experiment if it runs longer than this (0 disables timeout)")
    sp_resume.add_argument("--max-experiments", type=int, default=0, help="Refuse to resume a session above this count")
    sp_resume.add_argument("--worthy-r2-threshold", type=float, default=0.6, help="Save a reproducible model bundle only for R2 scores strictly above this gate")

    # smart
    sp2 = sub.add_parser("smart_search", help="Rerun top configurations from previous experiments")
    sp2.add_argument("--from-experiments", default="experiments", help="Directory with experiment summaries")
    sp2.add_argument("--top-n", type=int, default=50, help="Number of top configs to rerun")
    sp2.add_argument("--max-parallel", type=int, default=2, help="Maximum parallel experiments")

    args = p.parse_args()

    if args.mode == "plan":
        base_cfg = yaml_load(args.base_config)
        sweep_cfg = yaml_load(args.param_grid)
        if args.legacy:
            sweep_cfg['smart_generation'] = False
        plan = build_sweep_plan(base_cfg, sweep_cfg)
        if args.json:
            print(json.dumps(plan, indent=2, default=str))
        else:
            print(f"Generation mode:    {plan['generation_mode']}")
            print(f"Raw generated:      {plan['raw_generated']}")
            print(f"Unique experiments: {plan['unique_experiments']}")
            print(f"Duplicates removed: {plan['duplicates_removed']}")
            print("Per model:")
            for model_name, count in plan['per_model'].items():
                print(f"  {model_name}: {count}")

    elif args.mode == "bulk_search":
        sem = asyncio.Semaphore(args.max_parallel)
        base_cfg = yaml_load(args.base_config)
        sweep_cfg = yaml_load(args.param_grid)

        session_name = args.session_name or datetime.now().strftime("bulk_%Y%m%d_%H%M%S")
        execution_dir = Path(args.output_dir) / session_name
        summary_dir = Path(args.summary_dir) if args.summary_dir else execution_dir
        print(f"Execution folder: {execution_dir}")
        print(f"Summary folder:   {summary_dir}")
        
        # Enable/disable smart generation
        if args.legacy:
            sweep_cfg['smart_generation'] = False
            print("WARNING: Using legacy factorial expansion - may generate thousands of experiments!")
        else:
            sweep_cfg['smart_generation'] = True
            print("Using smart experiment generation (recommended)")

        plan = build_sweep_plan(base_cfg, sweep_cfg)
        print(
            f"Preflight: {plan['unique_experiments']} unique experiments "
            f"({plan['duplicates_removed']} duplicates removed)"
        )
        if args.max_experiments and plan['unique_experiments'] > args.max_experiments:
            raise SystemExit(
                f"[bulk_search] Plan has {plan['unique_experiments']} unique experiments, "
                f"above --max-experiments {args.max_experiments}."
            )
        
        engine = BulkSearchEngine(
            base_cfg,
            sweep_cfg,
            execution_dir,
            sem,
            summary_dir=summary_dir,
            restart=args.restart,
            experiment_timeout_sec=(args.experiment_timeout_sec or None),
            max_experiments=(args.max_experiments or None),
            worthy_r2_threshold=args.worthy_r2_threshold,
        )
        asyncio.run(engine.launch_all())

    elif args.mode == "resume_bulk":
        sem = asyncio.Semaphore(args.max_parallel)
        resume_cfg_dir = Path(args.config_dir) if args.config_dir else _find_latest_execution_dir(Path(args.configs_root))
        if resume_cfg_dir is None:
            raise SystemExit("[resume_bulk] No execution folder with cfg_*.yaml found.")

        if args.summary_dir:
            resume_summary_dir = Path(args.summary_dir)
        elif args.summaries_root:
            resume_summary_dir = Path(args.summaries_root) / resume_cfg_dir.name
        else:
            resume_summary_dir = resume_cfg_dir

        print(f"Resuming config folder:  {resume_cfg_dir}")
        print(f"Using summary folder:    {resume_summary_dir}")

        engine = BulkSearchEngine(
            {},
            {"smart_generation": True},
            resume_cfg_dir,
            sem,
            summary_dir=resume_summary_dir,
            restart=False,
            experiment_timeout_sec=(args.experiment_timeout_sec or None),
            max_experiments=(args.max_experiments or None),
            worthy_r2_threshold=args.worthy_r2_threshold,
        )
        asyncio.run(engine.launch_all())

    elif args.mode == "smart_search":
        sem = asyncio.Semaphore(args.max_parallel)
        engine = SmartSearchEngine(args.from_experiments, args.top_n, sem)
        asyncio.run(engine.launch_top())


if __name__ == "__main__":
    main_cli()
