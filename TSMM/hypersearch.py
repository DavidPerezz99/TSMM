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
import itertools
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
import yaml
import numpy as np
from utils.cache_management import clear_cache


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
    # Separate parameters by category
    flat_params = {}  # e.g., model_type, hidden_size, epochs
    stacks_params = {}  # e.g., stacks_config.0.type, stacks_config.0.num_blocks
    blackbox_params = {}  # e.g., blackbox_config.num_blocks, blackbox_config.num_layers
    
    for key, values in nbeats_params.items():
        if 'stacks_config' in key:
            stacks_params[key] = values
        elif 'blackbox_config' in key:
            blackbox_params[key] = values
        else:
            flat_params[key] = values
    
    variants = []
    
    # Build combinations of flat parameters
    flat_keys = list(flat_params.keys())
    flat_values = [flat_params[k] for k in flat_keys]
    
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
                variant = base_variant.copy()
                for key, value in zip(stacks_keys, stacks_combo):
                    set_nested_value(variant, key, value)
                variants.append(variant)
        
        elif model_type == 'blackbox' and blackbox_params:
            # Build blackbox_config variants
            blackbox_keys = list(blackbox_params.keys())
            blackbox_values = [blackbox_params[k] for k in blackbox_keys]
            
            for blackbox_combo in itertools.product(*blackbox_values):
                variant = base_variant.copy()
                for key, value in zip(blackbox_keys, blackbox_combo):
                    set_nested_value(variant, key, value)
                variants.append(variant)
        else:
            # No nested params or unknown mode
            variants.append(base_variant)
    
    return variants


def generate_smart_experiments(
    base_cfg: Dict[str, Any],
    sweep_cfg: Dict[str, Any]
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
    all_models = models_to_run.get('univariate', []) + models_to_run.get('multivariate', [])
    
    for model_name in all_models:
        # Get parameters specific to this model
        model_specific = model_params.get(model_name, {})
        
        # Build valid model configuration variants
        model_variants = build_model_config_variants(model_name, model_specific)
        
        print(f"  {model_name}: {len(model_variants)} model variants × {len(global_combinations)} global combinations")
        
        # Combine global and model-specific parameters
        for global_combo in global_combinations:
            for model_variant in model_variants:
                experiment = base_cfg.copy()
                
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
                        exp_with_set = experiment.copy()
                        exp_with_set.update(rec)
                        experiment_count += 1
                        yield exp_with_set
                else:
                    experiment_count += 1
                    yield experiment
    
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
                experiment = base_cfg.copy()
                deep_merge_dict(experiment, patch)
                experiment.update(rec)
                yield experiment
        else:
            experiment = base_cfg.copy()
            deep_merge_dict(experiment, patch)
            yield experiment


# -----------------------------------------------------------------------------
# Bulk Search Engine
# -----------------------------------------------------------------------------

class BulkSearchEngine:
    """Engine for running bulk hyperparameter searches."""
    
    def __init__(self, base_cfg, sweep_cfg, out_dir, sem):
        self.base_cfg = base_cfg
        self.sweep_cfg = sweep_cfg
        self.out_dir = Path(out_dir)
        self.sem = sem
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = []
        clear_cache()

    def expand_grid(self):
        """Yield {param: value} dictionaries for every combination."""
        # Check if smart generation is enabled (default: True)
        use_smart = self.sweep_cfg.get('smart_generation', True)
        
        if use_smart:
            print("Using SMART generation (recommended)")
            return generate_smart_experiments(self.base_cfg, self.sweep_cfg)
        else:
            print("WARNING: Using LEGACY factorial expansion (may generate many experiments)")
            return generate_factorial_experiments(self.base_cfg, self.sweep_cfg)

    def materialize_configs(self):
        """Generate and save experiment configuration files."""
        cfg_paths = []
        for idx, param_patch in enumerate(self.expand_grid(), 1):
            cfg = self.base_cfg.copy()
            deep_merge_dict(cfg, param_patch)
            cfg_name = f"cfg_{idx:05d}_{uuid.uuid4().hex[:6]}.yaml"
            cfg_path = self.out_dir / cfg_name
            yaml_dump(cfg, cfg_path)
            cfg_paths.append(cfg_path)
        return cfg_paths

    async def _run_one(self, cfg_path):
        """Run a single experiment."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        
        cmd = [sys.executable, "search_mode.py", "--config", str(cfg_path)]
        async with self.sem:
            proc = await asyncio.create_subprocess_exec(
                *cmd, 
                stdout=asyncio.subprocess.PIPE, 
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            out, err = await proc.communicate()
            status = "OK" if proc.returncode == 0 else "FAIL"
            print(f"[{cfg_path.name}] finished -> {status}")
            if status == "FAIL":
                print(err.decode()[:500], file=sys.stderr)

    async def launch_all(self):
        """Launch all experiments."""
        cfg_paths = self.materialize_configs()
        print(f"\nLaunching {len(cfg_paths)} experiments...")
        print("-" * 60)
        tasks = [asyncio.create_task(self._run_one(p)) for p in cfg_paths]
        await asyncio.gather(*tasks)


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
  # Run bulk search with smart experiment generation (recommended)
  python hypersearch.py bulk_search --base-config config/config.yaml --param-grid config/sweep_definition.yaml --output-dir experiments --max-parallel 4
  
  # Run bulk search with legacy factorial approach (NOT recommended - may create thousands of experiments)
  python hypersearch.py bulk_search --base-config config/config.yaml --param-grid config/sweep_definition.yaml --output-dir experiments --legacy
  
  # Rerun top configurations
  python hypersearch.py smart_search --from-experiments experiments --top-n 50 --max-parallel 2
        """
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # bulk
    sp = sub.add_parser("bulk_search", help="Run bulk hyperparameter search")
    sp.add_argument("--base-config", required=True, help="Path to base configuration file")
    sp.add_argument("--param-grid", required=True, help="Path to sweep definition file")
    sp.add_argument("--output-dir", default="generated_cfgs", help="Directory for generated configs")
    sp.add_argument("--max-parallel", type=int, default=4, help="Maximum parallel experiments")
    sp.add_argument("--legacy", action="store_true", help="Use legacy factorial expansion (WARNING: may generate many experiments)")

    # smart
    sp2 = sub.add_parser("smart_search", help="Rerun top configurations from previous experiments")
    sp2.add_argument("--from-experiments", default="experiments", help="Directory with experiment summaries")
    sp2.add_argument("--top-n", type=int, default=50, help="Number of top configs to rerun")
    sp2.add_argument("--max-parallel", type=int, default=2, help="Maximum parallel experiments")

    args = p.parse_args()
    sem = asyncio.Semaphore(args.max_parallel)

    if args.mode == "bulk_search":
        base_cfg = yaml_load(args.base_config)
        sweep_cfg = yaml_load(args.param_grid)
        
        # Enable/disable smart generation
        if args.legacy:
            sweep_cfg['smart_generation'] = False
            print("WARNING: Using legacy factorial expansion - may generate thousands of experiments!")
        else:
            sweep_cfg['smart_generation'] = True
            print("Using smart experiment generation (recommended)")
        
        engine = BulkSearchEngine(base_cfg, sweep_cfg, args.output_dir, sem)
        asyncio.run(engine.launch_all())

    elif args.mode == "smart_search":
        engine = SmartSearchEngine(args.from_experiments, args.top_n, sem)
        asyncio.run(engine.launch_top())


if __name__ == "__main__":
    main_cli()