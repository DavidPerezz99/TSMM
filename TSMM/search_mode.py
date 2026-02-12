"""
Search Mode Module

This module is used by hypersearch.py to run individual experiments.
It loads a configuration file and runs the forecasting pipeline,
saving metrics to the experiments folder.
"""

import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from utils.data_loader import load_data_cached
from models.univariate_models import train_univariate_models
from models.multivariate_models import train_multivariate_models
from utils.evaluator import evaluate_models, save_best_model
from utils.tracking import write_run_summary


def parse_cli():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml",
                   help="Path to the YAML config used for this run")
    return p.parse_args()


def _jsonify(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main():
    """Main entry point for search mode."""
    args = parse_cli()
    config_path = args.config
    
    print(f"[search_mode] Starting with config: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup logging (respect config log_dir when provided)
    log_dir = config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    cfg_stem = Path(config_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"search_{cfg_stem}_{timestamp}.log")
    logger = setup_logger(log_file)
    
    logger.info("Starting forecasting application (search mode)")
    logger.info(f"Configuration: {config}")
    
    # Load data
    try:
        df = load_data_cached(
            config['data_path'],
            config['date_col'],
            config['target_col'],
            config
        )
        print(f"[search_mode] Data loaded, shape: {df.shape}")
        print(df.iloc[-1])
        
        if config.get('records'):
            df = df.tail(config['records'])
        print(f"[search_mode] Using last {len(df)} records")
    except Exception as e:
        print(f"[search_mode] ERROR loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return
    
    all_model_results = {}
    
    # Train models
    try:
        if config['problem_type'] == "univariate":
            print("[search_mode] Training univariate models")
            model_results = train_univariate_models(
                df, 
                config, 
                logger, 
                config['input_features'],
                config['target_features'],
                config['exclude_cols'],
                config['n_steps'],
                config['m_steps'],
                config['split_ratio']
            )
            all_model_results.update(model_results)
            print(f"[search_mode] Univariate models trained: {list(model_results.keys())}")
        else:
            print("[search_mode] Training multivariate models")
            model_results = train_multivariate_models(df, config, logger)
            all_model_results.update(model_results)
            print(f"[search_mode] Multivariate models trained: {list(model_results.keys())}")
    except Exception as e:
        print(f"[search_mode] ERROR training models: {str(e)}")
        logger.error(f"Error training models: {str(e)}")
        return
    
    # Evaluate models
    try:
        logger.info("Starting evaluation")
        print("[search_mode] Starting evaluation")
        evaluation, future_forecasts = evaluate_models(all_model_results, df, config)
        print(f"[search_mode] Evaluation completed for models: {list(evaluation.keys())}")
    except Exception as e:
        print(f"[search_mode] ERROR evaluating models: {str(e)}")
        logger.error(f"Error evaluating models: {str(e)}")
        return

    # Save run summary
    try:
        clean_metrics = {}
        any_non_empty = False

        # Use evaluation metrics when available; otherwise fall back to
        # training-time metrics or model-level errors so each model
        # contributes something useful to the summary.
        for model_name, model_result in all_model_results.items():
            eval_data = evaluation.get(model_name, {}) or {}
            metrics_block = eval_data.get("metrics", {}) or {}

            # If evaluation failed for this model, surface the error text
            if (not metrics_block) and isinstance(eval_data, dict) and eval_data.get("error"):
                metrics_block = {"error": str(eval_data.get("error"))}

            # Fallback: if evaluation produced no metrics and no explicit
            # error, use training metrics (if any) so the run summary
            # still contains quantitative information.
            if not metrics_block:
                train_metrics = model_result.get("metrics", {}) or {}
                if train_metrics:
                    metrics_block = train_metrics

            # If we still have nothing and the model itself failed during
            # training, propagate that training error so the summary does
            # not end up with an empty metrics object.
            if (not metrics_block) and isinstance(model_result, dict) and model_result.get("error"):
                metrics_block = {"error": str(model_result.get("error"))}

            if metrics_block:
                any_non_empty = True

            clean_metrics[model_name] = _jsonify(metrics_block)

        print("[search_mode] Writing run summary to experiments folder")
        print(f"[search_mode] Metrics to write for models: {list(clean_metrics.keys())}")

        # Mark summaries with no usable metrics via a suffix for easier filtering
        summary_suffix = None if any_non_empty else "no_metrics"

        write_run_summary(
            config_path=config_path,
            metrics=clean_metrics,
            suffix=summary_suffix
        )
        print("[search_mode] Run summary written successfully")
    except Exception as e:
        print(f"[search_mode] ERROR writing run summary: {str(e)}")
        logger.error(f"Error writing run summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Save best model
    try:
        save_best_model(all_model_results, evaluation, "model_files", logger)
        print("[search_mode] Best model saved")
    except Exception as e:
        print(f"[search_mode] ERROR saving best model: {str(e)}")
        logger.error(f"Error saving best model: {str(e)}")
    
    logger.info("Application completed successfully")
    print("[search_mode] Application completed successfully")


if __name__ == "__main__":
    main()