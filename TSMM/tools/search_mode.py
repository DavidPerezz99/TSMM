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

# Add project root to path when launched as tools/search_mode.py.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import setup_logger
from utils.data_loader import load_data_cached
from models.univariate_models import train_univariate_models
from models.multivariate_models import train_multivariate_models
from utils.evaluator import evaluate_models, save_best_model
from utils.experiment_artifacts import export_worthy_experiment_bundle
from utils.tracking import write_run_summary


def parse_cli():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml",
                   help="Path to the YAML config used for this run")
    p.add_argument(
        "--summary-dir",
        default="experiments",
        help="Directory where run summaries are written"
    )
    p.add_argument(
        "--bulk-search",
        action="store_true",
        help="Run in bulk-search mode (skip model artifact persistence)"
    )
    p.add_argument("--worthy-artifact-dir", default=None,
                   help="Bulk-search directory for reproducible bundles above the R2 gate")
    p.add_argument("--worthy-r2-threshold", type=float, default=0.6,
                   help="Persist a bundle only when primary-target R2 is strictly above this value")
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


def _write_failure_summary(config_path: str, stage: str, error: Exception, summary_dir: str):
    """Write a failure summary so bulk-search always has a run record."""
    metrics = {
        "run_error": {
            "stage": stage,
            "error": str(error)
        }
    }
    write_run_summary(
        config_path=config_path,
        metrics=_jsonify(metrics),
        summary_dir=summary_dir,
        suffix="failed"
    )


def main():
    """Main entry point for search mode."""
    args = parse_cli()
    config_path = args.config
    summary_dir = args.summary_dir
    is_bulk_search = bool(args.bulk_search)
    
    print(f"[search_mode] Starting with config: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['_bulk_search'] = is_bulk_search
    
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
        try:
            _write_failure_summary(config_path, "data_loading", e, summary_dir)
            print("[search_mode] Failure summary written")
        except Exception as summary_err:
            print(f"[search_mode] ERROR writing failure summary: {summary_err}")
            logger.error(f"Error writing failure summary: {summary_err}")
        sys.exit(11)
    
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
        try:
            _write_failure_summary(config_path, "training", e, summary_dir)
            print("[search_mode] Failure summary written")
        except Exception as summary_err:
            print(f"[search_mode] ERROR writing failure summary: {summary_err}")
            logger.error(f"Error writing failure summary: {summary_err}")
        sys.exit(12)
    
    # Evaluate models
    try:
        logger.info("Starting evaluation")
        print("[search_mode] Starting evaluation")
        evaluation, future_forecasts = evaluate_models(all_model_results, df, config)
        print(f"[search_mode] Evaluation completed for models: {list(evaluation.keys())}")
    except Exception as e:
        print(f"[search_mode] ERROR evaluating models: {str(e)}")
        logger.error(f"Error evaluating models: {str(e)}")
        try:
            _write_failure_summary(config_path, "evaluation", e, summary_dir)
            print("[search_mode] Failure summary written")
        except Exception as summary_err:
            print(f"[search_mode] ERROR writing failure summary: {summary_err}")
            logger.error(f"Error writing failure summary: {summary_err}")
        sys.exit(13)

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
            summary_dir=summary_dir,
            suffix=summary_suffix
        )
        print("[search_mode] Run summary written successfully")
    except Exception as e:
        print(f"[search_mode] ERROR writing run summary: {str(e)}")
        logger.error(f"Error writing run summary: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(14)
    
    # Bulk runs retain model files only for experiments that clear the R2 gate.
    if is_bulk_search:
        if args.worthy_artifact_dir:
            try:
                bundle_path = export_worthy_experiment_bundle(
                    models=all_model_results, evaluation=evaluation,
                    future_forecasts=future_forecasts, config=config,
                    config_path=config_path, artifact_root=args.worthy_artifact_dir,
                    r2_threshold=args.worthy_r2_threshold, logger=logger, dataframe=df,
                )
                if bundle_path is None:
                    print(f"[search_mode] R2 gate not met; no artifacts persisted (requires R2 > {args.worthy_r2_threshold})")
                else:
                    print(f"[search_mode] Worthy artifact bundle saved: {bundle_path}")
            except Exception as e:
                logger.error(f"Error exporting worthy artifact bundle: {e}")
                _write_failure_summary(config_path, "artifact_export", e, summary_dir)
                sys.exit(15)
        else:
            print("[search_mode] Bulk-search mode: worthy artifact export is not configured")
    else:
        try:
            save_best_model(all_model_results, evaluation, "model_files", logger, config=config)
            print("[search_mode] Best model saved")
        except Exception as e:
            print(f"[search_mode] ERROR saving best model: {str(e)}")
            logger.error(f"Error saving best model: {str(e)}")
    
    logger.info("Application completed successfully")
    print("[search_mode] Application completed successfully")


if __name__ == "__main__":
    main()
