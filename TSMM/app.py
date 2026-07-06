"""
Time Series Forecasting Application - Single Run Mode

This application trains forecasting models based on configuration settings
and generates outputs in the specified format (PDF report, CSV, or Parquet).

Features:
- Config-driven model selection via config.yaml
- Configurable output format (PDF, CSV, Parquet)
- Comprehensive evaluation metrics with confusion matrices
- Confidence level prediction for forecasts
- Forecast explosion detection
"""

import os
import sys
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

from utils.logger import setup_logger
import logging
from utils.data_loader import load_data_cached
from models.univariate_models import train_univariate_models
from models.multivariate_models import train_multivariate_models
from utils.reporter import CompactPDFReport
from utils.evaluator import evaluate_models, save_best_model
from utils.metrics_saver import save_all_models_metrics, save_forecast_to_file
from utils.interpretability import add_interpretability
from utils.investing_agent import load_trading_config
from utils.trading_job import start_trading_job, resume_trading_job, stop_trading_job, kill_trading_job, resolve_trading_start_request
from utils.live_data import bootstrap_master_on_backend_start, sync_dataset_source_from_master, resolve_tiingo_token_candidates


def _resolve_app_path(path_value: str) -> str:
    path_str = str(path_value or '').strip()
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(BASE_DIR, path_str)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="TSMM Forecasting and Trading Job CLI")
    parser.add_argument("command", nargs="?", default="forecast", choices=["forecast", "trading-job"])
    parser.add_argument("action", nargs="?", default="start", choices=["start", "resume", "stop", "kill"])
    parser.add_argument("--plan-model", default=None, help="Optional model name to force for Agent A plan")
    parser.add_argument("--job-id", default=None, help="Optional trading job id for start/resume/stop/kill")
    parser.add_argument("--submission-mode", default=None, choices=["programmed", "market"], help="Optional order submission mode override for trading-job start")
    parser.add_argument("--autonomous-trigger", default=None, choices=["mandatory_session", "autonomous_followup", "opposing_countertrade"], help="Internal autonomous launcher context for trading-job start")
    return parser.parse_args()


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
    
    Returns:
    --------
    dict
        Configuration dictionary
    """
    config_path = _resolve_app_path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_output_directory(config: dict) -> str:
    """
    Create output directory if it doesn't exist.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    str
        Path to output directory
    """
    output_config = config.get('output', {})
    output_dir = output_config.get('directory', 'reports')
    output_dir = _resolve_app_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _extract_trading_job_state(job_result: dict) -> dict:
    if not isinstance(job_result, dict):
        return {}
    state = job_result.get("state")
    if isinstance(state, dict):
        return state
    if isinstance(job_result.get("job_id"), str) or "status" in job_result or "stage" in job_result:
        return job_result
    return {}


def _summarize_trading_job_result(job_result: dict) -> dict:
    state = _extract_trading_job_state(job_result)
    status = str(state.get("status") or "").strip().lower() if isinstance(state, dict) else ""

    if isinstance(job_result, dict) and "ok" in job_result:
        ok_value = bool(job_result.get("ok"))
    elif status:
        ok_value = status not in {"failed", "killed", "stopped"}
    else:
        ok_value = False

    summary = {
        "ok": ok_value,
    }

    if isinstance(job_result, dict):
        message = str(job_result.get("message") or "").strip()
        if message:
            summary["message"] = message
        error = str(job_result.get("error") or "").strip()
        if error:
            summary["error"] = error

    if not isinstance(state, dict) or not state:
        return summary

    summary.update(
        {
            "job_id": str(state.get("job_id") or ""),
            "status": str(state.get("status") or ""),
            "stage": str(state.get("stage") or ""),
            "mode": str(state.get("mode") or ""),
            "order_submission_mode": str(state.get("order_submission_mode") or ""),
            "runner_pid": int(state.get("runner_pid") or 0),
            "closed_reason": str(state.get("closed_reason") or ""),
            "state_path": str(state.get("state_path") or ""),
        }
    )

    position = state.get("position") if isinstance(state.get("position"), dict) else {}
    order = state.get("order") if isinstance(state.get("order"), dict) else {}
    position_ticket = int(position.get("ticket") or 0)
    order_ticket = int(order.get("order_ticket") or 0)
    if position_ticket > 0:
        summary["position_ticket"] = position_ticket
    if order_ticket > 0:
        summary["order_ticket"] = order_ticket
    return summary


def generate_output_filename(config: dict, output_format: str) -> str:
    """
    Generate output filename based on configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    output_format : str
        Output file format
    
    Returns:
    --------
    str
        Generated filename
    """
    output_config = config.get('output', {})
    prefix = output_config.get('filename_prefix', 'forecast_report')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_format.lower() == 'pdf':
        extension = 'pdf'
    elif output_format.lower() == 'csv':
        extension = 'csv'
    elif output_format.lower() == 'parquet':
        extension = 'parquet'
    else:
        extension = 'csv'
    
    return f"{prefix}_{timestamp}.{extension}"


def run_forecasting_pipeline(config: dict, logger: logging.Logger) -> dict:
    """
    Run the complete forecasting pipeline.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    dict
        Pipeline results including models, evaluation, and forecasts
    """
    logger.info("Starting forecasting application")
    logger.info(f"Configuration: {config}")
    
    # Load data
    logger.info("Loading data...")
    df = load_data_cached(
        config['data_path'],
        config['date_col'],
        config['target_col'],
        config
    )
    print(f"Last data point:\n{df.iloc[-1]}")
    
    # Limit records if specified
    if config.get('records'):
        df = df.tail(config['records'])
        logger.info(f"Using last {len(df)} records")
    
    # Store last date for forecast timestamping
    last_date = df.index[-1]
    logger.info(f"Data range: {df.index[0]} to {last_date}")
    
    # Train models based on problem type
    all_model_results = {}
    
    if config['problem_type'] == "univariate":
        logger.info("Training univariate models...")
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
    else:
        logger.info("Training multivariate models...")
        model_results = train_multivariate_models(df, config, logger)
        all_model_results.update(model_results)
    
    logger.info(f"Models trained: {list(all_model_results.keys())}")
    
    # Evaluate models
    logger.info("Starting evaluation...")
    evaluation, future_forecasts = evaluate_models(all_model_results, df, config)

    # Add interpretability information (best-effort, non-fatal)
    logger.info("Computing interpretability metrics for trained models...")
    add_interpretability(all_model_results, df, config, logger)

    # Save best model
    save_best_model(all_model_results, evaluation, "model_files", logger, config=config)

    return {
        'models': all_model_results,
        'evaluation': evaluation,
        'future_forecasts': future_forecasts,
        'last_date': last_date,
        'df': df
    }


def generate_pdf_report(
    config: dict,
    results: dict,
    output_path: str,
    logger: logging.Logger
) -> str:
    """
    Generate PDF report from forecasting results.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    results : dict
        Pipeline results
    output_path : str
        Path to save the PDF report
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    str
        Path to the generated report
    """
    logger.info(f"Generating PDF report at: {output_path}")
    
    pdf = CompactPDFReport(config)
    pdf.generate_report(
        results['models'],
        results['evaluation'],
        results['future_forecasts'],
        output_path
    )
    
    logger.info(f"PDF report generated successfully")
    return output_path


def generate_table_output(
    config: dict,
    results: dict,
    output_path: str,
    output_format: str,
    logger: logging.Logger
) -> str:
    """
    Generate table output (CSV or Parquet) from forecasting results.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    results : dict
        Pipeline results
    output_path : str
        Path to save the output file
    output_format : str
        Output format ('csv' or 'parquet')
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    str
        Path to the generated file
    """
    logger.info(f"Generating {output_format.upper()} output at: {output_path}")
    
    # Extract confidence levels from evaluation
    confidence_levels = {}
    for model_name, eval_data in results['evaluation'].items():
        if 'confidence_levels' in eval_data and eval_data['confidence_levels']:
            confidence_levels[model_name] = eval_data['confidence_levels']
    
    save_forecast_to_file(
        results['future_forecasts'],
        output_path,
        output_format,
        config,
        df_last_date=results['last_date'],
        confidence_levels=confidence_levels
    )
    
    logger.info(f"{output_format.upper()} output generated successfully")
    return output_path


def save_metrics(
    results: dict,
    output_dir: str,
    logger: logging.Logger
) -> str:
    """
    Save evaluation metrics to JSON file.
    
    Parameters:
    -----------
    results : dict
        Pipeline results
    output_dir : str
        Directory to save metrics
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    str
        Path to the saved metrics file
    """
    logger.info("Saving evaluation metrics...")
    
    metrics_path = save_all_models_metrics(
        results['evaluation'],
        output_dir
    )
    
    logger.info(f"Metrics saved to: {metrics_path}")
    return metrics_path


def maybe_sync_master_on_backend_start(active_config: dict, logger: logging.Logger):
    """Auto-sync the master source first, then refresh the active config source if needed."""
    trading_cfg_path = os.environ.get('TRADING_CONFIG_PATH', 'config/trading_agent.yaml')
    trading_cfg = load_trading_config(trading_cfg_path)
    dash_cfg = (trading_cfg.get('dashboard') or {})

    status_path = str(
        dash_cfg.get('startup_status_path')
        or os.path.join('reports', 'runtime', 'startup_sync_status.json')
    )

    def _save_startup_status(payload: dict):
        try:
            os.makedirs(os.path.dirname(status_path) or '.', exist_ok=True)
            with open(status_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning("Could not write startup sync status file %s: %s", status_path, e)

    if not bool(dash_cfg.get('startup_sync_enabled', True)):
        logger.info("Startup master sync disabled in trading config.")
        _save_startup_status({
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'enabled': False,
            'ok': False,
            'reason': 'startup_sync_disabled',
            'status_path': status_path,
        })
        return

    master_path = str(dash_cfg.get('master_table_path') or dash_cfg.get('raw_data_path') or '').strip()
    if not master_path:
        logger.warning("Startup master sync skipped: missing master path in dashboard config")
        _save_startup_status({
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'enabled': True,
            'ok': False,
            'reason': 'missing_master_path',
            'status_path': status_path,
        })
        return

    token_env = str(dash_cfg.get('tiingo_token_env', 'TIINGO_API_TOKEN')).strip() or 'TIINGO_API_TOKEN'
    token_envs_cfg = dash_cfg.get('tiingo_token_envs')
    token_rotation_state_path = str(dash_cfg.get('tiingo_token_rotation_state_path') or '').strip() or None
    token = os.environ.get(token_env, '')
    token_candidates = resolve_tiingo_token_candidates(
        token_env=token_env,
        token_envs=token_envs_cfg,
        token=token,
    )
    if not token_candidates:
        configured_envs = [token_env]
        if isinstance(token_envs_cfg, str):
            configured_envs.extend([x.strip() for x in token_envs_cfg.replace(';', ',').split(',') if str(x).strip()])
        elif isinstance(token_envs_cfg, (list, tuple, set)):
            configured_envs.extend([str(x).strip() for x in token_envs_cfg if str(x).strip()])
        configured_envs = list(dict.fromkeys(configured_envs))
        logger.warning("Startup master sync skipped: missing Tiingo token in configured env vars %s", configured_envs)
        _save_startup_status({
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'enabled': True,
            'ok': False,
            'reason': f"missing_token_envs:{','.join(configured_envs)}",
            'status_path': status_path,
        })
        return

    symbol = str(dash_cfg.get('tiingo_symbol', 'xauusd')).strip().lower()
    rate = str(dash_cfg.get('tiingo_rate', '1min')).strip()
    max_pulls = int(dash_cfg.get('startup_max_pulls', 2) or 2)
    freshness_lag_minutes = int(dash_cfg.get('startup_freshness_lag_minutes', 20) or 20)

    logger.info(
        "Running startup master sync: path=%s symbol=%s rate=%s max_pulls=%s freshness_lag_minutes=%s",
        master_path,
        symbol,
        rate,
        max_pulls,
        freshness_lag_minutes,
    )

    result = bootstrap_master_on_backend_start(
        master_table_path=master_path,
        rate=rate,
        symbol=symbol,
        token=token,
        max_pulls=max_pulls,
        freshness_lag_minutes=freshness_lag_minutes,
        token_env=token_env,
        token_envs=token_envs_cfg,
        token_rotation_state_path=token_rotation_state_path,
    )

    active_sync_result = None
    active_data_path = str((active_config or {}).get('data_path') or '').strip()
    active_sql_symbol = str(
        (active_config or {}).get('sql_symbol')
        or (active_config or {}).get('symbol')
        or dash_cfg.get('sql_symbol')
        or dash_cfg.get('tiingo_symbol')
        or 'xauusd'
    ).strip()
    active_records = int((active_config or {}).get('records', 5000) or 5000)
    active_rolling_windows = list((active_config or {}).get('rolling_windows') or [2, 7, 30, 60])
    active_n_steps = int((active_config or {}).get('n_steps', 1) or 1)
    active_horizon = int((active_config or {}).get('horizon', 1) or 1)
    active_tf_minutes = (active_config or {}).get('data_timeframe_minutes')

    if active_data_path and not active_data_path.lower().endswith(('.db', '.sqlite')):
        logger.info("Refreshing active config source file from updated master: %s", active_data_path)
        active_sync_result = sync_dataset_source_from_master(
            master_table_path=master_path,
            output_path=active_data_path,
            timeframe_minutes=active_tf_minutes,
            records=active_records,
            rolling_windows=active_rolling_windows,
            n_steps=active_n_steps,
            horizon=active_horizon,
            symbol=active_sql_symbol,
            logger=logger,
        )
    elif active_data_path:
        logger.info("Active config uses SQLite source %s; master sync already updated it if paths match.", active_data_path)
        if os.path.normcase(os.path.abspath(active_data_path)) == os.path.normcase(os.path.abspath(master_path)):
            active_sync_result = {
                'updated': bool(result.get('ok', False)),
                'output_path': active_data_path,
                'source': 'master_sync_shared_sqlite',
                'latest_date': result.get('latest_date'),
            }
        else:
            active_sync_result = {
                'updated': False,
                'output_path': active_data_path,
                'source': 'independent_sqlite_not_synced',
            }

    logger.info("Startup master sync result: %s", result)
    _save_startup_status({
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'enabled': True,
        'ok': bool(result.get('ok', False)),
        'master_path': master_path,
        'symbol': symbol,
        'rate': rate,
        'max_pulls': max_pulls,
        'freshness_lag_minutes': freshness_lag_minutes,
        'result': result,
        'token_env': token_env,
        'configured_token_envs': [item.get('env') for item in token_candidates],
        'active_config_data_path': active_data_path,
        'active_config_sync_result': active_sync_result,
        'status_path': status_path,
    })
    if not bool(result.get('ok', False)):
        logger.warning("Startup master sync did not reach aligned state after attempts")
    if isinstance(active_sync_result, dict) and not bool(active_sync_result.get('updated', False)):
        logger.warning("Active config source refresh did not update target: %s", active_sync_result)


def main():
    """Main entry point for the forecasting application."""
    args = parse_cli_args()

    # Normalize relative paths to the app directory so launches from parent
    # folders or other shells behave consistently.
    os.chdir(BASE_DIR)
    
    # Load configuration
    config_path = os.environ.get('CONFIG_PATH', 'config/config.yaml')
    config = load_config(config_path)
    output_dir = create_output_directory(config)
    
    # Setup logging
    log_dir = config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_file)
    
    try:
        # Trading job manual stop command does not require a forecast run.
        if args.command == "trading-job" and args.action in {"stop", "kill"}:
            trading_cfg_path = os.environ.get('TRADING_CONFIG_PATH', 'config/trading_agent.yaml')
            trading_cfg = load_trading_config(trading_cfg_path)
            output_dir = create_output_directory(config)
            action_result = stop_trading_job(output_dir, trading_cfg, job_id=args.job_id) if args.action == "stop" else kill_trading_job(output_dir, trading_cfg, job_id=args.job_id)
            print(str(action_result.get("message") or f"Trading job {args.action} request completed."))
            if bool(action_result.get("ok", False)):
                logger.info("Trading job %s requested. result=%s", args.action, action_result)
                return
            logger.warning("Trading job %s requested but no active jobs were found. requested_job_id=%s", args.action, args.job_id)
            return

        trading_cfg = None
        request_context = {}
        effective_config = dict(config)
        if args.command == "trading-job":
            trading_cfg_path = os.environ.get('TRADING_CONFIG_PATH', 'config/trading_agent.yaml')
            trading_cfg = load_trading_config(trading_cfg_path)
            if args.action == "start":
                request_context = resolve_trading_start_request(
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    requested_submission_mode=args.submission_mode,
                )
                if args.autonomous_trigger:
                    request_context["autonomous_trigger"] = str(args.autonomous_trigger).strip().lower()
                    request_context["autonomous"] = True
                effective_config["data_timeframe_minutes"] = int(request_context.get("grounding_timeframe_minutes") or effective_config.get("data_timeframe_minutes", 420) or 420)

        # Startup sync: refresh master from Tiingo before backend pipeline continues.
        maybe_sync_master_on_backend_start(effective_config, logger)

        # Run forecasting pipeline
        results = run_forecasting_pipeline(effective_config, logger)
        
        # Trading-job start/resume mode.
        if args.command == "trading-job":
            trading_cfg = trading_cfg or load_trading_config(os.environ.get('TRADING_CONFIG_PATH', 'config/trading_agent.yaml'))

            if args.action == "resume":
                job_result = resume_trading_job(
                    app_config=effective_config,
                    trading_cfg=trading_cfg,
                    output_dir=output_dir,
                    logger=logger,
                    job_id=args.job_id,
                )
            else:
                job_result = start_trading_job(
                    app_config=effective_config,
                    results=results,
                    trading_cfg=trading_cfg,
                    output_dir=output_dir,
                    logger=logger,
                    selected_model=args.plan_model,
                    job_id=args.job_id,
                    submission_mode_override=str((request_context.get("effective_submission_mode") or args.submission_mode or "programmed")),
                    request_context=request_context,
                )

            print("\n" + "=" * 60)
            print("TRADING JOB RESULT")
            print("=" * 60)
            summary = _summarize_trading_job_result(job_result)
            print(json.dumps(summary, indent=2, ensure_ascii=True))
            print("=" * 60)
            logger.info("Trading job completed with result summary: %s", summary)
            logger.debug("Trading job completed with full result: %s", job_result)
            return

        # Standard forecast/report mode.
        output_config = config.get('output', {})
        output_format = output_config.get('format', 'pdf').lower()

        if output_format == 'pdf':
            output_filename = generate_output_filename(config, 'pdf')
            output_path = os.path.join(output_dir, output_filename)
            generate_pdf_report(config, results, output_path, logger)

        elif output_format in ['csv', 'parquet']:
            output_filename = generate_output_filename(config, output_format)
            output_path = os.path.join(output_dir, output_filename)
            generate_table_output(config, results, output_path, output_format, logger)

        else:
            logger.warning(f"Unknown output format: {output_format}. Defaulting to PDF.")
            output_filename = generate_output_filename(config, 'pdf')
            output_path = os.path.join(output_dir, output_filename)
            generate_pdf_report(config, results, output_path, logger)

        metrics_path = save_metrics(results, output_dir, logger)
        
        # Print summary
        print("\n" + "="*60)
        print("FORECASTING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Output format: {output_format.upper()}")
        print(f"Output file: {output_path}")
        print(f"Metrics file: {metrics_path}")
        print(f"Models trained: {list(results['models'].keys())}")
        print("\nModel Performance Summary:")
        for model_name, eval_data in results['evaluation'].items():
            if 'metrics' in eval_data:
                metrics = eval_data['metrics']
                print(f"  {model_name}:")
                print(f"    MAE: {metrics.get('MAE', 'N/A'):.4f}" if isinstance(metrics.get('MAE'), (int, float)) else f"    MAE: N/A")
                print(f"    R²: {metrics.get('R2', 'N/A'):.4f}" if isinstance(metrics.get('R2'), (int, float)) else f"    R²: N/A")
        print("="*60)
        
        logger.info("Application completed successfully")
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        print(f"\nERROR: Application failed - {str(e)}")
        raise


if __name__ == "__main__":
    main()
