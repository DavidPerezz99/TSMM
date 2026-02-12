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
import yaml
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
import logging
from utils.data_loader import load_data_cached
from models.univariate_models import train_univariate_models
from models.multivariate_models import train_multivariate_models
from utils.reporter import CompactPDFReport
from utils.evaluator import evaluate_models, save_best_model
from utils.metrics_saver import save_all_models_metrics, save_forecast_to_file
from utils.interpretability import add_interpretability


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
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


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
    save_best_model(all_model_results, evaluation, "model_files", logger)

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


def main():
    """Main entry point for the forecasting application."""
    
    # Load configuration
    config_path = os.environ.get('CONFIG_PATH', 'config/config.yaml')
    config = load_config(config_path)
    
    # Setup logging
    log_dir = config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_file)
    
    try:
        # Run forecasting pipeline
        results = run_forecasting_pipeline(config, logger)
        
        # Create output directory
        output_dir = create_output_directory(config)
        
        # Determine output format
        output_config = config.get('output', {})
        output_format = output_config.get('format', 'pdf').lower()
        
        # Generate output based on format
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
        
        # Always save metrics regardless of output format
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
