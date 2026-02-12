import os
import yaml
from datetime import datetime
from utils.logger import setup_logger
from utils.data_loader import load_data
from models.univariate_models import train_univariate_models
from models.multivariate_models import train_multivariate_models
from utils.evaluator import evaluate_models, save_best_model
from utils.reporter import CompactPDFReport 
from utils.evaluator import evaluate_models, save_best_model, generate_future_forecast # Import the PDF reporter
relative_path = 'config/snowflake_credentials.yaml'
data_output_path = 'data/output'
def main():
 
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_file)
    
    logger.info("Starting forecasting application")
    logger.info(f"Configuration: {config}")
    

    df = load_data(config['data_path'], config['date_col'], config['target_col'])
    

    all_model_results = {}
    

    if config['problem_type'] == "univariate":
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
        model_results = train_multivariate_models(df, config, logger)
        all_model_results.update(model_results)
    
    evaluation, future_forecasts = evaluate_models(all_model_results, df, config)
    save_best_model(all_model_results, evaluation, "model_files", logger)
    
    # Generate PDF report
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = f"{report_dir}/forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    pdf = CompactPDFReport(config)
    pdf.generate_report(all_model_results, evaluation, future_forecasts, report_path)
    logger.info(f"Compact PDF report generated at: {report_path}")
    
    logger.info("Application completed successfully")

if __name__ == "__main__":
    main()