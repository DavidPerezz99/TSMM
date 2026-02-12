"""
Reporter Module

This module provides PDF report generation functionality for time series forecasting results.
It includes model performance summaries, configuration parameters, and visualization plots.
Enhanced to include both training and evaluation metrics, confusion matrices, and confidence levels.
"""

import os
import logging
from datetime import datetime
from fpdf import FPDF
from PIL import Image
from tempfile import NamedTemporaryFile
from typing import Dict, List, Any, Optional


class CompactPDFReport(FPDF):
    """
    Enhanced PDF report generator for time series forecasting results.
    Includes training metrics, evaluation metrics, confusion matrices, and confidence levels.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.set_auto_page_break(auto=True, margin=10)
        self.WIDTH = 210
        self.HEIGHT = 297
        
        # Watermark settings
        self.watermark_path = config.get('watermark_path')
        self.watermark_size = config.get('watermark_size', 80)
        self.watermark_opacity = config.get('watermark_opacity', 0.15)
        self.watermark_img = None
        
        if self.watermark_path and os.path.exists(self.watermark_path):
            try:
                self.watermark_img = self._process_watermark()
            except Exception as e:
                logging.error(f"Watermark processing failed: {str(e)}")
                self.watermark_img = None
        
        self.add_page()
        
    def _process_watermark(self):
        """Process watermark image with opacity adjustment."""
        try:
            img = Image.open(self.watermark_path).convert("RGBA")
            
            if self.watermark_opacity < 1.0:
                alpha = img.split()[3]
                alpha = alpha.point(lambda p: p * self.watermark_opacity)
                img.putalpha(alpha)
            
            temp_file = NamedTemporaryFile(delete=False, suffix='.png')
            img.save(temp_file.name, "PNG")
            return temp_file.name
        except Exception as e:
            logging.error(f"Error processing watermark: {str(e)}")
            return None

    def header(self):
        """Add header with watermark and title."""
        if self.watermark_img:
            try:
                x = (self.WIDTH - self.watermark_size) / 2
                y = (self.HEIGHT - self.watermark_size) / 2
                
                self.image(
                    self.watermark_img,
                    x=x,
                    y=y,
                    w=self.watermark_size,
                    h=self.watermark_size
                )
            except Exception as e:
                logging.error(f"Failed to add watermark: {str(e)}")

        self.set_font('Arial', 'B', 16)
        self.cell(0, 8, 'Forecasting Report Summary', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 6, f"Dataset: {self.config['data_path']}", 0, 1, 'C')
        self.cell(0, 6, f"Target: {self.config['target_col']}, Problem Type: {self.config['problem_type']}", 0, 1, 'C')
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        """Add footer with page number."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        
    def generate_report(self, models_data, evaluation, future_forecasts, output_path):
        """
        Generates report with summary page followed by individual model pages.
        
        Parameters:
        -----------
        models_data : dict
            Dictionary containing model training results
        evaluation : dict
            Dictionary containing evaluation results
        future_forecasts : dict
            Dictionary containing future forecasts
        output_path : str
            Path to save the PDF report
        """
        self._add_summary_page(models_data)
        
        for model_name in models_data.keys():
            self._add_model_page(
                model_name, 
                models_data.get(model_name, {}), 
                evaluation.get(model_name, {}), 
                future_forecasts.get(model_name, {})
            )
        
        self.output(output_path)
    
    def _add_summary_page(self, models_data):
        """Page 1: Model summary table and parameters."""
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 8, 'Model Performance Summary', 0, 1, 'C')
        self.ln(10)
        
        # Performance metrics table
        self.set_fill_color(220, 220, 220)
        self.set_font('Arial', 'B', 10)
        self.cell(50, 8, 'Model', 1, 0, 'C', 1)
        self.cell(30, 8, 'MAE', 1, 0, 'C', 1)
        self.cell(30, 8, 'RMSE', 1, 0, 'C', 1)
        self.cell(30, 8, 'R²', 1, 0, 'C', 1)
        self.cell(30, 8, 'MAPE', 1, 1, 'C', 1)
        
        self.set_font('Arial', '', 9)
        for model_name, model_data in models_data.items():
            metrics = model_data.get('metrics', {})
            display_name = model_name[:18] + "..." if len(model_name) > 20 else model_name
            
            mae = metrics.get('MAE', 'N/A')
            mae_str = f"{mae:.1f}" if isinstance(mae, (int, float)) else str(mae)[:8]
            
            rmse = metrics.get('RMSE', 'N/A')
            rmse_str = f"{rmse:.1f}" if isinstance(rmse, (int, float)) else str(rmse)[:8]
            
            r2 = metrics.get('R2', 'N/A')
            r2_str = f"{r2:.3f}" if isinstance(r2, (int, float)) else str(r2)[:6]
            
            mape = metrics.get('MAPE', 'N/A')
            mape_str = f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape)[:6]
            
            self.cell(50, 8, display_name, 1)
            self.cell(30, 8, mae_str, 1, 0, 'C')
            self.cell(30, 8, rmse_str, 1, 0, 'C')
            self.cell(30, 8, r2_str, 1, 0, 'C')
            self.cell(30, 8, mape_str, 1, 1, 'C')
        
        self.ln(15)
        
        # Confusion Matrix Summary (if available)
        has_cm = any(
            'confusion_matrix' in models_data.get(m, {}) and models_data[m].get('confusion_matrix')
            for m in models_data.keys()
        )
        
        if has_cm:
            self.set_font('Arial', 'B', 14)
            self.cell(0, 8, 'Trend Direction Classification Summary', 0, 1, 'C')
            self.ln(5)
            
            self.set_fill_color(220, 220, 220)
            self.set_font('Arial', 'B', 9)
            self.cell(40, 8, 'Model', 1, 0, 'C', 1)
            self.cell(30, 8, 'Accuracy', 1, 0, 'C', 1)
            self.cell(30, 8, 'Precision', 1, 0, 'C', 1)
            self.cell(30, 8, 'Recall', 1, 0, 'C', 1)
            self.cell(30, 8, 'F1 Score', 1, 1, 'C', 1)
            
            self.set_font('Arial', '', 9)
            for model_name, model_data in models_data.items():
                cm_data = model_data.get('confusion_matrix', {})
                if cm_data:
                    display_name = model_name[:15] + "..." if len(model_name) > 17 else model_name
                    
                    acc = cm_data.get('accuracy', 'N/A')
                    acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else 'N/A'
                    
                    prec = cm_data.get('precision', 'N/A')
                    prec_str = f"{prec:.3f}" if isinstance(prec, (int, float)) else 'N/A'
                    
                    rec = cm_data.get('recall', 'N/A')
                    rec_str = f"{rec:.3f}" if isinstance(rec, (int, float)) else 'N/A'
                    
                    f1 = cm_data.get('f1', 'N/A')
                    f1_str = f"{f1:.3f}" if isinstance(f1, (int, float)) else 'N/A'
                    
                    self.cell(40, 8, display_name, 1)
                    self.cell(30, 8, acc_str, 1, 0, 'C')
                    self.cell(30, 8, prec_str, 1, 0, 'C')
                    self.cell(30, 8, rec_str, 1, 0, 'C')
                    self.cell(30, 8, f1_str, 1, 1, 'C')
            
            self.ln(15)
        
        # Model Configuration Parameters
        self.set_font('Arial', 'B', 14)
        self.cell(0, 8, 'Model Configuration Parameters', 0, 1, 'C')
        self.ln(10)
        
        for model_name, model_data in models_data.items():
            self.set_font('Arial', 'B', 12)
            self.cell(0, 8, f"{model_name.upper()}", 0, 1, 'L')
            self.ln(2)
            
            params = model_data.get('parameters', {})
            if params:
                self.set_font('Arial', '', 9)
                
                param_items = list(params.items())
                for i in range(0, len(param_items), 2): 
                    param1 = param_items[i]
                    param2 = param_items[i+1] if i+1 < len(param_items) else None
                    
                    param_name = str(param1[0])[:20]
                    param_value = str(param1[1])[:25]
                    self.cell(95, 5, f"{param_name}: {param_value}", 1, 0, 'L')
                  
                    if param2:
                        param_name2 = str(param2[0])[:20]
                        param_value2 = str(param2[1])[:25]
                        self.cell(95, 5, f"{param_name2}: {param_value2}", 1, 1, 'L')
                    else:
                        self.cell(95, 5, "", 1, 1, 'L')
            else:
                self.set_font('Arial', 'I', 9)
                self.cell(0, 6, "No parameters available", 0, 1, 'L')
            
            self.ln(8)
           
            if self.get_y() > self.HEIGHT - 50:
                self.add_page()
                self.set_y(30)
    
    def _add_model_page(self, model_name, model_data, evaluation_data, forecast_data):
        """Individual model page with all plots and metrics."""
        self.add_page()
        
        self.set_font('Arial', 'B', 18)
        self.cell(0, 10, f"{model_name.upper()} - Detailed Analysis", 0, 1, 'C')
        self.ln(5)
        
        # Add metrics box
        self._add_metrics_box(model_data, evaluation_data)
        self.ln(10)

        # Add interpretability summary if available
        self._add_interpretability_box(model_data)
        self.ln(5)
        
        # Add confusion matrix metrics if available
        self._add_confusion_matrix_box(model_data)
        self.ln(5)
        
        # Add confidence levels if available
        self._add_confidence_box(model_data)
        self.ln(5)
        
        # Add explosion detection if available
        self._add_explosion_detection_box(model_data)
        self.ln(5)
        
        # Add all model plots
        self._add_all_model_plots(model_name, model_data, evaluation_data, forecast_data)

    def _add_interpretability_box(self, model_data: Dict[str, Any]):
        """Add model interpretability summary box if available.

        Expects ``model_data['interpretability']`` to be a dict that may
        contain keys such as ``permutation``, ``integrated_gradients``,
        and ``shap``, each with a ``top_features`` list.
        """

        interp = model_data.get('interpretability') or {}
        if not interp:
            return

        # Collect one combined list of feature importances/attributions
        collected = []
        for key in ('permutation', 'integrated_gradients', 'shap'):
            block = interp.get(key) or {}
            for item in block.get('top_features', [])[:5]:
                feat = str(item.get('feature', ''))
                val = item.get('importance', item.get('attribution'))
                if feat and isinstance(val, (int, float)):
                    collected.append((key, feat, float(val)))

        if not collected:
            return

        start_y = self.get_y()
        self.set_draw_color(80, 80, 120)
        self.set_fill_color(250, 250, 255)
        self.rect(10, start_y, 190, 22, 'DF')

        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, 'Model Interpretability (Top Signals)', 0, 1, 'C')

        self.set_font('Arial', '', 9)
        # Show up to four lines summarizing top features
        for idx, (method, feat, val) in enumerate(collected[:4]):
            label = {
                'permutation': 'Perm',
                'integrated_gradients': 'IG',
                'shap': 'SHAP',
            }.get(method, method)
            self.cell(95, 5, f"{label}: {feat}", 0, 0, 'L')
            self.cell(95, 5, f"Score: {val:.3f}", 0, 1, 'L')
    
    def _add_metrics_box(self, model_data, evaluation_data):
        """Add performance metrics in a box."""
        metrics = model_data.get('metrics', {})
        if not metrics and evaluation_data:
            metrics = evaluation_data.get('metrics', {})
        
        if not metrics:
            return
            
        start_y = self.get_y()
        self.set_draw_color(100, 100, 100)
        self.set_fill_color(245, 245, 245)
        self.rect(10, start_y, 190, 25, 'DF')
        
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, 'Performance Metrics', 0, 1, 'C')
        self.ln(2)
        
        self.set_font('Arial', '', 10)
        metric_items = list(metrics.items())
        
        for i in range(0, len(metric_items), 2):
            metric1 = metric_items[i]
            metric2 = metric_items[i+1] if i+1 < len(metric_items) else None
            
            value1 = f"{metric1[1]:.3f}" if isinstance(metric1[1], (int, float)) else str(metric1[1])
            self.cell(95, 6, f"{metric1[0]}: {value1}", 0, 0, 'L')
            
            if metric2:
                value2 = f"{metric2[1]:.3f}" if isinstance(metric2[1], (int, float)) else str(metric2[1])
                self.cell(95, 6, f"{metric2[0]}: {value2}", 0, 1, 'L')
            else:
                self.ln(6)
    
    def _add_confusion_matrix_box(self, model_data):
        """Add confusion matrix metrics in a box."""
        cm_data = model_data.get('confusion_matrix', {})
        
        if not cm_data or not any(k in cm_data for k in ['accuracy', 'precision', 'recall', 'f1']):
            return
        
        start_y = self.get_y()
        self.set_draw_color(100, 100, 100)
        self.set_fill_color(240, 248, 255)  # Light blue
        self.rect(10, start_y, 190, 20, 'DF')
        
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, 'Trend Direction Classification Metrics', 0, 1, 'C')
        self.ln(1)
        
        self.set_font('Arial', '', 9)
        
        # Row 1: Accuracy and Precision
        acc = cm_data.get('accuracy', 'N/A')
        acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else 'N/A'
        self.cell(47, 5, f"Accuracy: {acc_str}", 0, 0, 'L')
        
        prec = cm_data.get('precision', 'N/A')
        prec_str = f"{prec:.3f}" if isinstance(prec, (int, float)) else 'N/A'
        self.cell(47, 5, f"Precision: {prec_str}", 0, 0, 'L')
        
        # Row 1: Recall and F1
        rec = cm_data.get('recall', 'N/A')
        rec_str = f"{rec:.3f}" if isinstance(rec, (int, float)) else 'N/A'
        self.cell(47, 5, f"Recall: {rec_str}", 0, 0, 'L')
        
        f1 = cm_data.get('f1', 'N/A')
        f1_str = f"{f1:.3f}" if isinstance(f1, (int, float)) else 'N/A'
        self.cell(47, 5, f"F1 Score: {f1_str}", 0, 1, 'L')
        
        self.ln(1)
        
        # Row 2: TP, TN, FP, FN
        tp = cm_data.get('true_positives', 'N/A')
        tn = cm_data.get('true_negatives', 'N/A')
        fp = cm_data.get('false_positives', 'N/A')
        fn = cm_data.get('false_negatives', 'N/A')
        
        self.cell(47, 5, f"True Positives: {tp}", 0, 0, 'L')
        self.cell(47, 5, f"True Negatives: {tn}", 0, 0, 'L')
        self.cell(47, 5, f"False Positives: {fp}", 0, 0, 'L')
        self.cell(47, 5, f"False Negatives: {fn}", 0, 1, 'L')
    
    def _add_confidence_box(self, model_data):
        """Add confidence level information."""
        confidence_levels = model_data.get('confidence_levels', [])
        
        if not confidence_levels:
            return
        
        start_y = self.get_y()
        self.set_draw_color(100, 100, 100)
        self.set_fill_color(240, 255, 240)  # Light green
        self.rect(10, start_y, 190, 12, 'DF')
        
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, 'Forecast Confidence Levels', 0, 1, 'C')
        
        self.set_font('Arial', '', 9)
        avg_conf = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0
        min_conf = min(confidence_levels) if confidence_levels else 0
        max_conf = max(confidence_levels) if confidence_levels else 0
        
        self.cell(63, 5, f"Average: {avg_conf:.3f}", 0, 0, 'C')
        self.cell(63, 5, f"Min: {min_conf:.3f}", 0, 0, 'C')
        self.cell(63, 5, f"Max: {max_conf:.3f}", 0, 1, 'C')
    
    def _add_explosion_detection_box(self, model_data):
        """Add explosion detection results."""
        explosion_data = model_data.get('explosion_detection', {})
        
        if not explosion_data or not explosion_data.get('checks_performed'):
            return
        
        start_y = self.get_y()
        
        if explosion_data.get('explosion_detected'):
            self.set_draw_color(200, 50, 50)
            self.set_fill_color(255, 240, 240)  # Light red
            title = '⚠️ Forecast Explosion Detected!'
        else:
            self.set_draw_color(50, 150, 50)
            self.set_fill_color(240, 255, 240)  # Light green
            title = '✓ Forecast Validation Passed'
        
        self.rect(10, start_y, 190, 12, 'DF')
        
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, title, 0, 1, 'C')
        
        self.set_font('Arial', '', 9)
        dev_violations = explosion_data.get('n_deviation_violations', 0)
        growth_violations = explosion_data.get('n_growth_violations', 0)
        
        self.cell(95, 5, f"Deviation Violations: {dev_violations}", 0, 0, 'C')
        self.cell(95, 5, f"Growth Violations: {growth_violations}", 0, 1, 'C')
    
    def _add_all_model_plots(self, model_name, model_data, evaluation_data, forecast_data):
        """Add ALL plots for a specific model."""
        all_figures = []
        
        if 'figures' in model_data and model_data['figures']:
            all_figures.extend(model_data['figures'])
        
        if 'figures' in evaluation_data and evaluation_data['figures']:
            all_figures.extend(evaluation_data['figures'])
        
        if 'plot_path' in forecast_data and forecast_data['plot_path']:
            all_figures.append(forecast_data['plot_path'])
        
        seen = set()
        unique_figures = []
        for fig in all_figures:
            if fig not in seen and os.path.exists(fig):
                seen.add(fig)
                unique_figures.append(fig)
        
        if not unique_figures:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 8, 'No visualization plots available for this model', 0, 1, 'C')
            return
        
        self.set_font('Arial', 'B', 14)
        self.cell(0, 8, 'Model Visualizations', 0, 1, 'C')
        self.ln(5)
        
        plots_added = 0
        plots_per_row = 2 
        current_row_plots = 0
        
        for i, fig_path in enumerate(unique_figures):
            try:
                plot_name = os.path.basename(fig_path).replace('.png', '').replace('_', ' ').title()
                
                if self.get_y() > self.HEIGHT - 80:
                    self.add_page()
                    self.set_y(30)
                    current_row_plots = 0
                
                if current_row_plots == 0:
                    x_pos = 10
                    y_pos = self.get_y()
                else:
                    x_pos = 105
                    y_pos = self.get_y() - 55  
                
                self.set_xy(x_pos, y_pos)
                self.set_font('Arial', 'B', 9)
                
                if len(plot_name) > 25:
                    plot_name = plot_name[:22] + "..."
                self.cell(95, 5, plot_name, 0, 1, 'C')
                
                self.set_xy(x_pos, y_pos + 5)
                self.image(fig_path, x=x_pos, w=95, h=50)
                
                current_row_plots += 1
                plots_added += 1
                
                if current_row_plots == plots_per_row:
                    self.set_y(y_pos + 60)  
                    current_row_plots = 0
                
                try:
                    os.remove(fig_path)
                except Exception as e:
                    logging.warning(f"Could not remove temp file {fig_path}: {str(e)}")
                    
            except Exception as e:
                logging.error(f"Error adding plot {fig_path}: {str(e)}")
                self.set_font('Arial', 'I', 8)
                self.cell(0, 6, f"Error displaying plot: {os.path.basename(fig_path)}", 0, 1)
                self.ln(3)

        if current_row_plots > 0:
            self.set_y(self.get_y() + 10)
        
        if forecast_data and 'forecast_days' in forecast_data:
            self.ln(10)
            self.set_font('Arial', 'B', 12)
            self.cell(0, 8, 'Forecast Information', 0, 1, 'L')
            self.set_font('Arial', '', 10)
            self.cell(0, 6, f"Forecast Period: {forecast_data['forecast_days']} days", 0, 1)
            
            if 'confidence_intervals' in forecast_data:
                self.cell(0, 6, "Confidence intervals included in forecast plots", 0, 1)
        
        self.ln(5)
        
        self.set_font('Arial', 'I', 9)
        self.cell(0, 6, f"Total plots displayed: {plots_added}", 0, 1, 'C')
    
    def __del__(self):
        """Clean up temporary watermark file when instance is destroyed."""
        if self.watermark_img and os.path.exists(self.watermark_img):
            try:
                os.remove(self.watermark_img)
            except Exception as e:
                logging.warning(f"Could not remove watermark temp file: {str(e)}")
