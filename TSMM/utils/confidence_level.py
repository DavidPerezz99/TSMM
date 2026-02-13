"""
Confidence Level Module

This module provides functionality to calculate confidence levels for forecasts
using a discriminator model that predicts whether a given input window is likely
to produce a correct or incorrect prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import os


def calculate_features_from_window(window: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistical features from a single input window.
    
    Parameters:
    -----------
    window : np.ndarray
        Input window of shape (n_steps, n_features)
    
    Returns:
    --------
    dict
        Dictionary of calculated features
    """
    # Ensure window is 2D: (n_steps, n_features)
    if window.ndim == 1:
        # Treat 1D input as a single feature over time
        window = window.reshape(-1, 1)

    features = {}
    
    # Compute statistics for each feature/dimension
    for i in range(window.shape[1]):  # Iterate over dimensions
        dimension = window[:, i]
        features[f"mean_dim_{i}"] = np.mean(dimension)
        features[f"std_dim_{i}"] = np.std(dimension)
        features[f"min_dim_{i}"] = np.min(dimension)
        features[f"max_dim_{i}"] = np.max(dimension)
        features[f"sum_dim_{i}"] = np.sum(dimension)
        features[f"median_dim_{i}"] = np.median(dimension)
        features[f"range_dim_{i}"] = np.max(dimension) - np.min(dimension)
    
    # Overall statistics across all dimensions
    features["overall_mean"] = np.mean(window)
    features["overall_std"] = np.std(window)
    features["overall_min"] = np.min(window)
    features["overall_max"] = np.max(window)
    features["overall_sum"] = np.sum(window)
    features["overall_median"] = np.median(window)
    features["overall_range"] = np.max(window) - np.min(window)
    
    return features


def find_misclassifications(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_index: int = 0,
    threshold: float = 0.0
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Identifies misclassified inputs where predictions failed to match actual return direction.
    
    Parameters:
    -----------
    X_test : np.ndarray
        Input features (original scale or transformed)
    y_test : np.ndarray
        Actual target values
    y_pred : np.ndarray
        Predicted target values
    target_index : int
        Index of the target feature to analyze
    threshold : float
        Decision boundary for positive/negative classification
    
    Returns:
    --------
    tuple
        (misclassified_data, misclassified_indices, correct_indices)
    """
    # Extract the relevant target column
    if y_test.ndim > 1:
        y_actual = y_test[:, target_index]
    else:
        y_actual = y_test
        
    if y_pred.ndim > 1:
        y_predicted = y_pred[:, target_index]
    else:
        y_predicted = y_pred
    
    # Define the sign-based classification
    actual_sign = np.where(y_actual > threshold, 1, -1)  # 1 = positive return, -1 = negative return
    predicted_sign = np.where(y_predicted > threshold, 1, -1)
    
    # Find misclassified and correctly classified indices
    misclassified_indices = np.where(actual_sign != predicted_sign)[0]
    correct_indices = np.where(actual_sign == predicted_sign)[0]
    
    # Create a dictionary of misclassified data
    misclassified_data = {
        'Actual_Return': y_actual[misclassified_indices].ravel(),
        'Predicted_Return': y_predicted[misclassified_indices].ravel(),
        'Actual_Sign': actual_sign[misclassified_indices].ravel(),
        'Predicted_Sign': predicted_sign[misclassified_indices].ravel(),
    }
    
    # Add input features, ensuring they're 1D
    for i in range(X_test.shape[1]):
        misclassified_data[f'Feature_{i+1}'] = X_test[misclassified_indices, i].ravel()
    
    return misclassified_data, misclassified_indices, correct_indices


def create_feature_dataset(
    X_test: np.ndarray,
    wrong_indices: np.ndarray,
    right_indices: np.ndarray
) -> pd.DataFrame:
    """
    Create a feature dataset for training the discriminator.
    
    Parameters:
    -----------
    X_test : np.ndarray
        Input test data
    wrong_indices : np.ndarray
        Indices of misclassified samples
    right_indices : np.ndarray
        Indices of correctly classified samples
    
    Returns:
    --------
    pd.DataFrame
        Feature dataset with labels (0=wrong, 1=correct)
    """
    data = []
    
    # Create entries for misclassified samples
    for idx in wrong_indices:
        features = calculate_features_from_window(X_test[idx])
        features['label'] = 0  # 0 for wrong prediction
        data.append(features)
    
    # Create entries for correctly classified samples
    for idx in right_indices:
        features = calculate_features_from_window(X_test[idx])
        features['label'] = 1  # 1 for correct prediction
        data.append(features)
    
    return pd.DataFrame(data)


class ConfidenceDiscriminator:
    """
    A discriminator model that predicts the confidence level of forecasts.
    
    This model is trained to classify input windows as likely to produce
    correct or incorrect predictions.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        test_size: float = 0.4,
        random_state: int = 42
    ):
        """
        Initialize the ConfidenceDiscriminator.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the Random Forest
        max_depth : int
            Maximum depth of the trees
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
        
    def train(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        target_index: int = 0,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Train the discriminator model.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test input data (used to identify misclassifications)
        y_test : np.ndarray
            Actual target values
        y_pred : np.ndarray
            Predicted target values
        target_index : int
            Index of the target feature to analyze
        threshold : float
            Decision boundary for classification
        
        Returns:
        --------
        dict
            Training metrics
        """
        # Find misclassifications
        misclassified_data, wrong_indices, right_indices = find_misclassifications(
            X_test, y_test, y_pred, target_index, threshold
        )
        
        logging.info(f"Training discriminator: {len(wrong_indices)} misclassified, {len(right_indices)} correct")
        
        if len(wrong_indices) < 5 or len(right_indices) < 5:
            logging.warning("Not enough samples to train discriminator")
            self.is_trained = False
            return {'error': 'Insufficient samples for training'}
        
        # Create feature dataset
        feature_dataset = create_feature_dataset(X_test, wrong_indices, right_indices)
        
        # Separate features and labels
        X = feature_dataset.drop(columns=["label"])
        y = feature_dataset["label"]
        
        # Split data into training and testing sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Create and train a Random Forest Classifier
        self.model = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred_val = self.model.predict(X_val)
        
        self.training_metrics = {
            'accuracy': accuracy_score(y_val, y_pred_val),
            'classification_report': classification_report(y_val, y_pred_val, output_dict=True),
            'confusion_matrix': confusion_matrix(y_val, y_pred_val).tolist(),
            'n_correct_samples': len(right_indices),
            'n_wrong_samples': len(wrong_indices),
        }
        
        self.is_trained = True
        
        logging.info(f"Discriminator training complete. Accuracy: {self.training_metrics['accuracy']:.4f}")
        
        return self.training_metrics
    
    def predict_confidence(self, window: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict the confidence level for a given input window.
        
        Parameters:
        -----------
        window : np.ndarray
            Input window of shape (n_steps, n_features)
        
        Returns:
        --------
        tuple
            (prediction, probabilities) where prediction is 1 (likely correct) or 0 (likely misclassified)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Discriminator has not been trained yet")
        
        # Extract features from the window
        features = calculate_features_from_window(window)
        features_df = pd.DataFrame([features])
        
        # Predict
        prediction = self.model.predict(features_df)
        probabilities = self.model.predict_proba(features_df)
        
        return prediction[0], probabilities[0]
    
    def predict_confidence_batch(
        self,
        windows: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Predict confidence levels for multiple input windows.
        
        Parameters:
        -----------
        windows : np.ndarray
            Array of input windows of shape (n_samples, n_steps, n_features)
        
        Returns:
        --------
        list
            List of dictionaries containing prediction results
        """
        results = []
        
        for i in range(len(windows)):
            pred, probs = self.predict_confidence(windows[i])
            results.append({
                'window_index': i,
                'prediction': pred,  # 1 = likely correct, 0 = likely misclassified
                'confidence_correct': probs[1] if len(probs) > 1 else probs[0],
                'confidence_misclassified': probs[0] if len(probs) > 1 else 0,
                'is_likely_correct': pred == 1
            })
        
        return results
    
    def save(self, filepath: str):
        """Save the trained discriminator model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump({
            'model': self.model,
            'training_metrics': self.training_metrics,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'is_trained': self.is_trained
        }, filepath)
        
    def load(self, filepath: str):
        """Load a trained discriminator model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.training_metrics = data['training_metrics']
        self.n_estimators = data['n_estimators']
        self.max_depth = data['max_depth']
        self.is_trained = data['is_trained']


def train_confidence_discriminator(
    model_results: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    config: Dict
) -> Optional[ConfidenceDiscriminator]:
    """
    Train a confidence discriminator for a model.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing model results
    X_test : np.ndarray
        Test input data
    y_test : np.ndarray
        Actual target values
    y_pred : np.ndarray
        Predicted target values
    config : dict
        Configuration dictionary with confidence settings
    
    Returns:
    --------
    ConfidenceDiscriminator or None
        Trained discriminator or None if training failed
    """
    confidence_config = config.get('confidence', {})
    
    if not confidence_config.get('enabled', True):
        return None
    
    try:
        # Align X_test samples with y_test length to avoid index errors.
        # In some evaluation flows (e.g., ULR), X_test may represent a
        # single window or a shorter sequence than the number of labeled
        # samples in y_test/y_pred. We tile or repeat X_test so that
        # find_misclassifications can safely index it using sample indices
        # derived from y_test.
        try:
            n_samples = y_test.shape[0]
        except Exception:
            n_samples = None

        if n_samples is not None and isinstance(X_test, np.ndarray):
            # Ensure at least 2D
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)

            if X_test.ndim == 2 and X_test.shape[0] != n_samples:
                if X_test.shape[0] == 1:
                    # Single window: repeat for all samples
                    X_test = np.repeat(X_test, n_samples, axis=0)
                else:
                    # Tile rows to cover all samples, then truncate
                    reps = int(np.ceil(n_samples / X_test.shape[0]))
                    X_tiled = np.tile(X_test, (reps, 1))
                    X_test = X_tiled[:n_samples]

        discriminator = ConfidenceDiscriminator(
            n_estimators=confidence_config.get('n_estimators', 100),
            max_depth=confidence_config.get('max_depth', 10),
            test_size=confidence_config.get('test_size', 0.4),
            random_state=42
        )
        
        metrics = discriminator.train(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            target_index=0,
            threshold=confidence_config.get('threshold', 0.0)
        )
        
        if 'error' in metrics:
            logging.warning(f"Failed to train discriminator: {metrics['error']}")
            return None
        
        return discriminator
        
    except Exception as e:
        logging.error(f"Error training confidence discriminator: {str(e)}")
        return None


def get_forecast_confidence_levels(
    discriminator: ConfidenceDiscriminator,
    future_windows: np.ndarray
) -> List[float]:
    """
    Get confidence levels for future forecast windows.
    
    Parameters:
    -----------
    discriminator : ConfidenceDiscriminator
        Trained discriminator model
    future_windows : np.ndarray
        Array of future input windows
    
    Returns:
    --------
    list
        List of confidence levels (probability of correct prediction)
    """
    results = discriminator.predict_confidence_batch(future_windows)
    
    # Return the probability of correct prediction for each window
    return [r['confidence_correct'] for r in results]


# =============================================================================
# Forecast Explosion Detection
# =============================================================================

def detect_forecast_explosion(
    future_forecast: np.ndarray,
    last_real_values: np.ndarray,
    config: Dict
) -> Dict[str, Any]:
    """
    Detect if a forecast shows signs of explosion (unrealistic growth).
    
    Parameters:
    -----------
    future_forecast : np.ndarray
        Array of future forecast values
    last_real_values : np.ndarray
        Array of last real data points for comparison
    config : dict
        Configuration dictionary with forecast_validation settings
    
    Returns:
    --------
    dict
        Detection results with explosion indicators
    """
    validation_config = config.get('forecast_validation', {})
    
    if not validation_config.get('enabled', True):
        return {'explosion_detected': False, 'checks_performed': False}
    
    max_deviation = validation_config.get('max_deviation_percent', 50.0) / 100.0
    max_growth_rate = validation_config.get('max_growth_rate', 2.0)
    
    results = {
        'explosion_detected': False,
        'checks_performed': True,
        'deviation_violations': [],
        'growth_violations': [],
        'max_deviation_percent': max_deviation * 100,
        'max_growth_rate': max_growth_rate,
    }
    
    # Get reference value (mean of last real values)
    reference_value = np.mean(last_real_values) if len(last_real_values) > 0 else 0
    
    if reference_value == 0:
        reference_value = 1e-10  # Avoid division by zero
    
    # Check deviation from reference
    for i, forecast_val in enumerate(future_forecast):
        if isinstance(forecast_val, (list, np.ndarray)):
            forecast_val = forecast_val[0] if len(forecast_val) > 0 else 0
        
        deviation = abs(forecast_val - reference_value) / abs(reference_value)
        
        if deviation > max_deviation:
            results['deviation_violations'].append({
                'index': i,
                'forecast_value': float(forecast_val),
                'reference_value': float(reference_value),
                'deviation_percent': float(deviation * 100)
            })
    
    # Check growth rate between consecutive forecasts
    for i in range(1, len(future_forecast)):
        prev_val = future_forecast[i-1]
        curr_val = future_forecast[i]
        
        if isinstance(prev_val, (list, np.ndarray)):
            prev_val = prev_val[0] if len(prev_val) > 0 else 0
        if isinstance(curr_val, (list, np.ndarray)):
            curr_val = curr_val[0] if len(curr_val) > 0 else 0
        
        if prev_val != 0:
            growth_rate = abs(curr_val / prev_val)
            
            if growth_rate > max_growth_rate:
                results['growth_violations'].append({
                    'index': i,
                    'previous_value': float(prev_val),
                    'current_value': float(curr_val),
                    'growth_rate': float(growth_rate)
                })
    
    # Determine if explosion is detected
    if len(results['deviation_violations']) > len(future_forecast) * 0.2 or \
       len(results['growth_violations']) > len(future_forecast) * 0.1:
        results['explosion_detected'] = True
    
    results['n_deviation_violations'] = len(results['deviation_violations'])
    results['n_growth_violations'] = len(results['growth_violations'])
    
    return results
