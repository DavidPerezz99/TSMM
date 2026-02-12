def evaluate_models(models, df, config):
    """
    Evaluates models using recursive forecasting for validation and future predictions
    Handles ULR, MLR, RNN, and CNN-BiLSTM models with feature recalculation
    """
    evaluation = {}
    future_forecasts = {}
    
    # Extract config parameters
    n_steps = config['n_steps']
    m_steps = config['m_steps']
    horizon = config['horizon']
    test_size = config['test_size']
    input_features = config['input_features']
    target_features = config['target_features']
    target_col = config['target_col']
    n_features = len(input_features)
    rolling_windows = config.get('rolling_windows', [7, 30, 60, 10])
    max_window = max(rolling_windows) if rolling_windows else 60
    
    # Feature update function
    def update_features(predictions, last_gross, input_features, target_features, 
                        history_gross, history_diff):
        """
        Recalculates features based on new predictions during recursive forecasting
        Maintains history for rolling features and GROSS_ADDS/y_diff relationship
        """
        updated_features = []
        current_gross = last_gross
        
        # Process each prediction in the window
        for pred in predictions:
            # 1. Handle GROSS_ADDS and y_diff relationship
            if 'y_diff' in target_features:
                y_diff_idx = target_features.index('y_diff')
                y_diff = pred[y_diff_idx]
                current_gross += y_diff
            elif config['target_col'] in target_features:
                gross_idx = target_features.index(config['target_col'])
                current_gross = pred[gross_idx]
                y_diff = current_gross - last_gross
            else:
                y_diff = 0
            
            # Update histories
            history_gross.append(current_gross)
            history_diff.append(y_diff)
            last_gross = current_gross
            
            # 2. Build feature vector in input_features order
            feature_vector = []
            for feat in input_features:
                # Base features
                if feat == 'y_diff':
                    feature_vector.append(y_diff)
                elif feat == config['target_col']:
                    feature_vector.append(current_gross)
                
                # Rolling features for GROSS_ADDS
                elif feat.startswith('SMA_') and not feat.endswith('_diff'):
                    window = int(feat.split('_')[1])
                    if len(history_gross) >= window:
                        sma = np.mean(history_gross[-window:])
                        feature_vector.append(sma)
                    else:
                        feature_vector.append(0)
                        
                elif feat.startswith('EMA_') and not feat.endswith('_diff'):
                    window = int(feat.split('_')[1])
                    if len(history_gross) >= window:
                        ema = pd.Series(history_gross).ewm(span=window, adjust=False).mean().iloc[-1]
                        feature_vector.append(ema)
                    else:
                        feature_vector.append(0)
                        
                elif feat.startswith('Volatility_') and not feat.endswith('_diff'):
                    window = int(feat.split('_')[1])
                    if len(history_gross) >= window:
                        vol = np.std(history_gross[-window:])
                        feature_vector.append(vol)
                    else:
                        feature_vector.append(0)
                
                # Rolling features for y_diff
                elif feat.startswith('SMA_') and feat.endswith('_diff'):
                    window = int(feat.split('_')[1])
                    if len(history_diff) >= window:
                        sma = np.mean(history_diff[-window:])
                        feature_vector.append(sma)
                    else:
                        feature_vector.append(0)
                        
                elif feat.startswith('EMA_') and feat.endswith('_diff'):
                    window = int(feat.split('_')[1])
                    if len(history_diff) >= window:
                        ema = pd.Series(history_diff).ewm(span=window, adjust=False).mean().iloc[-1]
                        feature_vector.append(ema)
                    else:
                        feature_vector.append(0)
                        
                elif feat.startswith('Volatility_') and feat.endswith('_diff'):
                    window = int(feat.split('_')[1])
                    if len(history_diff) >= window:
                        vol = np.std(history_diff[-window:])
                        feature_vector.append(vol)
                    else:
                        feature_vector.append(0)
                
                # Exogenous features
                elif feat in target_features:
                    idx = target_features.index(feat)
                    feature_vector.append(pred[idx])
                    
                # Fallback for unknown features
                else:
                    feature_vector.append(0)
            
            updated_features.append(feature_vector)
        
        # Trim histories to max_window size
        if len(history_gross) > max_window * 2:
            history_gross = history_gross[-max_window:]
        if len(history_diff) > max_window * 2:
            history_diff = history_diff[-max_window:]
        
        return np.array(updated_features), current_gross, history_gross, history_diff

    # Unified recursive forecasting engine
    def recursive_forecast(model, scalers, initial_window, steps, model_type, n_steps, n_features):
        """Handles recursive forecasting for all model types with feature updates"""
        predictions = []
        current_window = initial_window.copy()
        
        # Initialize histories
        history_gross = []
        history_diff = []
        
        # Get last known GROSS_ADDS
        if config['target_col'] in input_features:
            last_gross = current_window[-1, input_features.index(config['target_col'])]
        else:
            last_gross = 0
        
        # Pre-fill histories from initial window
        if config['target_col'] in input_features:
            gross_idx = input_features.index(config['target_col'])
            history_gross = list(current_window[:, gross_idx])
            
        if 'y_diff' in input_features:
            diff_idx = input_features.index('y_diff')
            history_diff = list(current_window[:, diff_idx])
        
        # Calculate iterations needed
        iterations = (steps + m_steps - 1) // m_steps
        
        for _ in range(iterations):
            # Scale each time step individually
            scaled_steps = []
            for i in range(current_window.shape[0]):
                step_scaled = scalers['X'].transform(current_window[i].reshape(1, -1))
                scaled_steps.append(step_scaled[0])
            X_scaled = np.array(scaled_steps)
            
            # Model-specific prediction
            if model_type in ['ulr', 'mlr']:
                # Flatten for linear models
                X_flat = X_scaled.reshape(1, -1)
                y_pred_scaled = model.predict(X_flat)
                y_pred_2d = y_pred_scaled.reshape(-1, len(target_features))
            else:
                # RNN/CNN models need 3D input
                X_3d = X_scaled.reshape(1, n_steps, n_features)
                y_pred_scaled = model.predict(X_3d)
                y_pred_2d = y_pred_scaled.reshape(-1, len(target_features))
            
            # Inverse scale predictions
            y_pred = scalers['y'].inverse_transform(y_pred_2d)
            predictions.append(y_pred)
            
            # Update features with predictions
            new_features, last_gross, history_gross, history_diff = update_features(
                y_pred, last_gross, input_features, target_features,
                history_gross, history_diff
            )
            
            # Update window (remove oldest steps, add new features)
            current_window = np.vstack([current_window[new_features.shape[0]:], new_features])
        
        # Combine predictions and trim to requested steps
        all_predictions = np.vstack(predictions)
        return all_predictions[:steps]

    # Main evaluation loop
    for model_name, model_data in models.items():
        if 'error' in model_data or 'model' not in model_data:
            continue  # Skip failed models
            
        model_eval = {'metrics': {}, 'exog_metrics': {}, 'figures': []}
        model = model_data['model']
        scalers = model_data['scalers']
        
        try:
            # Prepare validation window (last n_steps before test set)
            val_start = -(test_size + n_steps)
            val_end = -test_size
            X_val = df[input_features].iloc[val_start:val_end].values
            
            # Prepare future window (last n_steps in data)
            X_last = df[input_features].iloc[-n_steps:].values
            
            # Check if recursive forecasting is possible
            exog_in_input = [f for f in input_features 
                             if f not in ['y_diff', config['target_col']] 
                             and not f.startswith(('SMA_', 'EMA_', 'Volatility_'))]
            exog_in_target = [f for f in exog_in_input if f in target_features]
            can_recursive = len(exog_in_input) == len(exog_in_target)
            
            if test_size > m_steps and not can_recursive:
                raise ValueError(
                    f"Cannot recursively forecast for {model_name} - "
                    "exogenous features not in target_features"
                )
            
            # Get predictions using recursive engine
            y_val_pred = recursive_forecast(
                model, scalers, X_val, test_size, 
                model_name, n_steps, n_features
            )
            
            future_pred = recursive_forecast(
                model, scalers, X_last, horizon, 
                model_name, n_steps, n_features
            )
            
            # Get true values for validation period
            y_val_true = df[target_features].iloc[-test_size:].values
            
            # Ensure matching lengths
            min_val_length = min(len(y_val_true), len(y_val_pred))
            
            # Generate figures and metrics for each target feature
            for i, feat in enumerate(target_features):
                # Validation plot
                fig, val_path = save_true_vs_pred_1d(
                    y_true=y_val_true[:min_val_length, i],
                    y_pred=y_val_pred[:min_val_length, i],
                    title=f"{model_name.upper()} Validation: {feat}",
                    ylabel=feat
                )
                model_eval['figures'].append(val_path)
                
                # Metrics calculation
                y_true_feat = y_val_true[:min_val_length, i]
                y_pred_feat = y_val_pred[:min_val_length, i]
                
                metrics = {
                    'MAE': mean_absolute_error(y_true_feat, y_pred_feat),
                    'RMSE': np.sqrt(mean_squared_error(y_true_feat, y_pred_feat)),
                    'R2': r2_score(y_true_feat, y_pred_feat),
                    'MAPE': np.mean(np.abs((y_true_feat - y_pred_feat) / 
                                        np.maximum(np.abs(y_true_feat), 1))) * 100
                }
                
                # Classify metrics
                if feat == 'y_diff' or feat == target_col:
                    model_eval['metrics'][feat] = metrics
                else:
                    model_eval['exog_metrics'][feat] = metrics
            
            # Future plot for first target feature
            if target_features:
                fig, future_path = save_future_timeseries(
                    future_pred[:, 0],
                    title=f"{model_name.upper()} Future: {target_features[0]}",
                    ylabel=target_features[0]
                )
                model_eval['figures'].append(future_path)
            
            # Scatter plot for main target
            if target_features:
                plt.figure(figsize=(10, 6))
                plt.scatter(y_val_true[:min_val_length, 0], 
                            y_val_pred[:min_val_length, 0], 
                            alpha=0.6)
                plt.plot([y_val_true[:, 0].min(), y_val_true[:, 0].max()],
                         [y_val_true[:, 0].min(), y_val_true[:, 0].max()],
                         'k--', lw=2)
                plt.title(f'{model_name} Predicted vs Actual: {target_features[0]}')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.grid(True)
                plt.show()
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight')
                    plt.close()
                    model_eval['figures'].append(tmpfile.name)
            
            # Store forecasts
            future_forecasts[model_name] = {
                'validation': y_val_pred.tolist(),
                'future': future_pred.tolist(),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            evaluation[model_name] = model_eval
            
        except Exception as e:
            logging.error(f"Evaluation failed for {model_name}: {str(e)}", exc_info=True)
            model_eval['error'] = str(e)
            evaluation[model_name] = model_eval
    
    return evaluation, future_forecasts


# ==================================================================
# ULR (Univariate Linear Regression) Specialized Section
# ==================================================================
def evaluate_ulr_model(model, scalers, df, config):
    """Specialized evaluation for ULR models with recursive forecasting"""
    n_steps = config['n_steps']
    m_steps = config['m_steps']
    horizon = config['horizon']
    test_size = config['test_size']
    input_features = config['input_features']
    target_features = config['target_features']
    
    # Prepare validation window (last n_steps before test set)
    X_val = df[input_features].iloc[-(test_size + n_steps):-test_size].values
    
    # Prepare future window (very last n_steps in data)
    X_last = df[input_features].iloc[-n_steps:].values
    
    # Initialize predictions
    y_val_pred = []
    future_pred = []
    
    # Track the last known GROSS_ADDS value
    last_gross = X_val[-1, input_features.index(config['target_col'])] if config['target_col'] in input_features else 0
    
    # Recursive forecasting for validation
    current_window = X_val.copy()
    for _ in range(test_size):
        # Prepare input
        X_flat = current_window.reshape(1, -1)
        X_scaled = scalers['X'].transform(X_flat)
        y_pred_scaled = model.predict(X_scaled)
        y_pred_2d = y_pred_scaled.reshape(-1, len(target_features))
        y_pred = scalers['y'].inverse_transform(y_pred_2d)
        
        # For ULR, we typically have one target
        prediction = y_pred[0, 0] if y_pred.size > 0 else 0
        y_val_pred.append(prediction)
        
        # Update features
        new_features, last_gross = update_features(
            y_pred, last_gross, input_features, target_features
        )
        
        # Update window
        current_window = np.vstack([current_window[1:], new_features])
    
    # Recursive forecasting for future
    current_window = X_last.copy()
    last_gross = X_last[-1, input_features.index(config['target_col'])] if config['target_col'] in input_features else 0
    for _ in range(horizon):
        X_flat = current_window.reshape(1, -1)
        X_scaled = scalers['X'].transform(X_flat)
        y_pred_scaled = model.predict(X_scaled)
        y_pred_2d = y_pred_scaled.reshape(-1, len(target_features))
        y_pred = scalers['y'].inverse_transform(y_pred_2d)
        
        prediction = y_pred[0, 0] if y_pred.size > 0 else 0
        future_pred.append(prediction)
        
        # Update features
        new_features, last_gross = update_features(
            y_pred, last_gross, input_features, target_features
        )
        
        # Update window
        current_window = np.vstack([current_window[1:], new_features])
    
    return np.array(y_val_pred), np.array(future_pred)

# ==================================================================
# RNN/CNN-BiLSTM Specialized Section
# ==================================================================
def evaluate_rnn_model(model, scalers, df, config):
    """Specialized evaluation for RNN models with recursive forecasting"""
    n_steps = config['n_steps']
    m_steps = config['m_steps']
    horizon = config['horizon']
    test_size = config['test_size']
    input_features = config['input_features']
    target_features = config['target_features']
    
    # Prepare validation window (last n_steps before test set)
    X_val = df[input_features].iloc[-(test_size + n_steps):-test_size].values
    
    # Prepare future window (very last n_steps in data)
    X_last = df[input_features].iloc[-n_steps:].values
    
    # Initialize predictions
    y_val_pred = []
    future_pred = []
    
    # Track the last known GROSS_ADDS value
    last_gross = X_val[-1, input_features.index(config['target_col'])] if config['target_col'] in input_features else 0
    
    # Recursive forecasting for validation
    current_window = X_val.copy()
    for _ in range(0, test_size, m_steps):
        # Prepare input
        X_scaled = scalers['X'].transform(current_window)
        X_3d = X_scaled.reshape(1, n_steps, len(input_features))
        y_pred_scaled = model.predict(X_3d)
        y_pred_2d = y_pred_scaled.reshape(-1, len(target_features))
        y_pred = scalers['y'].inverse_transform(y_pred_2d)
        
        # Store predictions
        for i in range(min(m_steps, test_size - len(y_val_pred))):
            y_val_pred.append(y_pred[0, i, 0] if len(target_features) > 0 else y_pred[0, i])
        
        # Update features
        new_features, last_gross = update_features(
            y_pred[0], last_gross, input_features, target_features
        )
        
        # Update window
        current_window = np.vstack([current_window[m_steps:], new_features])
    
    # Recursive forecasting for future
    current_window = X_last.copy()
    last_gross = X_last[-1, input_features.index(config['target_col'])] if config['target_col'] in input_features else 0
    for _ in range(0, horizon, m_steps):
        X_scaled = scalers['X'].transform(current_window)
        X_3d = X_scaled.reshape(1, n_steps, len(input_features))
        y_pred_scaled = model.predict(X_3d)
        y_pred_2d = y_pred_scaled.reshape(-1, len(target_features))
        y_pred = scalers['y'].inverse_transform(y_pred_2d)
        
        # Store predictions
        for i in range(min(m_steps, horizon - len(future_pred))):
            future_pred.append(y_pred[0, i, 0] if len(target_features) > 0 else y_pred[0, i])
        
        # Update features
        new_features, last_gross = update_features(
            y_pred[0], last_gross, input_features, target_features
        )
        
        # Update window
        current_window = np.vstack([current_window[m_steps:], new_features])
    
    return np.array(y_val_pred), np.array(future_pred)


def update_features(predictions, last_gross, input_features, target_features, history_gross=None, history_diff=None):
    """
    Recalculates features based on new predictions during recursive forecasting.
    
    Args:
        predictions: 2D array of predicted values (m_steps x n_target_features)
        last_gross: Last known GROSS_ADDS value
        input_features: List of input feature names
        target_features: List of target feature names
        history_gross: History of GROSS_ADDS values for rolling calculations
        history_diff: History of y_diff values for rolling calculations
        
    Returns:
        updated_features: 2D array of updated features (m_steps x n_input_features)
        new_gross: Updated GROSS_ADDS value after predictions
        new_history_gross: Updated GROSS_ADDS history
        new_history_diff: Updated y_diff history
    """
    # Initialize histories if not provided
    if history_gross is None:
        history_gross = []
    if history_diff is None:
        history_diff = []
    
    updated_features = []
    current_gross = last_gross
    window_sizes = config.get('rolling_windows', [7, 30, 60, 10])
    
    # Process each prediction in the window
    for pred in predictions:
        # 1. Handle GROSS_ADDS and y_diff relationship
        if 'y_diff' in target_features:
            y_diff = pred[target_features.index('y_diff')]
            current_gross += y_diff
        elif config['target_col'] in target_features:
            current_gross = pred[target_features.index(config['target_col'])]
            y_diff = current_gross - last_gross
        else:
            # Default to 0 if neither is predicted
            y_diff = 0
        
        # Update histories
        history_gross.append(current_gross)
        history_diff.append(y_diff)
        last_gross = current_gross
        
        # 2. Build feature vector in input_features order
        feature_vector = []
        for feat in input_features:
            # Handle base features
            if feat == 'y_diff':
                feature_vector.append(y_diff)
            elif feat == config['target_col']:
                feature_vector.append(current_gross)
                
            # Handle rolling features for GROSS_ADDS
            elif feat.startswith('SMA_') and config['target_col'] in feat:
                window = int(feat.split('_')[1])
                if len(history_gross) >= window:
                    sma = np.mean(history_gross[-window:])
                    feature_vector.append(sma)
                else:
                    feature_vector.append(0)  # Not enough history
                    
            elif feat.startswith('EMA_') and config['target_col'] in feat:
                window = int(feat.split('_')[1])
                if len(history_gross) >= window:
                    ema = pd.Series(history_gross).ewm(span=window, adjust=False).mean().iloc[-1]
                    feature_vector.append(ema)
                else:
                    feature_vector.append(0)
                    
            elif feat.startswith('Volatility_') and config['target_col'] in feat:
                window = int(feat.split('_')[1])
                if len(history_gross) >= window:
                    vol = np.std(history_gross[-window:])
                    feature_vector.append(vol)
                else:
                    feature_vector.append(0)
                    
            # Handle rolling features for y_diff
            elif feat.startswith('SMA_') and 'diff' in feat:
                window = int(feat.split('_')[1])
                if len(history_diff) >= window:
                    sma = np.mean(history_diff[-window:])
                    feature_vector.append(sma)
                else:
                    feature_vector.append(0)
                    
            elif feat.startswith('EMA_') and 'diff' in feat:
                window = int(feat.split('_')[1])
                if len(history_diff) >= window:
                    ema = pd.Series(history_diff).ewm(span=window, adjust=False).mean().iloc[-1]
                    feature_vector.append(ema)
                else:
                    feature_vector.append(0)
                    
            elif feat.startswith('Volatility_') and 'diff' in feat:
                window = int(feat.split('_')[1])
                if len(history_diff) >= window:
                    vol = np.std(history_diff[-window:])
                    feature_vector.append(vol)
                else:
                    feature_vector.append(0)
                    
            # Handle exogenous features
            elif feat in target_features:
                idx = target_features.index(feat)
                feature_vector.append(pred[idx])
                
            # Fallback for unhandled features
            else:
                feature_vector.append(0)
                
        updated_features.append(feature_vector)
    
    return (
        np.array(updated_features),
        current_gross,
        history_gross,
        history_diff
    )    