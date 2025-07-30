import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
from config import features

def get_scaler(X):
    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled
    
def split_data(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Ensure all outputs are numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return X_train, X_test, y_train, y_test

# Train Random Forest model
def build_rf_model(X_train, y_train):    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# Train LightGBM model
def build_lgb_model(X_train, y_train, params=None):
    """
    Build and train a LightGBM model
    """
    if params is None:
        # Calculate class weights for imbalanced dataset
        unique_classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        scale_pos_weight = counts[0] / counts[1] if len(counts) > 1 else 1.0
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'scale_pos_weight': scale_pos_weight
            # Removed 'is_unbalance': True as it conflicts with scale_pos_weight
        }
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    
    return model

# Enhanced prediction function that works with both models
def predict_model_with_proba(model, X_test):
    """
    Make predictions with both model types
    """
    # Validate inputs
    if model is None:
        raise ValueError("Model cannot be None")
    
    if X_test is None:
        raise ValueError("X_test cannot be None")
    
    # Ensure X_test is a clean numpy array
    if hasattr(X_test, 'values'):  # DataFrame
        X_test = X_test.values
    
    # Convert to numpy array and validate
    try:
        X_test_array = np.asarray(X_test)
        
        # Check for object dtype which might contain invalid data
        if X_test_array.dtype == 'object':
            # Try to examine what's in the array
            flat_array = X_test_array.flatten()
            for i, val in enumerate(flat_array[:10]):  # Check first 10 elements
                if hasattr(val, '__class__') and 'Booster' in str(val.__class__):
                    raise ValueError(f"Found LightGBM Booster object in X_test at position {i}. X_test should contain only numerical data.")
                if not isinstance(val, (int, float, np.number)):
                    raise ValueError(f"X_test contains non-numeric data at position {i}: {type(val)} = {val}")
            
            # Try to convert to float
            try:
                X_test_array = X_test_array.astype(float)
            except:
                raise ValueError("X_test contains data that cannot be converted to numeric")
        
        # Ensure it's numeric
        if not np.issubdtype(X_test_array.dtype, np.number):
            raise ValueError(f"X_test must be numeric, got dtype: {X_test_array.dtype}")
        
        X_test = X_test_array
        
    except Exception as e:
        raise ValueError(f"Invalid X_test data: {e}")
    
    try:
        if hasattr(model, 'predict_proba'):  # Random Forest
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:  # LightGBM
            num_iteration = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration is not None else None
            y_proba = model.predict(X_test, num_iteration=num_iteration)
            y_pred = (y_proba > 0.5).astype(int)
        
        return y_pred, y_proba
        
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}. Model type: {type(model)}, X_test shape: {X_test.shape}, X_test dtype: {X_test.dtype}")

# Model evaluation function
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred, y_proba = predict_model_with_proba(model, X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
    }
    
    return metrics, y_pred, y_proba

# Feature importance function
def get_feature_importance(model, model_type=None):
    """
    Get feature importance from both model types
    Returns numpy array of importance values
    """
    if hasattr(model, 'feature_importances_'):  # Random Forest
        importance = model.feature_importances_
    else:  # LightGBM
        importance = model.feature_importance(importance_type='gain')
    
    # Ensure we have the right number of features
    if len(importance) != len(features):
        # Pad or truncate to match feature names
        if len(importance) > len(features):
            importance = importance[:len(features)]
        else:
            # Pad with zeros if needed
            importance = list(importance) + [0.0] * (len(features) - len(importance))
    
    return np.array(importance)

# Feature importance DataFrame function for detailed analysis
def get_feature_importance_df(model, model_type=None):
    """
    Get feature importance as DataFrame for visualization
    """
    importance = get_feature_importance(model, model_type)
    
    # Create DataFrame for easy visualization
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df

# Model comparison function
def compare_models(X_train, X_test, y_train, y_test):
    """
    Train and compare Random Forest vs LightGBM models
    """
    results = {}
    
    # Train Random Forest
    rf_model = build_rf_model(X_train, y_train)
    rf_metrics, rf_pred, rf_proba = evaluate_model(rf_model, X_test, y_test)
    rf_importance = get_feature_importance_df(rf_model)
    
    results['RandomForest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'feature_importance': rf_importance
    }
    
    # Train LightGBM
    lgb_model = build_lgb_model(X_train, y_train)
    lgb_metrics, lgb_pred, lgb_proba = evaluate_model(lgb_model, X_test, y_test)
    lgb_importance = get_feature_importance_df(lgb_model)
    
    results['LightGBM'] = {
        'model': lgb_model,
        'metrics': lgb_metrics,
        'predictions': lgb_pred,
        'probabilities': lgb_proba,
        'feature_importance': lgb_importance
    }
    
    return results
# Predictions (legacy function for backward compatibility)
def predict_model(model, X_test):
    y_pred, _ = predict_model_with_proba(model, X_test)
    return y_pred