import numpy as np
import pandas as pd
from utils.model_builder import build_rf_model, build_lgb_model, predict_model_with_proba, evaluate_model
from config import features

# Create test data that matches your actual features
np.random.seed(42)
n_samples = 100
n_features = len(features)

X_train = np.random.rand(n_samples, n_features)
y_train = np.random.randint(0, 2, n_samples)
X_test = np.random.rand(20, n_features)
y_test = np.random.randint(0, 2, 20)

print(f'Test data created: {n_samples} train samples, 20 test samples, {n_features} features')

# Test RF
try:
    rf_model = build_rf_model(X_train, y_train)
    rf_metrics, rf_pred, rf_proba = evaluate_model(rf_model, X_test, y_test)
    print('Random Forest workflow: SUCCESS')
    print(f'RF Accuracy: {rf_metrics["accuracy"]:.3f}')
except Exception as e:
    print(f'Random Forest workflow: FAILED - {e}')

# Test LGB
try:
    lgb_model = build_lgb_model(X_train, y_train)
    lgb_metrics, lgb_pred, lgb_proba = evaluate_model(lgb_model, X_test, y_test)
    print('LightGBM workflow: SUCCESS')
    print(f'LGB Accuracy: {lgb_metrics["accuracy"]:.3f}')
except Exception as e:
    print(f'LightGBM workflow: FAILED - {e}')

print("All tests completed!")
