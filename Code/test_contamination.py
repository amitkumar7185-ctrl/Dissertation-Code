"""
Simple test to check if the session state contamination issue exists
"""
import streamlit as st
import numpy as np
from utils.model_builder import build_rf_model, build_lgb_model, predict_model_with_proba

# Create test data
X_test = np.random.rand(10, 5)
y_test = np.random.randint(0, 2, 10)

# Train simple models
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)

rf_model = build_rf_model(X_train, y_train)
lgb_model = build_lgb_model(X_train, y_train)

print("Testing with clean data...")
print(f"X_test type: {type(X_test)}")
print(f"X_test dtype: {X_test.dtype}")

# Test RF
try:
    rf_pred, rf_proba = predict_model_with_proba(rf_model, X_test)
    print("✅ RF prediction successful")
except Exception as e:
    print(f"❌ RF prediction failed: {e}")

# Test LGB
try:
    lgb_pred, lgb_proba = predict_model_with_proba(lgb_model, X_test)
    print("✅ LGB prediction successful")
except Exception as e:
    print(f"❌ LGB prediction failed: {e}")

# Now simulate the Streamlit session state issue
print("\n" + "="*50)
print("Simulating session state contamination...")

# Create a list that accidentally contains a model object
contaminated_data = [1.0, 2.0, lgb_model, 4.0, 5.0]
contaminated_array = np.array(contaminated_data, dtype=object)

print(f"Contaminated data type: {type(contaminated_array)}")
print(f"Contaminated data dtype: {contaminated_array.dtype}")

try:
    rf_pred, rf_proba = predict_model_with_proba(rf_model, contaminated_array)
    print("✅ Contaminated prediction successful (unexpected)")
except Exception as e:
    print(f"❌ Contaminated prediction failed (expected): {e}")

print("\nTest completed!")
