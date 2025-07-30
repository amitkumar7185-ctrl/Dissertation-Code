"""
Test script to verify LightGBM integration
"""
import sys
import os

# Add the Code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Code'))

try:
    from utils.model_builder import build_lgb_model, compare_models
    from utils.model_saver import save_model_and_scaler, load_model_and_scaler, get_available_models
    import lightgbm as lgb
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    
    print("✅ All imports successful!")
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Scale data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("📊 Sample data created")
    
    # Test LightGBM model building
    lgb_model = build_lgb_model(X_train, y_train)
    print("🌟 LightGBM model trained successfully")
    
    # Test predictions
    from utils.model_builder import predict_model_with_proba
    y_pred, y_proba = predict_model_with_proba(lgb_model, X_test)
    print(f"🎯 Predictions made: {len(y_pred)} samples")
    
    # Test model saving
    success, message = save_model_and_scaler(lgb_model, scaler, "LightGBM", "test_models")
    print(f"💾 Model saving: {message}")
    
    # Test model loading
    loaded_model, loaded_scaler, metadata, load_message = load_model_and_scaler("LightGBM", "test_models")
    print(f"📂 Model loading: {load_message}")
    
    # Test available models
    available = get_available_models("test_models")
    print(f"📋 Available models: {available}")
    
    print("\n🎉 LightGBM integration test completed successfully!")
    
except Exception as e:
    print(f"❌ Error during testing: {str(e)}")
    import traceback
    traceback.print_exc()
