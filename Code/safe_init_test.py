import numpy as np
import pandas as pd
from utils.model_builder import get_scaler, split_data, build_rf_model, build_lgb_model
from utils.model_saver import get_available_models, load_model_and_scaler
from config import features

def safe_initialize_data():
    """
    Safely initialize data without any session state contamination
    """
    # Load the pivot data fresh
    pivot_path = "./Data/Pivot/pivot_output.csv"
    
    try:
        pivot_df = pd.read_csv(pivot_path)
        print(f"Loaded pivot data: {pivot_df.shape}")
        
        # Prepare clean data
        X = pivot_df[features].copy()
        y = pivot_df['Fault'].copy()
        
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}, y dtype: {y.dtype}")
        
        # Get scaler and split data
        scaler, X_scaled = get_scaler(X)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y)
        
        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
        print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
        
        # Check if X_test is purely numeric
        if not np.issubdtype(X_test.dtype, np.number):
            print(f"ERROR: X_test has non-numeric dtype: {X_test.dtype}")
            return None
        
        # Try to load existing models
        available_models = get_available_models()
        
        rf_model = None
        lgb_model = None
        
        if 'RandomForest' in available_models:
            try:
                rf_model, rf_scaler, _, _ = load_model_and_scaler("RandomForest")
                print(f"Loaded RF model: {type(rf_model)}")
            except Exception as e:
                print(f"Could not load RF model: {e}")
        
        if 'LightGBM' in available_models:
            try:
                lgb_model, lgb_scaler, _, _ = load_model_and_scaler("LightGBM")
                print(f"Loaded LGB model: {type(lgb_model)}")
            except Exception as e:
                print(f"Could not load LGB model: {e}")
        
        # Train models if not available
        if rf_model is None:
            print("Training RF model...")
            rf_model = build_rf_model(X_train, y_train)
            
        if lgb_model is None:
            print("Training LGB model...")
            lgb_model = build_lgb_model(X_train, y_train)
        
        return {
            'pivot_df': pivot_df,
            'rf_model': rf_model,
            'lgb_model': lgb_model,
            'rf_scaler': scaler,
            'lgb_scaler': scaler,
            'X_test': X_test,
            'y_test': y_test
        }
        
    except Exception as e:
        print(f"Error in safe_initialize_data: {e}")
        return None

if __name__ == "__main__":
    result = safe_initialize_data()
    if result:
        print("Safe initialization successful!")
        
        # Test predictions
        from utils.model_builder import predict_model_with_proba
        
        try:
            rf_pred, rf_proba = predict_model_with_proba(result['rf_model'], result['X_test'])
            print(f"RF prediction successful: {rf_pred.shape}")
        except Exception as e:
            print(f"RF prediction failed: {e}")
            
        try:
            lgb_pred, lgb_proba = predict_model_with_proba(result['lgb_model'], result['X_test'])
            print(f"LGB prediction successful: {lgb_pred.shape}")
        except Exception as e:
            print(f"LGB prediction failed: {e}")
    else:
        print("Safe initialization failed!")
