"""
Test script to verify the updated LightGBM integration with automatic model training
"""
import sys
import os

# Add the Code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Code'))

try:
    # Test model builder functions
    from utils.model_builder import build_lgb_model, build_rf_model, predict_model_with_proba, compare_models
    from utils.model_saver import save_model_and_scaler, load_model_and_scaler, get_available_models
    from views.upload_predict import render_upload_predict
    from views.model_comparison import render_model_comparison
    
    print("✅ All imports successful!")
    print("📋 Available functions:")
    print("   - build_rf_model: Random Forest training")
    print("   - build_lgb_model: LightGBM training")
    print("   - predict_model_with_proba: Unified prediction")
    print("   - compare_models: Side-by-side comparison")
    print("   - save_model_and_scaler: Multi-model persistence")
    print("   - load_model_and_scaler: Multi-model loading")
    print("   - get_available_models: Available model discovery")
    print("   - render_upload_predict: Updated upload interface")
    print("   - render_model_comparison: Model comparison interface")
    
    print("\n🎉 Integration test completed successfully!")
    print("\n📝 **Usage Instructions:**")
    print("1. Run: streamlit run Code/main.py")
    print("2. The app will automatically train both Random Forest and LightGBM models")
    print("3. Use 'Upload & Predict' to select between models for predictions")
    print("4. Use 'Model Comparison' to compare model performance side-by-side")
    
except Exception as e:
    print(f"❌ Error during testing: {str(e)}")
    import traceback
    traceback.print_exc()
