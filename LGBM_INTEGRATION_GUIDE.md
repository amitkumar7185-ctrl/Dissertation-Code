# LightGBM Integration and Model Comparison Guide

## 🚀 New Features Added

### 1. **LightGBM Model Support**
- **Enhanced Model Builder** (`utils/model_builder.py`):
  - `build_lgb_model()` - Train LightGBM models with customizable parameters
  - `compare_models()` - Side-by-side comparison of Random Forest vs LightGBM
  - `evaluate_model()` - Unified evaluation for both model types
  - `get_feature_importance()` - Feature importance for both models
  - `predict_model_with_proba()` - Compatible predictions for both models

### 2. **Multi-Model Persistence** (`utils/model_saver.py`):
- **Save Models**: `save_model_and_scaler(model, scaler, model_type="RandomForest|LightGBM")`
- **Load Models**: `load_model_and_scaler(model_type="RandomForest|LightGBM")`
- **Check Availability**: `get_available_models()` - Returns list of trained model types
- **Model Info**: `get_model_info(model_type)` - Get metadata about saved models

### 3. **Model Comparison Interface** (`views/model_comparison.py`):
- **Hyperparameter Tuning**: Interactive sliders for both model types
- **Side-by-Side Training**: Train and compare models simultaneously
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visual Comparisons**:
  - ROC and Precision-Recall curves
  - Confusion matrices
  - Feature importance comparison
  - Prediction agreement analysis
- **Model Export**: Save trained models and download comparison reports

### 4. **Enhanced Upload & Predict** (`views/upload_predict.py`):
- **Dynamic Model Selection**: Choose between available trained models
- **Model Metadata Display**: Show model type, features, and estimators
- **Better Error Handling**: Comprehensive validation and error messages

## 🎯 How to Use

### **Navigation**
1. Start the application: `streamlit run Code/main.py`
2. New menu option: **"Model Comparison"** in the sidebar
3. Use **"Upload & Predict"** for predictions with model selection

### **Model Comparison Workflow**
1. **Access**: Click "Model Comparison" in sidebar
2. **Configure**: Adjust hyperparameters for both models:
   - **Random Forest**: Trees, Max Depth, Min Samples Split
   - **LightGBM**: Boost Rounds, Learning Rate, Number of Leaves
3. **Train**: Click "🏃‍♂️ Train and Compare Models"
4. **Analyze**: Review results in tabs:
   - 📈 ROC & PR Curves
   - 🎯 Confusion Matrix
   - 🔍 Feature Importance
   - 📊 Prediction Comparison
   - 💾 Save Models
5. **Export**: Save best performing models and download reports

### **Upload & Predict with Model Selection**
1. **Access**: Click "Upload & Predict" in sidebar
2. **Select Model**: Choose from available trained models (Random Forest/LightGBM)
3. **Upload Data**: CSV file with required feature columns
4. **Get Predictions**: Comprehensive report with risk categorization

## 🏆 Key Benefits

### **Performance Comparison**
- **Random Forest**: Robust, interpretable, handles mixed data well
- **LightGBM**: Fast training, high accuracy, memory efficient, handles large datasets

### **Side-by-Side Analysis**
- **Metrics Comparison**: Direct performance comparison
- **Feature Importance**: Compare what each model considers important
- **Prediction Agreement**: See where models agree/disagree
- **Visual Analytics**: ROC curves, confusion matrices, correlation plots

### **Flexible Deployment**
- **Model Choice**: Use the best performing model for your data
- **A/B Testing**: Compare predictions from both models
- **Ensemble Potential**: Combine predictions for better results

## 🔧 Technical Details

### **LightGBM Configuration**
```python
default_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
```

### **Model File Structure**
```
models/
├── rf_fault_model.pkl          # Random Forest model
├── rf_scaler.pkl               # Random Forest scaler
├── rf_model_metadata.json      # Random Forest metadata
├── lgb_fault_model.pkl         # LightGBM model
├── lgb_scaler.pkl              # LightGBM scaler
├── lgb_model_metadata.json     # LightGBM metadata
└── backups/                    # Versioned backups
```

## 🎉 Ready to Use!

The application now provides comprehensive machine learning capabilities with:
- ✅ Dual model support (Random Forest + LightGBM)
- ✅ Interactive model comparison
- ✅ Flexible model selection for predictions
- ✅ Enhanced performance analytics
- ✅ Professional model management

Simply run `streamlit run Code/main.py` and explore the new "Model Comparison" section!
