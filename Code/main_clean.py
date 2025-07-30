import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.load_data import load_data
from utils.model_builder import get_scaler, split_data, build_rf_model, build_lgb_model
from utils.model_saver import save_model_and_scaler, load_model_and_scaler, get_available_models
from ui.sidebar_menu import display_menu
from views.dashboard import render_dashboard
from views.eda import render_eda  
from views.upload_predict import render_upload_predict
from views.model_comparison import run_model_comparison

from config import features

# Configure the page
st.set_page_config(
    page_title="Elevator Fault Detection Dashboard", 
    page_icon="🔧",   
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_data_and_models():
    """Clean initialization of data and models"""
    
    if st.session_state.get('data_initialized', False):
        return
    
    # Load pivot data with correct path
    pivot_path = "../Data/Pivot/pivot_output.csv"
    
    if not os.path.exists(pivot_path):
        st.error(f"❌ Data file not found: {pivot_path}")
        st.stop()
    
    try:
        # Load and validate data
        pivot_df = pd.read_csv(pivot_path)
        required_columns = features + ['Fault']
        missing_columns = [col for col in required_columns if col not in pivot_df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing columns: {missing_columns}")
            st.stop()
        
        # Prepare clean data
        X = pivot_df[features].copy()
        y = pivot_df['Fault'].copy()
        scaler, X_scaled = get_scaler(X)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y)
        
        # Validate data types
        assert isinstance(X_test, np.ndarray), f"X_test wrong type: {type(X_test)}"
        assert isinstance(y_test, np.ndarray), f"y_test wrong type: {type(y_test)}"
        assert np.issubdtype(X_test.dtype, np.number), f"X_test wrong dtype: {X_test.dtype}"
        
        # Try to load existing models
        available_models = get_available_models()
        rf_model = None
        lgb_model = None
        
        if 'RandomForest' in available_models:
            try:
                rf_model, rf_scaler, _, _ = load_model_and_scaler("RandomForest")
            except:
                pass
        
        if 'LightGBM' in available_models:
            try:
                lgb_model, lgb_scaler, _, _ = load_model_and_scaler("LightGBM")
            except:
                pass
        
        # Train models if not available
        if rf_model is None:
            with st.spinner("Training Random Forest..."):
                rf_model = build_rf_model(X_train, y_train)
                save_model_and_scaler(rf_model, scaler, "RandomForest")
        
        if lgb_model is None:
            with st.spinner("Training LightGBM..."):
                lgb_model = build_lgb_model(X_train, y_train)
                save_model_and_scaler(lgb_model, scaler, "LightGBM")
        
        # Store clean data in session state
        st.session_state.pivot_df = pivot_df
        st.session_state.rf_model = rf_model
        st.session_state.lgb_model = lgb_model
        st.session_state.rf_scaler = scaler
        st.session_state.lgb_scaler = scaler
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.data_initialized = True
        
        st.success("✅ Data and models initialized successfully!")
        
    except Exception as e:
        st.error(f"❌ Initialization error: {e}")
        st.stop()

def main():
    menu = display_menu()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Initialize data and models
    initialize_data_and_models()
    
    # Route to different views
    if menu == "Dashboard":
        render_dashboard(
            st.session_state.pivot_df, 
            st.session_state.rf_model, 
            st.session_state.rf_scaler,
            st.session_state.X_test, 
            st.session_state.y_test, 
            st.session_state.lgb_model, 
            st.session_state.lgb_scaler
        )
        
    elif menu == "Upload & Predict":
        render_upload_predict()
        
    elif menu == "EDA":
        render_eda(st.session_state.pivot_df)
        
    elif menu == "Model Comparison":
        run_model_comparison(st.session_state.pivot_df)

if __name__ == "__main__":
    main()
