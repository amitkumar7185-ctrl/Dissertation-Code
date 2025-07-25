
import streamlit as st
import pandas as pd
import os
from ui.sidebar_menu import display_menu
from utils.model_loader import load_models
from utils.model_saver import save_model_and_scaler, load_model_and_scaler, check_model_exists, get_model_info
from views.dashboard import render_dashboard
from views.upload_predict import render_upload_predict
from views.eda import render_eda
from views.shap_explain import render_shap
from views.model_management import render_model_management
from utils.load_data import load_Data
from utils.print_helper import print_data_distribution, prin_df_head,print_dataset_info,print_classification,print_confusion_matrix
from utils.preprocessing import prepare_pivoted_data, label_faults
from utils.model_builder import split_data, build_rf_model, predict_model, get_scaler
from config import features

# Configure the page
st.set_page_config(
    page_title="Elevator Fault Detection Dashboard",
    page_icon="🔧",   
    initial_sidebar_state="expanded"
)

#from utils.save_util import save_pivot_df

def initialize_data_and_model():
    """Initialize data and model only once using session state"""
    if 'data_initialized' not in st.session_state:
        output_path = 'Data\\pivot\\pivot_output.csv'
        
        # Check if pivot file already exists
        if os.path.exists(output_path):
            with st.spinner("Loading existing pivot data..."):
                st.info("📁 Found existing pivot file. Loading from cache...")
                
                # Load pivot data from existing file
                try:
                    pivot_df = pd.read_csv(output_path, index_col=0)
                    
                    # Validate that required columns exist
                    required_columns = features + ['Fault']
                    missing_columns = [col for col in required_columns if col not in pivot_df.columns]
                    
                    if missing_columns:
                        st.warning(f"❌ Pivot file missing columns: {missing_columns}")
                        st.info("🔄 Reprocessing original data...")
                        pivot_df = process_original_data()
                    else:
                        st.session_state.pivot_df = pivot_df
                        
                        # Also load original data for backup (optional)
                        df = load_Data()
                        st.session_state.df = df
                        
                        st.success(f"✅ Loaded {len(pivot_df)} records from existing pivot file!")
                    
                except Exception as e:
                    st.warning(f"❌ Error loading pivot file: {e}")
                    st.info("🔄 Falling back to processing original data...")
                    # Fall back to original processing
                    pivot_df = process_original_data()
                    
        else:
            with st.spinner("Processing original data (first time setup)..."):
                st.info("🔄 No pivot file found. Processing original data...")
                pivot_df = process_original_data()
        
        # Check for force retrain flag
        force_retrain = st.session_state.get('force_retrain', False)
        
        # Try to load existing model first (unless forced to retrain)
        if not force_retrain and check_model_exists():
            with st.spinner("Loading existing trained model..."):
                rf_model, rf_scaler, metadata, message = load_model_and_scaler()
                
                if rf_model is not None and rf_scaler is not None:
                    # Validate model compatibility with current data
                    try:
                        X = pivot_df[features]
                        X_scaled = rf_scaler.transform(X[:5])  # Test with first 5 rows
                        _ = rf_model.predict(X_scaled)
                        
                        # Model is compatible, use it
                        st.info("🤖 Found existing trained model. Loading...")
                        st.success(message)
                        
                        if metadata:
                            st.info(f"📅 Model created: {metadata.get('created_at', 'Unknown')}")
                            st.info(f"🌳 Estimators: {metadata.get('n_estimators', 'Unknown')}")
                        
                        # Still need to split data for test metrics
                        y = pivot_df['Fault']
                        X_scaled_full = rf_scaler.transform(X)
                        X_train, X_test, y_train, y_test = split_data(X_scaled_full, y)
                        
                        # Store in session state
                        st.session_state.rf_model = rf_model
                        st.session_state.rf_scaler = rf_scaler
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.data_initialized = True
                        
                        # Clear force retrain flag
                        if 'force_retrain' in st.session_state:
                            del st.session_state.force_retrain
                        
                        return
                        
                    except Exception as e:
                        st.warning(f"⚠️ Existing model incompatible with current data: {e}")
                        st.info("🔄 Training new model...")
                else:
                    st.warning("❌ Failed to load existing model. Training new model...")
        
        # Train new model (either no existing model or forced retrain)
        with st.spinner("Training machine learning model..."):
            if force_retrain:
                st.info("🔄 Retraining model as requested...")
            else:
                st.info("🤖 No compatible model found. Training new model...")
            
            X = pivot_df[features]
            y = pivot_df['Fault']
            rf_scaler, X_scaled = get_scaler(X)
            X_train, X_test, y_train, y_test = split_data(X_scaled, y)
            rf_model = build_rf_model(X_train, y_train)
            
            # Save the newly trained model
            success, save_message = save_model_and_scaler(rf_model, rf_scaler)
            if success:
                st.success(save_message)
            else:
                st.warning(save_message)
            
            # Store in session state
            st.session_state.rf_model = rf_model
            st.session_state.rf_scaler = rf_scaler
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.data_initialized = True
            
            # Clear force retrain flag
            if 'force_retrain' in st.session_state:
                del st.session_state.force_retrain
            
            st.success("🤖 Model trained and ready for predictions!")

def process_original_data():
    """Process original data and save pivot file"""
    output_path = 'Data\\pivot\\pivot_output.csv'
    
    # Load the Data and pre process
    df = load_Data()
    st.session_state.df = df
    
    pivot_df = prepare_pivoted_data(df)   
    pivot_df = label_faults(pivot_df)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed data
    pivot_df.to_csv(output_path, index=True)
    st.session_state.pivot_df = pivot_df
    
    st.success(f"💾 Processed and saved {len(pivot_df)} records to pivot file!")
    
    return pivot_df

def main():
    menu = display_menu()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("Data\\pivot", exist_ok=True)
    
    # Initialize data and model only once
    initialize_data_and_model()
    
    # Clear the main area for each view
    if menu == "Dashboard":
        render_dashboard(st.session_state.pivot_df, st.session_state.rf_model, st.session_state.rf_scaler, 
                        st.session_state.X_test, st.session_state.y_test)
            
    elif menu == "Upload & Predict":
        st.subheader("Upload & Predict")
        st.write("Upload your data and get predictions")
        render_upload_predict(st.session_state.rf_model, st.session_state.rf_scaler)
        
    elif menu == "EDA":
        render_eda(st.session_state.pivot_df)
        
    elif menu == "SHAP Explainability":
        st.subheader("SHAP Explainability")
        st.write("Understand your model's predictions with SHAP")
        #render_shap(st.session_state.rf_model, st.session_state.rf_scaler, st.session_state.pivot_df)
    
    elif menu == "Model Management":
        render_model_management(
            st.session_state.pivot_df, 
            st.session_state.rf_model, 
            st.session_state.rf_scaler,
            st.session_state.X_test, 
            st.session_state.y_test
        )

if __name__ == "__main__":
    main()
