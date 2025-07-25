
import streamlit as st
import os
from utils.model_saver import check_model_exists, get_model_info, list_model_versions

def display_menu():
    st.sidebar.title("🔧 Elevator Fault Detection")
    
    # Data Management Section
    st.sidebar.markdown("### 📊 Data Management")
    
    # Check if pivot file exists
    pivot_path = 'Data\\pivot\\pivot_output.csv'
    if os.path.exists(pivot_path):
        file_size = os.path.getsize(pivot_path) / 1024  # Size in KB
        st.sidebar.success(f"✅ Pivot file exists ({file_size:.1f} KB)")
        
        # Option to refresh data
        if st.sidebar.button("🔄 Refresh Data", help="Reprocess original data and update pivot file"):
            # Clear the existing pivot file and session state
            if os.path.exists(pivot_path):
                os.remove(pivot_path)
            if 'data_initialized' in st.session_state:
                del st.session_state.data_initialized
            st.experimental_rerun()
    else:
        st.sidebar.info("📁 No pivot file found - will process original data")
    
    # Model Management Section
    st.sidebar.markdown("### 🤖 Model Management")
    
    # Check if trained model exists
    if check_model_exists():
        model_info = get_model_info()
        if model_info:
            created_date = model_info.get('created_at', 'Unknown')[:10]  # Just the date part
            n_estimators = model_info.get('n_estimators', 'Unknown')
            st.sidebar.success(f"✅ Model trained ({created_date})")
            st.sidebar.info(f"🌳 Trees: {n_estimators}")
        else:
            st.sidebar.success("✅ Model file exists")
        
        # Option to retrain model
        if st.sidebar.button("🔄 Retrain Model", help="Train a new model with current data"):
            st.session_state.force_retrain = True
            if 'data_initialized' in st.session_state:
                del st.session_state.data_initialized
            st.experimental_rerun()
        
        # Show model versions (expandable)
        with st.sidebar.expander("📋 Model Versions"):
            versions = list_model_versions()
            if versions:
                for version in versions[:5]:  # Show last 5 versions
                    timestamp = version['timestamp']
                    size_kb = version['size_kb']
                    st.write(f"🕐 {timestamp} ({size_kb:.1f} KB)")
            else:
                st.write("No backup versions found")
    else:
        st.sidebar.info("🤖 No trained model - will train new model")
    
    # Add a clear button
    if st.sidebar.button("🔄 Clear Output", help="Clear all outputs and refresh"):
        st.experimental_rerun()
    
    st.sidebar.markdown("---")
    
    selected_menu = st.sidebar.radio("Choose View", [
        "Dashboard",
        "Upload & Predict", 
        "EDA", 
        "SHAP Explainability",
        "Model Management"
    ])
    
    # Track menu changes to clear content
    if 'current_menu' not in st.session_state:
        st.session_state.current_menu = selected_menu
    elif st.session_state.current_menu != selected_menu:
        st.session_state.current_menu = selected_menu
        # You can add additional cleanup here if needed
    
    return selected_menu
