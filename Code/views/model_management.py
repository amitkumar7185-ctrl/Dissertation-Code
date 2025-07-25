import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.model_saver import (
    save_model_and_scaler, 
    load_model_and_scaler, 
    check_model_exists, 
    get_model_info, 
    list_model_versions,
    load_specific_model_version
)
from utils.model_builder import predict_model
from utils.print_helper import print_classification, print_confusion_matrix

def render_model_management(pivot_df, current_model=None, current_scaler=None, X_test=None, y_test=None):
    """Render model management interface"""
    st.header("🤖 Model Management")
    
    # Current Model Status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Current Model Status")
        if current_model is not None:
            model_info = get_model_info()
            if model_info:
                st.success("✅ Model Loaded")
                st.metric("Model Type", model_info.get('model_type', 'Unknown'))
                st.metric("Trees (Estimators)", model_info.get('n_estimators', 'Unknown'))
                st.metric("Features", model_info.get('n_features', 'Unknown'))
                
                created_at = model_info.get('created_at', 'Unknown')
                if created_at != 'Unknown':
                    created_date = datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M")
                    st.metric("Created", created_date)
            else:
                st.success("✅ Model Loaded (No metadata)")
        else:
            st.error("❌ No model loaded")
    
    with col2:
        st.subheader("⚙️ Model Actions")
        
        # Retrain model button
        if st.button("🔄 Retrain Model", help="Train a new model with current data"):
            st.session_state.force_retrain = True
            if 'data_initialized' in st.session_state:
                del st.session_state.data_initialized
            st.experimental_rerun()
        
        # Save current model button (if model exists)
        if current_model is not None and current_scaler is not None:
            if st.button("💾 Save Current Model", help="Save current model with timestamp"):
                success, message = save_model_and_scaler(current_model, current_scaler)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Export model info button
        if current_model is not None:
            if st.button("📋 Export Model Info", help="Download model information"):
                model_info = get_model_info()
                if model_info:
                    # Create a comprehensive model report
                    report = {
                        "Model Information": model_info,
                        "Model Parameters": current_model.get_params(),
                        "Feature Names": list(pivot_df.columns[:-1]),  # Exclude 'Fault' column
                        "Data Shape": pivot_df.shape,
                        "Export Time": datetime.now().isoformat()
                    }
                    
                    # Convert to JSON string for download
                    import json
                    json_string = json.dumps(report, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Model Report",
                        data=json_string,
                        file_name=f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    # Model Performance Section
    if current_model is not None and X_test is not None and y_test is not None:
        st.subheader("📈 Current Model Performance")
        
        if st.button("🎯 Evaluate Current Model", key="eval_current"):
            with st.spinner("Evaluating model performance..."):
                y_pred = predict_model(current_model, X_test)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Classification Report:**")
                    print_classification(y_test, y_pred)
                
                with col2:
                    st.write("**Confusion Matrix:**")
                    fig = print_confusion_matrix(y_test, y_pred)
                    st.pyplot(fig)
                    plt.close(fig)
    
    # Model Versions Section
    st.subheader("📋 Model Version History")
    
    versions = list_model_versions()
    if versions:
        st.write(f"Found {len(versions)} model versions:")
        
        # Create a DataFrame for better display
        version_data = []
        for version in versions:
            timestamp = version['timestamp']
            # Parse timestamp for better display
            try:
                dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_date = timestamp
            
            version_data.append({
                "Timestamp": formatted_date,
                "Size (KB)": f"{version['size_kb']:.1f}",
                "File": version['file']
            })
        
        version_df = pd.DataFrame(version_data)
        st.dataframe(version_df, use_container_width=True)
        
        # Option to load a specific version
        st.subheader("🔄 Load Specific Version")
        
        if len(versions) > 0:
            selected_version = st.selectbox(
                "Select a version to load:",
                options=[v['timestamp'] for v in versions],
                format_func=lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            )
            
            if st.button("📥 Load Selected Version"):
                with st.spinner("Loading model version..."):
                    model, scaler, message = load_specific_model_version(selected_version)
                    
                    if model is not None and scaler is not None:
                        # Update session state with loaded model
                        st.session_state.rf_model = model
                        st.session_state.rf_scaler = scaler
                        st.success(message)
                        st.info("🔄 Please refresh the page to use the loaded model.")
                    else:
                        st.error(message)
    else:
        st.info("No model versions found. Train a model to create versions.")
    
    # Model Comparison Section (if multiple models exist)
    if len(versions) > 1:
        st.subheader("⚖️ Model Comparison")
        st.info("🚧 Model comparison feature coming soon! This will allow you to compare performance across different model versions.")
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        st.subheader("🗑️ Cleanup Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Old Versions", help="Keep only the latest 3 model versions"):
                if len(versions) > 3:
                    st.warning("This will delete old model versions. This action cannot be undone.")
                    # Implementation for cleanup would go here
                else:
                    st.info("Less than 3 versions exist. No cleanup needed.")
        
        with col2:
            if st.button("🔍 Validate All Models", help="Check integrity of all saved models"):
                st.info("🚧 Model validation feature coming soon!")
        
        st.subheader("📊 Model Statistics")
        if versions:
            total_size = sum(v['size_kb'] for v in versions)
            st.metric("Total Storage", f"{total_size:.1f} KB")
            st.metric("Model Versions", len(versions))
            
            # Show storage usage chart
            if len(versions) > 1:
                fig, ax = plt.subplots(figsize=(10, 4))
                timestamps = [v['timestamp'] for v in versions]
                sizes = [v['size_kb'] for v in versions]
                
                ax.bar(range(len(timestamps)), sizes)
                ax.set_xlabel('Model Version')
                ax.set_ylabel('Size (KB)')
                ax.set_title('Model Size by Version')
                ax.set_xticks(range(len(timestamps)))
                ax.set_xticklabels([t[:8] for t in timestamps], rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
