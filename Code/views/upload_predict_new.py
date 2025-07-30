import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from config import features

def create_prediction_report(pivot_df):
    """Create a comprehensive, user-friendly prediction report"""
    
    st.header("📊 Prediction Report")
    st.markdown("---")
    
    # Overall Summary Section
    st.subheader("🎯 Executive Summary")
    
    total_elevators = len(pivot_df)
    fault_predictions = pivot_df['Predicted_Fault'].sum()
    avg_fault_prob = pivot_df['Fault_Probability'].mean()
    high_risk_elevators = len(pivot_df[pivot_df['Fault_Probability'] >= 0.7])
    
    # Create summary metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🏢 Total Elevators", 
            total_elevators,
            help="Number of elevators analyzed"
        )
    
    with col2:
        st.metric(
            "⚠️ Fault Predictions", 
            fault_predictions,
            delta=f"{(fault_predictions/total_elevators*100):.1f}%" if total_elevators > 0 else "0%",
            help="Elevators predicted to have faults"
        )
    
    with col3:
        st.metric(
            "📈 Avg Risk Score", 
            f"{avg_fault_prob:.2f}",
            help="Average fault probability across all elevators"
        )
    
    with col4:
        st.metric(
            "🚨 High Risk Units", 
            high_risk_elevators,
            delta=f"{(high_risk_elevators/total_elevators*100):.1f}%" if total_elevators > 0 else "0%",
            help="Elevators with >70% fault probability"
        )
    
    # Risk Level Categorization
    st.subheader("🚦 Risk Level Analysis")
    
    # Binary risk categorization as requested
    def categorize_risk(prob):
        if prob == 0:
            return '🟢 Low Risk'
        elif prob == 1:
            return '🔴 High Risk'
        else:
            return '🟡 Medium Risk'
    
    pivot_df['Risk_Level'] = pivot_df['Predicted_Fault'].apply(categorize_risk)
    
    risk_summary = pivot_df['Risk_Level'].value_counts()
    
    # Display risk distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Risk level metrics
        for risk_level in ['🟢 Low Risk', '🟡 Medium Risk', '🔴 High Risk']:
            count = risk_summary.get(risk_level, 0)
            percentage = (count / total_elevators * 100) if total_elevators > 0 else 0
            st.metric(risk_level, f"{count} units", f"{percentage:.1f}%")
    
    with col2:
        # Risk distribution pie chart
        if len(risk_summary) > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = ['#2E8B57', '#FFD700', '#DC143C']  # Green, Yellow, Red
            risk_summary.plot.pie(
                ax=ax, 
                autopct='%1.1f%%', 
                colors=colors[:len(risk_summary)],
                startangle=90
            )
            ax.set_ylabel('')
            ax.set_title('Risk Distribution')
            st.pyplot(fig)
            plt.close(fig)
    
    # Detailed Elevator Analysis
    st.subheader("🔍 Detailed Elevator Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Summary Table", "📊 Visual Analysis", "⚠️ Priority Actions"])
    
    with tab1:
        st.write("**Complete Prediction Results**")
        
        # Prepare display dataframe
        display_df = pivot_df.copy()
        
        # Round numerical columns for better display
        for col in features:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        display_df['Fault_Probability'] = display_df['Fault_Probability'].round(3)
        
        # Reorder columns for better presentation
        column_order = ['Risk_Level'] + features + ['Fault_Probability', 'Predicted_Fault']
        display_df = display_df[[col for col in column_order if col in display_df.columns]]
        
        # Color-code the dataframe
        def highlight_risk(row):
            if 'Risk_Level' in row:
                if '🔴 High Risk' in str(row['Risk_Level']):
                    return ['background-color: #ffebee'] * len(row)
                elif '🟡 Medium Risk' in str(row['Risk_Level']):
                    return ['background-color: #fff9c4'] * len(row)
                else:
                    return ['background-color: #e8f5e8'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Export options
        csv = display_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Detailed Report (CSV)", 
            csv, 
            f"elevator_fault_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            "text/csv"
        )
    
    with tab2:
        st.write("**Visual Analysis of Predictions**")
        
        # Feature vs Fault Probability Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Fault probability distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(pivot_df['Fault_Probability'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(avg_fault_prob, color='red', linestyle='--', label=f'Average: {avg_fault_prob:.3f}')
            ax.set_xlabel('Fault Probability')
            ax.set_ylabel('Number of Elevators')
            ax.set_title('Distribution of Fault Probabilities')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Feature correlation with fault probability
            fig, ax = plt.subplots(figsize=(8, 6))
            feature_corr = []
            for feature in features:
                if feature in pivot_df.columns:
                    corr = pivot_df[feature].corr(pivot_df['Fault_Probability'])
                    feature_corr.append((feature, corr))
            
            if feature_corr:
                features_names, correlations = zip(*feature_corr)
                colors = ['red' if x < 0 else 'green' for x in correlations]
                bars = ax.barh(features_names, correlations, color=colors, alpha=0.7)
                ax.set_xlabel('Correlation with Fault Probability')
                ax.set_title('Feature Correlation Analysis')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center')
                
                st.pyplot(fig)
                plt.close(fig)
        
        # Time series visualization if elevator index represents time
        st.write("**Trend Analysis**")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot key features and fault probability
        ax2 = ax.twinx()
        
        elevator_indices = range(len(pivot_df))
        ax.plot(elevator_indices, pivot_df['total_door_reversals'], 
               label="Door Reversals", marker='o', alpha=0.7)
        ax.plot(elevator_indices, pivot_df['slow_door_operations'], 
               label="Slow Operations", marker='s', alpha=0.7)
        
        ax2.plot(elevator_indices, pivot_df['Fault_Probability'], 
                label="Fault Probability", marker='x', color='red', linewidth=2)
        
        ax.set_xlabel('Elevator Index')
        ax.set_ylabel('Feature Values', color='blue')
        ax2.set_ylabel('Fault Probability', color='red')
        ax.set_title('Feature Trends vs Fault Probability')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    with tab3:
        st.write("**Priority Action Items**")
        
        # High-risk elevators that need immediate attention
        high_risk_df = pivot_df[pivot_df['Fault_Probability'] >= 0.7].copy()
        
        if len(high_risk_df) > 0:
            st.error(f"🚨 **URGENT**: {len(high_risk_df)} elevator(s) require immediate inspection!")
            
            # Sort by fault probability (highest first)
            high_risk_df = high_risk_df.sort_values('Fault_Probability', ascending=False)
            
            for idx, (_, row) in enumerate(high_risk_df.iterrows(), 1):
                with st.expander(f"🚨 Priority {idx}: Elevator {row.name} (Risk: {row['Fault_Probability']:.1%})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Current Metrics:**")
                        for feature in features:
                            if feature in row:
                                st.write(f"• {feature}: {row[feature]:.2f}")
                    
                    with col2:
                        st.write("**Recommended Actions:**")
                        # Specific recommendations based on feature analysis
                        if 'hoistway_faults' in row and row['hoistway_faults'] > pivot_df['hoistway_faults'].median():
                            st.write("🔧 **Hoistway Issues**: Inspect hoistway equipment, check for obstructions, verify proper alignment")
                        if 'safety_chain_issues' in row and row['safety_chain_issues'] > pivot_df['safety_chain_issues'].median():
                            st.write("⚠️ **Safety Chain**: Test safety circuits, inspect door locks, verify emergency systems")
                        if 'levelling_total_errors' in row and row['levelling_total_errors'] > pivot_df['levelling_total_errors'].median():
                            st.write("📏 **Levelling Problems**: Calibrate position sensors, check motor control, inspect guide rails")
                        if 'total_door_reversals' in row and row['total_door_reversals'] > pivot_df['total_door_reversals'].median():
                            st.write("🚪 **Door Reversals**: Check door sensors and alignment, inspect door mechanisms")
                        if 'slow_door_operations' in row and row['slow_door_operations'] > pivot_df['slow_door_operations'].median():
                            st.write("⏱️ **Slow Operations**: Inspect door motor and mechanisms, check for wear")
                        st.write("📞 Schedule immediate maintenance inspection within 24 hours")
        
        # Medium-risk elevators
        medium_risk_df = pivot_df[(pivot_df['Fault_Probability'] >= 0.3) & 
                                 (pivot_df['Fault_Probability'] < 0.7)].copy()
        
        if len(medium_risk_df) > 0:
            st.warning(f"⚠️ **MONITOR**: {len(medium_risk_df)} elevator(s) require monitoring")
            
            medium_risk_df = medium_risk_df.sort_values('Fault_Probability', ascending=False)
            
            with st.expander(f"View {len(medium_risk_df)} Medium Risk Elevators"):
                for _, row in medium_risk_df.iterrows():
                    st.write(f"🟡 Elevator {row.name}: {row['Fault_Probability']:.1%} risk - Schedule preventive maintenance within 2 weeks")
        
        # Low-risk elevators summary
        low_risk_count = len(pivot_df[pivot_df['Fault_Probability'] < 0.3])
        if low_risk_count > 0:
            st.success(f"✅ **GOOD**: {low_risk_count} elevator(s) operating normally")
        
        # Overall recommendations
        st.subheader("📋 General Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Immediate Actions (Next 24-48 hours):**")
            st.write("• Inspect all high-risk elevators")
            st.write("• Verify door sensor calibration")
            st.write("• Check door motor performance")
            st.write("• Review maintenance logs")
        
        with col2:
            st.write("**Preventive Measures (Next 1-2 weeks):**")
            st.write("• Schedule routine maintenance for medium-risk units")
            st.write("• Update preventive maintenance schedules")
            st.write("• Monitor door operation patterns")
            st.write("• Train maintenance staff on fault indicators")

def render_upload_predict():   
    st.info("📋 Please upload a file that contains the required feature columns and is ready for prediction.")
    
    # Model selection section
    st.header("🤖 Model Selection")
    
    from utils.model_saver import get_available_models, load_model_and_scaler
    
    # Get available trained models
    available_models = get_available_models()
    
    if not available_models:
        st.warning("⚠️ No trained models found. Please train models first in the Model Comparison section.")
        return
    
    # Model selection dropdown
    selected_model_type = st.selectbox(
        "Select Model Type:",
        available_models,
        help="Choose which trained model to use for predictions"
    )
    
    # Load the selected model
    try:
        model, scaler, metadata, message = load_model_and_scaler(selected_model_type)
        if model is None:
            st.error(message)
            return
        
        st.success(f"✅ Loaded {selected_model_type} model successfully!")
        
        # Show model info
        if metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", metadata.get('model_type', 'Unknown'))
            with col2:
                st.metric("Features", metadata.get('n_features', 'Unknown'))
            with col3:
                if selected_model_type == "RandomForest":
                    st.metric("Estimators", metadata.get('n_estimators', 'Unknown'))
                else:  # LightGBM
                    st.metric("Trees", metadata.get('n_estimators', 'Unknown'))
    
    except Exception as e:
        st.error(f"Error loading {selected_model_type} model: {str(e)}")
        return
    
    # File upload section
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            # Load the uploaded pivot file directly
            pivot_df = pd.read_csv(uploaded_file)
            
            st.write("📊 **Data Preview**")
            st.dataframe(pivot_df.head(), use_container_width=True)
            
            # Validate required columns
            missing_features = [col for col in features if col not in pivot_df.columns]
            
            if missing_features:
                st.error(f"❌ **Missing Required Columns:** {missing_features}")
                st.write("**Available columns in your file:**")
                st.write(list(pivot_df.columns))
                st.write("**Required columns:**")
                st.write(features)
                return            
                                
            # Proceed with predictions if data is valid
            st.write(f"🤖 **Making Predictions with {selected_model_type} Model**")
            
            # Ensure we only use the required features
            X_features = pivot_df[features]               
            
            try:
                # Scale the features
                X_scaled = scaler.transform(X_features)                    
                # Make predictions using the unified prediction function
                from utils.model_builder import predict_model_with_proba
                preds, probs = predict_model_with_proba(model, X_scaled)
                
                # Add predictions to the dataframe
                pivot_df["Predicted_Fault"] = preds
                pivot_df["Fault_Probability"] = np.round(probs, 3)
                
                st.success("🎯 Prediction Complete!")
                
                # Create comprehensive prediction report
                create_prediction_report(pivot_df)
                
                # Additional download options
                st.subheader("📥 Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Summary report download
                    summary_data = {
                        'Total_Elevators': [len(pivot_df)],
                        'Fault_Predictions': [pivot_df['Predicted_Fault'].sum()],
                        'High_Risk_Count': [len(pivot_df[pivot_df['Fault_Probability'] >= 0.7])],
                        'Average_Risk_Score': [pivot_df['Fault_Probability'].mean()],
                        'Report_Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    }
                    summary_csv = pd.DataFrame(summary_data).to_csv(index=False).encode()
                    st.download_button(
                        "📊 Download Summary Report", 
                        summary_csv, 
                        f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
                with col2:
                    # Full results download
                    csv = pivot_df.to_csv(index=False).encode()
                    st.download_button(
                        "📋 Download Full Results", 
                        csv, 
                        f"predicted_faults_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                        "text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ **Prediction Error:** {str(e)}")
                st.write("This might be due to:")
                st.write("• Data scaling issues")
                st.write("• Incompatible data format")
                st.write("• Model-data mismatch")
                return
        
        except Exception as e:
            st.error(f"❌ **File Processing Error:** {str(e)}")
            st.write("Please ensure your CSV file is properly formatted.")
            return
