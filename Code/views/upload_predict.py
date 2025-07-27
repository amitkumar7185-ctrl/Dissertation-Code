
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from config import features
from utils.preprocessing import prepare_pivoted_data

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
    
    # Categorize elevators by risk level
    pivot_df['Risk_Level'] = pd.cut(
        pivot_df['Fault_Probability'], 
        bins=[0, 0.3, 0.7, 1.0], 
        labels=['🟢 Low Risk', '🟡 Medium Risk', '🔴 High Risk'],
        include_lowest=True
    )
    
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
                        if row['total_door_reversals'] > pivot_df['total_door_reversals'].median():
                            st.write("🔧 Check door sensors and alignment")
                        if row['slow_door_operations'] > pivot_df['slow_door_operations'].median():
                            st.write("⚙️ Inspect door motor and mechanisms")
                        if row['door_operations'] > pivot_df['door_operations'].median():
                            st.write("🔍 Review door usage patterns")
                        st.write("📞 Schedule immediate maintenance inspection")
        
        # Medium-risk elevators
        medium_risk_df = pivot_df[(pivot_df['Fault_Probability'] >= 0.3) & 
                                 (pivot_df['Fault_Probability'] < 0.7)].copy()
        
        if len(medium_risk_df) > 0:
            st.warning(f"⚠️ **MONITOR**: {len(medium_risk_df)} elevator(s) require monitoring")
            
            medium_risk_df = medium_risk_df.sort_values('Fault_Probability', ascending=False)
            
            with st.expander(f"View {len(medium_risk_df)} Medium Risk Elevators"):
                for _, row in medium_risk_df.iterrows():
                    st.write(f"🟡 Elevator {row.name}: {row['Fault_Probability']:.1%} risk - Schedule preventive maintenance")
        
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
#def render_upload_predict(rf_model, rf_scaler, lstm_model, lstm_scaler):
def render_upload_predict(rf_model, rf_scaler):
    st.subheader("Upload 5-Day Sensor CSV")
    model_type = st.radio("Model Type", ["Random Forest", "LSTM (Time-Series)"])
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)        
        pivot_df= prepare_pivoted_data(df)
        st.write("Data Preview")
        st.dataframe(pivot_df)

        if not all(col in pivot_df.columns for col in features):
            st.error("CSV do not have required columns.")
        else:
            if model_type == "Random Forest":
                X_scaled = rf_scaler.transform(pivot_df[features])
                preds = rf_model.predict(X_scaled)
                probs = rf_model.predict_proba(X_scaled)[:, 1]
                pivot_df["Predicted_Fault"] = preds
                pivot_df["Fault_Probability"] = np.round(probs, 2)
            # else:
            #     X_seq = lstm_scaler.transform(df[features]).reshape(1, 5, len(features))
            #     prob = lstm_model.predict(X_seq)[0][0]
            #     df["Predicted_Fault"] = [""] * 4 + [int(prob > 0.5)]
            #     df["Fault_Probability"] = [""] * 4 + [round(prob, 2)]

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
