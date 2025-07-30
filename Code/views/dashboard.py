import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
from utils.print_helper import print_data_distribution, print_dataset_info

def render_dashboard(pivot_df, rf_model=None, rf_scaler=None, lgb_model=None, lgb_scaler=None, X_test=None, y_test=None):
    """Render the main dashboard with data overview and model insights for both RF and LightGBM"""
    st.title("Elevator Fault Detection Dashboard")
    
    # Model Status Section
    st.header("🤖 Model Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if rf_model is not None:
            st.success("✅ Random Forest Model Ready")
        else:
            st.error("❌ Random Forest Model Not Available")
    
    with col2:
        if lgb_model is not None:
            st.success("✅ LightGBM Model Ready")
        else:
            st.error("❌ LightGBM Model Not Available")
    
    with col3:
        available_models = []
        if rf_model is not None:
            available_models.append("Random Forest")
        if lgb_model is not None:
            available_models.append("LightGBM")
        st.info(f"📊 Available Models: {len(available_models)}")
    
    # Data Overview Section
    st.header("📊 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(pivot_df))
    
    with col2:
        fault_count = pivot_df['Fault'].sum()
        st.metric("Fault Cases", fault_count)
    
    with col3:
        fault_rate = (fault_count / len(pivot_df)) * 100
        st.metric("Fault Rate", f"{fault_rate:.1f}%")   
   
    
    # Show dataset info button
    if st.button("Show Dataset Info", key="dataset_info"):
        st.subheader("Dataset Information")
        temp_df = pd.DataFrame(pivot_df).head(10)
        st.dataframe(temp_df)  # Show only top 10 rows for quick overview
        print_dataset_info(pivot_df)

 
    # Show Data Distribution with comprehensive visualizations
    if st.button("Show Data Distribution", key="data_dist"):
        st.header("📊 Data Distribution Analysis")
        
        # Create tabs for different distribution analyses
        dist_tab1, dist_tab2, dist_tab3, dist_tab4 = st.tabs([
            "🎯 Fault Distribution", 
            "📈 Feature Distributions", 
            "🔗 Feature Correlations", 
            "📋 Statistical Summary"
        ])
        
        with dist_tab1:
            st.subheader("Fault vs Non-Fault Distribution")
            
            # Get fault distribution
            fault_counts = pivot_df['Fault'].value_counts()
            fault_percent = pivot_df['Fault'].value_counts(normalize=True) * 100
            
            # Create two columns for pie chart and bar chart
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                colors = ['#2E8B57', '#DC143C']  # Green for No Fault, Red for Fault
                labels = ['No Fault', 'Fault']
                ax1.pie(fault_counts.values, labels=labels, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
                ax1.set_title('Fault Distribution (Pie Chart)')
                st.pyplot(fig1)
                plt.close(fig1)
            
            # Display statistics
            st.subheader("Distribution Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(pivot_df))
            with col2:
                st.metric("Fault Cases", int(fault_counts[1]) if 1 in fault_counts else 0)
            with col3:
                st.metric("Non-Fault Cases", int(fault_counts[0]) if 0 in fault_counts else 0)
            
            # Show percentages
            st.write("**Percentage Distribution:**")
            for idx, (fault_type, percentage) in enumerate(fault_percent.items()):
                label = "Fault" if fault_type == 1 else "No Fault"
                st.write(f"- {label}: {percentage:.2f}%")
        
        with dist_tab2:
            st.subheader("Feature Distributions")
            
            from config import features
            
            # Create histograms for each feature
            n_features = len(features)
            n_cols = 2
            n_rows = (n_features + 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, feature in enumerate(features):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                
                # Plot histogram with different colors for fault vs non-fault
                pivot_df[pivot_df['Fault'] == 0][feature].hist(
                    alpha=0.7, label='No Fault', bins=20, ax=ax, color='green'
                )
                pivot_df[pivot_df['Fault'] == 1][feature].hist(
                    alpha=0.7, label='Fault', bins=20, ax=ax, color='red'
                )
                
                ax.set_title(f'Distribution of {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(n_features, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Box plots for outlier detection
            st.subheader("Box Plots for Outlier Detection")
            fig, axes = plt.subplots(1, len(features), figsize=(15, 6))
            if len(features) == 1:
                axes = [axes]
            
            for i, feature in enumerate(features):
                pivot_df.boxplot(column=feature, by='Fault', ax=axes[i])
                axes[i].set_title(f'{feature} by Fault Status')
                axes[i].set_xlabel('Fault (0=No, 1=Yes)')
            
            plt.suptitle('')  # Remove automatic title
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with dist_tab3:
            st.subheader("Feature Correlation Analysis")
            
            # Calculate correlation matrix
            correlation_features = features + ['Fault']
            corr_matrix = pivot_df[correlation_features].corr()
            
            # Create correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, mask=mask, ax=ax, fmt='.3f')
            ax.set_title('Feature Correlation Matrix')
            st.pyplot(fig)
            plt.close(fig)
            
            # Correlation with target variable
            st.subheader("Correlation with Fault (Target Variable)")
            fault_correlations = corr_matrix['Fault'].drop('Fault').sort_values(key=abs, ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['red' if x < 0 else 'green' for x in fault_correlations.values]
            bars = ax.barh(fault_correlations.index, fault_correlations.values, color=colors)
            ax.set_title('Feature Correlation with Fault')
            ax.set_xlabel('Correlation Coefficient')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center')
            
            st.pyplot(fig)
            plt.close(fig)
        
        with dist_tab4:
            st.subheader("Statistical Summary")
            
            # Overall statistics
            st.write("**Overall Dataset Statistics:**")
            st.dataframe(pivot_df[features + ['Fault']].describe())
            
            # Statistics by fault status
            st.write("**Statistics by Fault Status:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**No Fault Cases:**")
                no_fault_stats = pivot_df[pivot_df['Fault'] == 0][features].describe()
                st.dataframe(no_fault_stats)
            
            with col2:
                st.write("**Fault Cases:**")
                fault_stats = pivot_df[pivot_df['Fault'] == 1][features].describe()
                st.dataframe(fault_stats)
            
            # Feature ranges comparison
            st.subheader("Feature Range Comparison")
            range_comparison = pd.DataFrame({
                'Feature': features,
                'No_Fault_Mean': [pivot_df[pivot_df['Fault'] == 0][f].mean() for f in features],
                'Fault_Mean': [pivot_df[pivot_df['Fault'] == 1][f].mean() for f in features],
                'No_Fault_Std': [pivot_df[pivot_df['Fault'] == 0][f].std() for f in features],
                'Fault_Std': [pivot_df[pivot_df['Fault'] == 1][f].std() for f in features]
            })
            
            range_comparison['Mean_Difference'] = range_comparison['Fault_Mean'] - range_comparison['No_Fault_Mean']
            st.dataframe(range_comparison)
    
    # Model Performance Section (if models are available)
    if (rf_model is not None or lgb_model is not None) and X_test is not None and y_test is not None:
        st.header("🤖 Model Performance")
        
        # Quick comparison if both models are available
        if rf_model is not None and lgb_model is not None:
            st.subheader("📊 Quick Model Comparison")
            from utils.model_builder import predict_model_with_proba
            
            # Debug information
            st.write(f"DEBUG: X_test type: {type(X_test)}")
            st.write(f"DEBUG: y_test type: {type(y_test)}")
            if hasattr(X_test, 'shape'):
                st.write(f"DEBUG: X_test shape: {X_test.shape}")
            if hasattr(y_test, 'shape'):
                st.write(f"DEBUG: y_test shape: {y_test.shape}")
            
            # Get predictions from both models
            rf_pred, rf_pred_proba = predict_model_with_proba(rf_model, X_test)
            lgb_pred, lgb_pred_proba = predict_model_with_proba(lgb_model, X_test)
            
            # Calculate metrics for both models
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            rf_metrics = {
                'Accuracy': accuracy_score(y_test, rf_pred),
                'Precision': precision_score(y_test, rf_pred),
                'Recall': recall_score(y_test, rf_pred),
                'F1-Score': f1_score(y_test, rf_pred),
                'ROC AUC': roc_auc_score(y_test, rf_pred_proba)
            }
            
            lgb_metrics = {
                'Accuracy': accuracy_score(y_test, lgb_pred),
                'Precision': precision_score(y_test, lgb_pred),
                'Recall': recall_score(y_test, lgb_pred),
                'F1-Score': f1_score(y_test, lgb_pred),
                'ROC AUC': roc_auc_score(y_test, lgb_pred_proba)
            }
            
            # Display comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🌲 Random Forest**")
                for metric, value in rf_metrics.items():
                    st.metric(metric, f"{value:.3f}")
            
            with col2:
                st.write("**⚡ LightGBM**")
                for metric, value in lgb_metrics.items():
                    # Show comparison with RF
                    delta = value - rf_metrics[metric]
                    st.metric(metric, f"{value:.3f}", delta=f"{delta:+.3f}")
            
            st.divider()
        
        # Model selection for detailed analysis
        available_models = []
        if rf_model is not None:
            available_models.append("Random Forest")
        if lgb_model is not None:
            available_models.append("LightGBM")
        
        if len(available_models) > 1:
            selected_model = st.selectbox("Select Model for Detailed Analysis:", available_models, key="model_select")
        else:
            selected_model = available_models[0] if available_models else None
        
        # Show model status cards
        col1, col2 = st.columns(2)
        with col1:
            if rf_model is not None:
                st.success("✅ Random Forest Model: Ready")
            else:
                st.error("❌ Random Forest Model: Not Available")
        
        with col2:
            if lgb_model is not None:
                st.success("✅ LightGBM Model: Ready")
            else:
                st.error("❌ LightGBM Model: Not Available")
        
        if st.button("Show Model Metrics", key="model_metrics"):
            from utils.model_builder import predict_model_with_proba
            from utils.print_helper import print_classification, print_confusion_matrix
            
            # Get the selected model
            current_model = rf_model if selected_model == "Random Forest" else lgb_model
            model_name = selected_model
            
            # Make predictions
            y_pred, y_pred_proba = predict_model_with_proba(current_model, X_test)
            
            # Create tabs for different metrics
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Classification Report", "🎯 Confusion Matrix", "📈 Precision-Recall Curve", "📊 Precision & Recall", "📉 ROC Curve", "🔍 Feature Importance"])
            
            with tab1:
                st.subheader(f"Classification Report - {model_name}")
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision (Class 1)", f"{report['1']['precision']:.3f}")
                    st.metric("Recall (Class 1)", f"{report['1']['recall']:.3f}")
                with col2:
                    st.metric("F1-Score (Class 1)", f"{report['1']['f1-score']:.3f}")
                    st.metric("Support (Class 1)", f"{int(report['1']['support'])}")
                with col3:
                    st.metric("Accuracy", f"{report['accuracy']:.3f}")
                    st.metric("Macro Avg F1", f"{report['macro avg']['f1-score']:.3f}")
                
                # Show detailed classification report
                st.text(classification_report(y_test, y_pred))
            
            with tab2:
                st.subheader(f"Confusion Matrix - {model_name}")
                fig = print_confusion_matrix(y_test, y_pred)
                st.pyplot(fig)
                plt.close(fig)  
            
            with tab3:
                st.subheader(f"Precision-Recall Curve - {model_name}")
                
                # Calculate precision-recall curve
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot PR curve with markers for better visibility
                ax.plot(recall, precision, linewidth=2, marker='o', markersize=4, 
                       label=f'PR Curve (AUC = {pr_auc:.3f})', color='blue')
                
                # Add baseline (random classifier)
                baseline = np.sum(y_test) / len(y_test)  # Proportion of positive class
                ax.axhline(y=baseline, color='red', linestyle='--', 
                          label=f'Random Classifier (Baseline = {baseline:.3f})')
                
                # Add some threshold points for interpretation
                if len(thresholds) > 10:
                    # Show every 10th threshold point
                    step = len(thresholds) // 10
                    for i in range(0, len(thresholds), step):
                        if i < len(recall) and i < len(precision):
                            ax.annotate(f'T={thresholds[i]:.2f}', 
                                      (recall[i], precision[i]),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, alpha=0.7)
                
                ax.set_xlabel('Recall (Sensitivity)')
                ax.set_ylabel('Precision (Positive Predictive Value)')
                ax.set_title('Precision-Recall Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Display additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PR AUC", f"{pr_auc:.3f}")
                with col2:
                    st.metric("Baseline (Random)", f"{baseline:.3f}")
                with col3:
                    best_f1_idx = np.argmax(2 * (precision * recall) / (precision + recall + 1e-10))
                    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
                    st.metric("Best F1 Threshold", f"{best_threshold:.3f}")
                
                # Show diagnostic information
                st.write("**Diagnostic Information:**")
                st.write(f"- Number of test samples: {len(y_test)}")
                st.write(f"- Positive class ratio: {np.sum(y_test)/len(y_test):.3f}")
                st.write(f"- Unique prediction probabilities: {len(np.unique(y_pred_proba))}")
                st.write(f"- Min probability: {np.min(y_pred_proba):.3f}")
                st.write(f"- Max probability: {np.max(y_pred_proba):.3f}")
                st.write(f"- Number of PR curve points: {len(precision)}")
                
                if pr_auc > 0.95:
                    st.warning("⚠️ Very high PR AUC might indicate overfitting or perfect separation")
                elif len(np.unique(y_pred_proba)) < 10:
                    st.warning("⚠️ Limited unique probability values might indicate model issues")
            
            with tab4:
                st.subheader(f"Precision & Recall Analysis - {model_name}")
                
                # Calculate precision-recall curve for threshold analysis
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
                
                # Create two separate plots
                col1, col2 = st.columns(2)
                
                with col1:
                    # Precision vs Threshold
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    ax1.plot(thresholds, precision[:-1], linewidth=2, marker='o', markersize=3, 
                            color='blue', label='Precision')
                    ax1.set_xlabel('Decision Threshold')
                    ax1.set_ylabel('Precision')
                    ax1.set_title('Precision vs Decision Threshold')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_xlim([0.0, 1.0])
                    ax1.set_ylim([0.0, 1.05])
                    
                    # Add current model threshold line (usually 0.5)
                    current_threshold = 0.5
                    current_precision_idx = np.argmin(np.abs(thresholds - current_threshold))
                    if current_precision_idx < len(precision) - 1:
                        current_precision = precision[current_precision_idx]
                        ax1.axvline(x=current_threshold, color='red', linestyle='--', 
                                   label=f'Default Threshold (0.5)')
                        ax1.axhline(y=current_precision, color='red', linestyle=':', alpha=0.7)
                        ax1.text(current_threshold + 0.05, current_precision + 0.05, 
                                f'Precision: {current_precision:.3f}', fontsize=9)
                    
                    ax1.legend()
                    st.pyplot(fig1)
                    plt.close(fig1)
                
                with col2:
                    # Recall vs Threshold
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.plot(thresholds, recall[:-1], linewidth=2, marker='s', markersize=3, 
                            color='green', label='Recall')
                    ax2.set_xlabel('Decision Threshold')
                    ax2.set_ylabel('Recall')
                    ax2.set_title('Recall vs Decision Threshold')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_xlim([0.0, 1.0])
                    ax2.set_ylim([0.0, 1.05])
                    
                    # Add current model threshold line
                    current_recall_idx = np.argmin(np.abs(thresholds - current_threshold))
                    if current_recall_idx < len(recall) - 1:
                        current_recall = recall[current_recall_idx]
                        ax2.axvline(x=current_threshold, color='red', linestyle='--', 
                                   label=f'Default Threshold (0.5)')
                        ax2.axhline(y=current_recall, color='red', linestyle=':', alpha=0.7)
                        ax2.text(current_threshold + 0.05, current_recall + 0.05, 
                                f'Recall: {current_recall:.3f}', fontsize=9)
                    
                    ax2.legend()
                    st.pyplot(fig2)
                    plt.close(fig2)
                
                # Combined Precision and Recall vs Threshold
                st.subheader("Combined Precision & Recall vs Threshold")
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                
                ax3.plot(thresholds, precision[:-1], linewidth=2, marker='o', markersize=3, 
                        color='blue', label='Precision', alpha=0.8)
                ax3.plot(thresholds, recall[:-1], linewidth=2, marker='s', markersize=3, 
                        color='green', label='Recall', alpha=0.8)
                
                # F1 Score
                f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
                ax3.plot(thresholds, f1_scores, linewidth=2, marker='^', markersize=3, 
                        color='orange', label='F1-Score', alpha=0.8)
                
                # Find optimal F1 threshold
                best_f1_idx = np.argmax(f1_scores)
                best_f1_threshold = thresholds[best_f1_idx]
                best_f1_score = f1_scores[best_f1_idx]
                
                ax3.axvline(x=best_f1_threshold, color='purple', linestyle='--', 
                           label=f'Optimal F1 Threshold ({best_f1_threshold:.3f})')
                ax3.axvline(x=current_threshold, color='red', linestyle='--', alpha=0.7,
                           label=f'Default Threshold (0.5)')
                
                ax3.set_xlabel('Decision Threshold')
                ax3.set_ylabel('Score')
                ax3.set_title('Precision, Recall & F1-Score vs Decision Threshold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim([0.0, 1.0])
                ax3.set_ylim([0.0, 1.05])
                
                st.pyplot(fig3)
                plt.close(fig3)
                
                # Display threshold analysis metrics
                st.subheader("Threshold Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Best F1 Threshold", f"{best_f1_threshold:.3f}")
                    st.metric("Best F1 Score", f"{best_f1_score:.3f}")
                
                with col2:
                    if current_precision_idx < len(precision) - 1 and current_recall_idx < len(recall) - 1:
                        st.metric("Precision @ 0.5", f"{current_precision:.3f}")
                        st.metric("Recall @ 0.5", f"{current_recall:.3f}")
                
                with col3:
                    # High precision threshold (e.g., 0.9 precision)
                    high_prec_indices = np.where(precision[:-1] >= 0.9)[0]
                    if len(high_prec_indices) > 0:
                        high_prec_threshold = thresholds[high_prec_indices[0]]
                        high_prec_recall = recall[high_prec_indices[0]]
                        st.metric("Threshold for 90% Precision", f"{high_prec_threshold:.3f}")
                        st.metric("Recall @ 90% Precision", f"{high_prec_recall:.3f}")
                    else:
                        st.metric("Threshold for 90% Precision", "N/A")
                        st.metric("Recall @ 90% Precision", "N/A")
                
                with col4:
                    # High recall threshold (e.g., 0.9 recall)
                    high_recall_indices = np.where(recall[:-1] >= 0.9)[0]
                    if len(high_recall_indices) > 0:
                        high_recall_threshold = thresholds[high_recall_indices[-1]]
                        high_recall_precision = precision[high_recall_indices[-1]]
                        st.metric("Threshold for 90% Recall", f"{high_recall_threshold:.3f}")
                        st.metric("Precision @ 90% Recall", f"{high_recall_precision:.3f}")
                    else:
                        st.metric("Threshold for 90% Recall", "N/A")
                        st.metric("Precision @ 90% Recall", "N/A")
                
                # Threshold recommendation
                st.subheader("📋 Threshold Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**For High Precision (Minimize False Positives):**")
                    if len(high_prec_indices) > 0:
                        st.write(f"• Use threshold ≥ {high_prec_threshold:.3f}")
                        st.write(f"• Expected precision: ≥ 90%")
                        st.write(f"• Expected recall: {high_prec_recall:.1%}")
                    else:
                        st.write("• Model may not achieve 90% precision")
                    st.write("• Good for: Critical fault detection where false alarms are costly")
                
                with col2:
                    st.write("**For High Recall (Minimize False Negatives):**")
                    if len(high_recall_indices) > 0:
                        st.write(f"• Use threshold ≤ {high_recall_threshold:.3f}")
                        st.write(f"• Expected recall: ≥ 90%")
                        st.write(f"• Expected precision: {high_recall_precision:.1%}")
                    else:
                        st.write("• Model may not achieve 90% recall")
                    st.write("• Good for: Safety-critical applications where missing faults is dangerous")
            
            with tab5:
                st.subheader(f"ROC Curve - {model_name}")
                
                # Calculate ROC curve
                fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot ROC curve with markers
                ax.plot(fpr, tpr, linewidth=2, marker='o', markersize=4,
                       label=f'ROC Curve (AUC = {roc_auc:.3f})', color='blue')
                
                # Plot diagonal line (random classifier)
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')
                
                # Add some threshold points
                if len(roc_thresholds) > 10:
                    step = len(roc_thresholds) // 10
                    for i in range(0, len(roc_thresholds), step):
                        if i < len(fpr) and i < len(tpr):
                            ax.annotate(f'T={roc_thresholds[i]:.2f}', 
                                      (fpr[i], tpr[i]),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, alpha=0.7)
                
                ax.set_xlabel('False Positive Rate (1 - Specificity)')
                ax.set_ylabel('True Positive Rate (Sensitivity)')
                ax.set_title('ROC Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Display additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ROC AUC", f"{roc_auc:.3f}")
                with col2:
                    # Find optimal threshold using Youden's index
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_threshold = roc_thresholds[optimal_idx] if optimal_idx < len(roc_thresholds) else 0.5
                    st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
                with col3:
                    # Specificity at optimal threshold
                    specificity = 1 - fpr[optimal_idx]
                    st.metric("Specificity @ Optimal", f"{specificity:.3f}")
                
                # Show diagnostic information
                st.write("**ROC Diagnostic Information:**")
                st.write(f"- Number of ROC curve points: {len(fpr)}")
                st.write(f"- TPR at optimal threshold: {tpr[optimal_idx]:.3f}")
                st.write(f"- FPR at optimal threshold: {fpr[optimal_idx]:.3f}")
                
                if roc_auc > 0.95:
                    st.warning("⚠️ Very high ROC AUC might indicate overfitting")
                elif roc_auc < 0.6:
                    st.warning("⚠️ Low ROC AUC indicates poor model performance")
            
            with tab6:
                st.subheader(f"Feature Importance - {model_name}")
                
                # Get feature importance from the selected model
                from config import features
                from utils.model_builder import get_feature_importance
                
                feature_importance = get_feature_importance(current_model)
                feature_names = features
                
                # Create a dataframe for easier plotting
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=True)
                
                # Create horizontal bar plot
                fig, ax = plt.subplots(figsize=(10, 8))
                bars = ax.barh(importance_df['feature'], importance_df['importance'])
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'{model_name} Feature Importance')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.3f}', ha='left', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Show top features
                top_features = importance_df.tail(5)
                st.subheader("Top Most Important Features")
                for idx, row in top_features.iterrows():
                    st.metric(row['feature'], f"{row['importance']:.4f}")
    
    # # Quick Actions
    # st.header("🚀 Quick Actions")
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     if st.button("Go to Upload & Predict", key="goto_predict"):
    #         st.session_state.menu_selection = "Upload & Predict"
    #         st.experimental_rerun()
    
    # with col2:
    #     if st.button("View EDA", key="goto_eda"):
    #         st.session_state.menu_selection = "EDA"
    #         st.experimental_rerun()