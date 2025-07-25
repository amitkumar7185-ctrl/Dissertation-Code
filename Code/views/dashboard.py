import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
from utils.print_helper import print_data_distribution, print_dataset_info

def render_dashboard(pivot_df, rf_model=None, rf_scaler=None, X_test=None, y_test=None):
    """Render the main dashboard with data overview and model insights"""
    st.title("Elevator Fault Detection Dashboard")
    
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
        st.dataframe(pivot_df)
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
    
    # Model Performance Section (if model is available)
    if rf_model is not None and X_test is not None and y_test is not None:
        st.header("🤖 Model Performance")
        st.write("Model is trained and ready for predictions!")
        
        if st.button("Show Model Metrics", key="model_metrics"):
            from utils.model_builder import predict_model
            from utils.print_helper import print_classification, print_confusion_matrix
            
            # Make predictions
            y_pred = predict_model(rf_model, X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability for positive class
            
            # Create tabs for different metrics
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Classification Report", "🎯 Confusion Matrix", "📈 Precision-Recall Curve", "📉 ROC Curve", "🔍 Feature Importance"])
            
            with tab1:
                st.subheader("Classification Report")
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
                st.subheader("Confusion Matrix")
                fig = print_confusion_matrix(y_test, y_pred)
                st.pyplot(fig)
                plt.close(fig)  
            
            with tab3:
                st.subheader("Precision-Recall Curve")
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
                
                st.info(f"**Area Under PR Curve: {pr_auc:.3f}**")
            
            with tab4:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
                
                st.info(f"**Area Under ROC Curve: {roc_auc:.3f}**")
            
            with tab5:
                st.subheader("Feature Importance")
                
                # Get feature importance from Random Forest
                from config import features
                feature_importance = rf_model.feature_importances_
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
                ax.set_title('Random Forest Feature Importance')
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