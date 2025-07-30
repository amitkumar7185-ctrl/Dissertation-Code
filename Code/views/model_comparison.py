import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve

from utils.model_builder import compare_models, get_scaler, split_data
from utils.model_saver import save_model_and_scaler, get_available_models, load_model_and_scaler
from config import features

def run_model_comparison(pivot_df):
    """
    Run comprehensive comparison between Random Forest and LightGBM models
    """
    st.header("🔬 Model Comparison: Random Forest vs LightGBM")
    st.markdown("---")
    
    # Check for available pre-trained models
    available_models = get_available_models()
    has_rf_model = 'random_forest' in available_models
    has_lgb_model = 'lightgbm' in available_models
    
    # Data preparation
    X = pivot_df[features]
    y = pivot_df['Fault']
    
    # Get scaled data
    scaler, X_scaled = get_scaler(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    st.info(f"📊 **Dataset Overview:** {len(X)} samples, {len(features)} features")
    st.info(f"🔄 **Train/Test Split:** {len(X_train)} training, {len(X_test)} testing samples")
    
    # Model selection section
    st.subheader("🎯 Model Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if has_rf_model:
            st.success("✅ Pre-trained Random Forest found")
            use_pretrained_rf = st.checkbox("Use pre-trained Random Forest", value=True, key="use_rf")
        else:
            st.warning("⚠️ No pre-trained Random Forest found")
            use_pretrained_rf = False
    
    with col2:
        if has_lgb_model:
            st.success("✅ Pre-trained LightGBM found")
            use_pretrained_lgb = st.checkbox("Use pre-trained LightGBM", value=True, key="use_lgb")
        else:
            st.warning("⚠️ No pre-trained LightGBM found")
            use_pretrained_lgb = False
    
    # If using pre-trained models, show quick comparison
    if use_pretrained_rf and use_pretrained_lgb:
        st.subheader("⚡ Quick Comparison (Pre-trained Models)")
        if st.button("� Compare Pre-trained Models", type="primary"):
            with st.spinner("Loading pre-trained models and comparing..."):
                try:
                    # Load pre-trained models
                    rf_model, rf_scaler = load_model_and_scaler('random_forest')
                    lgb_model, lgb_scaler = load_model_and_scaler('lightgbm')
                    
                    # Use the same scaler for fair comparison (or handle scaling appropriately)
                    from utils.model_builder import evaluate_model, get_feature_importance
                    
                    # Apply scaling
                    X_test_rf = rf_scaler.transform(X[features])[:len(X_test)]
                    X_test_lgb = lgb_scaler.transform(X[features])[:len(X_test)]
                    
                    # Re-split to ensure consistency
                    from sklearn.model_selection import train_test_split
                    _, X_test_rf, _, y_test = train_test_split(X[features], y, test_size=0.2, random_state=42, stratify=y)
                    _, X_test_lgb, _, _ = train_test_split(X[features], y, test_size=0.2, random_state=42, stratify=y)
                    
                    X_test_rf = rf_scaler.transform(X_test_rf)
                    X_test_lgb = lgb_scaler.transform(X_test_lgb)
                    
                    # Evaluate both models
                    rf_metrics, rf_pred, rf_proba = evaluate_model(rf_model, X_test_rf, y_test)
                    lgb_metrics, lgb_pred, lgb_proba = evaluate_model(lgb_model, X_test_lgb, y_test)
                    
                    # Get feature importance
                    rf_importance = get_feature_importance(rf_model)
                    lgb_importance = get_feature_importance(lgb_model)
                    
                    # Store results in session state
                    st.session_state.comparison_results = {
                        'rf_model': rf_model,
                        'lgb_model': lgb_model,
                        'rf_scaler': rf_scaler,
                        'lgb_scaler': lgb_scaler,
                        'rf_metrics': rf_metrics,
                        'lgb_metrics': lgb_metrics,
                        'rf_pred': rf_pred,
                        'lgb_pred': lgb_pred,
                        'rf_proba': rf_proba,
                        'lgb_proba': lgb_proba,
                        'rf_importance': rf_importance,
                        'lgb_importance': lgb_importance,
                        'X_test': X_test_rf,
                        'y_test': y_test,
                        'pretrained': True
                    }
                    
                    st.success("✅ Pre-trained models compared successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error loading pre-trained models: {str(e)}")
                    st.info("💡 Please train the models first or use custom training below.")
    
    # Custom training section
    st.subheader("🔧 Custom Model Training")
    st.write("Train models with custom hyperparameters or if pre-trained models are not available")
    
    with st.expander("⚙️ Advanced Training Options", expanded=not (use_pretrained_rf and use_pretrained_lgb)):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🌳 Random Forest Configuration:**")
            rf_n_estimators = st.slider("Number of Trees", 50, 200, 100, step=25)
            rf_max_depth = st.selectbox("Max Depth", [None, 5, 10, 15, 20], index=0)
            rf_min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
        
        with col2:
            st.write("**⚡ LightGBM Configuration:**")
            lgb_n_estimators = st.slider("Number of Boost Rounds", 50, 200, 100, step=25)
            lgb_learning_rate = st.selectbox("Learning Rate", [0.01, 0.05, 0.1, 0.2], index=1)
            lgb_num_leaves = st.slider("Number of Leaves", 15, 63, 31)
        
        if st.button("🏃‍♂️ Train Custom Models", type="secondary"):
            with st.spinner("Training models with custom parameters..."):
                
                # Custom parameters for models
                from utils.model_builder import build_rf_model, build_lgb_model, evaluate_model, get_feature_importance
                from sklearn.ensemble import RandomForestClassifier
                
                # Train Random Forest with custom parameters
                rf_model = RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    min_samples_split=rf_min_samples_split,
                    random_state=42,
                    class_weight='balanced'
                )
                rf_model.fit(X_train, y_train)
                
                # Train LightGBM with custom parameters
                lgb_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': lgb_num_leaves,
                    'learning_rate': lgb_learning_rate,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42,
                    'n_estimators': lgb_n_estimators
                }
                
                lgb_model = build_lgb_model(X_train, y_train, lgb_params)
                
                # Evaluate both models
                rf_metrics, rf_pred, rf_proba = evaluate_model(rf_model, X_test, y_test)
                lgb_metrics, lgb_pred, lgb_proba = evaluate_model(lgb_model, X_test, y_test)
                
                # Get feature importance
                rf_importance = get_feature_importance(rf_model)
                lgb_importance = get_feature_importance(lgb_model)
                
                # Store results in session state
                st.session_state.comparison_results = {
                    'rf_model': rf_model,
                    'lgb_model': lgb_model,
                    'scaler': scaler,
                    'rf_metrics': rf_metrics,
                    'lgb_metrics': lgb_metrics,
                    'rf_pred': rf_pred,
                    'lgb_pred': lgb_pred,
                    'rf_proba': rf_proba,
                    'lgb_proba': lgb_proba,
                    'rf_importance': rf_importance,
                    'lgb_importance': lgb_importance,
                    'X_test': X_test,
                    'y_test': y_test,
                    'pretrained': False
                }
                
                st.success("✅ Custom models trained successfully!")
    
    # Display results if available
    if 'comparison_results' in st.session_state:
        display_comparison_results(st.session_state.comparison_results)

def display_comparison_results(results):
    """
    Display comprehensive comparison results
    """
    st.markdown("---")
    st.subheader("📊 Comparison Results")
    
    # Extract results
    rf_metrics = results['rf_metrics']
    lgb_metrics = results['lgb_metrics']
    rf_pred = results['rf_pred']
    lgb_pred = results['lgb_pred']
    rf_proba = results['rf_proba']
    lgb_proba = results['lgb_proba']
    rf_importance = results['rf_importance']
    lgb_importance = results['lgb_importance']
    X_test = results['X_test']
    y_test = results['y_test']
    
    # Performance metrics comparison
    st.subheader("🎯 Performance Metrics")
    
    # Create metrics comparison table
    metrics_df = pd.DataFrame({
        'Random Forest': [
            f"{rf_metrics['accuracy']:.3f}",
            f"{rf_metrics['precision']:.3f}",
            f"{rf_metrics['recall']:.3f}",
            f"{rf_metrics['f1']:.3f}",
            f"{rf_metrics['roc_auc']:.3f}"
        ],
        'LightGBM': [
            f"{lgb_metrics['accuracy']:.3f}",
            f"{lgb_metrics['precision']:.3f}",
            f"{lgb_metrics['recall']:.3f}",
            f"{lgb_metrics['f1']:.3f}",
            f"{lgb_metrics['roc_auc']:.3f}"
        ]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        # Determine better model
        rf_score = (rf_metrics['accuracy'] + rf_metrics['f1'] + rf_metrics['roc_auc']) / 3
        lgb_score = (lgb_metrics['accuracy'] + lgb_metrics['f1'] + lgb_metrics['roc_auc']) / 3
        
        if rf_score > lgb_score:
            st.success("🏆 **Winner: Random Forest**")
            st.metric("Advantage", f"+{(rf_score - lgb_score)*100:.1f}%")
        elif lgb_score > rf_score:
            st.success("🏆 **Winner: LightGBM**")
            st.metric("Advantage", f"+{(lgb_score - rf_score)*100:.1f}%")
        else:
            st.info("🤝 **Result: Tie**")
    
    # Detailed visualizations in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 ROC & PR Curves", 
        "🎯 Confusion Matrix", 
        "🔍 Feature Importance", 
        "📊 Prediction Comparison",
        "💾 Save Models"
    ])
    
    with tab1:
        st.subheader("ROC and Precision-Recall Curves")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC Curves
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Random Forest ROC
            rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
            ax.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC: {rf_metrics["roc_auc"]:.3f})', linewidth=2)
            
            # LightGBM ROC
            lgb_fpr, lgb_tpr, _ = roc_curve(y_test, lgb_proba)
            ax.plot(lgb_fpr, lgb_tpr, label=f'LightGBM (AUC: {lgb_metrics["roc_auc"]:.3f})', linewidth=2)
            
            # Diagonal line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Precision-Recall Curves
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Random Forest PR
            rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_proba)
            ax.plot(rf_recall, rf_precision, label=f'Random Forest', linewidth=2)
            
            # LightGBM PR
            lgb_precision, lgb_recall, _ = precision_recall_curve(y_test, lgb_proba)
            ax.plot(lgb_recall, lgb_precision, label=f'LightGBM', linewidth=2)
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig)
    
    with tab2:
        st.subheader("Confusion Matrix Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Random Forest**")
            rf_cm = confusion_matrix(y_test, rf_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Random Forest Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.write("**LightGBM**")
            lgb_cm = confusion_matrix(y_test, lgb_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(lgb_cm, annot=True, fmt='d', cmap='Greens', ax=ax)
            ax.set_title('LightGBM Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            st.pyplot(fig)
            plt.close(fig)
    
    with tab3:
        st.subheader("Feature Importance Comparison")
        
        # Combine feature importance
        importance_comparison = pd.merge(
            rf_importance[['feature', 'importance']].rename(columns={'importance': 'RF_importance'}),
            lgb_importance[['feature', 'importance']].rename(columns={'importance': 'LGB_importance'}),
            on='feature'
        )
        
        # Normalize importances to 0-1 scale
        importance_comparison['RF_importance_norm'] = importance_comparison['RF_importance'] / importance_comparison['RF_importance'].max()
        importance_comparison['LGB_importance_norm'] = importance_comparison['LGB_importance'] / importance_comparison['LGB_importance'].max()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(importance_comparison))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, importance_comparison['RF_importance_norm'], width, 
                      label='Random Forest', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, importance_comparison['LGB_importance_norm'], width,
                      label='LightGBM', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Normalized Importance')
        ax.set_title('Feature Importance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(importance_comparison['feature'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Show numerical comparison
        st.write("**Feature Importance Rankings:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Random Forest Top 5:**")
            st.dataframe(rf_importance.head(), use_container_width=True)
        
        with col2:
            st.write("**LightGBM Top 5:**")
            st.dataframe(lgb_importance.head(), use_container_width=True)
    
    with tab4:
        st.subheader("Prediction Comparison")
        
        # Create prediction comparison dataframe
        pred_comparison = pd.DataFrame({
            'Actual': y_test,
            'RF_Prediction': rf_pred,
            'RF_Probability': rf_proba,
            'LGB_Prediction': lgb_pred,
            'LGB_Probability': lgb_proba
        })
        
        # Add agreement column
        pred_comparison['Models_Agree'] = pred_comparison['RF_Prediction'] == pred_comparison['LGB_Prediction']
        pred_comparison['Both_Correct'] = (pred_comparison['RF_Prediction'] == pred_comparison['Actual']) & \
                                         (pred_comparison['LGB_Prediction'] == pred_comparison['Actual'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Agreement statistics
            agreement_rate = pred_comparison['Models_Agree'].mean()
            both_correct_rate = pred_comparison['Both_Correct'].mean()
            
            st.metric("Model Agreement", f"{agreement_rate:.1%}")
            st.metric("Both Models Correct", f"{both_correct_rate:.1%}")
            
            # Show disagreement cases
            disagreements = pred_comparison[~pred_comparison['Models_Agree']]
            st.write(f"**Disagreement Cases: {len(disagreements)}**")
            
            if len(disagreements) > 0:
                st.dataframe(disagreements.head(10), use_container_width=True)
        
        with col2:
            # Probability distribution comparison
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.scatter(rf_proba, lgb_proba, alpha=0.6, c=y_test, cmap='RdYlBu')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('Random Forest Probability')
            ax.set_ylabel('LightGBM Probability')
            ax.set_title('Prediction Probability Comparison')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig)
    
    with tab5:
        st.subheader("Save Trained Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Random Forest Model", type="primary"):
                success, message = save_model_and_scaler(
                    results['rf_model'], 
                    results['scaler'], 
                    "RandomForest"
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        with col2:
            if st.button("💾 Save LightGBM Model", type="primary"):
                success, message = save_model_and_scaler(
                    results['lgb_model'], 
                    results['scaler'], 
                    "LightGBM"
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Model download section
        st.markdown("---")
        st.subheader("📥 Export Model Comparison Report")
        
        # Create comparison summary
        summary_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Random_Forest': [rf_metrics['accuracy'], rf_metrics['precision'], 
                             rf_metrics['recall'], rf_metrics['f1'], rf_metrics['roc_auc']],
            'LightGBM': [lgb_metrics['accuracy'], lgb_metrics['precision'], 
                        lgb_metrics['recall'], lgb_metrics['f1'], lgb_metrics['roc_auc']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False).encode()
        
        st.download_button(
            "📊 Download Comparison Report",
            csv,
            f"model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

def render_model_comparison(pivot_df):
    """
    Main function to render the model comparison interface
    """
    st.title("🔬 Model Comparison Hub")
    st.markdown("Compare Random Forest and LightGBM models side-by-side")
    
    # Quick model status check
    available_models = get_available_models()
    
    if available_models:
        st.info(f"📁 **Available Models:** {', '.join(available_models)}")
        
        # Quick load and compare existing models
        if len(available_models) == 2:  # Both models available
            if st.button("🔄 Quick Compare Existing Models"):
                with st.spinner("Loading and comparing existing models..."):
                    try:
                        # Load both models
                        rf_model, rf_scaler, rf_metadata, _ = load_model_and_scaler("RandomForest")
                        lgb_model, lgb_scaler, lgb_metadata, _ = load_model_and_scaler("LightGBM")
                        
                        if rf_model and lgb_model:
                            # Prepare test data
                            X = pivot_df[features]
                            y = pivot_df['Fault']
                            scaler, X_scaled = get_scaler(X)
                            _, X_test, _, y_test = split_data(X_scaled, y)
                            
                            # Evaluate models
                            from utils.model_builder import evaluate_model, get_feature_importance
                            
                            rf_metrics, rf_pred, rf_proba = evaluate_model(rf_model, X_test, y_test)
                            lgb_metrics, lgb_pred, lgb_proba = evaluate_model(lgb_model, X_test, y_test)
                            
                            rf_importance = get_feature_importance(rf_model)
                            lgb_importance = get_feature_importance(lgb_model)
                            
                            # Store in session state
                            st.session_state.comparison_results = {
                                'rf_model': rf_model,
                                'lgb_model': lgb_model,
                                'scaler': scaler,
                                'rf_metrics': rf_metrics,
                                'lgb_metrics': lgb_metrics,
                                'rf_pred': rf_pred,
                                'lgb_pred': lgb_pred,
                                'rf_proba': rf_proba,
                                'lgb_proba': lgb_proba,
                                'rf_importance': rf_importance,
                                'lgb_importance': lgb_importance,
                                'X_test': X_test,
                                'y_test': y_test
                            }
                            
                            st.success("✅ Models loaded and compared successfully!")
                            
                    except Exception as e:
                        st.error(f"❌ Error comparing existing models: {str(e)}")
    else:
        st.warning("⚠️ No trained models found. Train models using the comparison tool below.")
    
    # Main comparison interface
    run_model_comparison(pivot_df)
