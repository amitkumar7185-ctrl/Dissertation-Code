
import streamlit as st
import shap
import pandas as pd
from config import features
from utils.preprocessing import prepare_pivoted_data, label_faults

def render_shap(rf_model, rf_scaler, pivot_df):
    st.subheader("SHAP Explainability (Random Forest)")   
    X = pivot_df[features]
    X_scaled = rf_scaler.transform(X)

    explainer = shap.Explainer(rf_model, X_scaled)
    shap_values = explainer(X_scaled)

    st.markdown("### Global Feature Importance")
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    st.pyplot()

    st.markdown("### Local Explanation (Sample Instance 0)")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot()
