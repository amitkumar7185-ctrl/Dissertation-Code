
import streamlit as st

def display_menu():
    st.title("Elevator Fault Detection & Insights Dashboard")
    return st.sidebar.radio("Choose View", ["Upload & Predict", "EDA", "SHAP Explainability"])
