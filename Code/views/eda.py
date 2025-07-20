
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from config import features
from utils.preprocessing import prepare_pivoted_data, label_faults

def render_eda(pivot_df):
    st.subheader("Exploratory Data Analysis")  
    
    st.markdown("**Feature Distribution by Fault**")
    sampled_df = pivot_df[features + ['Fault']].sample(n=min(300, len(pivot_df)), random_state=42)
  

    pairplot_fig = sns.pairplot(sampled_df, hue="Fault")
    st.pyplot(pairplot_fig.figure)

    st.markdown("**Correlation Heatmap**")
    plt.figure(figsize=(8, 5))

    heatmap_fig =sns.heatmap(pivot_df[features + ['Fault']].corr(), annot=True, cmap="coolwarm")
    st.pyplot(heatmap_fig.figure)
