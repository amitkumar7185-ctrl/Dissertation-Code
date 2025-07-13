import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# Load models and scalers
rf_model = joblib.load("rf_fault_model.pkl")
rf_scaler = joblib.load("rf_scaler.pkl")

lstm_model = load_model("fault_lstm_model.h5")
lstm_scaler = joblib.load("feature_scaler.pkl")

# Configuration
features = [
    'front_door_cycles', 'rear_door_cycles',
    'front_door_reversals', 'rear_door_reversals',
    'door_operations', 'total_door_cycles'
]

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Elevator Fault Detection & Insights Dashboard")

menu = st.sidebar.radio("Choose View", ["Upload & Predict", "EDA", "SHAP Explainability"])

# --- Upload & Predict ---
if menu == "Upload & Predict":
    st.subheader("Upload 5-Day Sensor CSV")
    model_type = st.radio("Model Type", ["Random Forest", "LSTM (Time-Series)"])
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview")
        st.dataframe(df)

        if df.shape[0] != 5 or not all(col in df.columns for col in features):
            st.error("CSV must have exactly 5 rows and required columns.")
        else:
            if model_type == "Random Forest":
                X_scaled = rf_scaler.transform(df[features])
                preds = rf_model.predict(X_scaled)
                probs = rf_model.predict_proba(X_scaled)[:, 1]
                df["Predicted_Fault"] = preds
                df["Fault_Probability"] = np.round(probs, 2)
            else:
                X_seq = lstm_scaler.transform(df[features]).reshape(1, 5, len(features))
                prob = lstm_model.predict(X_seq)[0][0]
                df["Predicted_Fault"] = [""] * 4 + [int(prob > 0.5)]
                df["Fault_Probability"] = [""] * 4 + [round(prob, 2)]

            st.success("Prediction Complete")
            st.dataframe(df)

            # Visualization
            st.subheader("Sensor Trends")
            fig, ax = plt.subplots()
            ax.plot(df['front_door_reversals'], label="Front Reversals", marker='o')
            ax.plot(df['rear_door_reversals'], label="Rear Reversals", marker='o')
            try:
                fp = pd.to_numeric(df['Fault_Probability'], errors='coerce')
                ax.plot(fp, label="Fault Probability", marker='x')
            except:
                pass
            ax.legend()
            st.pyplot(fig)

            # Download button
            csv = df.to_csv(index=False).encode()
            st.download_button("Download Results CSV", csv, "predicted_faults.csv", "text/csv")

# --- EDA ---
elif menu == "EDA":
    st.subheader("Exploratory Data Analysis")

    # Load and prepare full dataset
    df = pd.read_excel("sampleData.xlsx")
    df['PerformanceStartDate'] = pd.to_datetime(df['PerformanceStartDate'])
    df['Date'] = df['PerformanceStartDate'].dt.date
    pivot_df = df.pivot_table(
        index=['elevatorunitId', 'elevatorunitnumber', 'Date'],
        columns='ItemFieldId',
        values='Readvalue',
        aggfunc='sum'
    ).reset_index().fillna(0)
    pivot_df['Fault'] = (
        (pivot_df['front_door_reversals'] > 20) |
        (pivot_df['rear_door_reversals'] > 15) |
        (pivot_df['total_door_cycles'] < 10)
    ).astype(int)

    # Pairplot (optional sampling for speed)
    st.markdown("**Feature Distribution by Fault**")
    sampled_df = pivot_df[features + ['Fault']].sample(n=min(300, len(pivot_df)), random_state=42)
    sns.pairplot(sampled_df, hue="Fault")
    st.pyplot()

    # Correlation heatmap
    st.markdown("**Correlation Heatmap**")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_df[features + ['Fault']].corr(), annot=True, cmap="coolwarm")
    st.pyplot()

# --- SHAP Explainability ---
elif menu == "SHAP Explainability":
    st.subheader("SHAP Explainability (Random Forest)")

    # Load and scale dataset
    df = pd.read_excel("sampleData.xlsx")
    df['PerformanceStartDate'] = pd.to_datetime(df['PerformanceStartDate'])
    df['Date'] = df['PerformanceStartDate'].dt.date
    pivot_df = df.pivot_table(
        index=['elevatorunitId', 'elevatorunitnumber', 'Date'],
        columns='ItemFieldId',
        values='Readvalue',
        aggfunc='sum'
    ).reset_index().fillna(0)
    pivot_df['Fault'] = (
        (pivot_df['front_door_reversals'] > 20) |
        (pivot_df['rear_door_reversals'] > 15) |
        (pivot_df['total_door_cycles'] < 10)
    ).astype(int)
    X = pivot_df[features]
    X_scaled = rf_scaler.transform(X)

    # SHAP values
    explainer = shap.Explainer(rf_model, X_scaled)
    shap_values = explainer(X_scaled)

    # Summary plot
    st.markdown("### Global Feature Importance")
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    st.pyplot()

    # Waterfall for a single instance
    st.markdown("### Local Explanation (Sample Instance 0)")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot()
