
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import features
from utils.preprocessing import prepare_pivoted_data
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

            st.success("Prediction Complete")
            st.dataframe(pivot_df)

            fig, ax = plt.subplots()
            ax.plot(pivot_df['front_door_reversals'], label="Front Reversals", marker='o')
            ax.plot(pivot_df['rear_door_reversals'], label="Rear Reversals", marker='o')
            try:
                fp = pd.to_numeric(pivot_df['Fault_Probability'], errors='coerce')
                ax.plot(fp, label="Fault Probability", marker='x')
            except:
                pass
            ax.legend()
            st.pyplot(fig)

            csv = pivot_df.to_csv(index=False).encode()
            st.download_button("Download Results CSV", csv, "predicted_faults.csv", "text/csv")
