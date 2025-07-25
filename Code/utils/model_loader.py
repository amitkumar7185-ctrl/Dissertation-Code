
import joblib
#from tensorflow.keras.models import load_model

def load_models():
    rf_model = joblib.load("rf_fault_model.pkl")
    rf_scaler = joblib.load("rf_scaler.pkl")

    #lstm_model = load_model("fault_lstm_model.h5")
    #lstm_scaler = joblib.load("feature_scaler.pkl")

    return rf_model, rf_scaler #, lstm_model, lstm_scaler
