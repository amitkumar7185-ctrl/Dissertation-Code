import joblib
import os
import pickle
import streamlit as st
from datetime import datetime
import json

def save_model_and_scaler(rf_model, rf_scaler, model_dir="models"):
    """
    Save the trained model and scaler to disk with metadata
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Create timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define file paths
        model_path = os.path.join(model_dir, "rf_fault_model.pkl")
        scaler_path = os.path.join(model_dir, "rf_scaler.pkl")
        
        # Save versioned backups
        model_backup_path = os.path.join(model_dir, f"rf_fault_model_{timestamp}.pkl")
        scaler_backup_path = os.path.join(model_dir, f"rf_scaler_{timestamp}.pkl")
        
        # Save current models
        joblib.dump(rf_model, model_path)
        joblib.dump(rf_scaler, scaler_path)
        
        # Save backup versions
        joblib.dump(rf_model, model_backup_path)
        joblib.dump(rf_scaler, scaler_backup_path)
        
        # Save metadata
        metadata = {
            "model_type": "RandomForest",
            "created_at": datetime.now().isoformat(),
            "model_params": rf_model.get_params(),
            "n_features": rf_model.n_features_in_,
            "n_estimators": rf_model.n_estimators,
            "timestamp": timestamp
        }
        
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True, f"✅ Model and scaler saved successfully!\n📁 Model: {model_path}\n📁 Scaler: {scaler_path}"
        
    except Exception as e:
        return False, f"❌ Error saving model: {str(e)}"

def load_model_and_scaler(model_dir="models"):
    """
    Load the trained model and scaler from disk
    """
    try:
        model_path = os.path.join(model_dir, "rf_fault_model.pkl")
        scaler_path = os.path.join(model_dir, "rf_scaler.pkl")
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None, None, "❌ Model files not found. Please train a model first."
        
        # Load model and scaler
        rf_model = joblib.load(model_path)
        rf_scaler = joblib.load(scaler_path)
        
        # Load metadata if available
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return rf_model, rf_scaler, metadata, "✅ Model and scaler loaded successfully!"
        
    except Exception as e:
        return None, None, None, f"❌ Error loading model: {str(e)}"

def check_model_exists(model_dir="models"):
    """
    Check if trained models exist
    """
    model_path = os.path.join(model_dir, "rf_fault_model.pkl")
    scaler_path = os.path.join(model_dir, "rf_scaler.pkl")
    
    return os.path.exists(model_path) and os.path.exists(scaler_path)

def get_model_info(model_dir="models"):
    """
    Get information about saved models
    """
    try:
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            return None
    except:
        return None

def list_model_versions(model_dir="models"):
    """
    List all available model versions
    """
    try:
        if not os.path.exists(model_dir):
            return []
        
        model_files = [f for f in os.listdir(model_dir) if f.startswith("rf_fault_model_") and f.endswith(".pkl")]
        versions = []
        
        for file in model_files:
            timestamp = file.replace("rf_fault_model_", "").replace(".pkl", "")
            file_path = os.path.join(model_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            
            versions.append({
                "timestamp": timestamp,
                "file": file,
                "size_kb": file_size
            })
        
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)
    except:
        return []

def load_specific_model_version(timestamp, model_dir="models"):
    """
    Load a specific version of the model
    """
    try:
        model_path = os.path.join(model_dir, f"rf_fault_model_{timestamp}.pkl")
        scaler_path = os.path.join(model_dir, f"rf_scaler_{timestamp}.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None, f"❌ Model version {timestamp} not found."
        
        rf_model = joblib.load(model_path)
        rf_scaler = joblib.load(scaler_path)
        
        return rf_model, rf_scaler, f"✅ Loaded model version {timestamp}"
        
    except Exception as e:
        return None, None, f"❌ Error loading model version: {str(e)}"
