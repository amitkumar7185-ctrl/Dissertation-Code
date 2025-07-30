import joblib
import os
import pickle
import streamlit as st
from datetime import datetime
import json

def save_model_and_scaler(model, scaler, model_type="RandomForest", model_dir="models"):
    """
    Save the trained model and scaler to disk with metadata
    Supports both RandomForest and LightGBM models
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Create timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define file paths based on model type
        model_prefix = "rf" if model_type == "RandomForest" else "lgb"
        model_path = os.path.join(model_dir, f"{model_prefix}_fault_model.pkl")
        scaler_path = os.path.join(model_dir, f"{model_prefix}_scaler.pkl")
        
        # Save versioned backups
        model_backup_path = os.path.join(model_dir, f"{model_prefix}_fault_model_{timestamp}.pkl")
        scaler_backup_path = os.path.join(model_dir, f"{model_prefix}_scaler_{timestamp}.pkl")
        
        # Save current models
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save backup versions
        joblib.dump(model, model_backup_path)
        joblib.dump(scaler, scaler_backup_path)
        
        # Prepare metadata based on model type
        if model_type == "RandomForest":
            metadata = {
                "model_type": "RandomForest",
                "created_at": datetime.now().isoformat(),
                "model_params": model.get_params(),
                "n_features": model.n_features_in_,
                "n_estimators": model.n_estimators,
                "timestamp": timestamp
            }
        else:  # LightGBM
            metadata = {
                "model_type": "LightGBM",
                "created_at": datetime.now().isoformat(),
                "n_features": model.num_feature(),
                "n_estimators": model.num_trees(),
                "timestamp": timestamp
            }
        
        metadata_path = os.path.join(model_dir, f"{model_prefix}_model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True, f"✅ {model_type} model and scaler saved successfully!\n📁 Model: {model_path}\n📁 Scaler: {scaler_path}"
        
    except Exception as e:
        return False, f"❌ Error saving {model_type} model: {str(e)}"

def load_model_and_scaler(model_type="RandomForest", model_dir="models"):
    """
    Load the trained model and scaler from disk
    Supports both RandomForest and LightGBM models
    """
    try:
        # Define file paths based on model type
        model_prefix = "rf" if model_type == "RandomForest" else "lgb"
        model_path = os.path.join(model_dir, f"{model_prefix}_fault_model.pkl")
        scaler_path = os.path.join(model_dir, f"{model_prefix}_scaler.pkl")
        metadata_path = os.path.join(model_dir, f"{model_prefix}_model_metadata.json")
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None, None, f"❌ {model_type} model files not found. Please train a model first."
        
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load metadata if available
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, scaler, metadata, f"✅ {model_type} model and scaler loaded successfully!"
        
    except Exception as e:
        return None, None, None, f"❌ Error loading {model_type} model: {str(e)}"

def check_model_exists(model_type="RandomForest", model_dir="models"):
    """
    Check if trained models exist for specified model type
    """
    model_prefix = "rf" if model_type == "RandomForest" else "lgb"
    model_path = os.path.join(model_dir, f"{model_prefix}_fault_model.pkl")
    scaler_path = os.path.join(model_dir, f"{model_prefix}_scaler.pkl")
    
    return os.path.exists(model_path) and os.path.exists(scaler_path)

def get_model_info(model_type="RandomForest", model_dir="models"):
    """
    Get information about saved models for specified model type
    """
    try:
        model_prefix = "rf" if model_type == "RandomForest" else "lgb"
        metadata_path = os.path.join(model_dir, f"{model_prefix}_model_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            return None
    except:
        return None

def get_available_models(model_dir="models"):
    """
    Get list of all available model types
    """
    available_models = []
    
    # Check for RandomForest models
    if check_model_exists("RandomForest", model_dir):
        available_models.append("RandomForest")
    
    # Check for LightGBM models
    if check_model_exists("LightGBM", model_dir):
        available_models.append("LightGBM")
    
    return available_models

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
