import pandas as pd
import numpy as np
import os
import datetime
from sklearn.metrics import confusion_matrix, classification_report

def generate_complete_test_dataset_csv(rf_model, rf_scaler, X_test, y_test, 
                                     original_test_data=None, feature_names=None, 
                                     output_dir="Reports"):
    """
    Generate a comprehensive CSV file containing all test records (20% of dataset) 
    with original features, actual labels, and predicted labels.
    
    Parameters:
    - rf_model: Trained Random Forest model
    - rf_scaler: Fitted scaler used for preprocessing
    - X_test: Test features (scaled)
    - y_test: Actual test labels
    - original_test_data: Original unscaled test data (optional)
    - feature_names: List of feature names
    - output_dir: Directory to save the CSV file
    
    Returns:
    - csv_filename: Path to the generated CSV file
    - results_df: DataFrame with all test data and predictions
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions on test set
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability for positive class
    
    # Convert test data to DataFrame if it's not already
    if hasattr(X_test, 'values'):
        X_test_array = X_test.values
        if hasattr(X_test, 'index'):
            test_indices = X_test.index.tolist()
        else:
            test_indices = list(range(len(X_test)))
    else:
        X_test_array = np.array(X_test)
        test_indices = list(range(len(X_test)))
    
    # Convert labels to arrays
    if hasattr(y_test, 'values'):
        y_test_array = y_test.values
    else:
        y_test_array = np.array(y_test)
    
    # Prepare feature names
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_test_array.shape[1])]
    
    # Create the complete dataset
    complete_records = []
    
    for i in range(len(y_test_array)):
        # Basic information
        record = {
            'Record_ID': i + 1,
            'Original_Dataset_Index': test_indices[i],
            'Dataset_Split': 'Test (20%)',
            'Actual_Label': int(y_test_array[i]),
            'Actual_Label_Description': 'Fault' if y_test_array[i] == 1 else 'No Fault',
            'Predicted_Label': int(y_pred[i]),
            'Predicted_Label_Description': 'Fault' if y_pred[i] == 1 else 'No Fault',
            'Prediction_Probability': round(float(y_pred_proba[i]), 6),
            'Prediction_Confidence': get_confidence_level(y_pred_proba[i]),
            'Prediction_Correct': bool(y_test_array[i] == y_pred[i]),
        }
        
        # Classification category
        if y_test_array[i] == 1 and y_pred[i] == 1:
            record['Classification_Type'] = 'True Positive (TP)'
            record['Classification_Result'] = 'Correct - Fault Detected'
        elif y_test_array[i] == 0 and y_pred[i] == 0:
            record['Classification_Type'] = 'True Negative (TN)'
            record['Classification_Result'] = 'Correct - Normal Operation'
        elif y_test_array[i] == 0 and y_pred[i] == 1:
            record['Classification_Type'] = 'False Positive (FP)'
            record['Classification_Result'] = 'Incorrect - False Alarm'
        else:  # y_test_array[i] == 1 and y_pred[i] == 0
            record['Classification_Type'] = 'False Negative (FN)'
            record['Classification_Result'] = 'Incorrect - Missed Fault'
        
        # Add scaled feature values (as used by the model)
        for j, feature_name in enumerate(feature_names):
            record[f'Scaled_{feature_name}'] = round(float(X_test_array[i, j]), 6)
        
        # Add original feature values if available
        if original_test_data is not None:
            if hasattr(original_test_data, 'iloc'):
                original_row = original_test_data.iloc[i]
                for feature_name in feature_names:
                    if feature_name in original_test_data.columns:
                        record[f'Original_{feature_name}'] = round(float(original_row[feature_name]), 6)
        
        complete_records.append(record)
    
    # Create DataFrame
    results_df = pd.DataFrame(complete_records)
    
    # Add summary statistics at the end
    summary_stats = calculate_dataset_summary(y_test_array, y_pred, y_pred_proba)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"complete_test_dataset_{timestamp}.csv")
    
    # Save the CSV file
    results_df.to_csv(csv_filename, index=False)
    
    # Also save summary statistics
    summary_filename = os.path.join(output_dir, f"test_dataset_summary_{timestamp}.csv")
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(summary_filename, index=False)
    
    return csv_filename, results_df, summary_stats

def get_confidence_level(probability):
    """Convert probability to confidence level"""
    if probability >= 0.9 or probability <= 0.1:
        return "Very High"
    elif probability >= 0.8 or probability <= 0.2:
        return "High"
    elif probability >= 0.7 or probability <= 0.3:
        return "Medium"
    elif probability >= 0.6 or probability <= 0.4:
        return "Low"
    else:
        return "Very Low"

def calculate_dataset_summary(y_test, y_pred, y_pred_proba):
    """Calculate comprehensive summary statistics"""
    
    # Basic counts
    total_records = len(y_test)
    actual_faults = np.sum(y_test)
    actual_normal = total_records - actual_faults
    predicted_faults = np.sum(y_pred)
    predicted_normal = total_records - predicted_faults
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Performance metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / total_records
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Error rates
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Confidence distribution
    high_confidence = np.sum((y_pred_proba >= 0.8) | (y_pred_proba <= 0.2))
    medium_confidence = np.sum((y_pred_proba >= 0.6) & (y_pred_proba <= 0.8)) + np.sum((y_pred_proba >= 0.2) & (y_pred_proba <= 0.4))
    low_confidence = total_records - high_confidence - medium_confidence
    
    summary = {
        'Report_Generated_At': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Dataset_Split_Type': 'Test Set (20% of total data)',
        'Total_Test_Records': total_records,
        'Actual_Fault_Records': actual_faults,
        'Actual_Normal_Records': actual_normal,
        'Predicted_Fault_Records': predicted_faults,
        'Predicted_Normal_Records': predicted_normal,
        'Fault_Percentage_In_Test': round((actual_faults / total_records) * 100, 2),
        'True_Positives': int(tp),
        'True_Negatives': int(tn),
        'False_Positives': int(fp),
        'False_Negatives': int(fn),
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'Specificity': round(specificity, 4),
        'F1_Score': round(f1_score, 4),
        'False_Positive_Rate': round(false_positive_rate, 4),
        'False_Negative_Rate': round(false_negative_rate, 4),
        'High_Confidence_Predictions': int(high_confidence),
        'Medium_Confidence_Predictions': int(medium_confidence),
        'Low_Confidence_Predictions': int(low_confidence),
        'Average_Prediction_Probability': round(np.mean(y_pred_proba), 4),
        'Min_Prediction_Probability': round(np.min(y_pred_proba), 4),
        'Max_Prediction_Probability': round(np.max(y_pred_proba), 4),
        'Recall_vs_93_Percent_Target': 'ACHIEVED' if recall >= 0.93 else f'NEED {round(0.93 - recall, 3)} MORE',
        'Model_Threshold_Used': 0.5
    }
    
    return summary

def generate_original_test_data_if_needed(pivot_df, test_indices, feature_names):
    """
    Extract original test data from the complete dataset if test indices are available
    """
    try:
        if hasattr(test_indices, '__iter__') and len(test_indices) > 0:
            # Get original data for test indices
            original_test_data = pivot_df.iloc[test_indices][feature_names].copy()
            return original_test_data
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not extract original test data: {e}")
        return None
