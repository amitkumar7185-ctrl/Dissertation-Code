import pandas as pd
import numpy as np
import datetime
import os

def create_test_dataset_csv(X_test, y_test, y_pred, y_pred_proba, feature_names):
    """
    Create a simple CSV file with all test records (20% of dataset),
    their actual labels, predicted labels, and feature values.
    
    Parameters:
    - X_test: Test features 
    - y_test: Actual test labels
    - y_pred: Predicted labels
    - y_pred_proba: Prediction probabilities
    - feature_names: Names of features
    
    Returns:
    - csv_filename: Path to created CSV file
    - dataframe: The complete dataset as DataFrame
    """
    
    # Ensure we have numpy arrays for consistent indexing
    if hasattr(X_test, 'values'):
        X_test_array = X_test.values
    else:
        X_test_array = np.array(X_test)
    
    if hasattr(y_test, 'values'):
        y_test_array = y_test.values
    else:
        y_test_array = np.array(y_test)
    
    # Create the complete dataset
    all_records = []
    
    for i in range(len(y_test_array)):
        # Basic record information
        record = {
            'Record_ID': i + 1,
            'Dataset_Split': 'Test_Set_20_Percent',
            'Actual_Label': int(y_test_array[i]),
            'Actual_Label_Text': 'Fault' if y_test_array[i] == 1 else 'Normal',
            'Predicted_Label': int(y_pred[i]),
            'Predicted_Label_Text': 'Fault' if y_pred[i] == 1 else 'Normal',
            'Prediction_Probability': round(float(y_pred_proba[i]), 4),
            'Prediction_Correct': bool(y_test_array[i] == y_pred[i]),
            'Model_Threshold': 0.5,
            'Above_Threshold': 'Yes' if y_pred_proba[i] >= 0.5 else 'No'
        }
        
        # Add classification type
        if y_test_array[i] == 1 and y_pred[i] == 1:
            record['Classification_Type'] = 'True_Positive'
            record['Classification_Result'] = 'Correct_Fault_Detection'
        elif y_test_array[i] == 0 and y_pred[i] == 0:
            record['Classification_Type'] = 'True_Negative'
            record['Classification_Result'] = 'Correct_Normal_Classification'
        elif y_test_array[i] == 0 and y_pred[i] == 1:
            record['Classification_Type'] = 'False_Positive'
            record['Classification_Result'] = 'False_Alarm'
        else:
            record['Classification_Type'] = 'False_Negative'
            record['Classification_Result'] = 'Missed_Fault'
        
        # Add confidence level
        prob = y_pred_proba[i]
        if prob >= 0.9 or prob <= 0.1:
            record['Confidence_Level'] = 'Very_High'
        elif prob >= 0.8 or prob <= 0.2:
            record['Confidence_Level'] = 'High'
        elif prob >= 0.7 or prob <= 0.3:
            record['Confidence_Level'] = 'Medium'
        else:
            record['Confidence_Level'] = 'Low'
        
        # Add all feature values
        for j, feature_name in enumerate(feature_names):
            if j < X_test_array.shape[1]:
                record[f'Feature_{feature_name}'] = round(float(X_test_array[i, j]), 4)
        
        all_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Create Reports directory in the Code folder (absolute path)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils folder
    code_dir = os.path.dirname(current_dir)  # Code folder
    reports_dir = os.path.join(code_dir, 'Reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(reports_dir, f"complete_test_dataset_{timestamp}.csv")
    
    # Save to CSV
    df.to_csv(csv_filename, index=False)
    
    return csv_filename, df

def get_dataset_summary(df):
    """Get summary statistics for the dataset"""
    total_records = len(df)
    actual_faults = len(df[df['Actual_Label'] == 1])
    predicted_faults = len(df[df['Predicted_Label'] == 1])
    correct_predictions = len(df[df['Prediction_Correct'] == True])
    
    tp = len(df[df['Classification_Type'] == 'True_Positive'])
    tn = len(df[df['Classification_Type'] == 'True_Negative'])
    fp = len(df[df['Classification_Type'] == 'False_Positive'])
    fn = len(df[df['Classification_Type'] == 'False_Negative'])
    
    # Calculate metrics
    accuracy = correct_predictions / total_records if total_records > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    summary = {
        'total_records': total_records,
        'actual_faults': actual_faults,
        'predicted_faults': predicted_faults,
        'correct_predictions': correct_predictions,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'fault_percentage': round((actual_faults / total_records) * 100, 2) if total_records > 0 else 0
    }
    
    return summary
