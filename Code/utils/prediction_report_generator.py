import pandas as pd
import numpy as np
import os
import datetime
from sklearn.metrics import confusion_matrix

def generate_detailed_prediction_report(rf_model, X_test, y_test, y_pred, y_pred_proba, features, output_dir="Reports"):
    """
    Generate a comprehensive CSV report with actual vs predicted labels and detailed reasoning
    
    Parameters:
    - rf_model: Trained Random Forest model
    - X_test: Test features
    - y_test: Actual test labels
    - y_pred: Predicted labels
    - y_pred_proba: Prediction probabilities
    - features: List of feature names
    - output_dir: Directory to save the report
    
    Returns:
    - csv_filename: Path to the generated CSV file
    - results_df: DataFrame with detailed results
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature importance for reasoning
    feature_importance = rf_model.feature_importances_
    feature_importance_dict = dict(zip(features, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Define feature interpretation rules
    fault_indicators = {
        'door_reversals': {'high': 0.5, 'moderate': 0.2, 'description': 'door reversal frequency'},
        'door_failure_events': {'high': 0.3, 'moderate': 0.1, 'description': 'door failure incidents'},
        'safety_chain_issues': {'high': 0.4, 'moderate': 0.15, 'description': 'safety chain problems'},
        'hoistway_faults': {'high': 0.3, 'moderate': 0.1, 'description': 'hoistway mechanical issues'},
        'levelling_total_errors': {'high': 0.4, 'moderate': 0.2, 'description': 'leveling accuracy problems'},
        'startup_delays': {'high': 0.5, 'moderate': 0.25, 'description': 'system startup issues'},
        'door_reversal_rate': {'high': 0.3, 'moderate': 0.15, 'description': 'door reversal rate'},
        'safety_chain_issues_ratio': {'high': 0.2, 'moderate': 0.1, 'description': 'safety chain issue ratio'},
        'slow_door_operations_ratio': {'high': 0.3, 'moderate': 0.15, 'description': 'slow door operation ratio'}
    }
    
    detailed_results = []
    
    # Convert to numpy arrays if needed for consistent indexing
    if hasattr(y_test, 'values'):
        y_test_array = y_test.values
    else:
        y_test_array = np.array(y_test)
    
    if hasattr(X_test, 'values'):
        X_test_array = X_test.values
    else:
        X_test_array = np.array(X_test)
    
    for i in range(len(y_test_array)):
        actual_label = int(y_test_array[i])
        predicted_label = int(y_pred[i])
        prediction_probability = float(y_pred_proba[i])
        
        # Get feature values for this sample
        sample_features = X_test_array[i]
        
        # Determine prediction category and correctness
        if actual_label == 1 and predicted_label == 1:
            category = "True Positive"
            result = "Correct"
            correctness = "✓"
        elif actual_label == 0 and predicted_label == 0:
            category = "True Negative"
            result = "Correct"
            correctness = "✓"
        elif actual_label == 0 and predicted_label == 1:
            category = "False Positive"
            result = "Incorrect - False Alarm"
            correctness = "✗"
        else:  # actual_label == 1 and predicted_label == 0
            category = "False Negative"
            result = "Incorrect - Missed Fault"
            correctness = "✗"
        
        # Determine confidence level
        if prediction_probability >= 0.8:
            confidence = "Very High"
        elif prediction_probability >= 0.65:
            confidence = "High"
        elif prediction_probability >= 0.5:
            confidence = "Medium"
        elif prediction_probability >= 0.35:
            confidence = "Low"
        else:
            confidence = "Very Low"
        
        # Generate detailed reasoning
        reasoning_parts = []
        
        # 1. Probability-based reasoning
        reasoning_parts.append(f"Model confidence: {confidence} ({prediction_probability:.3f})")
        
        # 2. Feature-based reasoning
        feature_analysis = []
        top_5_features = [f[0] for f in sorted_features[:5]]
        
        for feat_name in top_5_features:
            if feat_name in features:
                feat_idx = features.index(feat_name)
                feat_value = sample_features[feat_idx]
                feat_importance = feature_importance_dict[feat_name]
                
                # Analyze feature contribution
                analysis = analyze_feature_contribution(feat_name, feat_value, feat_importance, fault_indicators)
                if analysis:
                    feature_analysis.append(analysis)
        
        if feature_analysis:
            reasoning_parts.append("Key factors: " + "; ".join(feature_analysis[:3]))
        
        # 3. Decision boundary reasoning
        if predicted_label == 1:
            if prediction_probability > 0.7:
                reasoning_parts.append("Strong fault pattern detected")
            elif prediction_probability > 0.5:
                reasoning_parts.append("Moderate fault risk identified")
            else:
                reasoning_parts.append("Borderline fault detection")
        else:
            if prediction_probability < 0.3:
                reasoning_parts.append("Normal operation pattern confirmed")
            elif prediction_probability < 0.5:
                reasoning_parts.append("Low fault risk, normal operation likely")
            else:
                reasoning_parts.append("Borderline case, leaning towards normal")
        
        # 4. Risk assessment
        if category == "False Negative":
            risk_level = "Critical"
            risk_description = "Undetected fault - high safety risk"
        elif category == "False Positive":
            risk_level = "Medium"
            risk_description = "False alarm - operational cost impact"
        elif category == "True Positive":
            risk_level = "Low"
            risk_description = "Correct fault detection - good catch"
        else:
            risk_level = "Very Low"
            risk_description = "Correct normal classification"
        
        # 5. Actionable insights
        if category == "False Negative":
            action = "URGENT: Investigate for potential fault, review maintenance schedule"
        elif category == "False Positive":
            action = "Verify alarm, consider threshold adjustment if pattern persists"
        elif category == "True Positive":
            action = "Schedule maintenance as recommended by model"
        else:
            action = "Continue normal operation, no action required"
        
        # Compile full reasoning
        full_reason = " | ".join(reasoning_parts)
        
        # Create feature summary
        feature_summary = {}
        for j, feat_name in enumerate(features[:10]):  # Top 10 features
            feature_summary[f'Feature_{feat_name}'] = round(sample_features[j], 4)
        
        # Build result record
        result_record = {
            'Sample_ID': i + 1,
            'Actual_Label': actual_label,
            'Actual_Label_Description': 'Fault Present' if actual_label == 1 else 'Normal Operation',
            'Predicted_Label': predicted_label,
            'Predicted_Label_Description': 'Fault Predicted' if predicted_label == 1 else 'Normal Predicted',
            'Prediction_Probability': round(prediction_probability, 4),
            'Confidence_Level': confidence,
            'Prediction_Category': category,
            'Correctness': correctness,
            'Result_Description': result,
            'Risk_Level': risk_level,
            'Risk_Description': risk_description,
            'Recommended_Action': action,
            'Detailed_Reasoning': full_reason,
            'Model_Decision_Threshold': 0.5,
            'Probability_Above_Threshold': 'Yes' if prediction_probability >= 0.5 else 'No'
        }
        
        # Add feature values
        result_record.update(feature_summary)
        
        # Add top contributing features
        for k, (feat_name, importance) in enumerate(sorted_features[:5]):
            feat_idx = features.index(feat_name)
            result_record[f'Top_{k+1}_Feature_Name'] = feat_name
            result_record[f'Top_{k+1}_Feature_Value'] = round(sample_features[feat_idx], 4)
            result_record[f'Top_{k+1}_Feature_Importance'] = round(importance, 4)
        
        detailed_results.append(result_record)
    
    # Create DataFrame
    results_df = pd.DataFrame(detailed_results)
    
    # Add summary statistics
    summary_stats = calculate_summary_statistics(results_df)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"detailed_prediction_analysis_{timestamp}.csv")
    
    # Save main results
    results_df.to_csv(csv_filename, index=False)
    
    # Save summary statistics
    summary_filename = os.path.join(output_dir, f"prediction_summary_{timestamp}.csv")
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(summary_filename, index=False)
    
    return csv_filename, results_df, summary_stats

def analyze_feature_contribution(feature_name, feature_value, importance, fault_indicators):
    """Analyze how a specific feature contributes to the prediction"""
    
    # Clean feature name for matching
    clean_name = feature_name.lower().replace('_', '').replace(' ', '')
    
    # Find matching fault indicator
    indicator_info = None
    for indicator_key, info in fault_indicators.items():
        if indicator_key.lower().replace('_', '') in clean_name:
            indicator_info = info
            break
    
    if indicator_info:
        if feature_value >= indicator_info['high']:
            level = "High"
            concern = "significant concern"
        elif feature_value >= indicator_info['moderate']:
            level = "Moderate"
            concern = "some concern"
        else:
            level = "Low"
            concern = "minimal concern"
        
        return f"{indicator_info['description']}: {level} ({feature_value:.3f}) - {concern}"
    else:
        # Generic analysis for unknown features
        if feature_value > 0.6:
            return f"{feature_name.replace('_', ' ')}: High value ({feature_value:.3f}) - potential fault indicator"
        elif feature_value > 0.3:
            return f"{feature_name.replace('_', ' ')}: Moderate value ({feature_value:.3f}) - monitor closely"
        else:
            return f"{feature_name.replace('_', ' ')}: Normal value ({feature_value:.3f})"

def calculate_summary_statistics(results_df):
    """Calculate summary statistics for the prediction report"""
    
    total_samples = len(results_df)
    
    # Count categories
    tp = len(results_df[results_df['Prediction_Category'] == 'True Positive'])
    tn = len(results_df[results_df['Prediction_Category'] == 'True Negative'])
    fp = len(results_df[results_df['Prediction_Category'] == 'False Positive'])
    fn = len(results_df[results_df['Prediction_Category'] == 'False Negative'])
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / total_samples
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Risk analysis
    critical_risk = len(results_df[results_df['Risk_Level'] == 'Critical'])
    medium_risk = len(results_df[results_df['Risk_Level'] == 'Medium'])
    
    # Confidence distribution
    high_confidence = len(results_df[results_df['Confidence_Level'].isin(['High', 'Very High'])])
    low_confidence = len(results_df[results_df['Confidence_Level'].isin(['Low', 'Very Low'])])
    
    summary = {
        'Report_Generated_At': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Total_Samples': total_samples,
        'True_Positives': tp,
        'True_Negatives': tn,
        'False_Positives': fp,
        'False_Negatives': fn,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'Specificity': round(specificity, 4),
        'Accuracy': round(accuracy, 4),
        'F1_Score': round(f1_score, 4),
        'Critical_Risk_Cases': critical_risk,
        'Medium_Risk_Cases': medium_risk,
        'High_Confidence_Predictions': high_confidence,
        'Low_Confidence_Predictions': low_confidence,
        'False_Alarm_Rate': round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        'Miss_Rate': round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
        'Recall_Target_93_Percent': 'ACHIEVED' if recall >= 0.93 else f'GAP: {0.93 - recall:.3f}',
    }
    
    return summary
