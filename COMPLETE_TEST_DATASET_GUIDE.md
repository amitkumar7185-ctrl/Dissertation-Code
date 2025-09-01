# Complete Test Dataset CSV Generation Guide

## Overview
This feature generates a comprehensive CSV file containing ALL test records (20% of the original dataset) with their actual labels, predicted labels, and complete feature information.

## 🎯 Purpose
Create a complete audit trail of all test predictions with the following information:
- **All test records** from the 20% test split
- **Original actual labels** from the dataset
- **Model predictions** and probabilities
- **Complete feature values** used for predictions
- **Classification results** (TP/TN/FP/FN)
- **Performance metrics** and confidence levels

## 📍 Location in Application
**Dashboard → Model Performance → Show Model Metrics → Actual vs Predicted Analysis Tab → "Generate Complete Test Dataset CSV"**

## 📋 Generated CSV Structure

### Basic Information Columns
- **`Record_ID`**: Sequential number for each test record (1, 2, 3, ...)
- **`Dataset_Split`**: Always "Test_Set_20_Percent" 
- **`Actual_Label`**: Original label from dataset (0 = Normal, 1 = Fault)
- **`Actual_Label_Text`**: Human-readable actual label ("Normal" or "Fault")
- **`Predicted_Label`**: Model prediction (0 = Normal, 1 = Fault)
- **`Predicted_Label_Text`**: Human-readable prediction ("Normal" or "Fault")

### Prediction Details
- **`Prediction_Probability`**: Model confidence score (0.0000 to 1.0000)
- **`Prediction_Correct`**: TRUE if prediction matches actual, FALSE otherwise
- **`Model_Threshold`**: Decision threshold used (default: 0.5)
- **`Above_Threshold`**: "Yes" if probability ≥ 0.5, "No" otherwise
- **`Confidence_Level`**: Categorical confidence (Very_High, High, Medium, Low)

### Classification Analysis
- **`Classification_Type`**: 
  - `True_Positive`: Correctly predicted fault
  - `True_Negative`: Correctly predicted normal
  - `False_Positive`: False alarm (predicted fault, actually normal)
  - `False_Negative`: Missed fault (predicted normal, actually fault)
- **`Classification_Result`**: Descriptive result text

### Feature Values
- **`Feature_[feature_name]`**: All feature values used by the model (scaled/preprocessed)
- Example columns: `Feature_total_door_reversals`, `Feature_safety_chain_issues`, etc.

## 🚀 How to Use

### Step 1: Navigate to the Feature
1. Open the Elevator Fault Detection Dashboard
2. Go to **Dashboard** menu
3. Click **"Show Model Metrics"** button
4. Select **"Actual vs Predicted Analysis"** tab
5. Scroll down to **"Complete Test Dataset Export"** section

### Step 2: Generate CSV
1. Click **"Generate Complete Test Dataset CSV"** button
2. Wait for processing (usually a few seconds)
3. Review the summary statistics displayed
4. Check the dataset preview table

### Step 3: Download and Use
1. Click **"Download Complete Test Dataset CSV"** button
2. Save the file to your desired location
3. Open in Excel, Python, R, or any data analysis tool

## 📊 What You Get

### File Information
- **File Name**: `complete_test_dataset_YYYYMMDD_HHMMSS.csv`
- **Location**: `Reports/` folder in your project
- **Size**: Typically 1-10 MB depending on number of features and test records
- **Format**: Standard CSV with headers

### Summary Statistics Displayed
- **Total Records**: Number of test samples (20% of original dataset)
- **Actual Faults**: Count and percentage of fault cases
- **Model Performance**: Accuracy, Precision, Recall
- **93% Recall Target**: Achievement status
- **Classification Breakdown**: TP, TN, FP, FN counts

### Sample Record Example
```csv
Record_ID,Dataset_Split,Actual_Label,Actual_Label_Text,Predicted_Label,Predicted_Label_Text,Prediction_Probability,Prediction_Correct,Classification_Type,Feature_total_door_reversals,Feature_safety_chain_issues,...
1,Test_Set_20_Percent,0,Normal,0,Normal,0.1234,TRUE,True_Negative,0.0892,0.0341,...
2,Test_Set_20_Percent,1,Fault,1,Fault,0.8756,TRUE,True_Positive,0.7234,0.4451,...
3,Test_Set_20_Percent,1,Fault,0,Normal,0.3421,FALSE,False_Negative,0.2341,0.3564,...
```

## 🎯 Use Cases

### 1. Model Validation and Auditing
- **Purpose**: Verify every prediction made by the model
- **Users**: Data scientists, quality assurance teams
- **Benefits**: Complete transparency in model decisions

### 2. External Analysis
- **Purpose**: Use external tools (Excel, Tableau, R, Python) for analysis
- **Users**: Business analysts, researchers
- **Benefits**: Full dataset portability

### 3. Regulatory Compliance
- **Purpose**: Document model performance for audits
- **Users**: Compliance teams, auditors
- **Benefits**: Complete prediction trail

### 4. Error Investigation
- **Purpose**: Deep dive into false positives and false negatives
- **Users**: Domain experts, maintenance teams
- **Benefits**: Understand prediction failures

### 5. Threshold Optimization
- **Purpose**: Analyze how different thresholds affect results
- **Users**: Model tuning teams
- **Benefits**: Data-driven threshold selection

### 6. Research and Documentation
- **Purpose**: Academic research or technical documentation
- **Users**: Researchers, technical writers
- **Benefits**: Complete dataset for analysis

## 🔍 Analysis Examples

### Find All Missed Faults (False Negatives)
```python
import pandas as pd
df = pd.read_csv('complete_test_dataset_20240901_123456.csv')

# Find all missed faults
missed_faults = df[df['Classification_Type'] == 'False_Negative']
print(f"Total missed faults: {len(missed_faults)}")

# Analyze their characteristics
low_prob_misses = missed_faults[missed_faults['Prediction_Probability'] < 0.3]
print(f"Low probability misses: {len(low_prob_misses)}")
```

### Calculate Custom Metrics
```python
# Calculate metrics at different thresholds
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    custom_predictions = (df['Prediction_Probability'] >= threshold).astype(int)
    custom_recall = ((df['Actual_Label'] == 1) & (custom_predictions == 1)).sum() / (df['Actual_Label'] == 1).sum()
    print(f"Recall at {threshold} threshold: {custom_recall:.3f}")
```

### Feature Analysis
```python
# Find features that differ most between correct and incorrect predictions
feature_cols = [col for col in df.columns if col.startswith('Feature_')]
correct_preds = df[df['Prediction_Correct'] == True][feature_cols].mean()
incorrect_preds = df[df['Prediction_Correct'] == False][feature_cols].mean()
feature_diff = abs(correct_preds - incorrect_preds).sort_values(ascending=False)
print("Features with biggest differences:")
print(feature_diff.head())
```

## ⚠️ Important Notes

### Data Consistency
- **Same Records**: This CSV contains the exact same test records used for model evaluation
- **Same Preprocessing**: Feature values are the scaled/preprocessed values fed to the model
- **Same Predictions**: Prediction probabilities and labels are identical to model output

### File Management
- **Timestamps**: Each file has a unique timestamp to prevent overwrites
- **Location**: Files are saved in the `Reports/` folder
- **Cleanup**: Consider periodically cleaning old CSV files

### Performance Considerations
- **Large Datasets**: For very large test sets, file generation may take longer
- **Memory Usage**: Large files may require more memory to process in Excel/other tools
- **File Size**: Files with many features can become quite large

### 93% Recall Target
- **Automatic Check**: Each CSV generation checks if the 93% recall target is achieved
- **Gap Analysis**: If not achieved, shows exactly how much improvement is needed
- **Business Context**: Helps prioritize safety (recall) vs cost (precision) decisions

## 🔧 Troubleshooting

### Common Issues
1. **"Error generating CSV"**: Check that model is trained and test data is available
2. **Empty preview**: Ensure model predictions were successful
3. **Download fails**: Check browser download settings and available disk space

### File Format Issues
- **Excel Compatibility**: CSV is compatible with Excel, but very large files may load slowly
- **Character Encoding**: Files use UTF-8 encoding
- **Decimal Precision**: Numbers are rounded to 4 decimal places for readability

### Performance Issues
- **Large Files**: Consider using tools like Python/R for very large CSV files
- **Memory Limits**: Excel has row limits (~1M rows) for very large datasets

## 📈 Next Steps After Download

### Immediate Analysis
1. **Open in Excel**: Quick overview and basic filtering
2. **Check Error Cases**: Filter for False_Negative and False_Positive records
3. **Verify 93% Target**: Confirm recall meets your requirements

### Advanced Analysis
1. **Python/R Analysis**: Deep statistical analysis and custom metrics
2. **Visualization**: Create charts and plots for presentations
3. **Feature Engineering**: Identify patterns for model improvement

### Business Actions
1. **Review False Negatives**: High priority - safety critical
2. **Analyze False Positives**: Cost optimization opportunities
3. **Threshold Tuning**: Adjust based on business requirements
4. **Documentation**: Use for compliance and audit documentation

This complete test dataset CSV provides the foundation for comprehensive model analysis, ensuring you have all the data needed to validate, improve, and document your elevator fault detection system's performance.
