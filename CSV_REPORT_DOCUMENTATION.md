# CSV Prediction Report Documentation

## Overview
This document describes the comprehensive CSV prediction report feature that generates detailed analysis of actual vs predicted labels with reasoning for each prediction.

## Feature Location
**Dashboard → Model Performance → Show Model Metrics → Actual vs Predicted Analysis Tab → Generate Comprehensive CSV Report**

## Generated Files

### 1. Main Report File
**Filename:** `detailed_prediction_analysis_YYYYMMDD_HHMMSS.csv`

**Contains:** Detailed analysis for every test sample with the following columns:

#### Basic Information
- `Sample_ID`: Unique identifier for each test sample
- `Actual_Label`: Ground truth (0 = No Fault, 1 = Fault)
- `Actual_Label_Description`: Human-readable actual label
- `Predicted_Label`: Model prediction (0 = No Fault, 1 = Fault)
- `Predicted_Label_Description`: Human-readable predicted label

#### Prediction Details
- `Prediction_Probability`: Model confidence score (0.0 to 1.0)
- `Confidence_Level`: Categorized confidence (Very Low, Low, Medium, High, Very High)
- `Model_Decision_Threshold`: Threshold used for prediction (default: 0.5)
- `Probability_Above_Threshold`: Whether probability exceeded threshold

#### Classification Analysis
- `Prediction_Category`: TP, TN, FP, or FN classification
- `Correctness`: ✓ for correct, ✗ for incorrect predictions
- `Result_Description`: Detailed result explanation

#### Risk Assessment
- `Risk_Level`: Critical, Medium, Low, Very Low
- `Risk_Description`: Business impact explanation
- `Recommended_Action`: Specific action recommendations

#### Detailed Reasoning
- `Detailed_Reasoning`: Comprehensive explanation of prediction
  - Model confidence analysis
  - Key contributing features
  - Decision boundary reasoning
  - Pattern recognition insights

#### Feature Analysis
- `Feature_[feature_name]`: Raw feature values for each sample
- `Top_1_Feature_Name/Value/Importance`: Most important feature details
- `Top_2_Feature_Name/Value/Importance`: Second most important feature
- `Top_3_Feature_Name/Value/Importance`: Third most important feature
- `Top_4_Feature_Name/Value/Importance`: Fourth most important feature
- `Top_5_Feature_Name/Value/Importance`: Fifth most important feature

### 2. Summary Statistics File
**Filename:** `prediction_summary_YYYYMMDD_HHMMSS.csv`

**Contains:** Overall performance metrics and insights:

#### Performance Metrics
- `Precision`, `Recall`, `Specificity`, `Accuracy`, `F1_Score`
- `False_Alarm_Rate`, `Miss_Rate`
- `Recall_Target_93_Percent`: Achievement status vs 93% target

#### Risk Analysis
- `Critical_Risk_Cases`: Count of false negatives
- `Medium_Risk_Cases`: Count of false positives
- `High_Confidence_Predictions`: High-certainty predictions
- `Low_Confidence_Predictions`: Low-certainty predictions

#### Sample Counts
- `Total_Samples`, `True_Positives`, `True_Negatives`, `False_Positives`, `False_Negatives`

## Key Features

### 1. Intelligent Reasoning Engine
The system provides human-readable explanations for each prediction based on:

#### Confidence Analysis
- **Very High (≥0.8)**: Strong model certainty
- **High (0.65-0.8)**: Good model confidence
- **Medium (0.5-0.65)**: Moderate confidence
- **Low (0.35-0.5)**: Limited confidence
- **Very Low (<0.35)**: Uncertain prediction

#### Feature Contribution Analysis
- Analyzes top 5 most important features for each prediction
- Provides domain-specific interpretations for elevator fault indicators:
  - **Door Reversals**: Frequency of door direction changes
  - **Safety Chain Issues**: Safety system problems
  - **Hoistway Faults**: Mechanical shaft issues
  - **Leveling Errors**: Floor alignment problems
  - **Startup Delays**: System initialization issues

#### Decision Pattern Recognition
- **Fault Predictions**: Identifies strong vs moderate fault patterns
- **Normal Predictions**: Confirms normal operation patterns
- **Borderline Cases**: Flags uncertain classifications

### 2. Risk Assessment Framework

#### Critical Risk (False Negatives)
- **Impact**: Undetected faults could lead to elevator failures
- **Action**: URGENT investigation and maintenance review
- **Business Cost**: High safety risk, potential accidents

#### Medium Risk (False Positives)
- **Impact**: Unnecessary maintenance alerts
- **Action**: Verify alarm, consider threshold adjustment
- **Business Cost**: Operational inefficiency, maintenance costs

#### Low Risk (Correct Classifications)
- **Impact**: Proper fault detection or normal operation
- **Action**: Continue monitoring or schedule maintenance
- **Business Cost**: Minimal, appropriate resource allocation

### 3. Actionable Insights

#### For Each Prediction
- **Specific Recommendations**: Tailored actions based on prediction type
- **Maintenance Scheduling**: Guidance for fault cases
- **Threshold Optimization**: Suggestions for borderline cases
- **Feature Monitoring**: Key indicators to watch

#### For 93% Recall Target
- **Achievement Status**: Clear indication if target is met
- **Gap Analysis**: Specific improvements needed
- **Trade-off Analysis**: Impact on precision and false alarms

## Business Use Cases

### 1. Model Validation and Auditing
- **Purpose**: Understand and validate model decisions
- **Users**: Data scientists, model validators
- **Benefits**: Transparent AI decision-making

### 2. Maintenance Planning
- **Purpose**: Prioritize maintenance based on risk levels
- **Users**: Maintenance managers, facility operators
- **Benefits**: Optimized resource allocation

### 3. False Positive Analysis
- **Purpose**: Reduce unnecessary maintenance alerts
- **Users**: Operations teams, cost managers
- **Benefits**: Reduced operational costs

### 4. Safety Compliance
- **Purpose**: Ensure critical faults are not missed
- **Users**: Safety officers, compliance teams
- **Benefits**: Enhanced safety records

### 5. Model Improvement
- **Purpose**: Identify patterns in model errors
- **Users**: Data scientists, ML engineers
- **Benefits**: Systematic model enhancement

## Technical Implementation

### Data Processing Pipeline
1. **Feature Extraction**: Retrieve test data and model predictions
2. **Importance Analysis**: Calculate feature contributions
3. **Reasoning Generation**: Create explanations using domain knowledge
4. **Risk Assessment**: Categorize and prioritize cases
5. **Report Generation**: Format and save comprehensive results

### Quality Assurance
- **Error Handling**: Graceful fallback to simple report if issues occur
- **Data Validation**: Ensures consistency across all generated fields
- **Performance Optimization**: Efficient processing for large datasets

## Sample Report Entries

### Example 1: True Positive (Correct Fault Detection)
```
Sample_ID: 156
Actual_Label: 1 (Fault Present)
Predicted_Label: 1 (Fault Predicted)
Prediction_Probability: 0.8756
Confidence_Level: Very High
Risk_Level: Low - Correct Detection
Detailed_Reasoning: Model confidence: Very High (0.876) | Key factors: door reversal frequency: High (0.723) - significant concern; safety chain problems: Moderate (0.445) - some concern | Strong fault pattern detected
Recommended_Action: Schedule maintenance as recommended by model
```

### Example 2: False Negative (Missed Fault - Critical)
```
Sample_ID: 89
Actual_Label: 1 (Fault Present)  
Predicted_Label: 0 (Normal Predicted)
Prediction_Probability: 0.3421
Confidence_Level: Low
Risk_Level: Critical
Detailed_Reasoning: Model confidence: Low (0.342) | Key factors: door reversal frequency: Low (0.234) - minimal concern | Borderline case, leaning towards normal
Recommended_Action: URGENT: Investigate for potential fault, review maintenance schedule
```

### Example 3: False Positive (False Alarm)
```
Sample_ID: 203
Actual_Label: 0 (Normal Operation)
Predicted_Label: 1 (Fault Predicted)  
Prediction_Probability: 0.6789
Confidence_Level: High
Risk_Level: Medium - False Alarm
Detailed_Reasoning: Model confidence: High (0.679) | Key factors: startup delays: High (0.612) - potential fault indicator | Moderate fault risk identified  
Recommended_Action: Verify alarm, consider threshold adjustment if pattern persists
```

## Usage Instructions

### Step 1: Access the Feature
1. Navigate to Dashboard in the application
2. Click "Show Model Metrics" 
3. Go to "Actual vs Predicted Analysis" tab
4. Click "Generate Comprehensive CSV Report"

### Step 2: Review Summary
- Check overall performance metrics
- Verify 93% recall target achievement
- Review critical and medium risk case counts

### Step 3: Analyze the Report
- Download the detailed CSV file
- Filter by risk level for prioritization
- Review false negatives for safety concerns
- Analyze false positives for cost optimization

### Step 4: Take Action
- Address critical risk cases immediately
- Use recommendations for maintenance planning
- Consider threshold adjustments for persistent patterns
- Monitor key features identified in the analysis

## Best Practices

### For Operations Teams
- **Daily Review**: Check critical risk cases daily
- **Trend Analysis**: Monitor false positive/negative rates over time
- **Threshold Tuning**: Adjust based on operational experience

### For Management
- **KPI Tracking**: Use recall and precision as key performance indicators
- **Cost-Benefit Analysis**: Balance safety vs operational costs
- **Resource Planning**: Allocate maintenance resources based on risk levels

### For Data Scientists
- **Model Monitoring**: Track performance degradation over time
- **Feature Engineering**: Use insights for improving model features
- **Hyperparameter Tuning**: Optimize based on business requirements

## Troubleshooting

### Common Issues
1. **No Data Generated**: Ensure model is trained and test data available
2. **Import Errors**: Check if prediction_report_generator.py is accessible
3. **Performance Issues**: Consider sampling for very large datasets

### Error Recovery
- Automatic fallback to simple CSV if comprehensive generation fails
- Clear error messages with suggested solutions
- Graceful handling of missing or corrupted data

## Future Enhancements

### Planned Features
- **Time-series Analysis**: Track prediction patterns over time
- **Cost Modeling**: Include actual maintenance costs
- **Interactive Filtering**: Web-based report filtering and analysis
- **Automated Alerts**: Real-time notifications for critical cases

### Integration Opportunities
- **CMMS Integration**: Direct connection to maintenance management systems
- **BI Tools**: Export to PowerBI, Tableau for advanced visualization
- **API Endpoints**: Programmatic access to report generation
- **Scheduled Reports**: Automated daily/weekly report generation

---

This comprehensive CSV report feature transforms raw model predictions into actionable business intelligence, supporting both operational efficiency and safety compliance in elevator fault detection systems.
