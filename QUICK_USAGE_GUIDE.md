# Quick Guide: Actual vs Predicted Analysis Tab

## How to Access
1. Open the Elevator Fault Detection Dashboard
2. Go to **Dashboard** menu
3. Click **"Show Model Metrics"** button
4. Select the **"🔍 Actual vs Predicted Analysis"** tab

## What You'll See

### 📊 Performance Metrics (Top Section)
```
Precision | Recall | Specificity | F1-Score
   0.856  |  0.932 |    0.891    |   0.892
```
- **Green**: Excellent (≥93%)
- **Blue**: Good (≥80%) 
- **Yellow**: Needs Improvement (<80%)

### 🎯 Confusion Matrix
```
                   Predicted
               No Fault | Fault
Actual No Fault:  TN   |  FP    
Actual Fault:     FN   |  TP    
```

### Key Numbers to Watch
- **False Negatives (FN)**: Missed faults - **Critical for Safety**
- **False Positives (FP)**: False alarms - **Cost Impact**
- **True Positives (TP)**: Correctly caught faults - **Success Rate**

## For 93% Recall Target

### ✅ If Target is Achieved
```
🎉 Target Achieved! Current Recall: 93.2%
• Out of 100 actual faults, 93 were correctly identified
• Only 7 faults were missed (7% miss rate)
• The model captures 93.2% of all elevator faults
```

### 📊 If Target is Not Met
```
📊 Target Not Met - Current Recall: 87.5%
• Recall gap: 5.5%
• Additional True Positives needed: ~6
• Current miss rate: 12.5%
```

## Quick Actions

### To Improve Recall (Catch More Faults)
1. **Lower Threshold**: Use the recommended threshold from the analysis
2. **Accept More False Alarms**: Trade-off for better fault detection
3. **Check Recommendations**: Review the optimization suggestions

### To Reduce False Alarms
1. **Raise Threshold**: Increase prediction confidence requirement
2. **Focus on Precision**: Accept missing some faults for fewer false alarms
3. **Balance Strategy**: Use F1-score optimization recommendations

## Business Impact Translation

### High Recall (93%+) = Safety Priority
- **Benefit**: Catches almost all faults before failure
- **Cost**: More maintenance alerts (some unnecessary)
- **Best For**: Critical safety applications

### High Precision (90%+) = Cost Efficiency
- **Benefit**: Very few false alarms
- **Cost**: Some faults might be missed
- **Best For**: Cost-sensitive operations

## Red Flags to Watch
- **High False Negatives**: Risk of unexpected failures
- **Very High False Positives**: Operational inefficiency
- **Low F1-Score (<0.8)**: Model needs improvement

## Quick Recommendations

### Immediate Actions
1. **Check Current Recall**: Is it ≥93%?
2. **Review Miss Rate**: How many faults are being missed?
3. **Assess False Alarms**: Are they manageable?
4. **Apply Recommended Threshold**: Use the optimal threshold suggestion

### For Operations Teams
- Monitor the **False Negative** count daily
- Track **Precision** to manage maintenance costs
- Use **Threshold Recommendations** for operational adjustments

### For Management
- **93% Recall = 93% Safety Coverage**
- **Precision = Cost Efficiency Indicator** 
- **F1-Score = Overall Performance Balance**

---
**Remember**: The goal is not perfect metrics, but the right balance for your operational needs!
