# Manual Screenshot Capture Guide
## Elevator Fault Detection System

**Application URL**: http://localhost:8501

---

## 📁 Screenshot Folder Setup

Please create a folder named `screenshots` and save all manual captures there for consistent organization.

---

## 📸 **REQUIRED SCREENSHOTS - STEP BY STEP**

### **1. Main Dashboard (01_main_dashboard.png)**
- Navigate to http://localhost:8501
- Wait for page to fully load
- Capture the main dashboard showing:
  - Total Records metric
  - Fault Cases count  
  - Fault Rate percentage
  - Data overview section

### **2. Dataset Information (02_dataset_info.png)**
- On the dashboard, look for and click **"Show Dataset Info"** button
- Wait for the dataset preview to expand/load
- Capture the screen showing:
  - Expanded dataset table/preview
  - Dataset statistics
  - Any data distribution information

### **3. Model Performance (03_model_performance.png)**
- Look for and click **"Show Model Metrics"** button
- Wait for model performance data to load
- Capture the screen showing:
  - Classification report
  - Confusion matrix
  - Accuracy, Precision, Recall metrics
  - Feature importance (if visible)

### **4. Upload & Predict Page (04_upload_predict.png)**
- Click **"Upload & Predict"** in the sidebar menu
- Wait for page to load completely
- Capture the screen showing:
  - File upload interface
  - Browse/Drop files area
  - Model selection options (Random Forest, LSTM)
  - Instructions for users

### **5. Upload & Predict - File Interface (05_upload_interface.png)**
- On the Upload & Predict page, scroll down if needed
- Capture the detailed view showing:
  - Complete file upload workflow
  - Model type selection radio buttons
  - Any prediction options or settings

### **6. EDA Page (06_eda_page.png)**
- Click **"EDA"** in the sidebar menu
- Wait for all visualizations to load (this may take 5-10 seconds)
- Capture the screen showing:
  - Feature distribution plots
  - Correlation heatmap
  - Pair plots
  - Any statistical summaries

### **7. EDA - Additional Plots (07_eda_plots.png)**
- On the EDA page, scroll down to see more visualizations
- Capture any additional plots or analysis that weren't visible in the first EDA screenshot

### **8. Model Management (08_model_management.png)**
- Click **"Model Management"** in the sidebar menu
- Capture the screen showing:
  - Model information and metadata
  - Retraining options
  - Model performance statistics
  - Administrative controls

---

## 📋 **SCREENSHOT CHECKLIST**

When capturing each screenshot, ensure:

- [ ] Browser window is maximized for best visibility
- [ ] All content is fully loaded before capturing
- [ ] Screenshots are saved as PNG files
- [ ] High resolution (1920x1080 or higher if possible)
- [ ] Clear visibility of all text and data
- [ ] Proper filename following the naming convention

---

## 📂 **FILE NAMING CONVENTION**

Save screenshots with these exact names in the `screenshots/` folder:

```
screenshots/
├── 01_main_dashboard.png
├── 02_dataset_info.png
├── 03_model_performance.png
├── 04_upload_predict.png
├── 05_upload_interface.png
├── 06_eda_page.png
├── 07_eda_plots.png
└── 08_model_management.png
```

---

## 💡 **TIPS FOR BEST RESULTS**

### **Navigation Tips:**
- Use sidebar menu to navigate between pages
- Wait for each page to fully load before taking screenshot
- If buttons don't appear immediately, try refreshing the page

### **Dashboard Interactions:**
- **Show Dataset Info**: Look for this button in the Data Overview section
- **Show Model Metrics**: Usually found in the Model Performance section
- If buttons are not visible, scroll down to find them

### **Upload & Predict Page:**
- Make sure to capture both the file upload area and model selection
- Include any instructions or help text that appears
- Show the complete workflow interface

### **EDA Page:**
- Wait for all plots to render completely
- If plots are large, take multiple screenshots to capture all visualizations
- Include correlation matrices and statistical summaries

### **Quality Checks:**
- Verify all text is readable
- Ensure metrics and numbers are clearly visible
- Check that charts and plots are fully rendered
- Make sure UI elements are not cut off

---

## 🎯 **WHAT TO FOCUS ON IN EACH SCREENSHOT**

### **Dashboard Screenshots Should Show:**
- Key performance metrics (Total Records, Fault Cases, Fault Rate)
- Expanded dataset preview with actual data
- Model performance statistics and metrics

### **Upload & Predict Screenshots Should Show:**
- Complete file upload interface
- Model selection options (Random Forest, LSTM)
- Prediction workflow and options

### **EDA Screenshots Should Show:**
- Data visualization plots and charts
- Correlation heatmaps
- Feature analysis and statistics

### **Model Management Screenshots Should Show:**
- Model information and metadata
- Administrative controls and options

---

## ✅ **AFTER CAPTURING ALL SCREENSHOTS**

Once you have manually captured all screenshots and saved them in the `screenshots/` folder:

1. **Verify all 8 screenshots are present**
2. **Check that all data and metrics are clearly visible**
3. **Ensure filenames match the convention exactly**
4. **Review that each screenshot shows the intended functionality**

The screenshots will then be ready for inclusion in your academic documentation!

---

**Note**: Manual capture ensures you get exactly the right data displays and interactions, which is perfect for academic documentation quality.
