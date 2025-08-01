import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests

def setup_driver():
    """Setup Chrome WebDriver for screenshot capture"""
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        print("Please ensure Chrome and ChromeDriver are installed")
        return None

def wait_for_streamlit_ready(url, timeout=30):
    """Wait for Streamlit app to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False

def capture_screenshots():
    """Capture screenshots of the Streamlit application"""
    
    # Create screenshots directory
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    
    # Setup driver
    driver = setup_driver()
    if not driver:
        print("Failed to setup WebDriver. Manual screenshots required.")
        return
    
    try:
        # Wait for app to be ready
        app_url = "http://localhost:8503"
        print("Waiting for Streamlit app to be ready...")
        
        if not wait_for_streamlit_ready(app_url):
            print("Streamlit app not accessible. Please ensure it's running on port 8503")
            return
        
        print("Streamlit app is ready. Capturing screenshots...")
        
        # Navigate to the application
        driver.get(app_url)
        time.sleep(5)  # Wait for initial load
        
        # Screenshot 1: Main Dashboard
        print("Capturing main dashboard...")
        wait = WebDriverWait(driver, 10)
        driver.save_screenshot(f"{screenshot_dir}/01_main_dashboard.png")
        
        # Screenshot 2: Dashboard with expanded metrics
        print("Capturing dashboard metrics...")
        try:
            # Look for and click dataset info button
            dataset_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Show Dataset Info')]")
            driver.execute_script("arguments[0].click();", dataset_btn)
            time.sleep(3)
            driver.save_screenshot(f"{screenshot_dir}/02_dataset_info.png")
        except:
            print("Dataset info button not found, skipping...")
        
        # Screenshot 3: Model performance
        try:
            model_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Show Model Metrics')]")
            driver.execute_script("arguments[0].click();", model_btn)
            time.sleep(3)
            driver.save_screenshot(f"{screenshot_dir}/03_model_performance.png")
        except:
            print("Model metrics button not found, skipping...")
        
        # Navigate to Upload & Predict page
        print("Navigating to Upload & Predict...")
        try:
            upload_link = driver.find_element(By.XPATH, "//div[contains(text(), 'Upload & Predict')]")
            driver.execute_script("arguments[0].click();", upload_link)
            time.sleep(3)
            driver.save_screenshot(f"{screenshot_dir}/04_upload_predict.png")
        except:
            print("Upload & Predict link not found, skipping...")
        
        # Navigate to EDA page
        print("Navigating to EDA...")
        try:
            eda_link = driver.find_element(By.XPATH, "//div[contains(text(), 'EDA')]")
            driver.execute_script("arguments[0].click();", eda_link)
            time.sleep(5)  # EDA takes longer to load
            driver.save_screenshot(f"{screenshot_dir}/05_eda_page.png")
        except:
            print("EDA link not found, skipping...")
        
        # Navigate to Model Management
        print("Navigating to Model Management...")
        try:
            model_mgmt_link = driver.find_element(By.XPATH, "//div[contains(text(), 'Model Management')]")
            driver.execute_script("arguments[0].click();", model_mgmt_link)
            time.sleep(3)
            driver.save_screenshot(f"{screenshot_dir}/06_model_management.png")
        except:
            print("Model Management link not found, skipping...")
        
        print(f"Screenshots saved in '{screenshot_dir}' directory")
        
    except Exception as e:
        print(f"Error during screenshot capture: {e}")
    
    finally:
        driver.quit()

def create_manual_screenshot_guide():
    """Create a guide for manual screenshot capture"""
    guide = """
# Manual Screenshot Guide for Elevator Fault Detection System

Since automated screenshot capture may not work on all systems, here's a manual guide:

## Required Screenshots:

### 1. Main Dashboard (01_main_dashboard.png)
- Navigate to http://localhost:8503
- Wait for the page to fully load
- Capture the main dashboard showing:
  - Total Records, Fault Cases, Fault Rate metrics
  - Data Overview section
  - Any loaded visualizations

### 2. Dataset Information (02_dataset_info.png)
- Click the "Show Dataset Info" button
- Capture the expanded view showing:
  - Dataset preview table
  - Data distribution information
  - Statistical summaries

### 3. Model Performance (03_model_performance.png)
- Click the "Show Model Metrics" button
- Capture the model performance section showing:
  - Classification report
  - Confusion matrix
  - Performance metrics

### 4. Upload & Predict Page (04_upload_predict.png)
- Click on "Upload & Predict" in the sidebar
- Capture the upload interface showing:
  - File upload area
  - Model type selection
  - Instructions for users

### 5. EDA Page (05_eda_page.png)
- Click on "EDA" in the sidebar
- Wait for visualizations to load
- Capture the page showing:
  - Feature distribution plots
  - Correlation heatmap
  - Pair plots

### 6. Model Management (06_model_management.png)
- Click on "Model Management" in the sidebar
- Capture the page showing:
  - Model information
  - Retraining options
  - Model statistics

## Screenshot Tips:
- Use full browser window (maximize)
- Ensure all content is visible
- Wait for loading to complete
- Use high resolution (1920x1080 or higher)
- Save in PNG format for best quality

## File Naming Convention:
Save screenshots with the exact names listed above for consistency.
"""
    
    with open("Manual_Screenshot_Guide.md", "w") as f:
        f.write(guide)
    
    print("Manual screenshot guide created: Manual_Screenshot_Guide.md")

if __name__ == "__main__":
    print("Elevator Fault Detection System - Screenshot Capture Tool")
    print("=" * 60)
    
    try:
        capture_screenshots()
    except Exception as e:
        print(f"Automated screenshot capture failed: {e}")
        print("Creating manual screenshot guide instead...")
        create_manual_screenshot_guide()
        
        print("\nTo capture screenshots manually:")
        print("1. Ensure the Streamlit app is running on http://localhost:8503")
        print("2. Follow the guide in Manual_Screenshot_Guide.md")
        print("3. Save screenshots in a 'screenshots' folder")
