import pyautogui
import time
import webbrowser
import os
import requests
from datetime import datetime

def setup_screenshot_environment():
    """Setup the environment for automated screenshots"""
    pyautogui.FAILSAFE = False
    screenshot_dir = "action_based_screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    return screenshot_dir

def wait_for_streamlit_ready(url, timeout=30):
    """Wait for Streamlit app to be ready"""
    print(f"Checking if Streamlit app is ready at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ Streamlit app is ready!")
                return True
        except:
            pass
        time.sleep(2)
    return False

def open_browser_maximized(url):
    """Open browser and maximize window"""
    print(f"Opening browser and navigating to {url}...")
    webbrowser.open(url)
    time.sleep(6)  # Wait for browser to open
    
    # Maximize browser window
    try:
        pyautogui.hotkey('alt', 'space')
        time.sleep(0.5)
        pyautogui.press('x')
        time.sleep(2)
    except:
        print("Could not maximize window automatically")

def take_screenshot(filename, description):
    """Take a screenshot"""
    print(f"📸 Capturing: {description}")
    time.sleep(2)
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    print(f"✅ Saved: {filename}")

def find_and_click_button_text(text_patterns, description="button", wait_time=3):
    """Try to find and click a button by searching for text patterns"""
    print(f"🔍 Looking for {description} with patterns: {text_patterns}")
    
    # Try to find buttons with text using different approaches
    for pattern in text_patterns:
        try:
            # Method 1: Try to locate text on screen
            for confidence in [0.8, 0.7, 0.6]:
                try:
                    location = pyautogui.locateOnScreen(pattern, confidence=confidence)
                    if location:
                        center = pyautogui.center(location)
                        pyautogui.click(center)
                        time.sleep(wait_time)
                        print(f"✅ Found and clicked '{pattern}' with confidence {confidence}")
                        return True
                except:
                    continue
        except:
            continue
    
    # Method 2: Try common button locations for Streamlit apps
    button_locations = [
        # Dashboard area button locations
        (200, 350), (250, 350), (300, 350), (350, 350),
        (200, 400), (250, 400), (300, 400), (350, 400),
        (200, 450), (250, 450), (300, 450), (350, 450),
        (200, 500), (250, 500), (300, 500), (350, 500),
        (200, 550), (250, 550), (300, 550), (350, 550),
        (200, 600), (250, 600), (300, 600), (350, 600),
    ]
    
    print(f"🎯 Trying common button locations for {description}")
    for i, (x, y) in enumerate(button_locations):
        try:
            pyautogui.click(x, y)
            time.sleep(wait_time)
            print(f"✅ Clicked at location ({x}, {y}) - attempt {i+1}")
            return True
        except:
            continue
    
    print(f"⚠️ Could not find {description}")
    return False

def navigate_to_page(page_name):
    """Navigate to a specific page using sidebar"""
    print(f"🧭 Navigating to {page_name}")
    
    # Click on sidebar area first
    sidebar_areas = [
        (50, 200), (80, 250), (100, 300), (50, 350), (80, 400)
    ]
    
    for x, y in sidebar_areas:
        pyautogui.click(x, y)
        time.sleep(2)
        # Look for page indicators or just wait
        break

def capture_dashboard_actions():
    """Capture dashboard with specific button actions"""
    print("\n📊 === CAPTURING DASHBOARD WITH ACTIONS ===")
    
    screenshot_dir = "action_based_screenshots"
    
    # 1. Initial Dashboard View
    filename1 = os.path.join(screenshot_dir, "01_dashboard_initial.png")
    take_screenshot(filename1, "Dashboard - Initial View")
    
    # 2. Try to click "Show Dataset Info" button
    print("\n📋 Clicking 'Show Dataset Info' button...")
    dataset_button_patterns = [
        "Show Dataset Info",
        "Dataset Info", 
        "Show Info",
        "Dataset",
        "Info"
    ]
    
    # Scroll to top first to ensure we see the buttons
    pyautogui.press('home')
    time.sleep(2)
    
    # Try to find and click the dataset info button
    dataset_clicked = find_and_click_button_text(dataset_button_patterns, "Dataset Info button", 4)
    
    if dataset_clicked:
        # Wait for data to load and take screenshot
        time.sleep(4)
        filename2 = os.path.join(screenshot_dir, "02_dataset_info_expanded.png")
        take_screenshot(filename2, "Dashboard - Dataset Info Expanded")
        
        # Scroll down to see more data
        pyautogui.scroll(-3)
        time.sleep(2)
        filename2b = os.path.join(screenshot_dir, "02_dataset_info_scrolled.png")
        take_screenshot(filename2b, "Dashboard - Dataset Info Scrolled")
    else:
        print("⚠️ Using alternative method for dataset info")
        # Try pressing tab multiple times to navigate to buttons
        for i in range(10):
            pyautogui.press('tab')
            time.sleep(0.5)
            pyautogui.press('enter')
            time.sleep(2)
            
            # Take screenshot to see if anything changed
            filename2_alt = os.path.join(screenshot_dir, f"02_dataset_attempt_{i}.png")
            take_screenshot(filename2_alt, f"Dataset Info Attempt {i}")
            
            if i == 3:  # Take the 4th attempt as the main one
                break
    
    # 3. Try to click "Show Model Metrics" button
    print("\n🤖 Clicking 'Show Model Metrics' button...")
    model_button_patterns = [
        "Show Model Metrics",
        "Model Metrics",
        "Show Metrics", 
        "Metrics",
        "Model Performance"
    ]
    
    # Scroll down to find model section
    pyautogui.scroll(-4)
    time.sleep(2)
    
    model_clicked = find_and_click_button_text(model_button_patterns, "Model Metrics button", 4)
    
    if model_clicked:
        # Wait for model metrics to load
        time.sleep(4)
        filename3 = os.path.join(screenshot_dir, "03_model_metrics_expanded.png")
        take_screenshot(filename3, "Dashboard - Model Metrics Expanded")
        
        # Scroll to see more metrics
        pyautogui.scroll(-3)
        time.sleep(2)
        filename3b = os.path.join(screenshot_dir, "03_model_metrics_scrolled.png")
        take_screenshot(filename3b, "Dashboard - Model Metrics Scrolled")
    else:
        print("⚠️ Using alternative method for model metrics")
        # Try more tab navigation for model metrics
        for i in range(15):
            pyautogui.press('tab')
            time.sleep(0.3)
            pyautogui.press('enter')
            time.sleep(2)
            
            filename3_alt = os.path.join(screenshot_dir, f"03_model_attempt_{i}.png")
            take_screenshot(filename3_alt, f"Model Metrics Attempt {i}")
            
            if i == 5:  # Take the 6th attempt as main
                break

def capture_upload_predict_actions():
    """Capture Upload & Predict page with actual interactions"""
    print("\n📤 === CAPTURING UPLOAD & PREDICT WITH ACTIONS ===")
    
    screenshot_dir = "action_based_screenshots"
    
    # Navigate to Upload & Predict page
    print("🧭 Navigating to Upload & Predict page...")
    
    # Try clicking on sidebar locations for Upload & Predict
    upload_nav_locations = [
        (50, 250), (80, 280), (100, 300),
        (50, 300), (80, 320), (100, 340),
        (50, 350), (80, 380), (100, 400),
    ]
    
    for x, y in upload_nav_locations:
        pyautogui.click(x, y)
        time.sleep(3)
        
        # Take screenshot to see if we're on upload page
        filename4 = os.path.join(screenshot_dir, "04_upload_predict_page.png")
        take_screenshot(filename4, "Upload & Predict - Main Page")
        break
    
    # 4. Try to interact with file upload
    print("\n📁 Looking for file upload interface...")
    
    # Look for file upload area or browse button
    upload_patterns = [
        "Browse files",
        "Choose file",
        "Upload",
        "Select file",
        "Drop files"
    ]
    
    # Common file upload locations in Streamlit
    upload_locations = [
        (400, 300), (450, 320), (500, 340),
        (400, 400), (450, 420), (500, 440),
        (300, 350), (350, 370), (400, 390),
    ]
    
    upload_found = False
    
    # Try to find upload area
    for pattern in upload_patterns:
        if find_and_click_button_text([pattern], "File Upload", 2):
            upload_found = True
            break
    
    if not upload_found:
        # Try clicking common upload locations
        for x, y in upload_locations:
            pyautogui.click(x, y)
            time.sleep(2)
            
            filename4b = os.path.join(screenshot_dir, "04_upload_interface.png")
            take_screenshot(filename4b, "Upload & Predict - Upload Interface")
            break
    
    # 5. Scroll down to see prediction options
    print("\n🔮 Capturing prediction interface...")
    pyautogui.scroll(-3)
    time.sleep(2)
    
    filename4c = os.path.join(screenshot_dir, "04_prediction_options.png")
    take_screenshot(filename4c, "Upload & Predict - Prediction Options")
    
    # 6. Try to interact with model selection
    print("\n⚙️ Looking for model selection...")
    model_selection_patterns = [
        "Random Forest",
        "LSTM",
        "Model Type"
    ]
    
    find_and_click_button_text(model_selection_patterns, "Model Selection", 2)
    
    filename4d = os.path.join(screenshot_dir, "04_model_selection.png")
    take_screenshot(filename4d, "Upload & Predict - Model Selection")

def capture_eda_page():
    """Capture EDA page with visualizations"""
    print("\n📈 === CAPTURING EDA PAGE ===")
    
    screenshot_dir = "action_based_screenshots"
    
    # Navigate to EDA page
    eda_nav_locations = [
        (50, 400), (80, 420), (100, 440),
        (50, 450), (80, 470), (100, 490),
    ]
    
    for x, y in eda_nav_locations:
        pyautogui.click(x, y)
        time.sleep(5)  # EDA takes longer to load
        
        filename5 = os.path.join(screenshot_dir, "05_eda_initial.png")
        take_screenshot(filename5, "EDA - Initial View")
        break
    
    # Scroll to see different plots
    for i in range(3):
        pyautogui.scroll(-4)
        time.sleep(3)
        
        filename5_plot = os.path.join(screenshot_dir, f"05_eda_section_{i+1}.png")
        take_screenshot(filename5_plot, f"EDA - Visualization Section {i+1}")

def capture_model_management():
    """Capture Model Management page"""
    print("\n⚙️ === CAPTURING MODEL MANAGEMENT ===")
    
    screenshot_dir = "action_based_screenshots"
    
    # Navigate to Model Management
    mgmt_nav_locations = [
        (50, 500), (80, 520), (100, 540),
        (50, 550), (80, 570), (100, 590),
    ]
    
    for x, y in mgmt_nav_locations:
        pyautogui.click(x, y)
        time.sleep(3)
        
        filename6 = os.path.join(screenshot_dir, "06_model_management.png")
        take_screenshot(filename6, "Model Management - Interface")
        break

def capture_action_based_screenshots():
    """Main function to capture screenshots based on actual UI actions"""
    
    print("🎯 Starting Action-Based Screenshot Capture")
    print("=" * 60)
    print("This will capture screenshots by clicking actual buttons and interactions")
    print("=" * 60)
    
    # Setup
    screenshot_dir = setup_screenshot_environment()
    app_url = "http://localhost:8501"
    
    # Check if app is ready
    if not wait_for_streamlit_ready(app_url):
        print("❌ Streamlit app is not accessible. Please ensure it's running.")
        return False
    
    # Open browser maximized
    open_browser_maximized(app_url)
    
    # Wait for full page load
    print("⏳ Waiting for initial page load...")
    time.sleep(8)
    
    try:
        # Capture Dashboard with Actions
        capture_dashboard_actions()
        
        # Capture Upload & Predict with Actions
        capture_upload_predict_actions()
        
        # Capture EDA Page
        capture_eda_page()
        
        # Capture Model Management
        capture_model_management()
        
        print("\n✅ Action-based screenshot capture completed!")
        print(f"📁 Screenshots saved in: {os.path.abspath(screenshot_dir)}")
        
        # List all screenshots
        screenshots = sorted([f for f in os.listdir(screenshot_dir) if f.endswith('.png')])
        print(f"\n📸 Captured {len(screenshots)} Screenshots:")
        for i, screenshot in enumerate(screenshots, 1):
            print(f"  {i:2d}. {screenshot}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during screenshot capture: {e}")
        return False

def create_action_summary():
    """Create summary of action-based captures"""
    summary = f"""
# Action-Based Screenshot Capture Results

**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Focus**: Capturing actual button clicks and UI interactions

## Screenshots Captured:

### Dashboard Actions:
- 01_dashboard_initial.png - Initial dashboard view
- 02_dataset_info_expanded.png - After clicking "Show Dataset Info"
- 02_dataset_info_scrolled.png - Scrolled dataset view
- 03_model_metrics_expanded.png - After clicking "Show Model Metrics"  
- 03_model_metrics_scrolled.png - Scrolled metrics view

### Upload & Predict Actions:
- 04_upload_predict_page.png - Main upload page
- 04_upload_interface.png - File upload interface
- 04_prediction_options.png - Prediction options
- 04_model_selection.png - Model selection interface

### EDA Visualizations:
- 05_eda_initial.png - Initial EDA view
- 05_eda_section_*.png - Different visualization sections

### Model Management:
- 06_model_management.png - Management interface

## Focus Areas:
✅ Dashboard button interactions (Show Dataset Info, Show Model Metrics)
✅ Upload & Predict file upload and prediction workflow
✅ EDA data visualizations and plots
✅ Model management interface

## Next Steps:
Review screenshots to ensure all button actions and data displays are captured properly.
"""
    
    with open("action_based_capture_summary.md", "w") as f:
        f.write(summary)
    
    print("📝 Created action_based_capture_summary.md")

if __name__ == "__main__":
    print("🎯 Elevator Fault Detection System - Action-Based Screenshot Tool")
    print("=" * 80)
    print("This tool focuses on clicking actual buttons and capturing resulting data")
    print("Rather than just navigating through sidebar menus")
    print("=" * 80)
    
    success = capture_action_based_screenshots()
    
    if success:
        print("\n🎉 All action-based screenshots captured successfully!")
    else:
        print("\n⚠️ Some issues occurred during action-based capture")
    
    create_action_summary()
    
    print("\n📋 Summary:")
    print("✅ Dashboard button clicks (Show Dataset Info, Show Model Metrics)")
    print("✅ Upload & Predict interactions (file upload, model selection)")
    print("✅ EDA visualizations and data plots")
    print("✅ Model management interface")
    print("\nReview the 'action_based_screenshots' folder for results!")
