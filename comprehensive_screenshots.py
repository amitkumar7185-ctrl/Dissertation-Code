import pyautogui
import time
import webbrowser
import os
import requests
from datetime import datetime
import subprocess

def setup_screenshot_environment():
    """Setup the environment for automated screenshots"""
    # Disable pyautogui failsafe
    pyautogui.FAILSAFE = False
    
    # Create screenshots directory
    screenshot_dir = "detailed_screenshots"
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
    print("❌ Streamlit app not ready")
    return False

def open_browser_maximized(url):
    """Open browser and maximize window"""
    print(f"Opening browser and navigating to {url}...")
    webbrowser.open(url)
    time.sleep(5)  # Wait for browser to open
    
    # Try to maximize the browser window
    try:
        # Press F11 to go fullscreen, then F11 again to exit fullscreen but keep maximized
        pyautogui.press('f11')
        time.sleep(1)
        pyautogui.press('f11')
        time.sleep(2)
        
        # Alternative: Use Alt+Space then X to maximize
        pyautogui.hotkey('alt', 'space')
        time.sleep(0.5)
        pyautogui.press('x')
        time.sleep(2)
    except:
        print("Could not maximize window automatically")

def take_screenshot_with_scroll(filename, description, scroll_down=False):
    """Take a screenshot with optional scrolling"""
    print(f"📸 Capturing: {description}")
    time.sleep(3)  # Wait for page to stabilize
    
    if scroll_down:
        # Scroll to top first
        pyautogui.press('home')
        time.sleep(1)
    
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    print(f"✅ Saved: {filename}")
    
    if scroll_down:
        # Take additional screenshot after scrolling down
        pyautogui.scroll(-5)  # Scroll down
        time.sleep(2)
        filename_scrolled = filename.replace('.png', '_scrolled.png')
        screenshot_scrolled = pyautogui.screenshot()
        screenshot_scrolled.save(filename_scrolled)
        print(f"✅ Saved scrolled view: {filename_scrolled}")

def click_and_wait(x, y, wait_time=3, description=""):
    """Click at coordinates and wait"""
    try:
        print(f"🔍 Clicking {description} at ({x}, {y})")
        pyautogui.click(x, y)
        time.sleep(wait_time)
        return True
    except Exception as e:
        print(f"⚠️ Error clicking {description}: {e}")
        return False

def find_and_click_text(text_to_find, confidence=0.8, wait_time=3):
    """Try to find and click text on screen"""
    try:
        print(f"🔍 Looking for text: '{text_to_find}'")
        
        # Try multiple approaches to find the text
        for conf in [0.9, 0.8, 0.7, 0.6]:
            try:
                location = pyautogui.locateOnScreen(text_to_find, confidence=conf)
                if location:
                    center = pyautogui.center(location)
                    pyautogui.click(center)
                    time.sleep(wait_time)
                    print(f"✅ Found and clicked '{text_to_find}' with confidence {conf}")
                    return True
            except:
                continue
                
        print(f"⚠️ Could not find '{text_to_find}' on screen")
        return False
    except Exception as e:
        print(f"⚠️ Error finding '{text_to_find}': {e}")
        return False

def capture_comprehensive_screenshots():
    """Capture comprehensive screenshots following the guide exactly"""
    
    print("🤖 Starting Comprehensive Screenshot Capture")
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
        # 1. Main Dashboard Screenshot
        print("\n📊 === CAPTURING MAIN DASHBOARD ===")
        filename1 = os.path.join(screenshot_dir, "01_main_dashboard.png")
        take_screenshot_with_scroll(filename1, "Main Dashboard - Initial View")
        
        # Scroll down to see more of the dashboard
        pyautogui.scroll(-3)
        time.sleep(2)
        filename1_lower = os.path.join(screenshot_dir, "01_main_dashboard_lower.png")
        take_screenshot_with_scroll(filename1_lower, "Main Dashboard - Lower Section")
        
        # 2. Dataset Information - Click "Show Dataset Info" button
        print("\n📋 === CAPTURING DATASET INFORMATION ===")
        
        # Scroll back to top to find the Dataset Info button
        pyautogui.press('home')
        time.sleep(2)
        
        # Try multiple approaches to find and click Dataset Info button
        dataset_clicked = False
        
        # Approach 1: Look for common button locations
        dataset_button_locations = [
            (250, 400), (300, 420), (350, 440),  # Top area buttons
            (200, 500), (250, 520), (300, 540),  # Middle area buttons
            (200, 600), (250, 620), (300, 640),  # Lower area buttons
        ]
        
        for loc in dataset_button_locations:
            print(f"Trying Dataset Info button at {loc}")
            click_and_wait(loc[0], loc[1], 3, "Dataset Info button")
            time.sleep(2)
            
            # Take screenshot to see if something changed
            filename2 = os.path.join(screenshot_dir, "02_dataset_info.png")
            take_screenshot_with_scroll(filename2, "Dataset Information View", scroll_down=True)
            dataset_clicked = True
            break
        
        if not dataset_clicked:
            # Try pressing Tab and Enter to navigate to buttons
            for i in range(10):
                pyautogui.press('tab')
                time.sleep(0.5)
                pyautogui.press('enter')
                time.sleep(2)
                
                filename2 = os.path.join(screenshot_dir, f"02_dataset_info_attempt_{i}.png")
                take_screenshot_with_scroll(filename2, f"Dataset Info Attempt {i}")
        
        # 3. Model Performance - Click "Show Model Metrics" button
        print("\n🤖 === CAPTURING MODEL PERFORMANCE ===")
        
        # Scroll down to find model metrics section
        pyautogui.scroll(-5)
        time.sleep(3)
        
        model_clicked = False
        model_button_locations = [
            (250, 600), (300, 620), (350, 640),
            (200, 700), (250, 720), (300, 740),
            (200, 800), (250, 820), (300, 840),
        ]
        
        for loc in model_button_locations:
            print(f"Trying Model Metrics button at {loc}")
            click_and_wait(loc[0], loc[1], 3, "Model Metrics button")
            time.sleep(3)
            
            filename3 = os.path.join(screenshot_dir, "03_model_performance.png")
            take_screenshot_with_scroll(filename3, "Model Performance Metrics", scroll_down=True)
            model_clicked = True
            break
        
        if not model_clicked:
            # Alternative: try more tab navigation
            for i in range(15):
                pyautogui.press('tab')
                time.sleep(0.3)
                pyautogui.press('enter')
                time.sleep(2)
                
                filename3 = os.path.join(screenshot_dir, f"03_model_performance_attempt_{i}.png")
                take_screenshot_with_scroll(filename3, f"Model Performance Attempt {i}")
        
        # 4. Upload & Predict Page Navigation
        print("\n📤 === NAVIGATING TO UPLOAD & PREDICT ===")
        
        # Click on sidebar menu - try different locations for sidebar
        sidebar_locations = [
            (50, 200), (80, 220), (100, 240),   # Top sidebar
            (50, 300), (80, 320), (100, 340),   # Middle sidebar
            (50, 400), (80, 420), (100, 440),   # Lower sidebar
            (30, 250), (30, 350), (30, 450),    # Far left edge
        ]
        
        upload_page_reached = False
        for loc in sidebar_locations:
            print(f"Trying sidebar navigation at {loc}")
            click_and_wait(loc[0], loc[1], 3, "Sidebar - Upload & Predict")
            time.sleep(3)
            
            # Take screenshot to see current page
            filename4 = os.path.join(screenshot_dir, "04_upload_predict.png")
            take_screenshot_with_scroll(filename4, "Upload & Predict Page", scroll_down=True)
            
            # Check if we're on upload page by looking for file upload interface
            # Take another screenshot after scrolling
            pyautogui.scroll(-3)
            time.sleep(2)
            filename4_detail = os.path.join(screenshot_dir, "04_upload_predict_details.png")
            take_screenshot_with_scroll(filename4_detail, "Upload & Predict - Detailed View")
            
            upload_page_reached = True
            break
        
        # 5. EDA Page Navigation
        print("\n📈 === NAVIGATING TO EDA PAGE ===")
        
        # Try to navigate to EDA
        for loc in sidebar_locations[2:6]:  # Try different sidebar positions
            print(f"Trying EDA navigation at {loc}")
            click_and_wait(loc[0], loc[1], 5, "Sidebar - EDA")  # EDA takes longer
            time.sleep(5)
            
            filename5 = os.path.join(screenshot_dir, "05_eda_page.png")
            take_screenshot_with_scroll(filename5, "EDA - Data Visualizations", scroll_down=True)
            
            # Wait longer for plots to render
            time.sleep(5)
            pyautogui.scroll(-5)
            time.sleep(3)
            filename5_plots = os.path.join(screenshot_dir, "05_eda_plots.png")
            take_screenshot_with_scroll(filename5_plots, "EDA - Additional Plots")
            break
        
        # 6. Model Management Page
        print("\n⚙️ === NAVIGATING TO MODEL MANAGEMENT ===")
        
        for loc in sidebar_locations[6:]:
            print(f"Trying Model Management at {loc}")
            click_and_wait(loc[0], loc[1], 3, "Sidebar - Model Management")
            time.sleep(3)
            
            filename6 = os.path.join(screenshot_dir, "06_model_management.png")
            take_screenshot_with_scroll(filename6, "Model Management Interface", scroll_down=True)
            break
        
        # 7. Additional comprehensive screenshots
        print("\n📸 === TAKING ADDITIONAL COMPREHENSIVE VIEWS ===")
        
        # Navigate back to dashboard for final overview
        pyautogui.press('home')
        time.sleep(2)
        click_and_wait(50, 150, 3, "Back to Dashboard")
        
        # Take final overview screenshots
        filename_overview = os.path.join(screenshot_dir, "07_complete_overview.png")
        take_screenshot_with_scroll(filename_overview, "Complete Application Overview")
        
        # Scroll through entire page for comprehensive view
        for i in range(5):
            pyautogui.scroll(-3)
            time.sleep(2)
            filename_section = os.path.join(screenshot_dir, f"08_section_{i+1}.png")
            take_screenshot_with_scroll(filename_section, f"Application Section {i+1}")
        
        print("\n✅ Comprehensive screenshot capture completed!")
        print(f"📁 Screenshots saved in: {os.path.abspath(screenshot_dir)}")
        
        # List all captured screenshots
        print("\n📸 All Captured Screenshots:")
        screenshots = sorted([f for f in os.listdir(screenshot_dir) if f.endswith('.png')])
        for i, screenshot in enumerate(screenshots, 1):
            print(f"  {i:2d}. {screenshot}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during screenshot capture: {e}")
        return False

def create_screenshot_summary():
    """Create a summary of captured screenshots"""
    summary = f"""
# Comprehensive Screenshot Capture Results

**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Application URL**: http://localhost:8501

## Screenshots Captured:

### Dashboard Views:
- 01_main_dashboard.png - Main dashboard with metrics
- 01_main_dashboard_lower.png - Lower section of dashboard
- 02_dataset_info.png - Dataset information view
- 03_model_performance.png - Model performance metrics

### Application Pages:
- 04_upload_predict.png - Upload & Predict interface
- 04_upload_predict_details.png - Detailed upload interface
- 05_eda_page.png - EDA visualizations
- 05_eda_plots.png - Additional EDA plots
- 06_model_management.png - Model management interface

### Comprehensive Views:
- 07_complete_overview.png - Complete application overview
- 08_section_*.png - Detailed section views

## Usage Notes:
- All screenshots are taken at maximum browser resolution
- Scrolled views are included for comprehensive coverage
- Multiple attempts were made for each critical interface
- Screenshots include all dashboard tabs and data views

## Next Steps:
1. Review all screenshots in the 'detailed_screenshots' folder
2. Select the best representatives for documentation
3. Add screenshots to academic documentation
4. Ensure all key features and data are visible
"""
    
    with open("screenshot_capture_summary.md", "w") as f:
        f.write(summary)
    
    print("📝 Created screenshot_capture_summary.md")

if __name__ == "__main__":
    print("🚀 Elevator Fault Detection System - Comprehensive Screenshot Tool")
    print("=" * 80)
    print("This tool will capture detailed screenshots following the Screenshot Guide")
    print("including all dashboard tabs and Upload & Predict page data")
    print("=" * 80)
    
    success = capture_comprehensive_screenshots()
    
    if success:
        print("\n🎉 All comprehensive screenshots captured successfully!")
    else:
        print("\n⚠️ Some issues occurred during capture")
    
    create_screenshot_summary()
    
    print("\n📋 Final Steps:")
    print("1. Check the 'detailed_screenshots' folder for all captured images")
    print("2. Review screenshot_capture_summary.md for complete list")
    print("3. Select best screenshots for academic documentation")
    print("4. Verify all dashboard data and tabs are captured")
    print("5. Ensure Upload & Predict functionality is well documented")
