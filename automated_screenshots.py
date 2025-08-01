import pyautogui
import time
import webbrowser
import os
import requests
from datetime import datetime

def setup_screenshot_environment():
    """Setup the environment for automated screenshots"""
    # Disable pyautogui failsafe
    pyautogui.FAILSAFE = False
    
    # Create screenshots directory
    screenshot_dir = "screenshots"
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

def open_browser_and_navigate(url):
    """Open browser and navigate to the URL"""
    print(f"Opening browser and navigating to {url}...")
    webbrowser.open(url)
    time.sleep(5)  # Wait for browser to open and load

def take_screenshot(filename, description):
    """Take a screenshot with given filename"""
    print(f"📸 Capturing: {description}")
    time.sleep(2)  # Wait for page to stabilize
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    print(f"✅ Saved: {filename}")

def click_element_by_text(text, wait_time=3):
    """Try to click an element by finding text on screen"""
    try:
        print(f"🔍 Looking for: {text}")
        # Try to find the text on screen
        location = pyautogui.locateOnScreen(text, confidence=0.8)
        if location:
            center = pyautogui.center(location)
            pyautogui.click(center)
            time.sleep(wait_time)
            return True
        else:
            print(f"⚠️ Could not find '{text}' on screen")
            return False
    except Exception as e:
        print(f"⚠️ Error clicking '{text}': {e}")
        return False

def simulate_navigation_clicks(x, y, wait_time=3):
    """Click at specific coordinates for navigation"""
    try:
        pyautogui.click(x, y)
        time.sleep(wait_time)
        return True
    except Exception as e:
        print(f"⚠️ Error clicking coordinates ({x}, {y}): {e}")
        return False

def capture_streamlit_screenshots():
    """Main function to capture all screenshots automatically"""
    
    print("🤖 Starting Automated Screenshot Capture")
    print("=" * 50)
    
    # Setup
    screenshot_dir = setup_screenshot_environment()
    app_url = "http://localhost:8501"
    
    # Check if app is ready
    if not wait_for_streamlit_ready(app_url):
        print("❌ Streamlit app is not accessible. Please ensure it's running.")
        return False
    
    # Open browser
    open_browser_and_navigate(app_url)
    
    # Wait for full page load
    print("⏳ Waiting for page to fully load...")
    time.sleep(8)
    
    try:
        # Screenshot 1: Main Dashboard
        filename1 = os.path.join(screenshot_dir, "01_main_dashboard.png")
        take_screenshot(filename1, "Main Dashboard")
        
        # Try to click "Show Dataset Info" button
        print("\n📊 Trying to capture Dataset Info...")
        time.sleep(2)
        
        # Method 1: Try to find and click Dataset Info button
        dataset_clicked = False
        try:
            # Look for common button coordinates (adjust based on your layout)
            # These are estimated coordinates - may need adjustment
            potential_coords = [
                (200, 400), (300, 450), (400, 500),  # Left side buttons
                (200, 500), (300, 550), (400, 600),  # Lower buttons
            ]
            
            for coord in potential_coords:
                print(f"🔍 Trying to click at {coord}")
                pyautogui.click(coord[0], coord[1])
                time.sleep(3)
                
                # Take screenshot to see if anything changed
                filename2 = os.path.join(screenshot_dir, "02_dataset_info.png")
                take_screenshot(filename2, "Dataset Information")
                dataset_clicked = True
                break
                
        except Exception as e:
            print(f"⚠️ Could not click dataset button: {e}")
        
        if not dataset_clicked:
            # Just take a screenshot of current state
            filename2 = os.path.join(screenshot_dir, "02_current_view.png")
            take_screenshot(filename2, "Current View")
        
        # Try to click "Show Model Metrics" button
        print("\n🤖 Trying to capture Model Metrics...")
        time.sleep(2)
        
        # Scroll down a bit to find model metrics
        pyautogui.scroll(-3)
        time.sleep(2)
        
        model_clicked = False
        try:
            # Try different coordinates for model metrics button
            model_coords = [
                (200, 600), (300, 650), (400, 700),
                (200, 700), (300, 750), (400, 800),
            ]
            
            for coord in model_coords:
                print(f"🔍 Trying to click Model Metrics at {coord}")
                pyautogui.click(coord[0], coord[1])
                time.sleep(3)
                
                filename3 = os.path.join(screenshot_dir, "03_model_performance.png")
                take_screenshot(filename3, "Model Performance")
                model_clicked = True
                break
                
        except Exception as e:
            print(f"⚠️ Could not click model metrics: {e}")
        
        if not model_clicked:
            filename3 = os.path.join(screenshot_dir, "03_current_model_view.png")
            take_screenshot(filename3, "Current Model View")
        
        # Navigate to Upload & Predict
        print("\n📤 Navigating to Upload & Predict...")
        time.sleep(2)
        
        # Try to click sidebar menu items
        sidebar_coords = [
            (50, 200), (100, 250), (150, 300),   # Left sidebar
            (50, 300), (100, 350), (150, 400),
            (50, 400), (100, 450), (150, 500),
        ]
        
        for coord in sidebar_coords:
            print(f"🔍 Trying sidebar click at {coord}")
            pyautogui.click(coord[0], coord[1])
            time.sleep(3)
            
            filename4 = os.path.join(screenshot_dir, "04_upload_predict.png")
            take_screenshot(filename4, "Upload & Predict Page")
            break
        
        # Navigate to EDA
        print("\n📈 Trying to capture EDA page...")
        time.sleep(2)
        
        # Try more sidebar coordinates
        for coord in sidebar_coords[3:6]:
            print(f"🔍 Trying EDA navigation at {coord}")
            pyautogui.click(coord[0], coord[1])
            time.sleep(5)  # EDA takes longer to load
            
            filename5 = os.path.join(screenshot_dir, "05_eda_page.png")
            take_screenshot(filename5, "EDA Page")
            break
        
        # Navigate to Model Management
        print("\n⚙️ Trying to capture Model Management...")
        time.sleep(2)
        
        for coord in sidebar_coords[6:]:
            print(f"🔍 Trying Model Management at {coord}")
            pyautogui.click(coord[0], coord[1])
            time.sleep(3)
            
            filename6 = os.path.join(screenshot_dir, "06_model_management.png")
            take_screenshot(filename6, "Model Management")
            break
        
        # Take a final full-page screenshot
        filename_final = os.path.join(screenshot_dir, "07_final_view.png")
        take_screenshot(filename_final, "Final Application View")
        
        print("\n✅ Screenshot capture completed!")
        print(f"📁 Screenshots saved in: {os.path.abspath(screenshot_dir)}")
        
        # List captured screenshots
        print("\n📸 Captured Screenshots:")
        for file in os.listdir(screenshot_dir):
            if file.endswith('.png'):
                print(f"  - {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during screenshot capture: {e}")
        return False

def create_manual_backup_instructions():
    """Create manual instructions if automation fails"""
    instructions = f"""
# Automated Screenshot Capture Results

**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## If Automated Capture Partially Failed:

The script attempted to capture screenshots automatically. If some screenshots 
are missing or incomplete, please manually capture the following:

### Required Screenshots:
1. **Main Dashboard**: Navigate to http://localhost:8501
2. **Dataset Info**: Click "Show Dataset Info" button
3. **Model Metrics**: Click "Show Model Metrics" button  
4. **Upload & Predict**: Click sidebar menu item
5. **EDA Page**: Click "EDA" in sidebar (wait for plots to load)
6. **Model Management**: Click "Model Management" in sidebar

### Tips for Better Results:
- Ensure browser window is maximized
- Wait for each page to fully load
- Use high resolution display if possible
- Clear browser cache if pages don't load properly

### Troubleshooting:
- If buttons are not found, try manual navigation
- Check that Streamlit app is running on correct port
- Ensure no popup blockers are interfering
"""
    
    with open("screenshot_results.md", "w") as f:
        f.write(instructions)
    
    print("📝 Created screenshot_results.md with backup instructions")

if __name__ == "__main__":
    print("🚀 Elevator Fault Detection System - Automated Screenshot Tool")
    print("=" * 70)
    
    success = capture_streamlit_screenshots()
    
    if success:
        print("\n🎉 All screenshots captured successfully!")
    else:
        print("\n⚠️ Some issues occurred during capture")
    
    create_manual_backup_instructions()
    
    print("\n📋 Next Steps:")
    print("1. Check the 'screenshots' folder for captured images")
    print("2. Review screenshot_results.md for any manual steps needed")
    print("3. Add screenshots to your academic documentation")
    print("4. Ensure all key features are visible in the screenshots")
