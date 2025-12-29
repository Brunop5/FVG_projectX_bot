# Project Installation and Usage Guide (For Non-Technical Users)

This guide will help you set up and run the project, even if you’ve never used Python before.

---

## 1. Install Python

### Windows
1. Open your web browser and go to: https://www.python.org/downloads/windows/
2. Click **“Download Python 3.x”** (latest version)
3. When the file finishes downloading, double-click it to run the installer
4. **Important:** Check the box **“Add Python 3.x to PATH”**
5. Click **Install Now** and wait until it finishes
6. Close the installer

### macOS
1. Go to https://www.python.org/downloads/macos/
2. Download the latest version
3. Open the file and follow the instructions to install Python

### Linux
1. Open Terminal (search for "Terminal" in your system)
2. If you use Ubuntu/Debian, type the following commands and press Enter after each:
   ```
   sudo apt update
   sudo apt install python3 python3-venv python3-pip -y
   ```
3. If you use Fedora, type:
   ```
   sudo dnf install python3 python3-venv python3-pip -y
   ```
4. Python is now installed

---

## 2. Run the Installation Script

1. Open the folder where you downloaded this project
2. Locate the file **install.py**
3. **Windows:**
   - Right-click the Start button and choose **“Windows Terminal”** or **“Command Prompt”**
   - In the window that opens, type:
     ```
     cd path\to\project\folder
     python install.py
     ```
     Replace `path\to\project\folder` with the actual path to the project folder
4. **macOS / Linux:**
   - Open **Terminal**
   - Type:
     ```
     cd /path/to/project/folder
     python3 install.py
     ```
     Replace `/path/to/project/folder` with the actual path
5. Wait while the script creates a virtual environment and installs all required packages

---

## 3. Run the Main Script

1. **Windows:**
   ```
   venv\Scripts\python strategyClass.py
   ```
2. **macOS / Linux:**
   ```
   source venv/bin/activate
   python strategyClass.py
   ```
3. The program should now start

---

## 4. Customize Your Strategy

You can edit the **strategy settings** in the `strategyClass.py` file with any text editor (like Notepad on Windows, TextEdit on macOS, or gedit on Linux).  

Here are the main inputs you can change:

```
IS_FVG_TO_SHOW = True              # Display FVG
FVG_HISTORY_NBR = 5                # Number of FVGs to show (1-50)
IS_MITIGATED_FVG_TO_REDUCE = True  # Reduce mitigated FVG
MIN_FVG_POWER_PCT = 0.1            # Min FVG Power %
HTF_TF = "240"                     # HTF Bias (4H)
ATR_PERIOD = 14                    # ATR Period (min 1)
SL_MULTIPLIER = 4.0                # SL ATR Multiplier
TP_MULTIPLIER = 25.0               # TP ATR Multiplier (Positional: Wider Targets)
USE_TRAILING = True                 # Use trailing stop
TRAIL_OFFSET_MULT = 8.0             # Trailing Offset ATR Multiplier
HOLD_UNTIL_OPPOSITE = True          # Hold Until Opposite BOS/CHoCH

ORDER_SIZE = 1
ASSETS = [("CON.F.US.BP6.H26","30min")]  # Asset ID and timeframe
USERNAME = "yourTopstepXUsername"        # Your account username
API_KEY = "yourApiKey"                   # Your account API key
LIVE = True   # True = real money account, False = demo account
```
Do not rename the names, just change the values!


### Important Notes:
- **ASSETS:** Replace with the asset ID and timeframe you want to trade. If you’re not sure what ID your asset has, you can contact me and we can figure it out together.
- **USERNAME / API_KEY:** Replace with your real money or demo account credentials.
- **LIVE:** Set to `True` if using a real account, `False` for demo.

---

## 5. Help and Support

- If you see any errors or problems while running the program, let me know.
- If you don’t know the asset ID for your desired asset, contact me and we’ll figure it out.
- Always save a copy of `strategyClass.py` before making changes, just in case.

