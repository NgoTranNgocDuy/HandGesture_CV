### Step 1: Install Python Dependencies
Open Terminal in the project directory and run:
```powershell
pip install -r requirements.txt
```
### Step 2: Run the Application
```powershell
python app.py
```
### Step 3: Open in Browser
Once the server starts, open your web browser and navigate to port 5000
##  How to Use
1. **Allow Camera Access**: When prompted, allow the browser to access your webcam
2. **Position Your Hand**: Place your hand in front of the camera
3. **Make Gestures**: Try different hand gestures to see them recognized in real-time
4. **Control Volume/Brightness**: Use pinch gestures to control virtual volume and brightness

##  Tech Stack
- **Backend**: Flask (Python)
- **Computer Vision**: OpenCV, MediaPipe
- **Frontend**: HTML5, CSS3, JS
- **Real-time Updates**: AJAX polling