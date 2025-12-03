from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time

app = Flask(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables
gesture_history = deque(maxlen=10)
current_gesture = "None"
gesture_confidence = 0
volume_level = 50
brightness_level = 50

class HandGestureDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 0.5  # seconds
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_up(self, landmarks, finger_tip, finger_pip):
        """Check if a finger is extended"""
        return landmarks[finger_tip].y < landmarks[finger_pip].y
    
    def detect_gesture(self, landmarks):
        """Detect hand gestures based on finger positions"""
        global current_gesture, gesture_confidence, volume_level, brightness_level
        
        # Get finger tips and PIPs
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Check which fingers are up
        thumb_up = landmarks[4].x < landmarks[3].x  # For thumb, check x-axis
        index_up = self.is_finger_up(landmarks, 8, 6)
        middle_up = self.is_finger_up(landmarks, 12, 10)
        ring_up = self.is_finger_up(landmarks, 16, 14)
        pinky_up = self.is_finger_up(landmarks, 20, 18)
        
        fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        fingers_count = sum(fingers_up)
        
        # Calculate distances for control gestures
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        index_middle_dist = self.calculate_distance(index_tip, middle_tip)
        
        # Gesture detection logic
        gesture = "None"
        confidence = 0
        
        # Volume Control (Thumb and Index fingers stretched - higher priority)
        if thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
            # Calculate volume based on distance between thumb and index
            # Normalize distance: 0.05 = 0%, 0.25+ = 100%
            volume_level = int(max(0, min(100, (thumb_index_dist - 0.05) * 500)))
            gesture = f"Volume Control 🔊 ({volume_level}%)"
            confidence = 92
            
        # Brightness Control (Index and Middle fingers stretched - higher priority)
        elif index_up and middle_up and not thumb_up and not ring_up and not pinky_up:
            # Calculate brightness based on distance between index and middle
            # Normalize distance: 0.02 = 0%, 0.15+ = 100%
            brightness_level = int(max(0, min(100, (index_middle_dist - 0.02) * 800)))
            gesture = f"Brightness Control 💡 ({brightness_level}%)"
            confidence = 92
            
        # Peace Sign (Victory) - Index and Middle up with thumb
        elif index_up and middle_up and thumb_up and not ring_up and not pinky_up:
            gesture = "Peace ✌️"
            confidence = 95
            
        # Thumbs Up (only thumb)
        elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            gesture = "Thumbs Up 👍"
            confidence = 90
            
        # Pointing (Only Index up)
        elif index_up and not thumb_up and not middle_up and not ring_up and not pinky_up:
            gesture = "Pointing ☝️"
            confidence = 85
            
        # OK Sign (Thumb and Index touching with other fingers up)
        elif thumb_index_dist < 0.05 and middle_up and ring_up and pinky_up:
            gesture = "OK Sign 👌"
            confidence = 93
            
        # Rock Sign (Index and Pinky up)
        elif index_up and pinky_up and not middle_up and not ring_up:
            gesture = "Rock 🤘"
            confidence = 90
            
        # Dynamic Finger Count - Show number of fingers raised
        elif fingers_count > 0:
            finger_emojis = ['0️⃣', '1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣']
            finger_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
            gesture = f"{finger_names[fingers_count]} Finger{'s' if fingers_count > 1 else ''} {finger_emojis[fingers_count]}"
            confidence = 88
            
        # Fist (All fingers down)
        else:
            gesture = "Fist ✊"
            confidence = 88
        
        current_gesture = gesture
        gesture_confidence = confidence
        
        return gesture, confidence
    
    def process_frame(self, frame):
        """Process video frame and detect hand gestures"""
        global current_gesture, gesture_confidence
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks and detect gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Detect gesture
                gesture, confidence = self.detect_gesture(hand_landmarks.landmark)
                
                # Add gesture to history
                gesture_history.append(gesture)
        else:
            current_gesture = "No Hand Detected"
            gesture_confidence = 0
        
        # Display gesture info on frame
        self.draw_info_panel(frame)
        
        return frame
    
    def draw_info_panel(self, frame):
        """Draw information panel on the frame"""
        height, width, _ = frame.shape
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display gesture info
        cv2.putText(frame, f"Gesture: {current_gesture}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {gesture_confidence}%", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Volume: {volume_level}% | Brightness: {brightness_level}%", 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Initialize detector
detector = HandGestureDetector()

def generate_frames():
    """Generate video frames with hand gesture detection"""
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame
        frame = detector.process_frame(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture_data')
def gesture_data():
    """Return current gesture data as JSON"""
    return jsonify({
        'gesture': current_gesture,
        'confidence': gesture_confidence,
        'volume': volume_level,
        'brightness': brightness_level,
        'history': list(gesture_history)
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Hand Gesture Control System Starting...")
    print("=" * 60)
    print("📹 Camera will initialize when you open the browser")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("=" * 60)
    print("\n👋 Supported Gestures:")
    print("  1️⃣-5️⃣  Dynamic Finger Counting - Shows number of raised fingers")
    print("  ✌️  Peace Sign - Index, Middle, and Thumb up")
    print("  👍 Thumbs Up - Only thumb extended")
    print("  ✊ Fist - All fingers closed")
    print("  ☝️  Pointing - Only index finger up")
    print("  👌 OK Sign - Thumb and index touching")
    print("  🤘 Rock Sign - Index and pinky up")
    print("  🔊 Volume Control - Stretch Thumb + Index apart (0-100%)")
    print("  💡 Brightness Control - Stretch Index + Middle apart (0-100%)")
    print("\n⚠️  Press Ctrl+C to stop the server\n")
    print("=" * 60)
    
    app.run(debug=True, threaded=True, use_reloader=False)
