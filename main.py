"""
Unified Hand Recognition System
Dual-mode app with landing page and integrated modes
"""

from flask import Flask, render_template, Response, jsonify, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import math
import torch
import torchvision.transforms as transforms
from pathlib import Path

app = Flask(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables for Mode 1 (Gesture Classification)
gesture_detector = None
gesture_active = False
gesture_camera = None
gesture_history = deque(maxlen=10)
current_gesture = "None"
gesture_confidence = 0
volume_level = 50
brightness_level = 50

# Deep Learning Model Setup for Mode 1
USE_DL_MODEL = False
DL_MODEL = None
DL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DL_GESTURES = [
    'Thumbs Up 👍',      # 0
    'Peace ✌️',          # 1
    'Fist ✊',           # 2
    'Pointing ☝️',       # 3
    'OK Sign 👌',        # 4
    'Rock 🤘',           # 5
    'One Finger 1️⃣',    # 6
    'Two Fingers 2️⃣',   # 7
    'Three Fingers 3️⃣', # 8
    'Open Palm ✋'       # 9
]

# Global variables for Mode 2 (Object Detection)
detection_active = False
detection_camera = None
yolo_model = None
yolo_available = False

# ============================================================================
# LANDING PAGE
# ============================================================================

@app.route('/')
def landing():
    """Landing page with mode selection"""
    return render_template('landing.html')

# ============================================================================
# MODE 1: GESTURE CLASSIFICATION
# ============================================================================

def load_gesture_model():
    """Load gesture classification model"""
    global DL_MODEL, DL_DEVICE, DL_TRANSFORM, DL_GESTURES, USE_DL_MODEL
    
    DL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DL_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    DL_GESTURES = [
        'Thumbs Up 👍', 'Peace ✌️', 'Fist ✊', 'Pointing ☝️', 'OK Sign 👌',
        'Rock 🤘', 'One Finger ☝', 'Two Fingers ✌', 'Three Fingers 🤟', 'Open Palm 🖐'
    ]
    
    # Try to load model
    model_files = list(Path('checkpoints').glob('best_model_*.pth'))
    if model_files:
        from models.gesture_cnn_lstm import HybridGestureModel
        DL_MODEL = HybridGestureModel(num_classes=10)
        checkpoint = torch.load(model_files[0], map_location=DL_DEVICE)
        DL_MODEL.load_state_dict(checkpoint['model_state_dict'])
        DL_MODEL.to(DL_DEVICE)
        DL_MODEL.eval()
        USE_DL_MODEL = True
        print(f"✅ Loaded gesture model from {model_files[0]}")
        return True
    else:
        USE_DL_MODEL = False
        print("⚠️ No gesture model found, using rule-based only")
        return False

@app.route('/gesture')
def gesture_mode():
    """Gesture classification mode"""
    return render_template('gesture.html')

@app.route('/gesture_video_feed')
def gesture_video_feed():
    """Video feed for gesture mode"""
    global gesture_active
    gesture_active = True
    return Response(
        generate_gesture_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

class HandGestureDetector:
    """Complete gesture detection class from app_with_dl.py"""
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 0.5
        self.landmark_buffer = deque(maxlen=30)
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_up(self, landmarks, finger_tip, finger_pip):
        """Check if a finger is extended"""
        return landmarks[finger_tip].y < landmarks[finger_pip].y
    
    def detect_gesture_rule_based(self, landmarks):
        """Rule-based gesture detection"""
        global current_gesture, gesture_confidence, volume_level, brightness_level
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        thumb_up = landmarks[4].x < landmarks[3].x
        index_up = self.is_finger_up(landmarks, 8, 6)
        middle_up = self.is_finger_up(landmarks, 12, 10)
        ring_up = self.is_finger_up(landmarks, 16, 14)
        pinky_up = self.is_finger_up(landmarks, 20, 18)
        
        fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        fingers_count = sum(fingers_up)
        
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        index_middle_dist = self.calculate_distance(index_tip, middle_tip)
        
        gesture = "None"
        confidence = 0
        
        # Volume Control
        if thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
            volume_level = int(max(0, min(100, (thumb_index_dist - 0.05) * 500)))
            gesture = f"Volume Control 🔊 ({volume_level}%)"
            confidence = 92
            
        # Brightness Control
        elif index_up and middle_up and not thumb_up and not ring_up and not pinky_up:
            brightness_level = int(max(0, min(100, (index_middle_dist - 0.02) * 800)))
            gesture = f"Brightness Control 💡 ({brightness_level}%)"
            confidence = 92
            
        # Peace Sign
        elif index_up and middle_up and thumb_up and not ring_up and not pinky_up:
            gesture = "Peace ✌️"
            confidence = 95
            
        # Thumbs Up
        elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            gesture = "Thumbs Up 👍"
            confidence = 90
            
        # Pointing
        elif index_up and not thumb_up and not middle_up and not ring_up and not pinky_up:
            gesture = "Pointing ☝️"
            confidence = 85
            
        # OK Sign
        elif thumb_index_dist < 0.05 and middle_up and ring_up and pinky_up:
            gesture = "OK Sign 👌"
            confidence = 93
            
        # Rock Sign
        elif index_up and pinky_up and not middle_up and not ring_up:
            gesture = "Rock 🤘"
            confidence = 90
            
        # Finger Count
        elif fingers_count > 0:
            finger_emojis = ['0️⃣', '1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣']
            finger_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
            gesture = f"{finger_names[fingers_count]} Finger{'s' if fingers_count > 1 else ''} {finger_emojis[fingers_count]}"
            confidence = 88
            
        # Fist
        else:
            gesture = "Fist ✊"
            confidence = 88
        
        current_gesture = gesture
        gesture_confidence = confidence
        
        return gesture, confidence
    
    def detect_gesture_dl(self, frame, landmarks):
        """Deep learning based gesture detection"""
        global DL_MODEL, DL_DEVICE, DL_TRANSFORM, current_gesture, gesture_confidence
        
        if DL_MODEL is None:
            return self.detect_gesture_rule_based(landmarks)
        
        try:
            # Prepare image
            image_tensor = DL_TRANSFORM(frame).unsqueeze(0).to(DL_DEVICE)
            
            # Prepare landmarks sequence
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            self.landmark_buffer.append(landmarks_array)
            
            # Pad if not enough frames
            if len(self.landmark_buffer) < 30:
                landmark_seq = np.tile(landmarks_array, (30, 1))
            else:
                landmark_seq = np.array(list(self.landmark_buffer))
            
            landmark_tensor = torch.FloatTensor(landmark_seq).unsqueeze(0).to(DL_DEVICE)
            
            # Inference
            with torch.no_grad():
                outputs = DL_MODEL(image_tensor, landmark_tensor, mode='hybrid')
                probs = torch.softmax(outputs, dim=1)
                
                top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)
                
                predicted = top2_indices[0][0].item()
                confidence_pct = int(top2_probs[0][0].item() * 100)
                second_conf = int(top2_probs[0][1].item() * 100)
                
                # Get rule-based prediction for validation
                rule_gesture, rule_conf = self.detect_gesture_rule_based(landmarks)
                
                confidence_gap = confidence_pct - second_conf
                dl_gesture = DL_GESTURES[predicted]
                
                if confidence_gap < 50:
                    return rule_gesture, rule_conf
                
                # Validate against rule-based
                dl_base = dl_gesture.split()[0].lower()
                rule_base = rule_gesture.split()[0].lower()
                
                if dl_base != rule_base:
                    return rule_gesture, rule_conf
                
                # High confidence AND agrees with rule-based
                current_gesture = f"{dl_gesture} [DL]"
                gesture_confidence = confidence_pct
                
                return current_gesture, gesture_confidence
        
        except Exception as e:
            print(f"DL inference error: {e}")
            return self.detect_gesture_rule_based(landmarks)
    
    def process_frame(self, frame):
        """Process video frame and detect hand gestures"""
        global current_gesture, gesture_confidence, USE_DL_MODEL
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Choose detection method
                if USE_DL_MODEL:
                    gesture, confidence = self.detect_gesture_dl(frame, hand_landmarks.landmark)
                else:
                    gesture, confidence = self.detect_gesture_rule_based(hand_landmarks.landmark)
                
                gesture_history.append(gesture)
        else:
            current_gesture = "No Hand Detected"
            gesture_confidence = 0
        
        self.draw_info_panel(frame)
        
        return frame
    
    def draw_info_panel(self, frame):
        """Draw information panel on the frame"""
        height, width, _ = frame.shape
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Detection method
        method = "Deep Learning" if USE_DL_MODEL else "Rule-Based"
        cv2.putText(frame, f"Method: {method}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        cv2.putText(frame, f"Gesture: {current_gesture}", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {gesture_confidence}%", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Volume: {volume_level}% | Brightness: {brightness_level}%", 
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def generate_gesture_frames():
    """Generate frames for gesture classification"""
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = HandGestureDetector()
    
    while gesture_active:
        success, frame = camera.read()
        if not success:
            break
        
        frame = detector.process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    camera.release()

@app.route('/gesture_data')
def gesture_data():
    """Return current gesture data as JSON"""
    return jsonify({
        'gesture': current_gesture,
        'confidence': gesture_confidence,
        'volume': volume_level,
        'brightness': brightness_level,
        'history': list(gesture_history),
        'using_dl_model': USE_DL_MODEL
    })

# ============================================================================
# MODE 2: OBJECT DETECTION
# ============================================================================

def load_detection_model():
    """Load YOLOv8 detection model"""
    global yolo_model, yolo_available
    
    try:
        from ultralytics import YOLO
        yolo_available = True
        
        custom_model = Path('models/hand_gesture_yolo.pt')
        if custom_model.exists():
            yolo_model = YOLO(str(custom_model))
            print(f"✅ Loaded custom YOLOv8 model from {custom_model}")
        else:
            yolo_model = YOLO('yolov8n.pt')
            print("✅ Loaded YOLOv8n pretrained model")
        return True
    except ImportError:
        yolo_available = False
        print("⚠️ ultralytics not installed")
        return False

@app.route('/detection')
def detection_mode():
    """Object detection mode"""
    if not yolo_available:
        return render_template('detection_error.html')
    return render_template('detection.html')

@app.route('/detection_video_feed')
def detection_video_feed():
    """Video feed for detection mode"""
    global detection_active
    detection_active = True
    return Response(
        generate_detection_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def generate_detection_frames():
    """Generate frames for object detection"""
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while detection_active:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        if yolo_model:
            results = yolo_model(frame, conf=0.25, verbose=False)
            annotated = results[0].plot()
        else:
            annotated = frame
        
        ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    camera.release()

# ============================================================================
# CONTROL ROUTES
# ============================================================================

@app.route('/stop_gesture')
def stop_gesture():
    global gesture_active
    gesture_active = False
    return jsonify({'status': 'stopped'})

@app.route('/stop_detection')
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({'status': 'stopped'})

@app.route('/status')
def status():
    return jsonify({
        'gesture_active': gesture_active,
        'detection_active': detection_active,
        'gesture_model_loaded': DL_MODEL is not None,
        'yolo_available': yolo_available
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("🤖 Hand Recognition System - Unified App")
    print("=" * 70)
    print()
    print("Loading models...")
    load_gesture_model()
    load_detection_model()
    print()
    print("🌐 Server starting at: http://localhost:5000")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
