"""
Improved Flask App with Deep Learning Model Integration
Hybrid approach: MediaPipe for real-time + DL model for enhanced accuracy
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time
import torch
import torchvision.transforms as transforms
from pathlib import Path

# Import deep learning model
from models.gesture_cnn_lstm import HybridGestureModel

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

# Deep Learning Model Setup
USE_DL_MODEL = False  # Toggle between rule-based and DL model
DL_MODEL = None
DL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Gesture class names for DL model
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

def load_dl_model():
    """Load trained deep learning model"""
    global DL_MODEL, USE_DL_MODEL
    
    model_path = Path('checkpoints')
    if not model_path.exists():
        print(" No trained model found. Using rule-based detection only.")
        USE_DL_MODEL = False
        return
    
    # Find latest checkpoint
    checkpoints = list(model_path.glob('best_model*.pth'))
    if not checkpoints:
        print(" No trained model found. Using rule-based detection only.")
        USE_DL_MODEL = False
        return
    
    try:
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        DL_MODEL = HybridGestureModel(num_classes=10).to(DL_DEVICE)
        checkpoint = torch.load(latest_checkpoint, map_location=DL_DEVICE)
        DL_MODEL.load_state_dict(checkpoint['model_state_dict'])
        DL_MODEL.eval()
        
        USE_DL_MODEL = True
        print(f" Loaded DL model from {latest_checkpoint}")
        print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A')}%")
    except Exception as e:
        print(f" Error loading model: {e}")
        print("  Falling back to rule-based detection")
        USE_DL_MODEL = False


class HandGestureDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 0.5
        
        # Landmark sequence buffer for LSTM
        self.landmark_buffer = deque(maxlen=30)
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_up(self, landmarks, finger_tip, finger_pip):
        """Check if a finger is extended"""
        return landmarks[finger_tip].y < landmarks[finger_pip].y
    
    def detect_gesture_rule_based(self, landmarks):
        """Rule-based gesture detection (original method)"""
        global current_gesture, gesture_confidence, volume_level, brightness_level
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
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
                
                # Get top 2 predictions to check confidence distribution
                top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)
                
                predicted = top2_indices[0][0].item()
                confidence_pct = int(top2_probs[0][0].item() * 100)
                second_conf = int(top2_probs[0][1].item() * 100)
                
                # Get rule-based prediction for validation
                rule_gesture, rule_conf = self.detect_gesture_rule_based(landmarks)
                
                # ALWAYS use rule-based for undertrained models
                # DL model with limited training data is unreliable
                # Only use DL if confidence gap is HUGE (>50%) indicating strong learning
                confidence_gap = confidence_pct - second_conf
                
                dl_gesture = DL_GESTURES[predicted]
                
                if confidence_gap < 50:
                    # Not enough confidence gap - model is guessing
                    print(f"⚠️ DL uncertain (top: {confidence_pct}%, 2nd: {second_conf}%), using rule-based: {rule_gesture}")
                    return rule_gesture, rule_conf
                
                # Even with high confidence, validate against rule-based
                # Extract base gesture names for comparison
                dl_base = dl_gesture.split()[0].lower()
                rule_base = rule_gesture.split()[0].lower()
                
                if dl_base != rule_base:
                    # DL disagrees with rule-based - likely wrong due to limited training
                    print(f"⚠️ DL ({dl_gesture}, {confidence_pct}%) conflicts with rule-based ({rule_gesture}), using rule-based")
                    return rule_gesture, rule_conf
                
                # High confidence AND agrees with rule-based - safe to use
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


# Initialize detector and load model
detector = HandGestureDetector()
load_dl_model()

def generate_frames():
    """Generate video frames with hand gesture detection"""
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = detector.process_frame(frame)
        
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
        'history': list(gesture_history),
        'using_dl_model': USE_DL_MODEL
    })

@app.route('/toggle_model', methods=['POST'])
def toggle_model():
    """Toggle between rule-based and DL model"""
    global USE_DL_MODEL
    if DL_MODEL is not None:
        USE_DL_MODEL = not USE_DL_MODEL
        return jsonify({'success': True, 'using_dl_model': USE_DL_MODEL})
    return jsonify({'success': False, 'message': 'DL model not available'})

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Hand Gesture Control System Starting...")
    print("=" * 60)
    print(f"📹 Detection Method: {'Deep Learning' if USE_DL_MODEL else 'Rule-Based'}")
    if USE_DL_MODEL:
        print(f"🧠 DL Device: {DL_DEVICE}")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("=" * 60)
    print("\n👋 Supported Gestures:")
    print("  1️⃣-5️⃣  Dynamic Finger Counting")
    print("  ✌️  Peace Sign")
    print("  👍 Thumbs Up")
    print("  ✊ Fist")
    print("  ☝️  Pointing")
    print("  👌 OK Sign")
    print("  🤘 Rock Sign")
    print("  🔊 Volume Control")
    print("  💡 Brightness Control")
    print("\n⚠️  Press Ctrl+C to stop the server\n")
    print("=" * 60)
    
    app.run(debug=True, threaded=True, use_reloader=False)
