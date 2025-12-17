"""
Object Detection App using YOLOv8 and Roboflow
Detects multiple hands and classifies gestures simultaneously
Following Roboflow.com best practices
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from pathlib import Path
import time

app = Flask(__name__)

# Global variables
detection_active = False
model = None
class_names = []

# Try to import ultralytics (YOLOv8)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not installed. Install with: pip install ultralytics")

def load_model():
    """
    Load YOLOv8 model - either from Roboflow or custom trained
    Following Roboflow's model loading pattern
    """
    global model, class_names
    
    if not YOLO_AVAILABLE:
        print("❌ YOLOv8 not available. Please install: pip install ultralytics")
        return False
    
    # Check for custom trained model first
    custom_model_path = Path('models/hand_gesture_yolo.pt')
    
    if custom_model_path.exists():
        print(f"📦 Loading custom trained model from {custom_model_path}")
        model = YOLO(str(custom_model_path))
        class_names = model.names
        print(f"✅ Loaded custom model with {len(class_names)} classes")
    else:
        # Use pre-trained YOLO model for hand detection
        print("📦 Loading YOLOv8n (nano) pretrained model...")
        print("   For better results, download from Roboflow:")
        print("   https://universe.roboflow.com/search?q=hand+gesture")
        model = YOLO('yolov8n.pt')  # Will auto-download on first run
        
        # Filter to detect hands/person only
        class_names = {
            0: 'hand',  # We'll treat person detection as hand for demo
        }
        print("✅ Loaded YOLOv8n pretrained model")
        print("⚠️  Note: Using generic detection. For gesture-specific detection:")
        print("   1. Download dataset from Roboflow Universe")
        print("   2. Train YOLOv8 model")
        print("   3. Save as models/hand_gesture_yolo.pt")
    
    return True

def detect_hands_in_frame(frame):
    """
    Detect hands using YOLOv8
    Returns: List of [x1, y1, x2, y2, confidence, class_id, class_name]
    """
    global model
    
    if model is None:
        return []
    
    # Run inference
    results = model(frame, conf=0.25, iou=0.45, verbose=False)
    
    detections = []
    
    # Parse results (Roboflow/YOLOv8 format)
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get class name
            if hasattr(model, 'names'):
                class_name = model.names[class_id]
            else:
                class_name = class_names.get(class_id, f'Class {class_id}')
            
            detections.append([
                int(x1), int(y1), int(x2), int(y2),
                float(confidence), class_id, class_name
            ])
    
    return detections

def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels (Roboflow visualization style)
    """
    frame_copy = frame.copy()
    
    # Roboflow color palette
    colors = [
        (0, 255, 0),    # Green
        (255, 144, 30), # Orange
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 0),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 128),# Light Pink
        (128, 255, 128),# Light Green
        (128, 128, 255),# Light Blue
        (255, 192, 203),# Pink
    ]
    
    for det in detections:
        x1, y1, x2, y2, confidence, class_id, class_name = det
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw bounding box (Roboflow style - thicker lines)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label with confidence
        label = f"{class_name} {confidence:.2f}"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Draw label background (Roboflow style)
        cv2.rectangle(
            frame_copy,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame_copy,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    # Draw detection count (Roboflow style)
    count_text = f"Detections: {len(detections)}"
    cv2.rectangle(frame_copy, (10, 10), (250, 50), (0, 0, 0), -1)
    cv2.putText(
        frame_copy,
        count_text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    
    return frame_copy

def generate_frames():
    """Generate video frames with hand detection"""
    global detection_active
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while detection_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        detections = detect_hands_in_frame(frame)
        
        # Draw detections
        annotated_frame = draw_detections(frame, detections)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Draw FPS (Roboflow style)
        cv2.rectangle(annotated_frame, 
                     (annotated_frame.shape[1] - 130, 10),
                     (annotated_frame.shape[1] - 10, 50),
                     (0, 0, 0), -1)
        cv2.putText(
            annotated_frame,
            f"FPS: {fps}",
            (annotated_frame.shape[1] - 120, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    """Main page for detection mode"""
    return render_template('detection.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    global detection_active
    detection_active = True
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stop_detection')
def stop_detection():
    """Stop detection"""
    global detection_active
    detection_active = False
    return jsonify({'status': 'stopped'})

@app.route('/status')
def status():
    """Get detection status"""
    return jsonify({
        'active': detection_active,
        'model_loaded': model is not None,
        'yolo_available': YOLO_AVAILABLE,
        'num_classes': len(class_names) if class_names else 0
    })

if __name__ == '__main__':
    print("=" * 70)
    print("Hand Gesture Object Detection - YOLOv8 + Roboflow")
    print("=" * 70)
    
    # Load model
    model_loaded = load_model()
    
    if model_loaded:
        print()
        print("📋 Roboflow Integration Guide:")
        print("   1. Visit: https://universe.roboflow.com/")
        print("   2. Search: 'hand gesture detection'")
        print("   3. Download dataset in YOLOv8 format")
        print("   4. Train model: yolo task=detect mode=train ...")
        print("   5. Save as: models/hand_gesture_yolo.pt")
        print()
        print("🌐 Server starting at: http://localhost:5001")
        print("=" * 70)
        
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
    else:
        print()
        print("❌ Failed to load model. Please install ultralytics:")
        print("   pip install ultralytics")
        print("=" * 70)
