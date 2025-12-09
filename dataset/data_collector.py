"""
Dataset Collection Module for Hand Gesture Recognition
Collects and preprocesses hand gesture data using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime
import pickle
from pathlib import Path

class GestureDataCollector:
    """
    Collects hand gesture data for training deep learning models
    Supports both image and landmark data collection
    """
    
    # Standard gesture classes for the dataset
    GESTURE_CLASSES = [
        'thumbs_up',      # 0
        'peace',          # 1
        'fist',           # 2
        'pointing',       # 3
        'ok_sign',        # 4
        'rock',           # 5
        'one_finger',     # 6
        'two_fingers',    # 7
        'three_fingers',  # 8
        'open_palm'       # 9
    ]
    
    def __init__(self, dataset_root='dataset'):
        """
        Initialize the data collector
        Args:
            dataset_root: Root directory for dataset storage
        """
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / 'images'
        self.landmarks_dir = self.dataset_root / 'landmarks'
        self.metadata_dir = self.dataset_root / 'metadata'
        
        # Create directories
        for class_name in self.GESTURE_CLASSES:
            (self.images_dir / class_name).mkdir(parents=True, exist_ok=True)
            (self.landmarks_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Collection statistics
        self.stats = {class_name: 0 for class_name in self.GESTURE_CLASSES}
        self.load_stats()
    
    def load_stats(self):
        """Load existing collection statistics"""
        stats_file = self.metadata_dir / 'collection_stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
    
    def save_stats(self):
        """Save collection statistics"""
        stats_file = self.metadata_dir / 'collection_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
    
    def extract_landmarks(self, frame):
        """
        Extract hand landmarks from frame
        Args:
            frame: BGR image from camera
        Returns:
            landmarks: Numpy array of shape (21, 3) or None if no hand detected
            annotated_frame: Frame with landmarks drawn
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.hands.process(rgb_frame)
        
        annotated_frame = frame.copy()
        landmarks = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            
            # Extract coordinates
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        return landmarks, annotated_frame
    
    def collect_samples(self, gesture_class, num_samples=100, preview=True):
        """
        Collect samples for a specific gesture class
        Args:
            gesture_class: Name of the gesture class
            num_samples: Number of samples to collect
            preview: Show camera preview
        """
        if gesture_class not in self.GESTURE_CLASSES:
            raise ValueError(f"Unknown gesture class: {gesture_class}")
        
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        collected = 0
        countdown = 3
        collecting = False
        start_time = None
        
        print(f"\n{'='*60}")
        print(f"Collecting samples for gesture: {gesture_class.upper()}")
        print(f"Target: {num_samples} samples")
        print(f"{'='*60}")
        print("\nPress SPACE to start collection")
        print("Press 'q' to quit\n")
        
        while collected < num_samples:
            ret, frame = camera.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks, annotated_frame = self.extract_landmarks(frame)
            
            # Display info
            display_frame = annotated_frame.copy()
            
            # Status overlay
            if not collecting:
                cv2.putText(display_frame, "Press SPACE to start", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                if countdown > 0:
                    # Countdown
                    cv2.putText(display_frame, str(countdown), (280, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
                else:
                    # Collecting
                    cv2.putText(display_frame, "COLLECTING!", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Progress
            cv2.putText(display_frame, f"Gesture: {gesture_class}", (20, 420),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Collected: {collected}/{num_samples}", (20, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            if preview:
                cv2.imshow('Gesture Data Collection', display_frame)
            
            # Handle collection
            if collecting and countdown == 0 and landmarks is not None:
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_path = self.images_dir / gesture_class / f"{timestamp}.jpg"
                cv2.imwrite(str(img_path), frame)
                
                # Save landmarks
                lm_path = self.landmarks_dir / gesture_class / f"{timestamp}.npy"
                np.save(str(lm_path), landmarks)
                
                collected += 1
                self.stats[gesture_class] += 1
                
                print(f"✓ Collected sample {collected}/{num_samples}")
            
            # Handle countdown
            if collecting and countdown > 0:
                if start_time is None:
                    start_time = datetime.now()
                
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= 1.0:
                    countdown -= 1
                    start_time = datetime.now()
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not collecting:
                collecting = True
                countdown = 3
                start_time = None
            elif key == ord('q'):
                break
        
        camera.release()
        cv2.destroyAllWindows()
        
        self.save_stats()
        
        print(f"\n{'='*60}")
        print(f"✓ Collection complete!")
        print(f"Total samples for '{gesture_class}': {self.stats[gesture_class]}")
        print(f"{'='*60}\n")
    
    def create_dataset_info(self):
        """Create dataset information file"""
        info = {
            'name': 'Hand Gesture Recognition Dataset',
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'classes': self.GESTURE_CLASSES,
            'num_classes': len(self.GESTURE_CLASSES),
            'statistics': self.stats,
            'total_samples': sum(self.stats.values()),
            'data_format': {
                'images': 'JPG, 640x480',
                'landmarks': 'NumPy array, shape (21, 3)',
                'coordinates': 'Normalized [0, 1] for x, y; relative depth for z'
            },
            'annotation': {
                'landmarks_per_hand': 21,
                'coordinates_per_landmark': 3,
                'coordinate_system': 'MediaPipe Hand Landmarks'
            }
        }
        
        info_file = self.metadata_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"✓ Dataset info saved to {info_file}")
    
    def show_statistics(self):
        """Display collection statistics"""
        print(f"\n{'='*60}")
        print("Dataset Collection Statistics")
        print(f"{'='*60}")
        
        for class_name in self.GESTURE_CLASSES:
            count = self.stats[class_name]
            bar = '█' * (count // 10)
            print(f"{class_name:15s}: {count:4d} {bar}")
        
        print(f"{'='*60}")
        print(f"Total samples: {sum(self.stats.values())}")
        print(f"{'='*60}\n")


def main():
    """Interactive data collection script"""
    collector = GestureDataCollector()
    
    print("\n" + "="*60)
    print("Hand Gesture Dataset Collection Tool")
    print("="*60)
    
    while True:
        print("\nMenu:")
        print("1. Collect samples for a gesture")
        print("2. Show statistics")
        print("3. Create dataset info file")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            print("\nAvailable gestures:")
            for i, gesture in enumerate(collector.GESTURE_CLASSES):
                print(f"{i}. {gesture}")
            
            gesture_idx = int(input("\nEnter gesture index: "))
            if 0 <= gesture_idx < len(collector.GESTURE_CLASSES):
                gesture = collector.GESTURE_CLASSES[gesture_idx]
                num_samples = int(input("Number of samples (default 100): ") or "100")
                collector.collect_samples(gesture, num_samples)
            else:
                print("Invalid gesture index!")
        
        elif choice == '2':
            collector.show_statistics()
        
        elif choice == '3':
            collector.create_dataset_info()
        
        elif choice == '4':
            print("\nGoodbye! 👋\n")
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
