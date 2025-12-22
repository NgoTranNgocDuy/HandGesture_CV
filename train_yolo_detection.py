"""
Train YOLOv8 Model on Roboflow Hand Gesture Dataset
Optimized for hand gesture detection with proper configuration
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml

def fix_data_yaml():
    """Fix the data.yaml file paths to be absolute"""
    data_yaml_path = Path('roboflow_data/data.yaml')
    
    if not data_yaml_path.exists():
        print("❌ Error: roboflow_data/data.yaml not found!")
        print("   Please run download_roboflow_dataset.py first")
        return None
    
    # Read the original yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get absolute path to roboflow_data folder
    roboflow_dir = Path('roboflow_data').absolute()
    
    # Update paths to absolute
    data['train'] = str(roboflow_dir / 'train' / 'images')
    data['val'] = str(roboflow_dir / 'valid' / 'images')
    data['test'] = str(roboflow_dir / 'test' / 'images')
    
    # Save the updated yaml
    updated_yaml_path = Path('roboflow_data/data_fixed.yaml')
    with open(updated_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✅ Fixed data.yaml saved to: {updated_yaml_path}")
    print(f"   Train: {data['train']}")
    print(f"   Val: {data['val']}")
    print(f"   Classes: {data['names']}")
    print()
    
    return str(updated_yaml_path)

def train_model(epochs=50, batch_size=16, img_size=640):
    """
    Train YOLOv8 model on the Roboflow dataset
    
    Args:
        epochs: Number of training epochs (default: 50)
        batch_size: Batch size (default: 16, adjust based on GPU memory)
        img_size: Image size (default: 640)
    """
    print("=" * 70)
    print("YOLOv8 Hand Gesture Detection Training")
    print("=" * 70)
    print()
    
    # Fix data.yaml paths
    print("📋 Step 1: Preparing dataset configuration...")
    data_yaml = fix_data_yaml()
    
    if data_yaml is None:
        return False
    
    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Initialize YOLOv8 model
    print("📦 Step 2: Loading YOLOv8n (nano) base model...")
    model = YOLO('yolov8n.pt')  # Start from pretrained COCO model
    print("✅ Model loaded successfully")
    print()
    
    # Training configuration
    print("⚙️  Step 3: Training Configuration")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Image Size: {img_size}x{img_size}")
    print(f"   Device: GPU if available, else CPU")
    print()
    
    print("🚀 Step 4: Starting training...")
    print("   This may take a while depending on your hardware...")
    print()
    
    try:
        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='hand_gesture_detection',
            patience=10,  # Early stopping patience
            save=True,
            device='0',  # Use GPU 0 if available
            workers=4,
            project='runs/detect',
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=10,
            resume=False,
            amp=True,  # Automatic Mixed Precision
            fraction=1.0,
            profile=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True
        )
        
        print()
        print("=" * 70)
        print("✅ Training completed successfully!")
        print("=" * 70)
        print()
        
        # Get the best model path
        best_model_path = Path('runs/detect/hand_gesture_detection/weights/best.pt')
        
        if best_model_path.exists():
            # Copy to models directory for easy access
            import shutil
            target_path = models_dir / 'hand_gesture_yolo.pt'
            shutil.copy(best_model_path, target_path)
            
            print(f"📦 Best model saved to: {target_path}")
            print()
            print("📊 Training Results:")
            print(f"   Results folder: runs/detect/hand_gesture_detection")
            print(f"   Weights: runs/detect/hand_gesture_detection/weights/")
            print(f"   - best.pt (best performing model)")
            print(f"   - last.pt (last epoch)")
            print()
            print("📋 Next Steps:")
            print("   1. Check training results in: runs/detect/hand_gesture_detection")
            print("   2. View plots: confusion_matrix.png, results.png, etc.")
            print("   3. Run the detection app:")
            print("      python app_detection.py")
            print()
            print("🎯 The app will automatically use the trained model!")
            print()
            
            return True
        else:
            print("⚠️  Warning: Best model not found at expected location")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print()
        print("Common issues:")
        print("  - Insufficient GPU/CPU memory → Reduce batch_size")
        print("  - CUDA out of memory → Reduce batch_size or image size")
        print("  - Missing dataset → Check roboflow_data folder")
        return False

def validate_model():
    """Validate the trained model"""
    model_path = Path('models/hand_gesture_yolo.pt')
    
    if not model_path.exists():
        print("❌ No trained model found!")
        return False
    
    print("=" * 70)
    print("Model Validation")
    print("=" * 70)
    print()
    
    try:
        model = YOLO(str(model_path))
        
        # Get model info
        print(f"✅ Model loaded: {model_path}")
        print(f"   Classes: {model.names}")
        print(f"   Number of classes: {len(model.names)}")
        print()
        
        # Run validation on test set
        data_yaml = Path('roboflow_data/data_fixed.yaml')
        if data_yaml.exists():
            print("📊 Running validation on test set...")
            results = model.val(data=str(data_yaml))
            print("✅ Validation complete!")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating model: {e}")
        return False

if __name__ == '__main__':
    import sys
    
    print()
    print("=" * 70)
    print("Hand Gesture Detection - YOLOv8 Training Script")
    print("=" * 70)
    print()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--validate':
            validate_model()
            sys.exit(0)
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python train_yolo_detection.py                # Train with default settings")
            print("  python train_yolo_detection.py --validate     # Validate trained model")
            print("  python train_yolo_detection.py --help         # Show this help")
            print()
            print("Training parameters (edit in the script):")
            print("  epochs: Number of training epochs (default: 50)")
            print("  batch_size: Batch size (default: 16)")
            print("  img_size: Image size (default: 640)")
            print()
            sys.exit(0)
    
    # Start training
    success = train_model(
        epochs=50,        # Adjust as needed
        batch_size=16,    # Reduce if GPU memory issues
        img_size=640      # Standard size for YOLOv8
    )
    
    if success:
        print("=" * 70)
        print("🎉 Training pipeline completed successfully!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("❌ Training failed. Please check the errors above.")
        print("=" * 70)
