from ultralytics import YOLO
import os

"""
YOLOv8 Hand Gesture Detection Training Script
This script trains a YOLOv8 model on hand gesture data from Roboflow
"""

def train_hand_gesture_model():
    print("=" * 70)
    print("YOLOv8 Hand Gesture Detection - Training Script")
    print("=" * 70)
    
    # Check if dataset exists
    dataset_path = 'roboflow_data/data.yaml'
    if not os.path.exists(dataset_path):
        print("\n⚠️  ERROR: Dataset not found!")
        print("\nPlease download the dataset first:")
        print("1. Run: python download_roboflow_dataset.py")
        print("2. Follow the prompts to download from Roboflow")
        print("3. Then run this script again")
        return
    
    print(f"\n✅ Dataset found: {dataset_path}")
    
    # Select model size
    print("\n📦 Available YOLOv8 Models:")
    print("1. yolov8n.pt - Nano (Fastest, smallest)")
    print("2. yolov8s.pt - Small (Balanced)")
    print("3. yolov8m.pt - Medium (Better accuracy)")
    print("4. yolov8l.pt - Large (Best accuracy, slower)")
    
    model_choice = input("\nSelect model [1-4] (default: 1): ").strip() or "1"
    
    models = {
        "1": "yolov8n.pt",
        "2": "yolov8s.pt",
        "3": "yolov8m.pt",
        "4": "yolov8l.pt"
    }
    
    model_name = models.get(model_choice, "yolov8n.pt")
    print(f"\n✅ Selected: {model_name}")
    
    # Training parameters - LIGHTNING FAST (30 seconds target)
    print("\n⚡ LIGHTNING FAST MODE - Training in ~30 seconds!")
    print("⚠️  Minimal training for quick testing only")
    epochs = int(input("Number of epochs (default: 3): ").strip() or "3")
    batch = int(input("Batch size (default: 16): ").strip() or "16")
    imgsz_input = input("Image size (default: 320): ").strip() or "320"
    imgsz = max(320, int(imgsz_input))
    fraction = float(input("Dataset fraction (default 0.05 = 5% of data): ").strip() or "0.05")
    
    print(f"\n📊 Training on only {int(fraction*100)}% of dataset ({epochs} epochs)")
    print("⚠️  This is for testing only - accuracy will be low!")
    if int(imgsz_input) < 320:
        print(f"⚠️  Image size adjusted from {imgsz_input} to {imgsz} (minimum required)")
    
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch}")
    print(f"Image Size: {imgsz}")
    print("=" * 70)
    
    # Detect available device
    import torch
    if torch.cuda.is_available():
        device = '0'
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("\n⚠️  No GPU detected, using CPU (training will be slower)")
    
    # Load pretrained model
    model = YOLO(model_name)
    
    # Train the model - LIGHTNING FAST (30 second target)
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='hand_gesture_yolo',
        patience=1,             # Stop after 1 epoch no improvement
        save=True,
        device=device,
        workers=0,              # No workers for CPU
        project='runs/detect',
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',
        verbose=False,
        seed=42,
        deterministic=False,
        rect=False,
        cos_lr=False,
        resume=False,
        amp=False,
        fraction=fraction,      # Only 5% of data
        cache='ram',
        profile=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0,       # Disabled for speed
        warmup_epochs=0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        val=False,              # SKIP VALIDATION for speed
        plots=False,
        save_period=-1,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        crop_fraction=1.0,
    )
    
    print("\n" + "=" * 70)
    print("✅ Lightning Fast Training Complete!")
    print("=" * 70)
    print("⚠️  Note: This model is only for testing - low accuracy expected")
    print("    For production, train with more epochs and data")
    
    # Copy model to models directory
    best_model_path = f'runs/detect/hand_gesture_yolo/weights/best.pt'
    
    if os.path.exists(best_model_path):
        os.makedirs('models', exist_ok=True)
        
        import shutil
        destination = 'models/hand_gesture_yolo.pt'
        shutil.copy(best_model_path, destination)
        
        print(f"\n✅ Model copied to: {destination}")
        print("\n🚀 You can now run the app with:")
        print("   python main.py")
    else:
        print(f"\n⚠️  Could not find trained model at: {best_model_path}")
    
    print("\n" + "=" * 70)
    print("Training session complete!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        train_hand_gesture_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
