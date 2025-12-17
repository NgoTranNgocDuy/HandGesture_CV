"""
Download Hand Gesture Dataset from Roboflow Universe
Following Roboflow's data download best practices
"""

import os
from pathlib import Path

def main():
    print("=" * 70)
    print("Roboflow Dataset Downloader for Hand Gesture Detection")
    print("=" * 70)
    print()
    
    # Check if roboflow is installed
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ Roboflow package not installed!")
        print()
        print("Install with:")
        print("  pip install roboflow")
        print()
        return
    
    print("📋 Setup Instructions:")
    print()
    print("1. Get your API key:")
    print("   → Visit: https://app.roboflow.com/settings/api")
    print("   → Copy your API key")
    print()
    print("2. Find a dataset:")
    print("   → Visit: https://universe.roboflow.com/")
    print("   → Search: 'hand gesture detection'")
    print("   → Choose a dataset")
    print()
    print("3. Get workspace and project name from URL:")
    print("   Example URL: https://universe.roboflow.com/david-lee/hand-gestures")
    print("   → Workspace: david-lee")
    print("   → Project: hand-gestures")
    print()
    print("-" * 70)
    print()
    
    # Get user input
    api_key = input("Enter your Roboflow API key: ").strip()
    
    if not api_key:
        print("❌ API key is required!")
        return
    
    workspace = input("Enter workspace name: ").strip()
    project_name = input("Enter project name: ").strip()
    version = input("Enter version (default: 1): ").strip() or "1"
    
    print()
    print("📦 Downloading dataset...")
    print(f"   Workspace: {workspace}")
    print(f"   Project: {project_name}")
    print(f"   Version: {version}")
    print()
    
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        
        # Get project
        project = rf.workspace(workspace).project(project_name)
        
        # Download dataset in YOLOv8 format
        dataset = project.version(int(version)).download("yolov8", location="./roboflow_data")
        
        print()
        print("✅ Dataset downloaded successfully!")
        print(f"📁 Location: {Path('roboflow_data').absolute()}")
        print()
        print("📋 Next steps:")
        print()
        print("Option 1: Use pre-trained model (if available)")
        print("  → Check if workspace provides trained model")
        print("  → Download and save to: models/hand_gesture_yolo.pt")
        print()
        print("Option 2: Train your own model")
        print("  yolo task=detect mode=train \\")
        print("       model=yolov8n.pt \\")
        print(f"       data=roboflow_data/data.yaml \\")
        print("       epochs=50 \\")
        print("       imgsz=640 \\")
        print("       batch=16")
        print()
        print("Option 3: Run with YOLOv8 pretrained (no gesture-specific classes)")
        print("  python app_detection.py")
        print()
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print()
        print("Common issues:")
        print("  - Invalid API key")
        print("  - Wrong workspace/project name")
        print("  - Network connection issues")
        print("  - Version doesn't exist")
        print()


def show_popular_datasets():
    """Show some popular hand gesture datasets on Roboflow"""
    print()
    print("=" * 70)
    print("Popular Hand Gesture Datasets on Roboflow Universe")
    print("=" * 70)
    print()
    
    datasets = [
        {
            "name": "Hand Gesture Recognition Computer Vision Project",
            "workspace": "david-lee-d0rhs",
            "project": "hand-gesture-recognition-computer-vision-project",
            "classes": "10 hand gestures",
            "url": "https://universe.roboflow.com/david-lee-d0rhs/hand-gesture-recognition-computer-vision-project"
        },
        {
            "name": "ASL Alphabet",
            "workspace": "roboflow-58fyf",
            "project": "asl-alphabet-b5fbn",
            "classes": "26 ASL signs",
            "url": "https://universe.roboflow.com/roboflow-58fyf/asl-alphabet-b5fbn"
        },
        {
            "name": "Hand Detection",
            "workspace": "roboflow-100",
            "project": "hand-detection-fubc9",
            "classes": "1 class (hand)",
            "url": "https://universe.roboflow.com/roboflow-100/hand-detection-fubc9"
        }
    ]
    
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds['name']}")
        print(f"   Workspace: {ds['workspace']}")
        print(f"   Project: {ds['project']}")
        print(f"   Classes: {ds['classes']}")
        print(f"   URL: {ds['url']}")
        print()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        show_popular_datasets()
    else:
        main()
        print()
        print("💡 Tip: Run 'python download_roboflow_dataset.py --list' to see popular datasets")
        print()
