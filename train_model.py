"""
Training Script for Hand Gesture Recognition Model
Includes comprehensive evaluation metrics and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import time
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.gesture_cnn_lstm import HybridGestureModel, GestureCNN, GestureLSTM
from dataset.gesture_dataset import get_data_loaders


class GestureTrainer:
    """
    Comprehensive training pipeline for gesture recognition models
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device='cuda', checkpoint_dir='checkpoints', results_dir='results'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for images, landmarks, labels in pbar:
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, landmarks, mode='hybrid')
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for images, landmarks, labels in pbar:
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, landmarks, mode='hybrid')
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels, all_probs
    
    def train(self, num_epochs=50, early_stopping_patience=15):
        """
        Complete training pipeline
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Stop if no improvement for this many epochs
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("="*60 + "\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels, val_probs = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = self.checkpoint_dir / f'best_model_acc{val_acc:.2f}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, self.best_model_path)
                print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break
            
            print("-" * 60)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("="*60 + "\n")
        
        # Save training history
        self.save_history()
        self.plot_training_curves()
    
    def test(self):
        """Comprehensive model evaluation on test set"""
        print("\n" + "="*60)
        print("Evaluating Model on Test Set")
        print("="*60 + "\n")
        
        # Load best model
        if self.best_model_path and self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, landmarks, labels in tqdm(self.test_loader, desc='Testing'):
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)
                
                outputs = self.model(images, landmarks, mode='hybrid')
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        test_acc = accuracy_score(all_labels, all_preds) * 100
        
        print(f"\n{'='*60}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"{'='*60}\n")
        
        # Detailed metrics
        self.compute_metrics(all_labels, all_preds, all_probs)
        
        return test_acc, all_preds, all_labels, all_probs
    
    def compute_metrics(self, labels, preds, probs):
        """Compute comprehensive evaluation metrics"""
        
        # Classification report
        class_names = ['thumbs_up', 'peace', 'fist', 'pointing', 'ok_sign', 
                       'rock', 'one_finger', 'two_fingers', 'three_fingers', 'open_palm']
        
        print("\nClassification Report:")
        print("="*60)
        report = classification_report(labels, preds, target_names=class_names, digits=4)
        print(report)
        
        # Save report
        with open(self.results_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"\nWeighted Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Save metrics
        with open(self.results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Confusion matrix
        self.plot_confusion_matrix(labels, preds, class_names)
        
        # ROC curves (one-vs-rest)
        self.plot_roc_curves(labels, probs, class_names)
    
    def plot_confusion_matrix(self, labels, preds, class_names):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Confusion matrix saved to {self.results_dir / 'confusion_matrix.png'}")
    
    def plot_roc_curves(self, labels, probs, class_names):
        """Plot ROC curves for each class"""
        from sklearn.preprocessing import label_binarize
        
        labels_bin = label_binarize(labels, classes=range(len(class_names)))
        probs = np.array(probs)
        
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            auc_score = roc_auc_score(labels_bin[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to {self.results_dir / 'roc_curves.png'}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(self.history['learning_rates'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(alpha=0.3)
        
        # Overfitting analysis
        gap = np.array(self.history['train_acc']) - np.array(self.history['val_acc'])
        axes[1, 1].plot(gap, linewidth=2, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Train Acc - Val Acc (%)', fontsize=12)
        axes[1, 1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to {self.results_dir / 'training_curves.png'}")
    
    def save_history(self):
        """Save training history"""
        with open(self.results_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"✓ Training history saved to {self.results_dir / 'training_history.json'}")


def main():
    """Main training script"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_root='dataset',
        batch_size=32,
        num_workers=4
    )
    
    # Create model
    model = HybridGestureModel(num_classes=10)
    
    # Create trainer
    trainer = GestureTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device
    )
    
    # Train
    trainer.train(num_epochs=50, early_stopping_patience=15)
    
    # Test
    trainer.test()
    
    print("\n" + "="*60)
    print("Training and Evaluation Complete! ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
