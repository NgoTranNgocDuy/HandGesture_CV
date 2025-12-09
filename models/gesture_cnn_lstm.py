"""
Deep Learning Model for Hand Gesture Recognition
Hybrid CNN-LSTM Architecture for Temporal Gesture Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GestureCNN(nn.Module):
    """
    Convolutional Neural Network for spatial feature extraction from hand images
    Based on ResNet18 architecture with custom classification head
    """
    def __init__(self, num_classes=10, feature_dim=512):
        super(GestureCNN, self).__init__()
        
        # Use pretrained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Remove the final fc layer
        self.backbone.fc = nn.Identity()
        
        # Custom feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        Returns:
            features: Feature vector of shape (batch_size, feature_dim)
            logits: Classification logits of shape (batch_size, num_classes)
        """
        # Extract features using ResNet backbone
        x = self.backbone(x)  # (batch_size, 512)
        
        # Additional feature extraction
        features = self.feature_extractor(x)  # (batch_size, feature_dim)
        
        # Classification
        logits = self.classifier(features)  # (batch_size, num_classes)
        
        return features, logits


class GestureLSTM(nn.Module):
    """
    LSTM Network for temporal gesture sequence modeling
    Processes sequences of hand landmarks over time
    """
    def __init__(self, input_dim=63, hidden_dim=256, num_layers=2, num_classes=10):
        super(GestureLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Actual sequence lengths for packed sequences
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_dim*2)
        
        # Classification
        logits = self.fc(context)  # (batch_size, num_classes)
        
        return logits, attention_weights


class HybridGestureModel(nn.Module):
    """
    Hybrid CNN-LSTM Model for robust gesture recognition
    Combines spatial (CNN) and temporal (LSTM) features
    """
    def __init__(self, num_classes=10, cnn_feature_dim=512, lstm_hidden_dim=256):
        super(HybridGestureModel, self).__init__()
        
        # CNN for image features
        self.cnn = GestureCNN(num_classes=num_classes, feature_dim=cnn_feature_dim)
        
        # LSTM for landmark sequences (21 landmarks * 3 coordinates = 63)
        self.lstm = GestureLSTM(
            input_dim=63, 
            hidden_dim=lstm_hidden_dim, 
            num_classes=num_classes
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image=None, landmarks=None, mode='hybrid'):
        """
        Forward pass with flexible input modes
        Args:
            image: Image tensor (batch_size, 3, 224, 224)
            landmarks: Landmark sequence (batch_size, seq_len, 63)
            mode: 'cnn', 'lstm', or 'hybrid'
        Returns:
            logits: Final classification logits
        """
        if mode == 'cnn':
            _, cnn_logits = self.cnn(image)
            return cnn_logits
        
        elif mode == 'lstm':
            lstm_logits, _ = self.lstm(landmarks)
            return lstm_logits
        
        elif mode == 'hybrid':
            # Get predictions from both models
            _, cnn_logits = self.cnn(image)
            lstm_logits, _ = self.lstm(landmarks)
            
            # Concatenate logits
            combined = torch.cat([cnn_logits, lstm_logits], dim=1)
            
            # Fused prediction
            fused_logits = self.fusion(combined)
            
            return fused_logits
        
        else:
            raise ValueError(f"Unknown mode: {mode}")


class GestureEnsemble(nn.Module):
    """
    Ensemble of multiple gesture recognition models
    """
    def __init__(self, num_classes=10):
        super(GestureEnsemble, self).__init__()
        
        self.model1 = HybridGestureModel(num_classes=num_classes)
        self.model2 = HybridGestureModel(num_classes=num_classes)
        self.model3 = HybridGestureModel(num_classes=num_classes)
        
    def forward(self, image, landmarks):
        """Average ensemble predictions"""
        logits1 = self.model1(image, landmarks, mode='hybrid')
        logits2 = self.model2(image, landmarks, mode='hybrid')
        logits3 = self.model3(image, landmarks, mode='hybrid')
        
        return (logits1 + logits2 + logits3) / 3.0


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("=" * 60)
    print("Testing Hand Gesture Recognition Models")
    print("=" * 60)
    
    # Test CNN
    cnn_model = GestureCNN(num_classes=10)
    test_image = torch.randn(4, 3, 224, 224)
    features, logits = cnn_model(test_image)
    print(f"\n✓ CNN Model:")
    print(f"  Input shape: {test_image.shape}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Parameters: {count_parameters(cnn_model):,}")
    
    # Test LSTM
    lstm_model = GestureLSTM(num_classes=10)
    test_landmarks = torch.randn(4, 30, 63)  # 30 frames
    lstm_logits, attention = lstm_model(test_landmarks)
    print(f"\n✓ LSTM Model:")
    print(f"  Input shape: {test_landmarks.shape}")
    print(f"  Output shape: {lstm_logits.shape}")
    print(f"  Attention shape: {attention.shape}")
    print(f"  Parameters: {count_parameters(lstm_model):,}")
    
    # Test Hybrid Model
    hybrid_model = HybridGestureModel(num_classes=10)
    hybrid_logits = hybrid_model(test_image, test_landmarks, mode='hybrid')
    print(f"\n✓ Hybrid Model:")
    print(f"  Output shape: {hybrid_logits.shape}")
    print(f"  Parameters: {count_parameters(hybrid_model):,}")
    
    print("\n" + "=" * 60)
    print("All models initialized successfully! ✓")
    print("=" * 60)
