"""
Inspired by YOLO v1 but adapted for barcode detection
"""

import torch
import torch.nn as nn
import torchvision.models as models

class BarcodeDetector(nn.Module):
    """
    1. Use pretrained ResNet as backbone (transfer learning)
    2. Simple grid-based detection (like YOLO v1)
    3. Single class -> simplified loss
    4. Direct bbox regression (no anchors initially)
    """
    
    def __init__(self, grid_size=7, num_boxes_per_cell=2, pretrained=True):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes_per_cell
        
        # Backbone: ResNet18 (lightweight, pretrained)
        resnet = models.resnet18(weights='DEFAULT')
        # Remove final FC layer, keep features
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Output: (batch, 512, H/32, W/32)
        
        # Adaptive pooling to fixed grid size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        # Output: (batch, 512, grid_size, grid_size)
        
        # Detection head
        # Each grid cell predicts: B boxes × (x, y, w, h, confidence) + 1 class prob
        self.num_outputs = num_boxes_per_cell * 5 + 1  # 2*5+1 = 11
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, self.num_outputs, kernel_size=1)
        )
        # Output: (batch, 11, grid_size, grid_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, H, W) - input images
        Returns:
            predictions: (batch, grid_size, grid_size, num_outputs)
        """
        # Extract features
        features = self.backbone(x)  # (B, 512, H/32, W/32)
        
        # Pool to grid size
        features = self.adaptive_pool(features)  # (B, 512, S, S)
        
        # Predict
        predictions = self.detection_head(features)  # (B, 11, S, S)
        
        # Permute to (B, S, S, 11) for easier processing
        predictions = predictions.permute(0, 2, 3, 1)
        
        return predictions