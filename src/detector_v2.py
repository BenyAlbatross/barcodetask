"""
BarcodeDetector V2 - Hybrid Loss Approach
- Sigmoid + BCE for objectness and class predictions (calibrated probabilities)
- MSE for bounding box coordinates (continuous regression)
- Same architecture as V1, just different output activations
"""

import torch
import torch.nn as nn
import torchvision.models as models

class BarcodeDetectorV2(nn.Module):
    """
    V2 improvements:
    - Sigmoid activation for confidence/class outputs
    - Enables BCE loss for proper probability calibration
    - MSE still used for bbox coordinates (regression task)
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
                - bbox coords [0:8]: raw values (no activation)
                - objectness [8:10]: sigmoid activated [0, 1]
                - class [10]: sigmoid activated [0, 1]
        """
        # Extract features
        features = self.backbone(x)  # (B, 512, H/32, W/32)
        
        # Pool to grid size
        features = self.adaptive_pool(features)  # (B, 512, S, S)
        
        # Predict
        predictions = self.detection_head(features)  # (B, 11, S, S)
        
        # Permute to (B, S, S, 11) for easier processing
        predictions = predictions.permute(0, 2, 3, 1)
        
        # Apply selective activations
        # Split into components with CORRECT indices
        # Layout: [box1_xywh(4), box1_conf(1), box2_xywh(4), box2_conf(1), class(1)]
        box1_coords = predictions[..., 0:4]      # Box 1: [x,y,w,h]
        box1_conf = predictions[..., 4:5]        # Box 1 confidence
        box2_coords = predictions[..., 5:9]      # Box 2: [x,y,w,h]
        box2_conf = predictions[..., 9:10]       # Box 2 confidence
        class_probs = predictions[..., 10:11]    # Class probability
        
        # Apply sigmoid to confidence and class (make them probabilities)
        box1_conf = torch.sigmoid(box1_conf)
        box2_conf = torch.sigmoid(box2_conf)
        class_probs = torch.sigmoid(class_probs)
        
        # Concatenate back in same order
        output = torch.cat([box1_coords, box1_conf, box2_coords, box2_conf, class_probs], dim=-1)
        
        return output
