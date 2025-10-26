from dataclasses import dataclass, asdict
from typing import List, Optional
import yaml

@dataclass
class AugmentationConfig:
    """Track all augmentation parameters"""
    # Geometric transforms
    rotation_degrees: float = 15.0
    perspective: float = 0.0
    scale: float = 0.5
    translate: float = 0.1
    shear: float = 0.0
    flipud: float = 0.0  # vertical flip
    fliplr: float = 0.5  # horizontal flip
    
    # Color transforms
    hsv_h: float = 0.015  # hue
    hsv_s: float = 0.7    # saturation
    hsv_v: float = 0.4    # value
    
    # Image quality
    blur: float = 0.0      # motion blur probability
    brightness: float = 0.0
    contrast: float = 0.0
    
    # Advanced
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    
    def to_dict(self):
        return asdict(self)

@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Model
    model_name: str = "yolov8n.pt"
    
    # Training
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 0.01
    
    # Optimizer
    optimizer: str = "SGD"
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # Data
    data_path: str = "data/barcode_data.yaml"
    
    # Augmentation
    augmentation: AugmentationConfig = None
    
    # Experiment tracking
    experiment_name: str = "baseline"
    notes: str = ""
    
    def __post_init__(self):
        if self.augmentation is None:
            self.augmentation = AugmentationConfig()
    
    def to_dict(self):
        config_dict = asdict(self)
        config_dict['augmentation'] = self.augmentation.to_dict()
        return config_dict