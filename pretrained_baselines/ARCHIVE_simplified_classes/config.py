from dataclasses import dataclass, asdict
from typing import List, Optional
import yaml

@dataclass
class AugmentationConfig:
    """Track all augmentation parameters - ALL VALUES MUST BE EXPLICITLY SET"""
    # Geometric transforms
    rotation_degrees: float = 0.0
    perspective: float = 0.0
    scale: float = 0.0
    translate: float = 0.0
    shear: float = 0.0
    flipud: float = 0.0  # vertical flip
    fliplr: float = 0.0  # horizontal flip
    
    # Color transforms
    hsv_h: float = 0.0  # hue
    hsv_s: float = 0.0  # saturation
    hsv_v: float = 0.0  # value
    
    # Image quality
    blur: float = 0.0      # motion blur probability
    brightness: float = 0.0
    contrast: float = 0.0
    
    # Advanced
    mosaic: float = 0.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    
    def to_dict(self):
        return asdict(self)
    
    @staticmethod
    def get_yolo_default():
        """Get YOLO's recommended default augmentations"""
        return AugmentationConfig(
            rotation_degrees=15.0,
            scale=0.5,
            translate=0.1,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            mosaic=1.0,
        )
    
    @staticmethod
    def get_light():
        """Get light augmentations for simple tasks"""
        return AugmentationConfig(
            rotation_degrees=10.0,
            fliplr=0.5,
            hsv_v=0.2,
        )
    
    @staticmethod
    def get_none():
        """No augmentation - all zeros (already the default)"""
        return AugmentationConfig()

@dataclass
class TrainingConfig:
    # Model
    model_name: str = None
    
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
    
    # Augmentation - MUST be explicitly set (no hidden defaults)
    augmentation: AugmentationConfig = None
    
    # Experiment tracking
    experiment_name: str = None # Set in either train script or randomly assigned in trainer class
    notes: str = "" # Can be set in train script
    
    def __post_init__(self):
        # Force user to explicitly set augmentation
        if self.augmentation is None:
            raise ValueError(
                "Augmentation must be explicitly specified! Use:\n"
                "  - AugmentationConfig.get_yolo_default() for YOLO defaults\n"
                "  - AugmentationConfig.get_light() for light augmentation\n"
                "  - AugmentationConfig.get_none() for no augmentation\n"
                "  - AugmentationConfig(...) for custom settings"
            )
    
    def to_dict(self):
        config_dict = asdict(self)
        config_dict['augmentation'] = self.augmentation.to_dict()
        return config_dict