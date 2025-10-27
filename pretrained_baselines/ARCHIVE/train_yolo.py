"""
Simplified YOLO training script
"""

from config import TrainingConfig, AugmentationConfig
from yolo_trainer_class import SimpleYOLOTrainer

# Create configuration with EXPLICIT augmentation settings
config = TrainingConfig(
    model_name="yolo12n",  # Options: yolo11n, yolo11s, yolo12s, yolov10s, etc.
    epochs=1,
    batch_size=8,  # 4GB VRAM
    img_size=640,
    data_path="barcode_dataset/data.yaml",
    augmentation=AugmentationConfig.get_yolo_default(),  # EXPLICIT: Use YOLO defaults
    # Alternative options:
    # augmentation=AugmentationConfig.get_none(),  # No augmentation
    # augmentation=AugmentationConfig.get_light(),  # Light augmentation
)

print("="*60)
print("TRAINING YOLO MODEL")
print("="*60)
print(f"Model: {config.model_name}")
print(f"Epochs: {config.epochs}")
print(f"Batch size: {config.batch_size}")
print(f"Image size: {config.img_size}")
print("="*60)

# Train the model
trainer = SimpleYOLOTrainer(
    config=config,
    # project_name set in class
)

result = trainer.train()

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Results saved to: {result.training_dir}")
print(f"Best weights: {result.model_path}")
print("\nCheck W&B dashboard for training curves and metrics!")
print("="*60)