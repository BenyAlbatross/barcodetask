"""
Script for RT-DETR training
"""

from config import TrainingConfig, AugmentationConfig
from detr_trainer_class import SimpleDETRTrainer

# Create configuration with EXPLICIT augmentation settings
config = TrainingConfig(
    model_name="rtdetr-l",
    epochs=1,
    batch_size=8,
    img_size=640,
    data_path="barcode_dataset/data.yaml",
    augmentation=AugmentationConfig.get_yolo_default(),  # EXPLICIT: Use YOLO defaults
    # Alternative options:
    # augmentation=AugmentationConfig.get_none(),  # No augmentation
    # augmentation=AugmentationConfig.get_light(),  # Light augmentation
    notes="Baseline RT-DETR test"
)

print("="*60)
print("TRAINING RT-DETR")
print("="*60)
print(f"Model: {config.model_name}")
print(f"Epochs: {config.epochs}")
print(f"Batch size: {config.batch_size}")
print(f"Image size: {config.img_size}")
print("="*60)

# Train the model
trainer = SimpleDETRTrainer(
    config=config,
    # project_name set in class
)

result = trainer.train()

print("\n" + "="*60)
print("TEST COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Results saved to: {result.training_dir}")
print(f"Best weights: {result.model_path}")
print("\nCheck W&B dashboard for training curves and metrics!")
print("="*60)
