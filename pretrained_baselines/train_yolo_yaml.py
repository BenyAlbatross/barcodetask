"""
Train YOLO using YAML configuration files
"""

from ultralytics import YOLO

# Choose your configuration
CONFIG = "configs/yolo_baseline.yaml"  # or "configs/yolo_no_aug.yaml"

print("="*60)
print("TRAINING YOLO WITH YAML CONFIG")
print("="*60)
print(f"Config file: {CONFIG}")
print("="*60)

# Load model specified in YAML and train
model = YOLO("yolo12s")  # Model from YAML config
results = model.train(cfg=CONFIG)

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print("Check W&B dashboard for metrics and curves!")
print("="*60)
