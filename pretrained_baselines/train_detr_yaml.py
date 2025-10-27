"""
Train RT-DETR using YAML configuration files
"""

from ultralytics import RTDETR

# Choose your configuration
CONFIG = "configs/rtdetr_baseline.yaml"

print("="*60)
print("TRAINING RT-DETR WITH YAML CONFIG")
print("="*60)
print(f"Config file: {CONFIG}")
print("="*60)

# Load model and train
model = RTDETR()  # Will use model specified in YAML
results = model.train(cfg=CONFIG)

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print("Check W&B dashboard for metrics and curves!")
print("="*60)
