"""
Test script for comparing multiple YOLO model versions and sizes
This will train different YOLO variants and compare them
"""

from config import TrainingConfig
from yolo_trainer_class import train_multiple_models

# Create configurations for different YOLO models
configs = []

# YOLO11 Nano
config_11n = TrainingConfig(
    model_name="yolo11n",
    experiment_name="yolo11n_comparison",
    epochs=20,
    batch_size=16,  # Nano is smaller, can use bigger batch
    img_size=640,
    data_path="barcode_dataset/data.yaml",
    notes="YOLO11 Nano model for barcode detection comparison"
)
configs.append(config_11n)

# YOLO11 Small
config_11s = TrainingConfig(
    model_name="yolo11s",
    experiment_name="yolo11s_comparison",
    epochs=20,
    batch_size=8,
    img_size=640,
    data_path="barcode_dataset/data.yaml",
    notes="YOLO11 Small model for barcode detection comparison"
)
configs.append(config_11s)

# Uncomment to add more models:
# YOLO12 Small
# config_12s = TrainingConfig(
#     model_name="yolo12s",
#     experiment_name="yolo12s_comparison",
#     epochs=20,
#     batch_size=8,
#     img_size=640,
#     data_path="barcode_dataset/data.yaml",
#     notes="YOLO12 Small model for barcode detection comparison"
# )
# configs.append(config_12s)

# YOLOv10 Small
# config_v10s = TrainingConfig(
#     model_name="yolov10s",
#     experiment_name="yolov10s_comparison",
#     epochs=20,
#     batch_size=8,
#     img_size=640,
#     data_path="barcode_dataset/data.yaml",
#     notes="YOLOv10 Small model for barcode detection comparison"
# )
# configs.append(config_v10s)

print("="*60)
print("YOLO MODEL COMPARISON TEST")
print("="*60)
print(f"Number of models to train: {len(configs)}")
for i, cfg in enumerate(configs, 1):
    print(f"\n{i}. {cfg.model_name}")
    print(f"   - Epochs: {cfg.epochs}")
    print(f"   - Batch size: {cfg.batch_size}")
    print(f"   - Experiment: {cfg.experiment_name}")
print("="*60)

# Run comparison
results = train_multiple_models(configs, project_name="barcode-yolo-comparison")

print("\n✅ All models trained successfully!")
print("📊 Check your W&B dashboard to compare the runs")
