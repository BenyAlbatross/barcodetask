"""
Test script for comparing multiple RT-DETR model sizes
This will train both rtdetr-l and rtdetr-x (if available)
"""

from config import TrainingConfig, AugmentationConfig
from simple_detr_trainer import train_multiple_models

# Create configurations for different model sizes
configs = []

# RT-DETR Large
config_l = TrainingConfig(
    model_name="rtdetr-l.pt",
    experiment_name="rtdetr_l_comparison",
    epochs=20,  # Full training
    batch_size=8,
    img_size=640,
    data_path="barcode_dataset/data.yaml",
    notes="RT-DETR Large model for barcode detection comparison"
)
configs.append(config_l)

# Uncomment if you want to test rtdetr-x as well
# Note: rtdetr-x is much larger and may not fit in 4GB VRAM
# config_x = TrainingConfig(
#     model_name="rtdetr-x.pt",
#     experiment_name="rtdetr_x_comparison",
#     epochs=20,
#     batch_size=4,  # Smaller batch for larger model
#     img_size=640,
#     data_path="barcode_dataset/data.yaml",
#     notes="RT-DETR XLarge model for barcode detection comparison"
# )
# configs.append(config_x)

print("="*60)
print("RT-DETR MODEL COMPARISON TEST")
print("="*60)
print(f"Number of models to train: {len(configs)}")
for i, cfg in enumerate(configs, 1):
    print(f"\n{i}. {cfg.model_name}")
    print(f"   - Epochs: {cfg.epochs}")
    print(f"   - Batch size: {cfg.batch_size}")
    print(f"   - Experiment: {cfg.experiment_name}")
print("="*60)

# Run comparison
results = train_multiple_models(configs, project_name="barcode-detr-comparison")

print("\n✅ All models trained successfully!")
print("📊 Check your W&B dashboard to compare the runs")
