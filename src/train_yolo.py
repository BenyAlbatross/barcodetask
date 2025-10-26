from config import TrainingConfig, AugmentationConfig
from yolo_comparison_trainer import YOLOComparisonTrainer

# Create base configuration
base_config = TrainingConfig(
    epochs=50,
    batch_size=16, # 4GB VRAM
    img_size=640,
    data_path="barcode_dataset/data.yaml"  # Updated to use the correct YAML file
)

# Initialize comparison trainer
trainer = YOLOComparisonTrainer(
    base_config=base_config,
    yolo_versions=['v10', 'v12'],
    model_sizes=['s'],
    project_name="barcode-detection-comparison"
)

# Run comparison
results = trainer.run_comparison()

# Get best model by different metrics
best_map50 = trainer.get_best_model('test_mAP50')
best_precision = trainer.get_best_model('precision')
fastest = trainer.get_best_model('training_time')

# Save results
trainer.save_comparison_results("comparison_results.json")