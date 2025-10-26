from config import TrainingConfig, AugmentationConfig
from detr_comparison_trainer import RTDETRComparisonTrainer

# Create base configuration
base_config = TrainingConfig(
    epochs=100,
    batch_size=4, # 4GB VRAM
    img_size=640,
    data_path="barcode_dataset/data.yaml"  # Updated to use the correct YAML file
)

# Initialize comparison trainer
trainer = RTDETRComparisonTrainer(
    base_config=base_config,
    model_sizes=['l'],  # RT-DETR available sizes: 'l' (large) and 'x' (extra-large)
    project_name="barcode-detection-rtdetr-comparison"
)

# Run comparison
results = trainer.run_comparison()

# Get best model by different metrics
best_map50 = trainer.get_best_model('test_mAP50')
best_precision = trainer.get_best_model('precision')
fastest = trainer.get_best_model('training_time')

# Save results
trainer.save_comparison_results("rtdetr_comparison_results.json")
