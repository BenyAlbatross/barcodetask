import sys
sys.path.append('..')

from src.config import TrainingConfig, AugmentationConfig
from src.train_with_wandb import run_experiment

def create_experiments():
    """Define all augmentation experiments to run"""
    
    experiments = []
    
    # Experiment 1: Baseline (minimal augmentation)
    experiments.append(TrainingConfig(
        experiment_name="exp01_baseline",
        model_name="yolov8n.pt",
        epochs=50,
        augmentation=AugmentationConfig(
            rotation_degrees=0,
            mosaic=0,
            mixup=0,
            blur=0,
        ),
        notes="Baseline with minimal augmentation"
    ))
    
    # Experiment 2: Add rotation (barcodes may be tilted)
    experiments.append(TrainingConfig(
        experiment_name="exp02_rotation",
        model_name="yolov8n.pt",
        epochs=50,
        augmentation=AugmentationConfig(
            rotation_degrees=15,
            mosaic=0,
            mixup=0,
            blur=0,
        ),
        notes="Added rotation augmentation for tilted barcodes"
    ))
    
    # Experiment 3: Add blur (simulates camera shake/focus)
    experiments.append(TrainingConfig(
        experiment_name="exp03_rotation_blur",
        model_name="yolov8n.pt",
        epochs=50,
        augmentation=AugmentationConfig(
            rotation_degrees=15,
            mosaic=0,
            mixup=0,
            blur=0.01,  # Small probability of blur
        ),
        notes="Added blur to simulate poor camera focus"
    ))
    
    # Experiment 4: Add perspective transforms
    experiments.append(TrainingConfig(
        experiment_name="exp04_rotation_blur_perspective",
        model_name="yolov8n.pt",
        epochs=50,
        augmentation=AugmentationConfig(
            rotation_degrees=15,
            perspective=0.0005,
            mosaic=0,
            mixup=0,
            blur=0.01,
        ),
        notes="Added perspective for different camera angles"
    ))
    
    # Experiment 5: Add mosaic (multi-barcode scenes)
    experiments.append(TrainingConfig(
        experiment_name="exp05_full_augmentation",
        model_name="yolov8n.pt",
        epochs=50,
        augmentation=AugmentationConfig(
            rotation_degrees=15,
            perspective=0.0005,
            mosaic=1.0,
            mixup=0,
            blur=0.01,
            scale=0.5,
            translate=0.1,
        ),
        notes="Full augmentation pipeline"
    ))
    
    # Experiment 6: Best augmentation with larger model
    experiments.append(TrainingConfig(
        experiment_name="exp06_full_aug_yolov8s",
        model_name="yolov8s.pt",
        epochs=50,
        augmentation=AugmentationConfig(
            rotation_degrees=15,
            perspective=0.0005,
            mosaic=1.0,
            blur=0.01,
            scale=0.5,
            translate=0.1,
        ),
        notes="Best augmentation with YOLOv8-small"
    ))
    
    # Experiment 7: Higher resolution
    experiments.append(TrainingConfig(
        experiment_name="exp07_full_aug_highres",
        model_name="yolov8s.pt",
        epochs=50,
        img_size=1280,  # Higher resolution for small barcodes
        batch_size=8,   # Reduce batch size for memory
        augmentation=AugmentationConfig(
            rotation_degrees=15,
            perspective=0.0005,
            mosaic=1.0,
            blur=0.01,
            scale=0.5,
            translate=0.1,
        ),
        notes="High resolution for detecting small barcodes"
    ))
    
    return experiments

def run_all_experiments():
    """Run all experiments sequentially"""
    
    experiments = create_experiments()
    results = []
    
    for i, config in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Running Experiment {i+1}/{len(experiments)}: {config.experiment_name}")
        print(f"{'='*60}\n")
        
        try:
            model, metrics = run_experiment(config)
            results.append({
                'name': config.experiment_name,
                'mAP50': metrics.box.map50,
                'mAP50-95': metrics.box.map,
            })
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}\n")
    
    for result in results:
        print(f"{result['name']}: mAP@0.5={result['mAP50']:.4f}, mAP@0.5:0.95={result['mAP50-95']:.4f}")

if __name__ == "__main__":
    # Login to wandb
    import wandb
    wandb.login()
    
    # Run experiments
    run_all_experiments()