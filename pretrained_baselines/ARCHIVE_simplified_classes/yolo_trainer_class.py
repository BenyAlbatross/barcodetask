from ultralytics import YOLO
import torch
from pathlib import Path
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from config import TrainingConfig


@dataclass
class ModelResult:
    """Simple result storage for a single model"""
    model_name: str
    model_version: str
    model_size: str
    test_mAP50: float
    test_mAP50_95: float
    test_precision: float
    test_recall: float
    model_path: str
    training_dir: str


class SimpleYOLOTrainer:
    """Simplified YOLO trainer - let YOLO handle W&B automatically"""
    
    def __init__(self, config: TrainingConfig, project_name: str = "barcode-yolo"):
        self.config = config
        self.project_name = project_name
    
    def train(self) -> ModelResult:
        """Train single model - YOLO handles W&B automatically"""
        
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.config.model_name}")
        print(f"{'='*60}\n")
        
        # Initialize model - remove .pt extension if present
        model_name = self.config.model_name.replace('.pt', '')
        model = YOLO(model_name)
        
        # Prepare training arguments
        aug = self.config.augmentation
        
        # Generate experiment name if not provided
        experiment_name = self.config.experiment_name
        if not experiment_name:
            experiment_name = f"{model_name}_{datetime.now().strftime('%H%M%S')}"
        
        train_args = {
            'data': 'barcode_dataset/data.yaml',
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.img_size,
            'lr0': self.config.learning_rate,
            'optimizer': self.config.optimizer,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            
            # Augmentation parameters
            'degrees': aug.rotation_degrees,
            'translate': aug.translate,
            'scale': aug.scale,
            'shear': aug.shear,
            'perspective': aug.perspective,
            'flipud': aug.flipud,
            'fliplr': aug.fliplr,
            'mosaic': aug.mosaic,
            'mixup': aug.mixup,
            'copy_paste': aug.copy_paste,
            'hsv_h': aug.hsv_h,
            'hsv_s': aug.hsv_s,
            'hsv_v': aug.hsv_v,
            
            # Local storage
            'project': f'runs/{self.project_name}',
            'name': experiment_name,
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'save_period': 10,
        }
        
        # Train - YOLO will auto-log to W&B if wandb is installed
        print("📈 Starting training...")
        train_results = model.train(**train_args)
        
        # Evaluate on test set
        print(f"\n📊 Evaluating on test set...")
        test_metrics = model.val(
            data='barcode_dataset/data.yaml',
            split='test',
            batch=1,
            imgsz=self.config.img_size,
        )
        
        # Extract model version and size from model name
        # e.g., 'yolo11n' -> version='11', size='n'
        # e.g., 'yolov10s' -> version='v10', size='s'
        model_version = ''
        model_size = ''
        
        if 'yolo11' in model_name or 'yolo12' in model_name:
            # YOLO11 or YOLO12 format
            model_version = model_name[4:6]  # '11' or '12'
            model_size = model_name[6:]  # 'n', 's', 'm', etc.
        elif 'yolov' in model_name:
            # YOLOv8, YOLOv10 format
            parts = model_name.replace('yolov', '').replace('yolo', '')
            if parts[0].isdigit():
                # Extract version number
                version_end = 1
                if len(parts) > 1 and parts[1].isdigit():
                    version_end = 2
                model_version = 'v' + parts[:version_end]
                model_size = parts[version_end:]
        
        # Get the training directory path
        training_dir = f"runs/{self.project_name}/{experiment_name}"
        
        # Create result
        result = ModelResult(
            model_name=self.config.model_name,
            model_version=model_version,
            model_size=model_size,
            test_mAP50=float(test_metrics.box.map50),
            test_mAP50_95=float(test_metrics.box.map),
            test_precision=float(test_metrics.box.mp),
            test_recall=float(test_metrics.box.mr),
            model_path=str(Path(training_dir) / "weights" / "best.pt"),
            training_dir=training_dir
        )
        
        # Print results
        print(f"\n{'='*60}")
        print(f"✅ {self.config.model_name} Training Complete!")
        print(f"{'='*60}")
        print(f"📊 Test Set Results:")
        print(f"   mAP@0.5:      {result.test_mAP50:.4f}")
        print(f"   mAP@0.5:0.95: {result.test_mAP50_95:.4f}")
        print(f"   Precision:    {result.test_precision:.4f}")
        print(f"   Recall:       {result.test_recall:.4f}")
        print(f"💾 Saved to:     {result.model_path}")
        print(f"{'='*60}\n")
        
        return result


def train_multiple_models(configs: list, project_name: str = "barcode-yolo-comparison"):
    """Train multiple models and compare results"""
    
    results = []
    
    for config in configs:
        trainer = SimpleYOLOTrainer(config, project_name)
        result = trainer.train()
        results.append(asdict(result))
    
    # Save comparison
    output_file = f"{project_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison summary
    print("\n" + "="*90)
    print("COMPARISON SUMMARY")
    print("="*90)
    print(f"{'Model':<20} | {'Version':<8} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<13} | {'Precision':<10} | {'Recall':<10}")
    print("-"*90)
    for r in results:
        print(f"{r['model_name']:<20} | {r['model_version']:<8} | {r['test_mAP50']:<10.4f} | {r['test_mAP50_95']:<13.4f} | {r['test_precision']:<10.4f} | {r['test_recall']:<10.4f}")
    print("="*90)
    print(f"\n💾 Results saved to: {output_file}\n")
    
    return results
