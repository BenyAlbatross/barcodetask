from ultralytics import RTDETR
import torch
from pathlib import Path
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from config import TrainingConfig


@dataclass
class ModelResult:
    model_name: str
    model_size: str
    test_mAP50: float
    test_mAP50_95: float
    test_precision: float
    test_recall: float
    model_path: str
    training_dir: str


class SimpleDETRTrainer:    
    def __init__(self, config: TrainingConfig, project_name: str = "barcode-detr"):
        self.config = config
        self.project_name = project_name
    
    def train(self) -> ModelResult:
        """Train single model - YOLO handles W&B automatically"""
        
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.config.model_name}")
        print(f"{'='*60}\n")
        
        # Initialize model
        model = RTDETR(self.config.model_name)
        
        # Prepare training arguments
        aug = self.config.augmentation
        
        # Generate experiment name if not provided
        experiment_name = self.config.experiment_name
        if not experiment_name:
            experiment_name = f"{self.config.model_name.replace('.pt', '')}_{datetime.now().strftime('%H%M%S')}"
        
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
        
        # Extract model size from model name (e.g., 'rtdetr-l.pt' -> 'l')
        model_size = self.config.model_name.replace('rtdetr-', '').replace('.pt', '')
        
        # Get the training directory path
        training_dir = f"runs/{self.project_name}/{experiment_name}"
        
        # Create result
        result = ModelResult(
            model_name=self.config.model_name,
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


def train_multiple_models(configs: list, project_name: str = "barcode-detr-comparison"):
    """Train multiple models and compare results"""
    
    results = []
    
    for config in configs:
        trainer = SimpleDETRTrainer(config, project_name)
        result = trainer.train()
        results.append(asdict(result))
    
    # Save comparison
    output_file = f"{project_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<13} | {'Precision':<10} | {'Recall':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['model_name']:<20} | {r['test_mAP50']:<10.4f} | {r['test_mAP50_95']:<13.4f} | {r['test_precision']:<10.4f} | {r['test_recall']:<10.4f}")
    print("="*80)
    print(f"\n💾 Results saved to: {output_file}\n")
    
    return results
