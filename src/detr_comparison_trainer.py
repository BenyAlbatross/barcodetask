import wandb
from ultralytics import RTDETR
import torch
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from config import TrainingConfig, AugmentationConfig


@dataclass
class ModelComparisonResult:
    """Results for a single model in the comparison"""
    model_name: str
    model_size: str
    training_time: float
    best_mAP50: float
    best_mAP50_95: float
    final_mAP50: float
    final_mAP50_95: float
    test_mAP50: float
    test_mAP50_95: float
    test_precision: float
    test_recall: float
    model_path: str
    training_metrics: Dict


class BarcodeDetectionExperiment:
    """Single RT-DETR model training experiment with wandb logging"""

    def __init__(self, config: TrainingConfig, project_name: str = "barcode-detection-detr"):
        self.config = config
        self.project_name = project_name
        self.run = None

    def initialize_wandb(self):
        """Initialize wandb with comprehensive config tracking"""
        self.run = wandb.init(
            project=self.project_name,
            name=self.config.experiment_name,
            config=self.config.to_dict(),
            notes=self.config.notes,
            tags=self._generate_tags(),
            save_code=True
        )

        # Log system info
        wandb.config.update({
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__,
        })

    def _generate_tags(self) -> list:
        """Generate meaningful tags for filtering experiments"""
        tags = ['rtdetr']

        # Model size tag
        if 'l' in self.config.model_name:
            tags.append('large')
        elif 'x' in self.config.model_name:
            tags.append('xlarge')

        # Augmentation tags
        aug = self.config.augmentation
        if aug.mosaic > 0:
            tags.append('mosaic')
        if aug.mixup > 0:
            tags.append('mixup')
        if aug.blur > 0:
            tags.append('blur')
        if aug.rotation_degrees > 0:
            tags.append('rotation')

        # Image size tag
        tags.append(f'img{self.config.img_size}')

        return tags

    def train(self):
        """Train model with wandb logging"""

        # Initialize RT-DETR model
        model = RTDETR(self.config.model_name)

        # Log model architecture
        wandb.watch(model.model, log='all', log_freq=100)

        # Prepare training arguments
        aug = self.config.augmentation
        train_args = {
            'data': 'barcode_dataset/data.yaml',  # Hardcoded path to data.yaml
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

            # Logging
            'project': 'runs/detect',
            'name': self.config.experiment_name,
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'save_period': 10,
        }

        # Train with custom callback for wandb
        results = model.train(**train_args)

        return model, results

    def evaluate_on_test(self, model):
        """Evaluate on test set and log comprehensive metrics"""

        # Run validation on test set
        test_metrics = model.val(
            data='barcode_dataset/data.yaml',  # Hardcoded path to data.yaml
            split='test',
            batch=1,
            imgsz=self.config.img_size,
        )

        # Log test metrics
        wandb.log({
            'test/mAP50': test_metrics.box.map50,
            'test/mAP50-95': test_metrics.box.map,
            'test/precision': test_metrics.box.mp,
            'test/recall': test_metrics.box.mr,
        })

        # Log per-class metrics if available
        if hasattr(test_metrics.box, 'maps'):
            for i, class_map in enumerate(test_metrics.box.maps):
                wandb.log({f'test/mAP50_class_{i}': class_map})

        return test_metrics

    def finish(self):
        """Clean up and finish wandb run"""
        if self.run:
            wandb.finish()


class RTDETRComparisonTrainer:
    """
    Class to train and compare multiple RT-DETR model sizes
    for bounding box detection tasks.
    """

    def __init__(self,
                 base_config: TrainingConfig,
                 model_sizes: List[str] = None,
                 project_name: str = "rtdetr-comparison"):
        """
        Initialize the comparison trainer.

        Args:
            base_config: Base training configuration
            model_sizes: List of RT-DETR model sizes to test (e.g., ['l', 'x'])
            project_name: Wandb project name
        """
        self.base_config = base_config
        self.project_name = project_name

        # Default model sizes for RT-DETR (typically only 'l' and 'x' are available)
        if model_sizes is None:
            self.model_sizes = ['l', 'x']
        else:
            self.model_sizes = model_sizes

        self.results: List[ModelComparisonResult] = []
        self.comparison_run = None

        # Generate all model combinations
        self.model_configs = self._generate_model_configs()

    def _generate_model_configs(self) -> List[Tuple[str, str]]:
        """Generate all RT-DETR model configurations"""
        configs = []
        for size in self.model_sizes:
            model_name = f"rtdetr-{size}.pt"
            configs.append(('rtdetr', size, model_name))
        return configs

    def _is_model_available(self, model_name: str) -> bool:
        """Check if an RT-DETR model is available for download"""
        try:
            # Try to load the model to check availability
            model = RTDETR(model_name)
            return True
        except Exception as e:
            print(f"Model {model_name} not available: {e}")
            return False

    def initialize_comparison_run(self):
        """Initialize wandb run for the entire comparison"""
        self.comparison_run = wandb.init(
            project=self.project_name,
            name=f"rtdetr_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": "RT-DETR",
                "model_sizes": self.model_sizes,
                "base_config": self.base_config.to_dict(),
                "total_models": len(self.model_configs)
            },
            notes="Comparing RT-DETR model sizes on barcode detection"
        )

    def train_single_model(self, version: str, size: str, model_name: str) -> Optional[ModelComparisonResult]:
        """Train a single RT-DETR model and return results"""

        if not self._is_model_available(model_name):
            print(f"Skipping {model_name} - not available")
            return None

        print(f"🚀 Training {model_name}")

        # Create config for this specific model
        if isinstance(self.base_config, dict):
            base_dict = self.base_config.copy()
        else:
            base_dict = self.base_config.to_dict()
        base_dict.pop('model_name', None)  # Remove model_name to avoid conflict
        base_dict.pop('experiment_name', None)  # Remove experiment_name to avoid conflict
        
        # Handle augmentation separately to ensure it's AugmentationConfig object
        aug_dict = base_dict.pop('augmentation', None)
        config = TrainingConfig(
            model_name=model_name,
            experiment_name=f"rtdetr_{size}_{datetime.now().strftime('%H%M%S')}",
            **base_dict
        )
        if aug_dict:
            config.augmentation = AugmentationConfig(**aug_dict)

        # Initialize experiment
        experiment = BarcodeDetectionExperiment(config, project_name=self.project_name)

        try:
            # Track training time
            start_time = datetime.now()

            # Initialize wandb for this model
            experiment.initialize_wandb()

            # Train model
            model, train_results = experiment.train()

            training_time = (datetime.now() - start_time).total_seconds()

            # Evaluate on test set
            test_metrics = experiment.evaluate_on_test(model)

            # Create result object
            result = ModelComparisonResult(
                model_name=model_name,
                model_size=size,
                training_time=training_time,
                best_mAP50=train_results.results_dict.get('metrics/mAP50(B)', 0),
                best_mAP50_95=train_results.results_dict.get('metrics/mAP50-95(B)', 0),
                final_mAP50=test_metrics.box.map50,
                final_mAP50_95=test_metrics.box.map,
                test_mAP50=test_metrics.box.map50,
                test_mAP50_95=test_metrics.box.map,
                test_precision=test_metrics.box.mp,
                test_recall=test_metrics.box.mr,
                model_path=str(model.ckpt_path) if hasattr(model, 'ckpt_path') else "",
                training_metrics=train_results.results_dict if hasattr(train_results, 'results_dict') else {}
            )

            print(f"✅ {model_name} completed - Test mAP@0.5: {result.test_mAP50:.4f}")

            return result

        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}")
            return None

        finally:
            experiment.finish()

    def run_comparison(self) -> List[ModelComparisonResult]:
        """Run training comparison for all model configurations"""

        self.initialize_comparison_run()

        print(f"🔬 Starting RT-DETR model comparison")
        print(f"📊 Models to test: {len(self.model_configs)}")
        print(f"📈 View progress at: {wandb.run.get_url()}")

        for version, size, model_name in self.model_configs:
            result = self.train_single_model(version, size, model_name)
            if result:
                self.results.append(result)

                # Log to comparison run
                wandb.log({
                    f"rtdetr_{size}/training_time": result.training_time,
                    f"rtdetr_{size}/test_mAP50": result.test_mAP50,
                    f"rtdetr_{size}/test_mAP50_95": result.test_mAP50_95,
                    f"rtdetr_{size}/test_precision": result.test_precision,
                    f"rtdetr_{size}/test_recall": result.test_recall,
                })

        # Generate comparison summary
        self._generate_comparison_summary()

        # Finish comparison run
        if self.comparison_run:
            wandb.finish()

        return self.results

    def _generate_comparison_summary(self):
        """Generate and log comparison visualizations"""

        if not self.results:
            return

        # Convert results to DataFrame for analysis
        df = pl.DataFrame([{
            'model': r.model_name,
            'size': r.model_size,
            'training_time': r.training_time,
            'test_mAP50': r.test_mAP50,
            'test_mAP50_95': r.test_mAP50_95,
            'precision': r.test_precision,
            'recall': r.test_recall
        } for r in self.results])

        # Create comparison plots
        self._create_performance_comparison_plot(df)
        self._create_training_time_comparison_plot(df)
        self._create_precision_recall_plot(df)

        # Log summary table (convert to pandas for wandb compatibility)
        summary_table = wandb.Table(dataframe=df.to_pandas())
        wandb.log({"model_comparison_summary": summary_table})

        # Log best performing models using polars operations
        best_map50_row = df.filter(pl.col('test_mAP50') == pl.col('test_mAP50').max()).row(0, named=True)
        best_map50_95_row = df.filter(pl.col('test_mAP50_95') == pl.col('test_mAP50_95').max()).row(0, named=True)
        fastest_training_row = df.filter(pl.col('training_time') == pl.col('training_time').min()).row(0, named=True)

        wandb.summary.update({
            "best_mAP50_model": best_map50_row['model'],
            "best_mAP50_score": best_map50_row['test_mAP50'],
            "best_mAP50_95_model": best_map50_95_row['model'],
            "best_mAP50_95_score": best_map50_95_row['test_mAP50_95'],
            "fastest_training_model": fastest_training_row['model'],
            "fastest_training_time": fastest_training_row['training_time'],
            "total_models_tested": len(self.results)
        })

        print("\n📊 Comparison Summary:")
        print(f"   Best mAP@0.5: {best_map50_row['model']} ({best_map50_row['test_mAP50']:.4f})")
        print(f"   Best mAP@0.5:0.95: {best_map50_95_row['model']} ({best_map50_95_row['test_mAP50_95']:.4f})")
        print(f"   Fastest training: {fastest_training_row['model']} ({fastest_training_row['training_time']:.1f}s)")

    def _create_performance_comparison_plot(self, df):
        """Create performance comparison plot"""
        plt.figure(figsize=(12, 6))

        # Convert to pandas for plotting
        df_pd = df.to_pandas()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # mAP@0.5 bar plot
        ax1.bar(df_pd['model'], df_pd['test_mAP50'], color='coral')
        ax1.set_title('Test mAP@0.5 by RT-DETR Model Size')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('mAP@0.5')
        ax1.tick_params(axis='x', rotation=45)

        # mAP@0.5:0.95 bar plot
        ax2.bar(df_pd['model'], df_pd['test_mAP50_95'], color='skyblue')
        ax2.set_title('Test mAP@0.5:0.95 by RT-DETR Model Size')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('mAP@0.5:0.95')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        wandb.log({"performance_comparison": wandb.Image(plt)})
        plt.close()

    def _create_training_time_comparison_plot(self, df):
        """Create training time comparison plot"""
        plt.figure(figsize=(10, 6))

        # Convert to pandas for seaborn barplot
        df_pd = df.to_pandas()
        ax = sns.barplot(data=df_pd, x='model', y='training_time', palette='viridis')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Training Time Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Training Time (seconds)')

        plt.tight_layout()
        wandb.log({"training_time_comparison": wandb.Image(plt)})
        plt.close()

    def _create_precision_recall_plot(self, df):
        """Create precision-recall scatter plot"""
        plt.figure(figsize=(10, 8))

        # Use Polars Series directly with matplotlib
        scatter = plt.scatter(
            x=df['recall'],
            y=df['precision'],
            s=df['test_mAP50'] * 1000,  # Size by mAP@0.5
            c=df['training_time'],       # Color by training time
            cmap='viridis',
            alpha=0.7
        )

        # Add model labels using Polars iteration
        for row in df.iter_rows(named=True):
            plt.annotate(
                row['model'],
                (row['recall'], row['precision']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

        plt.colorbar(scatter, label='Training Time (s)')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs Recall (bubble size = mAP@0.5)')

        plt.tight_layout()
        wandb.log({"precision_recall_comparison": wandb.Image(plt)})
        plt.close()

    def get_best_model(self, metric: str = 'test_mAP50') -> Optional[ModelComparisonResult]:
        """Get the best performing model by specified metric"""
        if not self.results:
            return None

        if metric == 'test_mAP50':
            return max(self.results, key=lambda x: x.test_mAP50)
        elif metric == 'test_mAP50_95':
            return max(self.results, key=lambda x: x.test_mAP50_95)
        elif metric == 'precision':
            return max(self.results, key=lambda x: x.test_precision)
        elif metric == 'recall':
            return max(self.results, key=lambda x: x.test_recall)
        elif metric == 'training_time':
            return min(self.results, key=lambda x: x.training_time)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def save_comparison_results(self, output_path: str = "rtdetr_comparison_results.json"):
        """Save comparison results to JSON file"""
        results_dict = {
            "comparison_date": datetime.now().isoformat(),
            "model_type": "RT-DETR",
            "models_tested": len(self.results),
            "results": [result.__dict__ for result in self.results]
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        print(f"💾 Results saved to {output_path}")


# Example usage:
"""
Example usage of RTDETRComparisonTrainer:

from config import TrainingConfig, AugmentationConfig
from detr_comparison_trainer import RTDETRComparisonTrainer

# Create base configuration
base_config = TrainingConfig(
    epochs=100,
    batch_size=16,
    img_size=640,
    data_path="barcode_dataset/data.yaml"
)

# Initialize comparison trainer
trainer = RTDETRComparisonTrainer(
    base_config=base_config,
    model_sizes=['l', 'x'],  # RT-DETR available sizes
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
"""
