"""
Model loader singleton for BarcodeDetector V2
Ensures model is loaded only once and reused across requests
"""

import torch
from pathlib import Path
import sys

# Add parent directory to path to import detector
sys.path.append(str(Path(__file__).parent.parent / "src"))
from detector_v2 import BarcodeDetectorV2


class ModelLoader:
    """Singleton pattern for model loading"""
    
    _instance = None
    _model = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(
        self, 
        checkpoint_path: str = "checkpoints_v2/latest_model.pt",
        grid_size: int = 7,
        num_boxes: int = 2,
        force_cpu: bool = True
    ):
        """
        Load the trained model from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            grid_size: Grid size for detection (default: 7)
            num_boxes: Number of boxes per grid cell (default: 2)
            force_cpu: Force CPU usage even if GPU available
        
        Returns:
            Tuple of (model, device)
        """
        if self._model is not None:
            print("Model already loaded, returning cached instance")
            return self._model, self._device
        
        # Determine device
        if force_cpu:
            self._device = torch.device('cpu')
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model on device: {self._device}")
        
        # Initialize model
        self._model = BarcodeDetectorV2(
            grid_size=grid_size,
            num_boxes_per_cell=num_boxes,
            pretrained=False  # Don't need ImageNet weights for inference
        ).to(self._device)
        
        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()
        
        # Log checkpoint info
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        print(f"✓ Loaded checkpoint from epoch {epoch}")
        if val_loss != 'unknown':
            print(f"  Validation loss: {val_loss:.4f}")
        
        return self._model, self._device
    
    def get_model(self):
        """Get cached model and device"""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model, self._device


# Global instance
model_loader = ModelLoader()
