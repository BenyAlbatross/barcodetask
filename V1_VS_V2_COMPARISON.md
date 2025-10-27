# BarcodeDetector V2 - Key Differences from V1

## Architecture Changes

### V1 (detector_v1.py)
- **Output**: Raw logits (no activation functions)
- **Forward pass**: Returns unbounded values
- **Confidence scores**: Can exceed [0, 1] range

### V2 (detector_v2.py)
- **Output**: Selective activation
  - Bbox coordinates [0:8]: Raw values (no activation)
  - Objectness [8:10]: `torch.sigmoid()` activation → [0, 1]
  - Class [10]: `torch.sigmoid()` activation → [0, 1]
- **Forward pass**: Applies sigmoid to confidence/class predictions
- **Confidence scores**: True probabilities [0, 1]

## Loss Function Changes

### V1 (YOLOLoss)
- **All components**: MSE loss
- **Objectness**: `F.mse_loss(pred_conf, target_conf)`
- **Class**: `F.mse_loss(pred_class, target_class)`
- **Bbox**: MSE on coordinates

### V2 (YOLOLossV2) - Hybrid Approach
- **Objectness**: `F.binary_cross_entropy(pred_conf, target_conf)` ← BCE
- **Class**: `F.binary_cross_entropy(pred_class, target_class)` ← BCE
- **Bbox**: MSE on coordinates (unchanged)

**Why this matters:**
- BCE loss expects probabilities [0, 1] → requires sigmoid activation
- MSE loss works on raw values → no activation needed
- BCE provides better calibration for binary classification tasks

## Inference Changes

### V1 (inference.py)
```python
# No activation applied (model outputs raw values)
conf = cell_pred[offset + 4].item()  # Can be >1.0
class_prob = cell_pred[10].item()

# Manual threshold tuning needed (e.g., 0.5 might be too high or low)
if conf < conf_threshold:
    continue
```

### V2 (inference_v2.py)
```python
# Sigmoid already applied in model forward pass
conf = cell_pred[offset + 4].item()  # Always [0, 1]
class_prob = cell_pred[10].item()  # Always [0, 1]

# Threshold is now interpretable (0.5 = 50% confidence)
if conf < conf_threshold:
    continue
```

**Key improvement**: Confidence threshold is now meaningful!
- V1: Threshold is arbitrary (depends on training dynamics)
- V2: Threshold is a true probability (0.5 = 50% confidence)

## Training Script Differences

### Checkpoints
- **V1**: Saves to `checkpoints/`
- **V2**: Saves to `checkpoints_v2/` ← Separate folder!

### Metrics
- **V1**: Basic loss tracking
- **V2**: Comprehensive metrics
  - Precision
  - Recall
  - mAP@0.5
  - mAP@0.75
  - Computed every 5 epochs

### Loss Components
Both track individual loss components, but V2 uses:
- `conf_obj` and `conf_noobj` (BCE-based)
- V1 uses MSE-based confidence loss

## Usage Examples

### Training
```bash
# V1
python src/train.py

# V2
python src/train_v2.py
```

### Inference
```bash
# V1
python src/visualize_single.py [image_path] [threshold]

# V2
python src/visualize_single_v2.py [image_path] [threshold]
```

## When to Use Each Version

### Use V1 if:
- You want the simplest possible implementation
- You're studying YOLO v1 paper directly
- You don't mind manual threshold tuning

### Use V2 if:
- You want modern best practices
- You need interpretable confidence scores
- You want proper probability calibration
- You need standard metrics (mAP, precision, recall)

## Migration Notes

**Models are NOT compatible!**
- V1 and V2 have different output interpretations
- V1 checkpoints cannot be used with V2 inference (and vice versa)
- Always use matching model + inference scripts

**Threshold tuning:**
- V1: Start with 0.3-0.5, adjust based on results
- V2: Start with 0.5 (50% confidence), more intuitive

## Expected Performance

Both should achieve similar final accuracy, but:
- **V2** will converge faster (BCE is better for classification)
- **V2** confidence scores are more reliable
- **V2** metrics provide better insight into model performance

## Summary

**V2 is recommended for production use** because:
1. ✓ Calibrated probabilities (confidence scores are meaningful)
2. ✓ Industry-standard hybrid loss (BCE for classification, MSE for regression)
3. ✓ Comprehensive metrics (precision, recall, mAP)
4. ✓ Better training dynamics (BCE converges faster than MSE for classification)
5. ✓ More interpretable results (threshold = actual confidence level)

V1 remains useful for educational purposes or if you specifically need pure YOLO v1 implementation.
