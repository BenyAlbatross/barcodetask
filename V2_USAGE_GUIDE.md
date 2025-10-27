# BarcodeDetector V2 - Usage Guide

## Quick Start

### 1. Train the Model

```bash
python src/train_v2.py
```

**What happens:**
- Loads COCO format dataset from `barcode_dataset/`
- Trains for 50 epochs with hybrid loss (BCE + MSE)
- Saves checkpoints to `checkpoints_v2/` (separate from V1!)
- Computes mAP every 5 epochs
- Saves best model based on validation loss

**Expected output:**
```
Using device: cuda
Train dataset: XXX images
Val dataset: XXX images

Epoch 1/50
================================================================
Train Loss: X.XXXX
  Coord: X.XXXX
  Conf (obj): X.XXXX
  Conf (noobj): X.XXXX
  Class: X.XXXX

Val Loss: X.XXXX
  Coord: X.XXXX
  Conf (obj): X.XXXX
  Conf (noobj): X.XXXX
  Class: X.XXXX

Metrics:
  Precision: 0.XXXX
  Recall: 0.XXXX
  mAP@0.5: 0.XXXX
  mAP@0.75: 0.XXXX

✓ Saved best model (val_loss: X.XXXX)
```

### 2. Test on Random Image

```bash
python src/visualize_single_v2.py
```

**What happens:**
- Selects random image from `barcode_dataset/images/val/`
- Loads best V2 model from `checkpoints_v2/best_model.pth`
- Runs inference with default threshold 0.5
- Displays image with bounding boxes
- Saves result with suffix `_detected_v2.jpg`

### 3. Test on Specific Image

```bash
python src/visualize_single_v2.py path/to/image.jpg
```

### 4. Adjust Confidence Threshold

```bash
python src/visualize_single_v2.py path/to/image.jpg 0.7
```

**Note:** In V2, threshold is a true probability!
- 0.5 = 50% confidence
- 0.7 = 70% confidence
- 0.3 = 30% confidence

## Understanding V2 Outputs

### Confidence Scores (Now Calibrated!)

```python
# V2 output
Detected 2 barcodes:
  Box 1:
    Confidence: 0.873 (true probability)  ← This means 87.3% confident
    Class prob: 0.921
    Location: (45.2, 120.3) -> (234.5, 189.7)
```

**Interpretation:**
- Confidence = probability that this box contains an object
- Class prob = probability that the object is a barcode
- Both are now calibrated probabilities [0, 1]

### Metrics Explained

**Precision**: Of all predicted boxes, what % were correct?
- High precision = few false positives
- Low precision = many wrong detections

**Recall**: Of all ground truth boxes, what % were detected?
- High recall = found most barcodes
- Low recall = missed many barcodes

**mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- Standard object detection metric
- Higher is better (max = 1.0)

**mAP@0.75**: Mean Average Precision at IoU threshold 0.75
- More strict (requires better localization)
- Higher is better (max = 1.0)

## File Structure

```
barcodetask/
├── src/
│   ├── detector_v2.py          # V2 model (with sigmoid activations)
│   ├── train_v2.py             # V2 training script (hybrid loss + metrics)
│   ├── inference_v2.py         # V2 inference utilities
│   └── visualize_single_v2.py  # V2 single image demo
├── checkpoints_v2/             # V2 model checkpoints (separate from V1!)
│   ├── best_model.pth          # Best model (lowest val loss)
│   └── latest_model.pth        # Latest epoch
└── V1_VS_V2_COMPARISON.md      # Detailed comparison
```

## Advanced Usage

### Load Model Programmatically

```python
from src.inference_v2 import load_model, predict_single_image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('checkpoints_v2/best_model.pth', device)

# Run inference
image_with_boxes, boxes = predict_single_image(
    model, 
    'path/to/image.jpg',
    conf_threshold=0.6,
    iou_threshold=0.5,
    device=device
)

# boxes format: [x1, y1, x2, y2, confidence, class_prob]
for box in boxes:
    print(f'Confidence: {box[4]:.3f}, Class: {box[5]:.3f}')
```

### Resume Training

```python
# In train_v2.py, modify main() to load checkpoint:

checkpoint = torch.load('checkpoints_v2/latest_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Then continue training from start_epoch
```

### Batch Inference

```python
from pathlib import Path
from src.inference_v2 import load_model, predict_single_image

model = load_model('checkpoints_v2/best_model.pth', 'cuda')

# Process all images in a folder
image_dir = Path('barcode_dataset/images/test')
for img_path in image_dir.glob('*.jpg'):
    image_with_boxes, boxes = predict_single_image(
        model, str(img_path), conf_threshold=0.5
    )
    
    # Save result
    output_path = str(img_path).replace('.jpg', '_detected.jpg')
    image_with_boxes.save(output_path)
    print(f'Processed {img_path.name}: {len(boxes)} detections')
```

## Troubleshooting

### Error: Checkpoint not found
```
Error: Checkpoint not found at checkpoints_v2/best_model.pth
Please train the V2 model first using: python src/train_v2.py
```
**Solution:** Train the model first!

### Too many detections
**Solution:** Increase confidence threshold
```bash
python src/visualize_single_v2.py image.jpg 0.7  # Try 0.7 instead of 0.5
```

### Missing detections
**Solution:** Decrease confidence threshold
```bash
python src/visualize_single_v2.py image.jpg 0.3  # Try 0.3 instead of 0.5
```

### Low mAP during training
- **Check data augmentation**: Too aggressive augmentation can hurt
- **Check learning rate**: May need to adjust
- **Check dataset**: Ensure annotations are correct
- **Train longer**: 50 epochs may not be enough

## Comparison with V1

| Feature | V1 | V2 |
|---------|----|----|
| Loss (Objectness) | MSE | BCE ✓ |
| Loss (Class) | MSE | BCE ✓ |
| Loss (Bbox) | MSE | MSE |
| Confidence Range | Unbounded | [0, 1] ✓ |
| Metrics | Basic | mAP, P, R ✓ |
| Checkpoints | `checkpoints/` | `checkpoints_v2/` |
| Threshold Tuning | Hard | Easy ✓ |

**Recommendation:** Use V2 for all new projects!

## Next Steps

1. **Train the model**: `python src/train_v2.py`
2. **Test inference**: `python src/visualize_single_v2.py`
3. **Tune threshold**: Adjust based on your precision/recall needs
4. **Deploy**: Use `inference_v2.py` functions in your application

## Questions?

- See `V1_VS_V2_COMPARISON.md` for detailed technical comparison
- See `TRAINING_CODE_EXPLANATION.md` for line-by-line explanation (V1, but concepts apply)
- Check training logs in console output for metrics tracking
