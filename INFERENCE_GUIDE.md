# Inference Scripts for Barcode Detector

## Overview

Two scripts are provided for running inference with your trained BarcodeDetector model:

1. **`inference.py`** - Full-featured inference script with multiple modes
2. **`visualize_single.py`** - Simple script to visualize a single image

---

## Quick Start

### 1. Visualize a Single Image (Simplest)

```bash
python src/visualize_single.py path/to/image.jpg
```

**Example:**
```bash
python src/visualize_single.py barcode_dataset/images/val/barcode_val_012337.jpg
```

This will:
- Load the best trained model from `checkpoints/detector_best.pt`
- Run inference on the image
- Draw bounding boxes on detected barcodes
- Save result to `inference_results/pred_<filename>.jpg`
- Display the result in your default image viewer

---

### 2. Random Validation Image

Run inference on ONE random validation image:

```bash
python src/inference.py
```

This picks a random image from the validation set and visualizes it.

---

### 3. Batch Inference on Random Subset

Run inference on multiple random validation images:

```bash
python src/inference.py --random-val --num-samples 20
```

This will:
- Select 20 random images from validation set
- Run inference on each
- Save visualizations to `inference_results/`
- Print summary statistics

**Adjust number of samples:**
```bash
python src/inference.py --random-val --num-samples 50
```

---

### 4. Specific Image Path

Run inference on a specific image:

```bash
python src/inference.py --image path/to/your/image.jpg
```

---

## Command-Line Options

### `inference.py` Options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `checkpoints/detector_best.pt` | Path to trained model checkpoint |
| `--image` | None | Path to single image for inference |
| `--val-dir` | `barcode_dataset/images/val` | Validation images directory |
| `--val-ann` | `barcode_dataset/images/val/_annotations.coco.json` | Validation annotations file |
| `--num-samples` | 10 | Number of random images to process |
| `--confidence` | 0.3 | Confidence threshold (0-1) |
| `--iou-threshold` | 0.5 | IoU threshold for NMS (0-1) |
| `--output-dir` | `inference_results` | Output directory for visualizations |
| `--random-val` | False | Run on random validation images (flag) |

---

## Examples

### Lower Confidence Threshold (More Detections)
```bash
python src/inference.py --random-val --confidence 0.2 --num-samples 10
```

### Higher Confidence Threshold (Fewer, More Confident Detections)
```bash
python src/inference.py --random-val --confidence 0.5 --num-samples 10
```

### Use Different Checkpoint
```bash
python src/inference.py --checkpoint checkpoints/detector_epoch_30.pt --random-val
```

### Process 100 Random Images
```bash
python src/inference.py --random-val --num-samples 100 --output-dir results_100
```

### Specific Image with Custom Confidence
```bash
python src/inference.py --image my_barcode.jpg --confidence 0.25
```

---

## Output

### Visualization
- Bounding boxes drawn in **red**
- Label shows: `"Barcode {confidence}"`
- Saved to `inference_results/` (or custom `--output-dir`)

### Console Output

For each image, you'll see:
```
Image: barcode_val_012337.jpg
Detected 2 barcode(s)
  Box 1: conf=0.856, pos=(0.234, 0.456), size=(0.123, 0.089)
  Box 2: conf=0.742, pos=(0.567, 0.234), size=(0.145, 0.098)
```

For batch inference, you'll also see summary:
```
Batch Inference Summary:
  Images processed: 20
  Total predictions: 38
  Total ground truth boxes: 42
  Avg predictions per image: 1.90
  Avg ground truth per image: 2.10
  Visualizations saved to: inference_results
```

---

## How It Works

### 1. **Model Loading**
- Loads checkpoint containing trained weights
- Sets model to evaluation mode (`.eval()`)

### 2. **Image Preprocessing**
- Resizes to 448×448 pixels
- Normalizes using ImageNet statistics
- Converts to PyTorch tensor

### 3. **Prediction Decoding**
- Model outputs 7×7 grid with predictions
- Each grid cell can predict up to 2 bounding boxes
- Extracts boxes above confidence threshold

### 4. **Non-Maximum Suppression (NMS)**
- Removes overlapping duplicate detections
- Keeps box with highest confidence
- Controlled by `--iou-threshold`

### 5. **Visualization**
- Draws bounding boxes on original image
- Adds confidence labels
- Saves annotated image

---

## Adjusting Detection Sensitivity

### If you get TOO MANY false positives:
```bash
# Increase confidence threshold
python src/inference.py --confidence 0.5 --random-val
```

### If you get TOO FEW detections:
```bash
# Decrease confidence threshold
python src/inference.py --confidence 0.2 --random-val
```

### If you get many duplicate boxes on same barcode:
```bash
# Decrease IoU threshold (more aggressive NMS)
python src/inference.py --iou-threshold 0.3 --random-val
```

### If NMS is removing correct detections:
```bash
# Increase IoU threshold (less aggressive NMS)
python src/inference.py --iou-threshold 0.7 --random-val
```

---

## Troubleshooting

### Error: "No module named 'detector_v1'"
Make sure you run from the project root directory:
```bash
cd /home/benjamin/barcodetask
python src/inference.py
```

### Error: "Checkpoint not found"
Train the model first:
```bash
python src/train.py
```

Or specify a different checkpoint:
```bash
python src/inference.py --checkpoint checkpoints/detector_epoch_10.pt
```

### No bounding boxes detected
Try lowering the confidence threshold:
```bash
python src/inference.py --confidence 0.1 --random-val
```

---

## Next Steps

1. **Start simple**: Test on a single image first
   ```bash
   python src/visualize_single.py barcode_dataset/images/val/barcode_val_012337.jpg
   ```

2. **Check random samples**: See how model performs on variety of images
   ```bash
   python src/inference.py --random-val --num-samples 20
   ```

3. **Adjust thresholds**: Fine-tune confidence and IoU based on results

4. **Evaluate systematically**: Run on larger subset to get statistics
   ```bash
   python src/inference.py --random-val --num-samples 100
   ```
