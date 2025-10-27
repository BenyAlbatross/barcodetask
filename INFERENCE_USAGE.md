# Inference Scripts Usage

## Overview

Two inference scripts are provided to test your trained barcode detector:

1. **`visualize_single.py`** - Detect barcodes in a single image (with optional visualization)
2. **`run_inference.py`** - Run inference on a random subset of validation images

---

## 1. Single Image Visualization (`visualize_single.py`)

### Quick Start

**Option A: Random image from validation set**
```bash
python src/visualize_single.py
```
This will automatically pick a random image from the validation set.

**Option B: Specify an image**
```bash
python src/visualize_single.py path/to/your/image.jpg
```

### Examples

```bash
# Use a random validation image
python src/visualize_single.py

# Use a specific validation image
python src/visualize_single.py barcode_dataset/images/val/barcode_val_012337.jpg

# Use a test image
python src/visualize_single.py barcode_dataset/images/test/barcode_test_020206.jpg

# Use your own image
python src/visualize_single.py my_barcode_photo.jpg
```

### Output

- **Console**: Shows detected bounding boxes with confidence scores
- **File**: Saves visualization to `inference_results/pred_<filename>.jpg`
- **Display**: Opens the image in your default image viewer

Example output:
```
==============================================================
Barcode Detection - Single Image Visualization
==============================================================

No image specified, selecting random image from validation set...
(Tip: You can specify an image with: python visualize_single.py <image_path>)
Randomly selected: barcode_val_012450.jpg
Using device: cuda
Loading model from: checkpoints/detector_best.pt
Loaded checkpoint from epoch 25

==============================================================
Detected 2 barcodes (confidence > 0.3):
  Box 1: [x=125, y=230, w=180, h=95] conf=0.89
  Box 2: [x=310, y=120, w=150, h=80] conf=0.76
==============================================================
✓ Visualization saved to: inference_results/pred_barcode_val_012450.jpg
Opening image viewer...
```

---

## 2. Batch Inference (`run_inference.py`)

### Quick Start

```bash
python src/run_inference.py --num_images 10
```

### Options

```bash
python src/run_inference.py --help
```

**Available arguments:**
- `--checkpoint`: Path to model checkpoint (default: `checkpoints/detector_best.pt`)
- `--val_dir`: Validation images directory (default: `barcode_dataset/images/val`)
- `--num_images`: Number of random images to test (default: `10`)
- `--conf_threshold`: Confidence threshold (default: `0.3`)
- `--iou_threshold`: IoU threshold for NMS (default: `0.5`)
- `--output_dir`: Where to save visualizations (default: `inference_results`)
- `--save_images`: Save visualization images (default: `True`)

### Examples

```bash
# Test on 5 random images
python src/run_inference.py --num_images 5

# Use higher confidence threshold
python src/run_inference.py --num_images 10 --conf_threshold 0.5

# Use a different checkpoint
python src/run_inference.py --checkpoint checkpoints/detector_epoch_30.pt

# Don't save images, just print results
python src/run_inference.py --num_images 20 --save_images False
```

### Output

Creates visualizations in `inference_results/` and prints a summary:

```
==============================================================
Batch Inference Results
==============================================================

Processing 10 random images from validation set...

Image 1/10: barcode_val_012337.jpg
  Detected 1 barcode(s)

Image 2/10: barcode_val_012450.jpg
  Detected 2 barcode(s)

...

Summary:
  Total images: 10
  Total barcodes detected: 15
  Average per image: 1.5
  Images with detections: 9 (90.0%)

Results saved to: inference_results/
```

---

## Understanding Confidence Thresholds

- **`conf_threshold`**: Minimum confidence score to consider a detection valid
  - Lower (e.g., 0.2): More detections, but more false positives
  - Higher (e.g., 0.5): Fewer false positives, but might miss some barcodes
  - Default: 0.3 (balanced)

- **`iou_threshold`**: Threshold for Non-Maximum Suppression (NMS)
  - Removes duplicate detections of the same barcode
  - Lower (e.g., 0.3): Keeps fewer overlapping boxes
  - Higher (e.g., 0.7): Allows more overlapping boxes
  - Default: 0.5 (standard)

---

## Visualization Features

The visualizations show:
- **Green boxes**: Detected barcodes
- **Labels**: Confidence scores (e.g., "0.89")
- **Red text**: "No detections" if no barcodes found

---

## Troubleshooting

### "Model checkpoint not found"
Make sure you've trained the model first:
```bash
python src/train.py
```
This creates `checkpoints/detector_best.pt`

### "No images found in validation directory"
Check that your dataset is in the correct location:
```
barcode_dataset/
  images/
    val/
      barcode_val_*.jpg
```

### "CUDA out of memory"
If using GPU and running out of memory, the scripts automatically use CPU. You can also force CPU:
```bash
CUDA_VISIBLE_DEVICES="" python src/visualize_single.py
```

---

## Next Steps

1. **Experiment with thresholds** to find optimal settings for your use case
2. **Try different images** to see how the model generalizes
3. **Compare with validation annotations** to assess accuracy
4. **Fine-tune** if results aren't satisfactory

For training, see: `TRAINING_CODE_EXPLANATION.md`
