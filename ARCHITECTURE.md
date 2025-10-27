# Custom Barcode Detector - Architecture Explanation

## Overview
This is a custom-built barcode detector inspired by YOLO v1, implemented from scratch in PyTorch for an internship assessment.

## Architecture Decisions

### 1. **Why YOLO-like Grid-Based Detection?**
   - **Single-pass detection**: Unlike two-stage detectors (R-CNN family), YOLO processes the image once, making it fast
   - **Grid-based**: Divides image into SxS grid (7x7), each cell predicts bounding boxes
   - **Simple for single class**: Since we only detect barcodes (one class), we don't need complex multi-class logic
   - **Good for internship**: Shows understanding of modern object detection principles

### 2. **ResNet-18 Backbone (Transfer Learning)**
   - **Pretrained on ImageNet**: Leverages learned features from millions of images
   - **Why ResNet-18?** Lightweight (44M params vs ResNet-50's 98M), faster training, good for limited time
   - **Feature extraction**: Remove final FC layer, keep convolutional features (512 channels)
   - **Trade-off**: Could use ResNet-50 for better accuracy, but ResNet-18 trains faster

### 3. **Detection Head Design**
   ```
   Conv2d(512 → 256) + BatchNorm + LeakyReLU
   Conv2d(256 → 128) + BatchNorm + LeakyReLU  
   Conv2d(128 → 5) + Sigmoid
   ```
   - **Why 3 layers?** Balance between capacity and overfitting
   - **BatchNorm**: Stabilizes training, allows higher learning rates
   - **LeakyReLU**: Better gradient flow than ReLU for negative values
   - **Sigmoid output**: Ensures predictions are in [0, 1] range (normalized coordinates)
   - **5 outputs per cell**: [x, y, w, h, confidence] - simplified from YOLO's multi-box approach

### 4. **Simplified from YOLO v1**
   - **Original YOLO**: 2 boxes per cell + class probabilities = 2×5 + 20 = 30 outputs per cell
   - **Our version**: 1 box per cell = 5 outputs (single class, simpler)
   - **Why simplify?** 
     - Barcodes are typically well-separated (low overlap)
     - Faster training and inference
     - Easier to debug and understand
     - Sufficient for this use case

### 5. **Loss Function (YOLO-inspired)**
   ```python
   Total Loss = λ_coord × Coordinate Loss 
              + Confidence Loss (object cells)
              + λ_noobj × Confidence Loss (empty cells)
   ```
   - **Coordinate Loss** (λ_coord = 5.0):
     - Higher weight because bbox coordinates are critical
     - Separate x,y and w,h (YOLO uses sqrt for w,h for better gradients)
   - **Confidence Loss**:
     - Object cells: Penalize incorrect confidence (should be high)
     - No-object cells: Penalize false positives (should be low, weighted by λ_noobj = 0.5)
   - **Why different weights?** Most cells are empty, need to balance the loss

### 6. **Input Size: 448×448**
   - **Why this size?** YOLO v1 standard, divisible by 32 (ResNet downsampling)
   - **Trade-off**: Larger = more detail but slower; smaller = faster but less detail
   - **Grid size 7×7**: Each cell covers 64×64 pixels, reasonable for barcode sizes

### 7. **Training Choices**
   - **Optimizer**: Adam (adaptive learning rate, good for limited data)
   - **Learning Rate**: 1e-4 (safe starting point with pretrained backbone)
   - **Scheduler**: ReduceLROnPlateau (auto-adjust when validation plateaus)
   - **Batch Size**: 16 (balance between GPU memory and stable gradients)

## What I Would Do With More Time

1. **Multi-scale training**: Train on different image sizes (320, 416, 512, 608)
2. **Data augmentation**: Random crops, flips, color jitter, mosaic augmentation
3. **Anchor boxes**: Use 2-3 boxes per cell with different aspect ratios
4. **Non-maximum suppression**: Remove duplicate detections
5. **Deeper backbone**: ResNet-50 or EfficientNet for better accuracy
6. **Focal loss**: Better handling of class imbalance (many empty cells)
7. **IoU-based metrics**: Calculate mAP, precision, recall during training

## Why This Approach?

✓ **Demonstrates understanding**: Shows knowledge of:
  - Transfer learning
  - Grid-based detection
  - Loss function design
  - PyTorch implementation

✓ **Practical**: Can train in reasonable time (hours, not days)

✓ **Explainable**: Every design decision has a clear rationale

✓ **Extensible**: Easy to add improvements incrementally

## Quick Start

### Training
```bash
python src/train.py
```

### Inference
```bash
python src/inference.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg --output result.jpg
```

## File Structure
```
src/
├── detector_v1.py  # Model architecture
├── dataset.py      # Data loader (YOLO format)
├── loss.py         # Custom loss function
├── train.py        # Training script
└── inference.py    # Inference + visualization
```
