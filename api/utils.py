"""
Utility functions for barcode detection API
- Bounding box decoding
- Non-Maximum Suppression (NMS)
- Image preprocessing
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Tuple


def get_inference_transform():
    """
    Get image transformation for inference
    Matches training preprocessing
    """
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def decode_predictions(
    predictions: torch.Tensor,
    confidence_threshold: float = 0.3,
    grid_size: int = 7,
    num_boxes: int = 2
) -> List[Tuple[float, float, float, float, float, float]]:
    """
    Convert model output to bounding boxes
    
    Args:
        predictions: (S, S, B*5 + 1) tensor from model
        confidence_threshold: minimum confidence to keep a box
        grid_size: grid size (default 7)
        num_boxes: boxes per cell (default 2)
    
    Returns:
        List of boxes: [(x1, y1, x2, y2, confidence, class_prob), ...]
        Coordinates are normalized (0 to 1)
    """
    S = grid_size
    B = num_boxes
    boxes = []
    
    # Reshape predictions
    pred_boxes = predictions[:, :, :B*5].reshape(S, S, B, 5)  # (S, S, B, 5)
    pred_class = predictions[:, :, -1]  # (S, S)
    
    for row in range(S):
        for col in range(S):
            for b in range(B):
                # Get raw confidence
                confidence = pred_boxes[row, col, b, 4].item()
                
                # Only keep boxes above confidence threshold
                if confidence < confidence_threshold:
                    continue
                
                # Get box coordinates
                x_cell = pred_boxes[row, col, b, 0].item()  # Offset in cell (0-1)
                y_cell = pred_boxes[row, col, b, 1].item()
                w = pred_boxes[row, col, b, 2].item()  # Width relative to image
                h = pred_boxes[row, col, b, 3].item()  # Height relative to image
                
                # Skip boxes with invalid dimensions
                if w <= 0 or h <= 0 or w > 2 or h > 2:
                    continue
                
                # Convert to image coordinates (0 to 1)
                x_center = (col + x_cell) / S
                y_center = (row + y_cell) / S
                
                # Convert to corner coordinates
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                
                # Clamp to [0, 1]
                x1 = max(0, min(1, x1))
                y1 = max(0, min(1, y1))
                x2 = max(0, min(1, x2))
                y2 = max(0, min(1, y2))
                
                # Get class probability
                class_prob = pred_class[row, col].item()
                
                boxes.append((x1, y1, x2, y2, confidence, class_prob))
    
    return boxes


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """
    Compute IoU between two boxes [x1, y1, x2, y2, ...]
    """
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]
    
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def non_maximum_suppression(
    boxes: List[Tuple],
    iou_threshold: float = 0.4
) -> List[Tuple]:
    """
    Apply Non-Maximum Suppression to remove duplicate detections
    
    Args:
        boxes: List of (x1, y1, x2, y2, confidence, class_prob)
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Filtered list of boxes
    """
    if len(boxes) == 0:
        return []
    
    # Sort by confidence (descending)
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    keep = []
    while len(boxes) > 0:
        # Keep the box with highest confidence
        current = boxes.pop(0)
        keep.append(current)
        
        # Remove boxes with high IoU
        boxes = [
            box for box in boxes
            if compute_iou(current, box) < iou_threshold
        ]
    
    return keep


def boxes_to_pixel_coords(
    boxes: List[Tuple],
    img_width: int,
    img_height: int
) -> List[dict]:
    """
    Convert normalized boxes to pixel coordinates and format as JSON
    
    Args:
        boxes: List of (x1, y1, x2, y2, confidence, class_prob) in [0, 1]
        img_width: Original image width
        img_height: Original image height
    
    Returns:
        List of dicts with bbox, label, confidence
    """
    results = []
    for box in boxes:
        x1, y1, x2, y2, confidence, class_prob = box
        
        results.append({
            "bbox": [
                int(x1 * img_width),
                int(y1 * img_height),
                int(x2 * img_width),
                int(y2 * img_height)
            ],
            "label": "barcode",
            "confidence": float(confidence),
            "class_probability": float(class_prob)
        })
    
    return results


def preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Preprocess PIL Image for model inference
    
    Args:
        image: PIL Image
        device: torch device
    
    Returns:
        Preprocessed tensor ready for model
    """
    transform = get_inference_transform()
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return img_tensor.to(device)
