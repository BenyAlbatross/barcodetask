"""
Inference utilities for BarcodeDetectorV2

Key differences from V1:
- Model outputs are already sigmoid activated for objectness/class
- No need to apply sigmoid during inference
- Confidence scores are true probabilities [0, 1]
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from detector_v2 import BarcodeDetectorV2


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
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


def decode_predictions(output, conf_threshold=0.5, grid_size=7, img_size=448):
    """
    Convert model output to bounding boxes
    
    Args:
        output: (S, S, 11) tensor - model output for single image
                Note: objectness [8:10] and class [10] are already sigmoid activated
        conf_threshold: confidence threshold (now a true probability)
        grid_size: grid size (default 7)
        img_size: image size (default 448)
    
    Returns:
        boxes: list of [x1, y1, x2, y2, confidence, class_prob]
    """
    S = grid_size
    cell_size = img_size / S
    boxes = []
    
    for i in range(S):
        for j in range(S):
            cell_pred = output[i, j]
            
            # Check both boxes in this cell
            for b in range(2):
                offset = b * 5
                x_cell = cell_pred[offset].item()
                y_cell = cell_pred[offset + 1].item()
                w = cell_pred[offset + 2].item()
                h = cell_pred[offset + 3].item()
                conf = cell_pred[offset + 4].item()  # Already sigmoid activated in model
                class_prob = cell_pred[10].item()  # Already sigmoid activated in model
                
                # Filter by confidence threshold
                if conf < conf_threshold:
                    continue
                
                # Convert to image coordinates
                x_center = (j + x_cell) * cell_size
                y_center = (i + y_cell) * cell_size
                width = w * img_size
                height = h * img_size
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                # Clamp to image boundaries
                x1 = max(0, min(x1, img_size))
                y1 = max(0, min(y1, img_size))
                x2 = max(0, min(x2, img_size))
                y2 = max(0, min(y2, img_size))
                
                # Validate box dimensions
                if width <= 0 or height <= 0:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                
                boxes.append([x1, y1, x2, y2, conf, class_prob])
    
    return boxes


def non_max_suppression(boxes, iou_threshold=0.5):
    """
    Apply NMS to remove duplicate detections
    
    Args:
        boxes: list of [x1, y1, x2, y2, confidence, class_prob]
        iou_threshold: IoU threshold for NMS
    
    Returns:
        filtered_boxes: list of boxes after NMS
    """
    if len(boxes) == 0:
        return []
    
    # Sort by confidence (descending)
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    keep = []
    while len(boxes) > 0:
        # Keep highest confidence box
        best_box = boxes.pop(0)
        keep.append(best_box)
        
        # Remove boxes with high IoU overlap
        boxes = [
            box for box in boxes
            if compute_iou(best_box[:4], box[:4]) < iou_threshold
        ]
    
    return keep


def draw_boxes(image, boxes, color='red', width=3):
    """
    Draw bounding boxes on image
    
    Args:
        image: PIL Image
        boxes: list of [x1, y1, x2, y2, confidence, class_prob]
        color: box color
        width: line width
    
    Returns:
        image_with_boxes: PIL Image with boxes drawn
    """
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for box in boxes:
        x1, y1, x2, y2, conf, class_prob = box
        
        # Validate coordinates
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        # Draw label with confidence (now a true probability!)
        label = f'Barcode {conf:.2f}'
        
        # Get text bounding box
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background for text
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
    
    return image


def predict_single_image(model, image_path, conf_threshold=0.5, iou_threshold=0.5, device='cpu'):
    """
    Run inference on a single image
    
    Args:
        model: BarcodeDetectorV2 model
        image_path: path to image
        conf_threshold: confidence threshold (true probability [0, 1])
        iou_threshold: IoU threshold for NMS
        device: 'cpu' or 'cuda'
    
    Returns:
        image_with_boxes: PIL Image with boxes drawn
        boxes: list of detected boxes
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size
    
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    # Decode predictions (objectness/class already sigmoid activated)
    output = output[0].cpu()  # (S, S, 11)
    boxes = decode_predictions(output, conf_threshold=conf_threshold)
    
    # Apply NMS
    boxes = non_max_suppression(boxes, iou_threshold=iou_threshold)
    
    # Scale boxes to original image size
    scale_x = original_size[0] / 448
    scale_y = original_size[1] / 448
    
    scaled_boxes = []
    for box in boxes:
        x1, y1, x2, y2, conf, class_prob = box
        scaled_boxes.append([
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
            conf,
            class_prob
        ])
    
    # Draw boxes
    image_with_boxes = draw_boxes(original_image.copy(), scaled_boxes)
    
    return image_with_boxes, scaled_boxes


def load_model(checkpoint_path, device='cpu'):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: path to checkpoint file
        device: 'cpu' or 'cuda'
    
    Returns:
        model: loaded model
    """
    model = BarcodeDetectorV2(grid_size=7, num_boxes_per_cell=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == '__main__':
    import argparse
    import os
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Barcode Detector V2 Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v2/latest_model.pt',
                       help='Path to model checkpoint (default: checkpoints_v2/latest_model.pt)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image for inference')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--nms-threshold', type=float, default=0.4,
                       help='NMS IoU threshold (default: 0.4)')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                       help='Output directory for results (default: inference_results)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, device)
    
    # Run inference
    print(f'Running inference on {args.image}...')
    image_with_boxes, boxes = predict_single_image(
        model, args.image, 
        conf_threshold=args.confidence,
        iou_threshold=args.nms_threshold,
        device=device
    )
    
    # Print results
    print(f'\nDetected {len(boxes)} barcodes:')
    for i, box in enumerate(boxes):
        print(f'  Box {i+1}: conf={box[4]:.3f}, class_prob={box[5]:.3f}')
    
    # Save result to output folder
    os.makedirs(args.output_dir, exist_ok=True)
    image_filename = os.path.basename(args.image)
    output_filename = image_filename.replace('.jpg', '_detected_v2.jpg').replace('.png', '_detected_v2.png')
    output_path = os.path.join(args.output_dir, output_filename)
    image_with_boxes.save(output_path)
    print(f'\nSaved result to {output_path}')
    
    # Display result
    if image_with_boxes:
        image_with_boxes.show()
