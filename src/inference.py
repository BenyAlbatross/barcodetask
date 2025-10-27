"""
Inference script for BarcodeDetector
Runs trained model on validation set and visualizes predictions
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from tqdm import tqdm
import json

from detector_v1 import BarcodeDetector


def load_model(checkpoint_path, device, grid_size=7, num_boxes=2):
    """Load trained model from checkpoint"""
    model = BarcodeDetector(
        grid_size=grid_size,
        num_boxes_per_cell=num_boxes,
        pretrained=False  # Don't need ImageNet weights for inference
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model


def get_inference_transform():
    """Same transform as validation (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def decode_predictions(predictions, confidence_threshold=0.3, grid_size=7, num_boxes=2):
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
                # Get raw confidence (model trained with MSE, targets were 0.0 and 1.0)
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
                if w <= 0 or h <= 0:
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
                
                # Skip if box became invalid after clamping
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Get class probability (also trained with MSE)
                class_prob = pred_class[row, col].item()
                
                boxes.append((x1, y1, x2, y2, confidence, class_prob))
    
    return boxes


def non_max_suppression(boxes, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes
    
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
    while boxes:
        # Take the box with highest confidence
        best = boxes.pop(0)
        keep.append(best)
        
        # Remove boxes that overlap too much with best
        boxes = [
            box for box in boxes
            if compute_iou(best[:4], box[:4]) < iou_threshold
        ]
    
    return keep


def compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def draw_boxes(image, boxes, color='red', width=3):
    """
    Draw bounding boxes on image
    
    Args:
        image: PIL Image
        boxes: List of (x1, y1, x2, y2, confidence, class_prob) in normalized coords
        color: Box color
        width: Line width
    
    Returns:
        PIL Image with boxes drawn
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    img_width, img_height = image.size
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for box in boxes:
        x1, y1, x2, y2, conf, class_prob = box
        
        # Convert to pixel coordinates
        x1_px = int(x1 * img_width)
        y1_px = int(y1 * img_height)
        x2_px = int(x2 * img_width)
        y2_px = int(y2 * img_height)
        
        # Ensure valid rectangle (x2 >= x1, y2 >= y1)
        if x2_px <= x1_px or y2_px <= y1_px:
            # Skip invalid boxes
            continue
        
        # Draw rectangle
        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline=color, width=width)
        
        # Draw label
        label = f"Barcode {conf:.2f}"
        
        # Draw text background
        text_bbox = draw.textbbox((x1_px, y1_px - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1_px, y1_px - 20), label, fill='white', font=font)
    
    return img_draw


def predict_single_image(image_path, model, device, confidence_threshold=0.3, 
                        iou_threshold=0.5, visualize=True, output_path=None):
    """
    Run inference on a single image
    
    Args:
        image_path: Path to image file
        model: Trained BarcodeDetector model
        device: torch device
        confidence_threshold: Minimum confidence to keep a box
        iou_threshold: IoU threshold for NMS
        visualize: Whether to draw boxes
        output_path: Where to save visualization (optional)
    
    Returns:
        boxes: List of detected boxes
        visualization: PIL Image with boxes (if visualize=True)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_inference_transform()
    
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)  # (1, S, S, 11)
    
    # Decode predictions
    predictions = predictions.squeeze(0)  # Remove batch dimension (S, S, 11)
    boxes = decode_predictions(predictions, confidence_threshold=confidence_threshold)
    
    # Apply NMS
    boxes = non_max_suppression(boxes, iou_threshold=iou_threshold)
    
    print(f"\nImage: {Path(image_path).name}")
    print(f"Detected {len(boxes)} barcode(s)")
    for i, box in enumerate(boxes, 1):
        x1, y1, x2, y2, conf, class_prob = box
        w = x2 - x1
        h = y2 - y1
        print(f"  Box {i}: conf={conf:.3f}, pos=({x1:.3f}, {y1:.3f}), size=({w:.3f}, {h:.3f})")
    
    # Visualize
    result_img = None
    if visualize:
        result_img = draw_boxes(image, boxes)
        if output_path:
            result_img.save(output_path)
            print(f"Saved visualization to: {output_path}")
    
    return boxes, result_img


def batch_inference(model, val_dir, annotation_file, device, num_samples=10,
                   confidence_threshold=0.3, iou_threshold=0.5, output_dir=None):
    """
    Run inference on random validation images
    
    Args:
        model: Trained model
        val_dir: Validation images directory
        annotation_file: COCO annotations file
        device: torch device
        num_samples: Number of random images to process
        confidence_threshold: Minimum confidence threshold
        iou_threshold: IoU threshold for NMS
        output_dir: Directory to save visualizations
    """
    # Load annotations to get image list
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get valid images
    images = {img['id']: img for img in coco_data['images']}
    annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)
    
    # Filter images that exist and have annotations
    valid_image_ids = []
    for img_id in images.keys():
        img_path = Path(val_dir) / images[img_id]['file_name']
        if img_path.exists() and img_id in annotations:
            valid_image_ids.append(img_id)
    
    # Sample random images
    num_samples = min(num_samples, len(valid_image_ids))
    sampled_ids = random.sample(valid_image_ids, num_samples)
    
    print(f"\nRunning inference on {num_samples} random validation images...")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print("=" * 60)
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    total_predictions = 0
    total_ground_truth = 0
    
    for img_id in tqdm(sampled_ids, desc="Processing images"):
        img_info = images[img_id]
        img_path = Path(val_dir) / img_info['file_name']
        
        # Get ground truth count
        gt_boxes = annotations[img_id]
        total_ground_truth += len(gt_boxes)
        
        # Run inference
        output_path = None
        if output_dir:
            output_path = output_dir / f"pred_{img_info['file_name']}"
        
        boxes, _ = predict_single_image(
            img_path, model, device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            visualize=True,
            output_path=output_path
        )
        
        total_predictions += len(boxes)
    
    print("\n" + "=" * 60)
    print("Batch Inference Summary:")
    print(f"  Images processed: {num_samples}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Total ground truth boxes: {total_ground_truth}")
    print(f"  Avg predictions per image: {total_predictions / num_samples:.2f}")
    print(f"  Avg ground truth per image: {total_ground_truth / num_samples:.2f}")
    if output_dir:
        print(f"  Visualizations saved to: {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Barcode Detector Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v1/detector_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--val-dir', type=str, default='barcode_dataset/images/val',
                       help='Validation images directory')
    parser.add_argument('--val-ann', type=str, 
                       default='barcode_dataset/images/val/_annotations.coco.json',
                       help='Validation annotations file')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of random validation images to process')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for NMS (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                       help='Output directory for visualizations')
    parser.add_argument('--random-val', action='store_true',
                       help='Run on random validation images')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    if args.image:
        # Single image inference
        print("\n" + "=" * 60)
        print("Single Image Inference")
        print("=" * 60)
        
        output_path = Path(args.output_dir) / f"pred_{Path(args.image).name}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        boxes, result_img = predict_single_image(
            args.image, model, device,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            visualize=True,
            output_path=output_path
        )
        
        # Display result
        if result_img:
            result_img.show()
    
    elif args.random_val:
        # Batch inference on random validation images
        batch_inference(
            model, args.val_dir, args.val_ann, device,
            num_samples=args.num_samples,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            output_dir=args.output_dir
        )
    
    else:
        # Pick a random validation image
        print("\n" + "=" * 60)
        print("Random Validation Image Inference")
        print("=" * 60)
        
        with open(args.val_ann, 'r') as f:
            coco_data = json.load(f)
        
        images = {img['id']: img for img in coco_data['images']}
        annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations:
                annotations[img_id] = []
            annotations[img_id].append(ann)
        
        valid_ids = [
            img_id for img_id in images.keys()
            if (Path(args.val_dir) / images[img_id]['file_name']).exists()
            and img_id in annotations
        ]
        
        random_id = random.choice(valid_ids)
        img_path = Path(args.val_dir) / images[random_id]['file_name']
        
        output_path = Path(args.output_dir) / f"pred_{images[random_id]['file_name']}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        boxes, result_img = predict_single_image(
            img_path, model, device,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            visualize=True,
            output_path=output_path
        )
        
        if result_img:
            result_img.show()


if __name__ == "__main__":
    main()
