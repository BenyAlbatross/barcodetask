"""
Training script for BarcodeDetectorV2 - Hybrid Loss Approach

Key improvements over V1:
- Sigmoid + BCE for objectness and class (calibrated probabilities)
- MSE for bbox coordinates (regression)
- Proper metrics: Precision, Recall, mAP, mAP50
- Separate checkpoint folder (checkpoints_v2/)
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import wandb

from detector_v2 import BarcodeDetectorV2


class BarcodeDataset(Dataset):
    """
    COCO format dataset loader for barcode detection
    Converts COCO annotations to YOLO-v1 style grid targets
    """
    
    def __init__(self, img_dir, annotation_file, grid_size=7, num_boxes=2, transform=None):
        """
        Args:
            img_dir: Directory containing images
            annotation_file: Path to _annotations.coco.json
            grid_size: Grid size for detection (default: 7)
            num_boxes: Number of bounding boxes per grid cell (default: 2)
            transform: Torchvision transforms
        """
        self.img_dir = Path(img_dir)
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.transform = transform
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image_id to filename mapping
        self.images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image_id
        self.annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # List of valid image IDs (those with annotations)
        self.image_ids = list(self.annotations.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, H, W) tensor
            target: (S, S, B*5 + 1) tensor
                Each cell: [box1_x, box1_y, box1_w, box1_h, box1_conf,
                           box2_x, box2_y, box2_w, box2_h, box2_conf,
                           class_prob]
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = self.img_dir / img_info['file_name']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Get annotations for this image
        anns = self.annotations[img_id]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Create target grid: (S, S, B*5 + 1)
        S = self.grid_size
        B = self.num_boxes
        target = torch.zeros((S, S, B * 5 + 1))
        
        # Fill in ground truth boxes
        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            
            # Normalize coordinates
            x_center = (bbox[0] + bbox[2] / 2) / orig_w
            y_center = (bbox[1] + bbox[3] / 2) / orig_h
            width = bbox[2] / orig_w
            height = bbox[3] / orig_h
            
            # Find which grid cell this bbox center falls into
            col = int(x_center * S)
            row = int(y_center * S)
            
            # Clip to grid boundaries
            col = min(col, S - 1)
            row = min(row, S - 1)
            
            # Compute bbox coordinates relative to grid cell
            # x, y are offsets within the cell (0 to 1)
            x_cell = x_center * S - col
            y_cell = y_center * S - row
            
            # Width and height are relative to image
            w_cell = width
            h_cell = height
            
            # Check if this cell already has an object
            # If not, assign to first box; otherwise second box
            if target[row, col, 4] == 0:  # First box is empty
                box_idx = 0
            else:  # First box occupied, use second
                box_idx = 1
            
            # Only assign if we have space
            if box_idx < B:
                offset = box_idx * 5
                target[row, col, offset:offset+5] = torch.tensor([
                    x_cell, y_cell, w_cell, h_cell, 1.0  # confidence = 1
                ])
                target[row, col, -1] = 1.0  # Class probability (always barcode)
        
        return image, target


class YOLOLossV2(nn.Module):
    """
    YOLO V2 hybrid loss function:
    - MSE for bbox coordinates (regression)
    - BCE for objectness (binary classification)
    - BCE for class prediction (binary classification)
    """
    
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord  # Weight for bbox coordinate loss
        self.lambda_noobj = lambda_noobj  # Weight for no-object confidence loss
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss(reduction='sum')  # For objectness and class
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch, S, S, B*5 + 1)
                - Box1: [x, y, w, h] raw + [conf] sigmoid  [0:5]
                - Box2: [x, y, w, h] raw + [conf] sigmoid  [5:10]
                - Class: sigmoid [10]
            targets: (batch, S, S, B*5 + 1)
        Returns:
            total_loss: scalar
            loss_dict: dictionary with individual loss components
        """
        batch_size = predictions.size(0)
        S = predictions.size(1)
        B = 2  # num_boxes_per_cell
        
        # Extract predictions - careful with indices!
        # Box 1: coords [0:4], conf [4]
        # Box 2: coords [5:9], conf [9]
        pred_box1_coords = predictions[..., 0:4]
        pred_box1_conf = predictions[..., 4:5]
        pred_box2_coords = predictions[..., 5:9]
        pred_box2_conf = predictions[..., 9:10]
        pred_class = predictions[..., 10:11]
        
        # Extract targets
        target_box1_coords = targets[..., 0:4]
        target_box1_conf = targets[..., 4:5]
        target_box2_coords = targets[..., 5:9]
        target_box2_conf = targets[..., 9:10]
        target_class = targets[..., 10:11]
        
        # Masks for cells with objects
        obj_mask_box1 = (target_box1_conf > 0).squeeze(-1)  # (batch, S, S)
        obj_mask_box2 = (target_box2_conf > 0).squeeze(-1)  # (batch, S, S)
        
        # ===== 1. LOCALIZATION LOSS (MSE for bbox coords) =====
        coord_loss = 0
        
        # Box 1 coordinate loss
        if obj_mask_box1.sum() > 0:
            mask1 = obj_mask_box1.unsqueeze(-1)  # (batch, S, S, 1)
            
            # x, y loss
            xy_loss = self.mse(
                pred_box1_coords[..., :2] * mask1,
                target_box1_coords[..., :2] * mask1
            )
            
            # w, h loss (use square root as in YOLO v1)
            pred_wh = torch.sqrt(torch.clamp(pred_box1_coords[..., 2:4], min=1e-6))
            target_wh = torch.sqrt(torch.clamp(target_box1_coords[..., 2:4], min=1e-6))
            wh_loss = self.mse(pred_wh * mask1, target_wh * mask1)
            
            coord_loss += xy_loss + wh_loss
        
        # Box 2 coordinate loss
        if obj_mask_box2.sum() > 0:
            mask2 = obj_mask_box2.unsqueeze(-1)  # (batch, S, S, 1)
            
            # x, y loss
            xy_loss = self.mse(
                pred_box2_coords[..., :2] * mask2,
                target_box2_coords[..., :2] * mask2
            )
            
            # w, h loss
            pred_wh = torch.sqrt(torch.clamp(pred_box2_coords[..., 2:4], min=1e-6))
            target_wh = torch.sqrt(torch.clamp(target_box2_coords[..., 2:4], min=1e-6))
            wh_loss = self.mse(pred_wh * mask2, target_wh * mask2)
            
            coord_loss += xy_loss + wh_loss
        
        coord_loss *= self.lambda_coord
        
        # ===== 2. OBJECTNESS LOSS (BCE for confidence) =====
        conf_loss_obj = 0
        conf_loss_noobj = 0
        
        # DEBUG: Check ranges
        # print(f"pred_box1_conf range: [{pred_box1_conf.min():.4f}, {pred_box1_conf.max():.4f}]")
        # print(f"pred_box2_conf range: [{pred_box2_conf.min():.4f}, {pred_box2_conf.max():.4f}]")
        # print(f"pred_class range: [{pred_class.min():.4f}, {pred_class.max():.4f}]")
        
        # Box 1 confidence loss
        if obj_mask_box1.sum() > 0:
            pred_conf_obj = pred_box1_conf[obj_mask_box1]
            target_conf_obj = target_box1_conf[obj_mask_box1]
            # Clamp predictions to valid range [0, 1] for numerical stability
            pred_conf_obj = torch.clamp(pred_conf_obj, min=1e-7, max=1.0 - 1e-7)
            conf_loss_obj += self.bce(pred_conf_obj, target_conf_obj)
        
        noobj_mask_box1 = ~obj_mask_box1
        if noobj_mask_box1.sum() > 0:
            pred_conf_noobj = pred_box1_conf[noobj_mask_box1]
            target_conf_noobj = target_box1_conf[noobj_mask_box1]
            # Clamp predictions to valid range [0, 1]
            pred_conf_noobj = torch.clamp(pred_conf_noobj, min=1e-7, max=1.0 - 1e-7)
            conf_loss_noobj += self.bce(pred_conf_noobj, target_conf_noobj)
        
        # Box 2 confidence loss
        if obj_mask_box2.sum() > 0:
            pred_conf_obj = pred_box2_conf[obj_mask_box2]
            target_conf_obj = target_box2_conf[obj_mask_box2]
            pred_conf_obj = torch.clamp(pred_conf_obj, min=1e-7, max=1.0 - 1e-7)
            conf_loss_obj += self.bce(pred_conf_obj, target_conf_obj)
        
        noobj_mask_box2 = ~obj_mask_box2
        if noobj_mask_box2.sum() > 0:
            pred_conf_noobj = pred_box2_conf[noobj_mask_box2]
            target_conf_noobj = target_box2_conf[noobj_mask_box2]
            pred_conf_noobj = torch.clamp(pred_conf_noobj, min=1e-7, max=1.0 - 1e-7)
            conf_loss_noobj += self.bce(pred_conf_noobj, target_conf_noobj)
        
        conf_loss_noobj *= self.lambda_noobj
        
        # ===== 3. CLASSIFICATION LOSS (BCE for class prob) =====
        # Only penalize classification in cells with objects
        obj_mask_any = obj_mask_box1 | obj_mask_box2  # (batch, S, S)
        
        class_loss = 0
        if obj_mask_any.sum() > 0:
            pred_class_obj = pred_class[obj_mask_any]
            target_class_obj = target_class[obj_mask_any]
            # Clamp predictions to valid range [0, 1]
            pred_class_obj = torch.clamp(pred_class_obj, min=1e-7, max=1.0 - 1e-7)
            class_loss = self.bce(pred_class_obj, target_class_obj)
        
        # Total loss
        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        
        # Normalize by batch size
        total_loss = total_loss / batch_size
        
        loss_dict = {
            'total': total_loss.item(),
            'coord': coord_loss.item() / batch_size,
            'conf_obj': conf_loss_obj.item() / batch_size,
            'conf_noobj': conf_loss_noobj.item() / batch_size,
            'class': class_loss.item() / batch_size
        }
        
        return total_loss, loss_dict


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
        conf_threshold: confidence threshold
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
                conf = cell_pred[offset + 4].item()  # Already sigmoid activated
                class_prob = cell_pred[10].item()  # Already sigmoid activated
                
                # Filter by confidence
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
    
    # Sort by confidence
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    keep = []
    while len(boxes) > 0:
        # Keep highest confidence box
        best_box = boxes.pop(0)
        keep.append(best_box)
        
        # Remove boxes with high IoU
        boxes = [
            box for box in boxes
            if compute_iou(best_box[:4], box[:4]) < iou_threshold
        ]
    
    return keep


def compute_metrics(all_predictions, all_targets, iou_thresholds=None):
    """
    Compute Precision, Recall, mAP with COCO-style evaluation
    
    Args:
        all_predictions: list of lists of [x1, y1, x2, y2, conf, class_prob]
        all_targets: list of lists of [x1, y1, x2, y2]
        iou_thresholds: IoU thresholds for evaluation (default: COCO 0.5:0.95:0.05)
    
    Returns:
        metrics: dict with precision, recall, mAP values
    """
    if iou_thresholds is None:
        # COCO evaluation: 10 IoU thresholds from 0.5 to 0.95
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    results = {}
    all_aps = []
    
    for iou_thresh in iou_thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for preds, targets in zip(all_predictions, all_targets):
            matched_targets = set()
            
            # For each prediction, find best matching target
            for pred in preds:
                pred_box = pred[:4]
                best_iou = 0
                best_idx = -1
                
                for idx, target in enumerate(targets):
                    if idx in matched_targets:
                        continue
                    
                    iou = compute_iou(pred_box, target)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                
                if best_iou >= iou_thresh:
                    true_positives += 1
                    matched_targets.add(best_idx)
                else:
                    false_positives += 1
            
            # Remaining targets are false negatives
            false_negatives += len(targets) - len(matched_targets)
        
        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        ap = (precision + recall) / 2 if (precision + recall) > 0 else 0
        
        results[f'precision@{iou_thresh:.2f}'] = precision
        results[f'recall@{iou_thresh:.2f}'] = recall
        results[f'AP@{iou_thresh:.2f}'] = ap
        all_aps.append(ap)
    
    # Compute aggregate metrics
    results['precision'] = np.mean([results[f'precision@{t:.2f}'] for t in iou_thresholds])
    results['recall'] = np.mean([results[f'recall@{t:.2f}'] for t in iou_thresholds])
    results['mAP'] = np.mean(all_aps)  # COCO mAP (average across all IoU thresholds)
    results['mAP@0.5'] = results['AP@0.50']  # PASCAL VOC metric
    results['mAP@0.75'] = results['AP@0.75']  # Strict localization metric
    
    return results


def validate(model, val_loader, criterion, device, compute_map=True):
    """
    Validation loop with metrics computation
    """
    model.eval()
    total_loss = 0
    loss_components = {'coord': 0, 'conf_obj': 0, 'conf_noobj': 0, 'class': 0}
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)
            
            total_loss += loss.item()
            for k in loss_components:
                loss_components[k] += loss_dict[k]
            
            # Decode predictions for metrics
            if compute_map:
                for i in range(outputs.size(0)):
                    # Decode predictions
                    pred_boxes = decode_predictions(outputs[i].cpu(), conf_threshold=0.01)
                    pred_boxes = non_max_suppression(pred_boxes, iou_threshold=0.5)
                    all_predictions.append(pred_boxes)
                    
                    # Decode targets
                    target_output = targets[i].cpu()
                    target_boxes = []
                    S = target_output.size(0)
                    cell_size = 448 / S
                    
                    for row in range(S):
                        for col in range(S):
                            for b in range(2):
                                offset = b * 5
                                conf = target_output[row, col, offset + 4].item()
                                if conf > 0:
                                    x_cell = target_output[row, col, offset].item()
                                    y_cell = target_output[row, col, offset + 1].item()
                                    w = target_output[row, col, offset + 2].item()
                                    h = target_output[row, col, offset + 3].item()
                                    
                                    x_center = (col + x_cell) * cell_size
                                    y_center = (row + y_cell) * cell_size
                                    width = w * 448
                                    height = h * 448
                                    
                                    x1 = x_center - width / 2
                                    y1 = y_center - height / 2
                                    x2 = x_center + width / 2
                                    y2 = y_center + height / 2
                                    
                                    target_boxes.append([x1, y1, x2, y2])
                    
                    all_targets.append(target_boxes)
    
    # Average losses
    avg_loss = total_loss / len(val_loader)
    for k in loss_components:
        loss_components[k] /= len(val_loader)
    
    # Compute metrics
    metrics = {}
    if compute_map:
        metrics = compute_metrics(all_predictions, all_targets)
    
    return avg_loss, loss_components, metrics


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    loss_components = {'coord': 0, 'conf_obj': 0, 'conf_noobj': 0, 'class': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss, loss_dict = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += loss_dict[k]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'coord': f'{loss_dict["coord"]:.4f}',
            'obj': f'{loss_dict["conf_obj"]:.4f}',
            'noobj': f'{loss_dict["conf_noobj"]:.4f}',
            'cls': f'{loss_dict["class"]:.4f}'
        })
    
    # Average losses
    avg_loss = total_loss / len(train_loader)
    for k in loss_components:
        loss_components[k] /= len(train_loader)
    
    return avg_loss, loss_components


def main():
    # Configuration
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    GRID_SIZE = 7
    NUM_BOXES = 2
    LAMBDA_COORD = 5.0
    LAMBDA_NOOBJ = 0.5
    COMPUTE_METRICS_EVERY_N_EPOCHS = 1  # Set to 5 if training is too slow
    
    # Paths - V2 uses separate checkpoint folder
    TRAIN_IMG_DIR = 'barcode_dataset/images/train'
    TRAIN_ANN_FILE = 'barcode_dataset/images/train/_annotations.coco.json'
    VAL_IMG_DIR = 'barcode_dataset/images/val'
    VAL_ANN_FILE = 'barcode_dataset/images/val/_annotations.coco.json'
    CHECKPOINT_DIR = 'checkpoints_v2'  # Separate from V1
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize W&B
    wandb.init(
        project="barcode-detector-v2",
        name=f"v2-{wandb.util.generate_id()}",  # Random name with v2 prefix
        config={
            "architecture": "BarcodeDetectorV2",
            "loss_type": "Hybrid (BCE+MSE)",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "grid_size": GRID_SIZE,
            "num_boxes_per_cell": NUM_BOXES,
            "lambda_coord": LAMBDA_COORD,
            "lambda_noobj": LAMBDA_NOOBJ,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "image_size": 448,
            "backbone": "ResNet18"
        }
    )
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = BarcodeDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, GRID_SIZE, NUM_BOXES, train_transform)
    val_dataset = BarcodeDataset(VAL_IMG_DIR, VAL_ANN_FILE, GRID_SIZE, NUM_BOXES, val_transform)
    
    print(f'Train dataset: {len(train_dataset)} images')
    print(f'Val dataset: {len(val_dataset)} images')
    
    # Log dataset sizes to W&B
    wandb.config.update({
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset)
    })
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model - V2 with sigmoid activations
    model = BarcodeDetectorV2(grid_size=GRID_SIZE, num_boxes_per_cell=NUM_BOXES)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = YOLOLossV2(lambda_coord=LAMBDA_COORD, lambda_noobj=LAMBDA_NOOBJ)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Watch model with W&B
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{EPOCHS}')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_components = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        print(f'\nTrain Loss: {train_loss:.4f}')
        print(f'  Coord: {train_components["coord"]:.4f}')
        print(f'  Conf (obj): {train_components["conf_obj"]:.4f}')
        print(f'  Conf (noobj): {train_components["conf_noobj"]:.4f}')
        print(f'  Class: {train_components["class"]:.4f}')
        
        # Validate (compute mAP based on config)
        compute_map = (epoch % COMPUTE_METRICS_EVERY_N_EPOCHS == 0) or (epoch == EPOCHS)
        val_loss, val_components, metrics = validate(model, val_loader, criterion, device, compute_map)
        
        print(f'\nVal Loss: {val_loss:.4f}')
        print(f'  Coord: {val_components["coord"]:.4f}')
        print(f'  Conf (obj): {val_components["conf_obj"]:.4f}')
        print(f'  Conf (noobj): {val_components["conf_noobj"]:.4f}')
        print(f'  Class: {val_components["class"]:.4f}')
        
        if metrics:
            print(f'\nMetrics:')
            print(f'  Precision (avg): {metrics["precision"]:.4f}')
            print(f'  Recall (avg): {metrics["recall"]:.4f}')
            print(f'  mAP (COCO): {metrics["mAP"]:.4f}')
            print(f'  mAP@0.5: {metrics["mAP@0.5"]:.4f}')
            print(f'  mAP@0.75: {metrics["mAP@0.75"]:.4f}')
        
        # ===== SINGLE W&B LOG CALL PER EPOCH =====
        wandb_log = {
            "epoch": epoch,
            "train/loss_total": train_loss,
            "train/loss_coord": train_components["coord"],
            "train/loss_conf_obj": train_components["conf_obj"],
            "train/loss_conf_noobj": train_components["conf_noobj"],
            "train/loss_class": train_components["class"],
            "val/loss_total": val_loss,
            "val/loss_coord": val_components["coord"],
            "val/loss_conf_obj": val_components["conf_obj"],
            "val/loss_conf_noobj": val_components["conf_noobj"],
            "val/loss_class": val_components["class"],
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        # Add metrics if computed
        if metrics:
            # Add aggregate metrics
            wandb_log.update({
                "metrics/precision": metrics["precision"],
                "metrics/recall": metrics["recall"],
                "metrics/mAP": metrics["mAP"],
                "metrics/mAP@0.5": metrics["mAP@0.5"],
                "metrics/mAP@0.75": metrics["mAP@0.75"]
            })
            
            # Add detailed metrics for each IoU threshold
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
            for thresh in iou_thresholds:
                wandb_log[f"metrics_detailed/precision@{thresh:.2f}"] = metrics[f"precision@{thresh:.2f}"]
                wandb_log[f"metrics_detailed/recall@{thresh:.2f}"] = metrics[f"recall@{thresh:.2f}"]
                wandb_log[f"metrics_detailed/AP@{thresh:.2f}"] = metrics[f"AP@{thresh:.2f}"]
        
        # Log everything at once (1 step per epoch)
        wandb.log(wandb_log)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'metrics': metrics
            }, checkpoint_path)
            print(f'✓ Saved best model (val_loss: {val_loss:.4f})')
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'metrics': metrics
        }, checkpoint_path)
    
    print('\n' + '='*60)
    print('Training complete!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Checkpoints saved in: {CHECKPOINT_DIR}/')
    print('='*60)
    
    # Finish W&B run
    wandb.finish()


if __name__ == '__main__':
    main()
