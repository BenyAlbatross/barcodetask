"""
Training script for BarcodeDetector (YOLO v1-style detector)

Dataset: COCO format barcode annotations
Model: Custom ResNet18-based grid detector
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import wandb

from detector_v1 import BarcodeDetector


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
        
        # Filter images that exist and have annotations
        self.image_ids = []
        for img_id in self.images.keys():
            img_path = self.img_dir / self.images[img_id]['file_name']
            if img_path.exists() and img_id in self.annotations:
                self.image_ids.append(img_id)
        
        print(f"Loaded {len(self.image_ids)} images with annotations")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Get annotations for this image
        anns = self.annotations[img_id]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Create target tensor: (S, S, B*5 + 1)
        # B*5: B boxes × (x, y, w, h, confidence)
        # +1: class probability (always 1 for single class)
        S = self.grid_size
        B = self.num_boxes
        target = torch.zeros((S, S, B * 5 + 1))
        
        # Convert COCO bboxes to grid targets
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height] in COCO format
            
            # Normalize bbox coordinates (0 to 1)
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


class YOLOLoss(nn.Module):
    """
    YOLO v1 style loss function for barcode detection
    
    Components:
    - Localization loss (x, y, w, h)
    - Confidence loss (objectness)
    - Classification loss (single class)
    """
    
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord  # Weight for bbox coordinate loss
        self.lambda_noobj = lambda_noobj  # Weight for no-object confidence loss
        self.mse = nn.MSELoss(reduction='sum')
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch, S, S, B*5 + 1)
            targets: (batch, S, S, B*5 + 1)
        Returns:
            total_loss: scalar
        """
        batch_size = predictions.size(0)
        S = predictions.size(1)
        B = 2  # num_boxes_per_cell
        
        # Split predictions and targets
        # Each box: [x, y, w, h, conf]
        pred_boxes = predictions[..., :B*5].reshape(batch_size, S, S, B, 5)
        pred_class = predictions[..., -1]
        
        target_boxes = targets[..., :B*5].reshape(batch_size, S, S, B, 5)
        target_class = targets[..., -1]
        
        # Mask for cells with objects
        obj_mask = target_boxes[..., 4] > 0  # (batch, S, S, B)
        
        # Localization loss (only for cells with objects)
        coord_loss = 0
        for b in range(B):
            mask = obj_mask[..., b].unsqueeze(-1)  # (batch, S, S, 1)
            
            # x, y loss
            xy_loss = self.mse(
                pred_boxes[..., b, :2] * mask,
                target_boxes[..., b, :2] * mask
            )
            
            # w, h loss (use square root as in YOLO v1)
            pred_wh = torch.sqrt(torch.clamp(pred_boxes[..., b, 2:4], min=1e-6))
            target_wh = torch.sqrt(torch.clamp(target_boxes[..., b, 2:4], min=1e-6))
            wh_loss = self.mse(pred_wh * mask, target_wh * mask)
            
            coord_loss += xy_loss + wh_loss
        
        coord_loss *= self.lambda_coord
        
        # Confidence loss
        conf_loss_obj = 0
        conf_loss_noobj = 0
        
        for b in range(B):
            obj_b = obj_mask[..., b]  # (batch, S, S)
            
            # Object confidence loss
            conf_loss_obj += self.mse(
                pred_boxes[..., b, 4] * obj_b,
                target_boxes[..., b, 4] * obj_b
            )
            
            # No-object confidence loss
            noobj_b = ~obj_b
            conf_loss_noobj += self.mse(
                pred_boxes[..., b, 4] * noobj_b,
                target_boxes[..., b, 4] * noobj_b
            )
        
        conf_loss_noobj *= self.lambda_noobj
        
        # Classification loss (only for cells with objects)
        class_mask = (target_class > 0)  # (batch, S, S)
        class_loss = self.mse(
            pred_class * class_mask,
            target_class * class_mask
        )
        
        # Total loss
        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        
        # Normalize by batch size
        total_loss = total_loss / batch_size
        
        return total_loss, {
            'coord': coord_loss / batch_size,
            'conf_obj': conf_loss_obj / batch_size,
            'conf_noobj': conf_loss_noobj / batch_size,
            'class': class_loss / batch_size
        }


def get_transforms(train=True):
    """Data augmentation and normalization"""
    if train:
        return transforms.Compose([
            transforms.Resize((448, 448)),  # YOLO v1 uses 448x448
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


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


def decode_predictions(output, conf_threshold=0.01, grid_size=7, img_size=448):
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
                conf = cell_pred[offset + 4].item()  # Raw value (no sigmoid in V1)
                class_prob = cell_pred[10].item()  # Raw value (no sigmoid in V1)
                
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



def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_metrics = {'coord': 0, 'conf_obj': 0, 'conf_noobj': 0, 'class': 0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        
        # Compute loss
        loss, metrics = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        for key in metrics:
            total_metrics[key] += metrics[key].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'coord': f"{metrics['coord'].item():.4f}",
            'conf': f"{metrics['conf_obj'].item():.4f}"
        })
    
    # Average metrics
    n = len(dataloader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


def validate(model, dataloader, criterion, device, compute_map=True):
    """Validation loop with metrics computation"""
    model.eval()
    total_loss = 0
    total_metrics = {'coord': 0, 'conf_obj': 0, 'conf_noobj': 0, 'class': 0}
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss, metrics = criterion(predictions, targets)
            
            # Accumulate metrics
            total_loss += loss.item()
            for key in metrics:
                total_metrics[key] += metrics[key].item()
            
            # Decode predictions for metrics
            if compute_map:
                for i in range(predictions.size(0)):
                    # Decode predictions
                    pred_boxes = decode_predictions(predictions[i].cpu(), conf_threshold=0.01)
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
    
    # Average metrics
    n = len(dataloader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    
    # Compute mAP metrics
    metrics = {}
    if compute_map:
        metrics = compute_metrics(all_predictions, all_targets)
    
    return avg_loss, avg_metrics, metrics


def main():
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    GRID_SIZE = 7
    NUM_BOXES = 2
    LAMBDA_COORD = 5.0
    LAMBDA_NOOBJ = 0.5
    COMPUTE_METRICS_EVERY_N_EPOCHS = 1  # Set to 5 if training is too slow
    
    # Paths
    DATA_ROOT = Path("barcode_dataset")
    TRAIN_IMG_DIR = DATA_ROOT / "images" / "train"
    VAL_IMG_DIR = DATA_ROOT / "images" / "val"
    TRAIN_ANN = TRAIN_IMG_DIR / "_annotations.coco.json"
    VAL_ANN = VAL_IMG_DIR / "_annotations.coco.json"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize W&B
    wandb.init(
        project="barcode-detector-v2",  # Same project as V2
        name=f"v1-{wandb.util.generate_id()}",  # Random name with v1 prefix
        config={
            "architecture": "BarcodeDetectorV1",
            "loss_type": "MSE (YOLO v1 style)",
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
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
    
    # Datasets
    print("Loading datasets...")
    train_dataset = BarcodeDataset(
        TRAIN_IMG_DIR,
        TRAIN_ANN,
        grid_size=GRID_SIZE,
        num_boxes=NUM_BOXES,
        transform=get_transforms(train=True)
    )
    
    val_dataset = BarcodeDataset(
        VAL_IMG_DIR,
        VAL_ANN,
        grid_size=GRID_SIZE,
        num_boxes=NUM_BOXES,
        transform=get_transforms(train=False)
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    
    # Log dataset sizes to W&B
    wandb.config.update({
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset)
    })
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print("Initializing model...")
    model = BarcodeDetector(
        grid_size=GRID_SIZE,
        num_boxes_per_cell=NUM_BOXES,
        pretrained=True
    ).to(device)
    
    # Loss and optimizer
    criterion = YOLOLoss(lambda_coord=LAMBDA_COORD, lambda_noobj=LAMBDA_NOOBJ)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Watch model with W&B
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate (compute mAP based on config)
        compute_map = (epoch % COMPUTE_METRICS_EVERY_N_EPOCHS == 0) or (epoch == NUM_EPOCHS)
        val_loss, val_metrics, metrics = validate(model, val_loader, criterion, device, compute_map)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    - Coord: {train_metrics['coord']:.4f}")
        print(f"    - Conf (obj): {train_metrics['conf_obj']:.4f}")
        print(f"    - Conf (noobj): {train_metrics['conf_noobj']:.4f}")
        print(f"    - Class: {train_metrics['class']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"    - Coord: {val_metrics['coord']:.4f}")
        print(f"    - Conf (obj): {val_metrics['conf_obj']:.4f}")
        print(f"    - Conf (noobj): {val_metrics['conf_noobj']:.4f}")
        print(f"    - Class: {val_metrics['class']:.4f}")
        
        if metrics:
            print(f"\nMetrics:")
            print(f"  Precision (avg): {metrics['precision']:.4f}")
            print(f"  Recall (avg): {metrics['recall']:.4f}")
            print(f"  mAP (COCO): {metrics['mAP']:.4f}")
            print(f"  mAP@0.5: {metrics['mAP@0.5']:.4f}")
            print(f"  mAP@0.75: {metrics['mAP@0.75']:.4f}")
        
        # Log to W&B
        wandb_log = {
            "epoch": epoch,
            "train/loss_total": train_loss,
            "train/loss_coord": train_metrics['coord'],
            "train/loss_conf_obj": train_metrics['conf_obj'],
            "train/loss_conf_noobj": train_metrics['conf_noobj'],
            "train/loss_class": train_metrics['class'],
            "val/loss_total": val_loss,
            "val/loss_coord": val_metrics['coord'],
            "val/loss_conf_obj": val_metrics['conf_obj'],
            "val/loss_conf_noobj": val_metrics['conf_noobj'],
            "val/loss_class": val_metrics['class'],
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        # Add metrics if computed
        if metrics:
            wandb_log.update({
                "metrics/precision": metrics["precision"],
                "metrics/recall": metrics["recall"],
                "metrics/mAP": metrics["mAP"],
                "metrics/mAP@0.5": metrics["mAP@0.5"],
                "metrics/mAP@0.75": metrics["mAP@0.75"]
            })
            
            # Log detailed metrics for each IoU threshold
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
            for thresh in iou_thresholds:
                wandb_log.update({
                    f"metrics_detailed/precision@{thresh:.2f}": metrics[f"precision@{thresh:.2f}"],
                    f"metrics_detailed/recall@{thresh:.2f}": metrics[f"recall@{thresh:.2f}"],
                    f"metrics_detailed/AP@{thresh:.2f}": metrics[f"AP@{thresh:.2f}"]
                })
        
        wandb.log(wandb_log)

        
        # Save checkpoint
        checkpoint_dir = Path("checkpoints_v1")
        checkpoint_dir.mkdir(exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_dir / f"latest_model.pt")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_dir / "best_model.pt")
            print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
