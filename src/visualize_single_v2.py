"""
Visualize single image with BarcodeDetectorV2

Usage:
    python visualize_single_v2.py [image_path] [conf_threshold]
    
    If no image_path is provided, a random validation image will be used.
    Default confidence threshold: 0.5 (now a true probability!)
"""

import sys
import random
from pathlib import Path
import torch

from inference_v2 import load_model, predict_single_image


def get_random_image(data_dir='barcode_dataset/images/val'):
    """
    Get a random image from the validation directory
    
    Args:
        data_dir: directory containing images
    
    Returns:
        image_path: path to random image
    """
    data_path = Path(data_dir)
    image_files = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png'))
    
    if not image_files:
        raise ValueError(f'No images found in {data_dir}')
    
    return str(random.choice(image_files))


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print('No image path provided, selecting random validation image...')
        image_path = get_random_image()
        print(f'Selected: {image_path}')
    
    confidence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    # Checkpoint path (V2 checkpoints in separate folder)
    checkpoint_path = 'checkpoints_v2/best_model.pt'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading V2 model from {checkpoint_path}...')
    try:
        model = load_model(checkpoint_path, device)
    except FileNotFoundError:
        print(f'Error: Checkpoint not found at {checkpoint_path}')
        print('Please train the V2 model first using: python src/train_v2.py')
        sys.exit(1)
    
    # Run inference
    print(f'Running inference with confidence threshold: {confidence_threshold}')
    print('(V2 note: confidence scores are now calibrated probabilities [0, 1])')
    image_with_boxes, boxes = predict_single_image(
        model, 
        image_path,
        conf_threshold=confidence_threshold,
        iou_threshold=0.5,
        device=device
    )
    
    # Print results
    print(f'\n✓ Detected {len(boxes)} barcodes:')
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, class_prob = box
        print(f'  Box {i+1}:')
        print(f'    Confidence: {conf:.3f} (true probability)')
        print(f'    Class prob: {class_prob:.3f}')
        print(f'    Location: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})')
    
    # Display result
    image_with_boxes.show()
    
    # Save result
    output_path = image_path.replace('.jpg', '_detected_v2.jpg').replace('.png', '_detected_v2.png')
    image_with_boxes.save(output_path)
    print(f'\n✓ Saved result to {output_path}')


if __name__ == '__main__':
    main()
