"""
Simple script to visualize barcode detection on a single image
"""

import sys
from pathlib import Path
import torch
from PIL import Image
import random

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from inference import load_model, predict_single_image


def get_random_image(val_dir="barcode_dataset/images/val"):
    """Get a random image from the validation set"""
    val_path = Path(val_dir)
    
    # Get all jpg images (excluding the annotation file)
    image_files = [f for f in val_path.glob("*.jpg") if f.is_file()]
    
    if not image_files:
        print(f"Error: No images found in {val_dir}")
        sys.exit(1)
    
    # Select random image
    random_image = random.choice(image_files)
    print(f"Randomly selected: {random_image.name}")
    return str(random_image)


def main():
    # Configuration
    checkpoint_path = "checkpoints_v1/detector_best.pt"
    confidence_threshold = 0.5  # Increased since model outputs raw values (not probabilities)
    iou_threshold = 0.5
    
    print("=" * 60)
    print("Barcode Detection - Single Image Visualization")
    print("=" * 60)
    
    # Check if image path provided, otherwise use random
    if len(sys.argv) < 2:
        print("\nNo image specified, selecting random image from validation set...")
        print("(Tip: You can specify an image with: python visualize_single.py <image_path>)")
        image_path = get_random_image()
    else:
        image_path = sys.argv[1]
        print(f"\nUsing specified image: {Path(image_path).name}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    
    # Run inference
    print("\n" + "=" * 60)
    output_path = Path("inference_results") / f"pred_{Path(image_path).name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    boxes, result_img = predict_single_image(
        image_path, model, device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        visualize=True,
        output_path=output_path
    )
    
    print("\n" + "=" * 60)
    print(f"✓ Visualization saved to: {output_path}")
    
    # Display result
    if result_img:
        print("Opening image viewer...")
        result_img.show()


if __name__ == "__main__":
    main()
