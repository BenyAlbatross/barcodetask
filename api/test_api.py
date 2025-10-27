"""
Test script for Barcode Detection API
Sends sample image to API and visualizes predictions
"""

import requests
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
import argparse
import subprocess
import os


def test_api(
    image_path: str,
    api_url: str = "http://localhost:8000/predict",
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    output_path: str = "prediction_result.jpg",
    show_image: bool = True
):
    """
    Test the barcode detection API
    
    Args:
        image_path: Path to input image
        api_url: API endpoint URL
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold
        output_path: Path to save result image
        show_image: Whether to automatically open the result image
    """
    
    # Check if image exists
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"❌ Error: Image not found at {image_path}")
        return False
    
    print(f"📸 Loading image: {image_path}")
    
    # Open image
    image = Image.open(img_path).convert("RGB")
    print(f"   Image size: {image.size}")
    
    # Prepare request
    print(f"\n🚀 Sending request to {api_url}")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   NMS threshold: {nms_threshold}")
    
    try:
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            params = {
                "conf_threshold": conf_threshold,
                "nms_threshold": nms_threshold
            }
            
            response = requests.post(api_url, files=files, params=params)
        
        # Check response status
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            return False
        
        # Parse response
        result = response.json()
        
        print("\n✅ Prediction successful!")
        print(f"   Inference time: {result['inference_time_seconds']:.3f}s")
        print(f"   Detections: {result['num_detections']}")
        
        # Print predictions
        if result['num_detections'] > 0:
            print("\n📊 Detected barcodes:")
            for i, pred in enumerate(result['predictions'], 1):
                bbox = pred['bbox']
                conf = pred['confidence']
                print(f"   {i}. bbox: {bbox}, confidence: {conf:.3f}")
        else:
            print("\n⚠️  No barcodes detected")
        
        # Visualize results
        print(f"\n🎨 Drawing bounding boxes...")
        visualize_predictions(image, result['predictions'], output_path)
        print(f"   Saved result to: {output_path}")
        
        # Automatically open the image
        if show_image:
            open_image(output_path)
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to API at {api_url}")
        print("   Make sure the API server is running!")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def visualize_predictions(image: Image.Image, predictions: list, output_path: str):
    """
    Draw bounding boxes on image and save
    
    Args:
        image: PIL Image
        predictions: List of prediction dicts
        output_path: Path to save annotated image
    """
    # Create a copy to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw each prediction
    for pred in predictions:
        bbox = pred['bbox']
        confidence = pred['confidence']
        label = pred['label']
        
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        
        # Draw label with confidence
        text = f"{label}: {confidence:.2f}"
        
        # Draw text background
        bbox_text = draw.textbbox((x1, y1 - 25), text, font=font)
        draw.rectangle(bbox_text, fill="green")
        draw.text((x1, y1 - 25), text, fill="white", font=font)
    
    # Save result
    img_draw.save(output_path)
    print(f"   ✓ Visualization saved")


def open_image(image_path: str):
    """
    Open image using the default system viewer
    Works on Linux, macOS, and Windows
    """
    try:
        if sys.platform.startswith('linux'):
            # Linux - try common image viewers
            subprocess.run(['xdg-open', image_path], check=False)
            print(f"   🖼️  Opening image in default viewer...")
        elif sys.platform == 'darwin':
            # macOS
            subprocess.run(['open', image_path], check=False)
            print(f"   🖼️  Opening image in default viewer...")
        elif sys.platform == 'win32':
            # Windows
            os.startfile(image_path)
            print(f"   🖼️  Opening image in default viewer...")
        else:
            print(f"   ℹ️  Auto-open not supported on {sys.platform}")
    except Exception as e:
        print(f"   ⚠️  Could not auto-open image: {e}")
        print(f"   💡 Manually open: {image_path}")


def test_health_check(base_url: str = "http://localhost:8000"):
    """Test the health endpoint"""
    print(f"🏥 Testing health endpoint...")
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API is healthy")
            print(f"   Device: {data.get('device', 'unknown')}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Could not connect to {base_url}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Barcode Detection API")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--url", type=str, default="http://localhost:8000/predict", help="API URL")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS threshold")
    parser.add_argument("--output", type=str, default="prediction_result.jpg", help="Output image path")
    parser.add_argument("--health-only", action="store_true", help="Only test health endpoint")
    parser.add_argument("--no-show", action="store_true", help="Don't automatically open the result image")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 BARCODE DETECTION API TEST")
    print("=" * 60)
    
    # Test health first
    base_url = args.url.replace("/predict", "")
    if not test_health_check(base_url):
        print("\n⚠️  API is not running or not healthy")
        print("   Start the API with: docker-compose up")
        sys.exit(1)
    
    if args.health_only:
        sys.exit(0)
    
    # Find a test image if not provided
    if not args.image:
        # Try to find an image in barcode_dataset
        dataset_path = Path("barcode_dataset/images/valid")
        if dataset_path.exists():
            images = list(dataset_path.glob("*.jpg"))
            if images:
                args.image = str(images[0])
                print(f"\n📁 Using sample image: {args.image}")
            else:
                print("\n❌ No test image provided and none found in barcode_dataset/images/valid")
                print("   Usage: python api/test_api.py --image path/to/image.jpg")
                sys.exit(1)
        else:
            print("\n❌ No test image provided")
            print("   Usage: python api/test_api.py --image path/to/image.jpg")
            sys.exit(1)
    
    print()
    
    # Test prediction
    success = test_api(
        image_path=args.image,
        api_url=args.url,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        output_path=args.output,
        show_image=not args.no_show
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
