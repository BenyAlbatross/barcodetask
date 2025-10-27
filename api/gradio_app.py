"""
Gradio Web Interface for Barcode Detection API
Interactive UI for testing barcode detection
"""

import gradio as gr
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np


def detect_barcodes(image, conf_threshold, nms_threshold):
    """
    Send image to API and return annotated result
    
    Args:
        image: PIL Image or numpy array
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold
    
    Returns:
        Annotated image with bounding boxes
        Text summary of detections
    """
    if image is None:
        return None, "❌ Please upload an image"
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    try:
        # Save image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Send to API
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": ("image.jpg", img_byte_arr, "image/jpeg")},
            params={
                "conf_threshold": conf_threshold,
                "nms_threshold": nms_threshold
            }
        )
        
        if response.status_code != 200:
            return None, f"❌ API Error: {response.status_code}\n{response.text}"
        
        result = response.json()
        
        # Draw bounding boxes
        annotated_image = draw_boxes(image, result['predictions'])
        
        # Create summary text
        summary = f"✅ **Detection Complete**\n\n"
        summary += f"⏱️ **Inference Time:** {result['inference_time_seconds']:.3f}s\n"
        summary += f"📦 **Detections:** {result['num_detections']}\n"
        summary += f"📐 **Image Size:** {result['image_shape']['width']}x{result['image_shape']['height']}\n\n"
        
        if result['num_detections'] > 0:
            summary += "### Detected Barcodes:\n"
            for i, pred in enumerate(result['predictions'], 1):
                bbox = pred['bbox']
                conf = pred['confidence']
                summary += f"{i}. **Confidence:** {conf:.3f} | **BBox:** [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n"
        else:
            summary += "⚠️ No barcodes detected. Try lowering the confidence threshold."
        
        return annotated_image, summary
        
    except requests.exceptions.ConnectionError:
        return None, "❌ **Error:** Cannot connect to API.\n\nMake sure the API is running:\n```\nuvicorn api.app:app --reload\n```"
    except Exception as e:
        return None, f"❌ **Error:** {str(e)}"


def draw_boxes(image, predictions):
    """
    Draw bounding boxes on image
    
    Args:
        image: PIL Image
        predictions: List of prediction dicts
    
    Returns:
        Annotated PIL Image
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load a nice font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # Color palette for boxes
    colors = ["#00FF00", "#00FFFF", "#FF00FF", "#FFFF00", "#FF6600"]
    
    for idx, pred in enumerate(predictions):
        bbox = pred['bbox']
        confidence = pred['confidence']
        label = pred['label']
        
        x1, y1, x2, y2 = bbox
        color = colors[idx % len(colors)]
        
        # Draw rectangle with thicker border
        for offset in range(4):
            draw.rectangle(
                [x1-offset, y1-offset, x2+offset, y2+offset], 
                outline=color, 
                width=1
            )
        
        # Draw label with confidence
        text = f"{label}: {confidence:.2f}"
        
        # Get text bounding box
        bbox_text = draw.textbbox((x1, y1 - 30), text, font=font)
        
        # Draw text background
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 30), text, fill="black", font=font)
    
    return img_draw


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return f"✅ API is healthy | Device: {data.get('device', 'unknown')}"
        else:
            return f"⚠️ API returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ API is not running. Start with: uvicorn api.app:app --reload"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Barcode Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 Barcode Detection System
        Upload an image to detect barcodes.
        
        **Model:** BarcodeDetector V2 (ResNet18 backbone)
        """
    )
    
    # API status
    api_status = gr.Textbox(
        label="API Status",
        value=check_api_health(),
        interactive=False,
        max_lines=1
    )
    
    with gr.Row():
        with gr.Column():
            # Input
            input_image = gr.Image(
                label="Upload Image",
                type="pil",
                height=400
            )
            
            # Parameters
            conf_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Confidence Threshold",
                info="Minimum confidence to keep a detection"
            )
            
            nms_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.4,
                step=0.05,
                label="NMS Threshold",
                info="IoU threshold for Non-Maximum Suppression"
            )
            
            # Buttons
            with gr.Row():
                detect_btn = gr.Button("🔍 Detect Barcodes", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", size="lg")
                refresh_status = gr.Button("🔄 Refresh Status", size="sm")
        
        with gr.Column():
            # Output
            output_image = gr.Image(
                label="Detection Result",
                type="pil",
                height=400
            )
            
            output_text = gr.Markdown(
                label="Detection Summary",
                value="Upload an image and click **Detect Barcodes** to get started."
            )
    
    # Examples (optional - uncomment if you want to add example images)
    # gr.Markdown("### 📸 Try Sample Images")
    # gr.Examples(
    #     examples=[
    #         ["barcode_dataset/images/val/barcode_detector_test_025569.jpg", 0.5, 0.4],
    #     ],
    #     inputs=[input_image, conf_threshold, nms_threshold],
    # )
    
    # Event handlers
    detect_btn.click(
        fn=detect_barcodes,
        inputs=[input_image, conf_threshold, nms_threshold],
        outputs=[output_image, output_text]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, "Ready for new image."),
        outputs=[input_image, output_image, output_text]
    )
    
    refresh_status.click(
        fn=check_api_health,
        outputs=[api_status]
    )
    
    gr.Markdown(
        """
        ---
        ### 🚀 API Endpoints
        - Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)
        - Health check: [http://localhost:8000/health](http://localhost:8000/health)
        """
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Gradio Interface for Barcode Detection")
    print("=" * 60)
    print("\n⚠️  Make sure the API is running first:")
    print("   uvicorn api.app:app --reload")
    print("\n" + check_api_health())
    print("\n" + "=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
