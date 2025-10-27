"""
FastAPI application for Barcode Detection
Provides REST API for barcode detection using trained model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from typing import Optional
import time

from api.model_loader import model_loader
from api.utils import (
    preprocess_image,
    decode_predictions,
    non_maximum_suppression,
    boxes_to_pixel_coords
)

# Initialize FastAPI app
app = FastAPI(
    title="Barcode Detection API",
    description="REST API for barcode detection",
    version="1.0.0"
)

# Model will be loaded on first request (lazy loading)
MODEL_LOADED = False


def ensure_model_loaded():
    """Ensure model is loaded before processing requests"""
    global MODEL_LOADED
    if not MODEL_LOADED:
        try:
            model_loader.load_model(
                checkpoint_path="checkpoints_v2/latest_model.pt",
                grid_size=7,
                num_boxes=2,
                force_cpu=True
            )
            MODEL_LOADED = True
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("🚀 Starting Barcode Detection API...")
    ensure_model_loaded()
    print("✓ Model loaded and ready!")


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Barcode Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Detect barcodes in image",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        },
        "model": "BarcodeDetector V2",
        "checkpoint": "checkpoints_v2/latest_model.pt"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model, device = model_loader.get_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": str(device)
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e)
            }
        )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf_threshold: Optional[float] = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold"),
    nms_threshold: Optional[float] = Query(0.4, ge=0.0, le=1.0, description="NMS IoU threshold")
):
    """
    Detect barcodes in uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
        conf_threshold: Confidence threshold for detections (0.0 - 1.0)
        nms_threshold: IoU threshold for Non-Maximum Suppression (0.0 - 1.0)
    
    Returns:
        JSON with predictions, bounding boxes, and metadata
    """
    start_time = time.time()
    
    # Ensure model is loaded
    ensure_model_loaded()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}"
        )
    
    try:
        # Read and open image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_width, original_height = image.size
        
        # Get model and device
        model, device = model_loader.get_model()
        
        # Preprocess image
        img_tensor = preprocess_image(image, device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(img_tensor)  # (1, S, S, 11)
        
        # Decode predictions
        predictions = predictions.squeeze(0)  # Remove batch dimension (S, S, 11)
        boxes = decode_predictions(
            predictions,
            confidence_threshold=conf_threshold,
            grid_size=7,
            num_boxes=2
        )
        
        # Apply NMS
        boxes_nms = non_maximum_suppression(boxes, iou_threshold=nms_threshold)
        
        # Convert to pixel coordinates
        results = boxes_to_pixel_coords(boxes_nms, original_width, original_height)
        
        inference_time = time.time() - start_time
        
        return {
            "success": True,
            "predictions": results,
            "num_detections": len(results),
            "image_shape": {
                "width": original_width,
                "height": original_height
            },
            "parameters": {
                "confidence_threshold": conf_threshold,
                "nms_threshold": nms_threshold
            },
            "inference_time_seconds": round(inference_time, 3)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
