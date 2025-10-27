# 🔍 Barcode Detection API

A production-ready REST API for barcode detection using a custom deep learning model.

## 📋 Overview

This API provides barcode detection capabilities through a simple REST interface. It uses a trained BarcodeDetector V2 model to detect barcodes in images.

**Key Features:**
- 🚀 **FastAPI** - Modern, fast web framework with automatic documentation
- 🐳 **Docker** - Fully containerized for easy deployment
- 🎯 **Single-class detection** - Optimized for barcode detection
- 🔧 **Configurable** - Adjustable confidence and NMS thresholds
- 📊 **JSON responses** - Easy integration with any application

## 🛠️ Technology Stack

- **Framework:** FastAPI + Uvicorn
- **Model:** Custom YOLO-v1 style detector (ResNet18 backbone)
- **Deep Learning:** PyTorch (CPU-only for lightweight deployment)
- **Container:** Docker + Docker Compose
- **Python:** 3.12+

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

The easiest way to run the API is using Docker Compose:

```bash
# Ensure you are in the root of the barcodetask folder
cd /path/to/barcodetask

# Build and start the API (one command!)
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

The API will be available at `http://localhost:8000`

### Option 2: Docker Only

```bash
# Ensure you are in the root of the barcodetask folder
cd /path/to/barcodetask

# Build the Docker image
docker build -t barcode-api .

# Run the container
docker run -p 8000:8000 barcode-api
```

### Option 3: Local Development

```bash
# Install dependencies
pip install -r api/requirements.txt

# Run the API
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## 🎨 Gradio Web Interface

For an easy-to-use graphical interface, use the Gradio web app:

### Quick Start with Gradio

```bash
# 1. First, start the API server (in one terminal)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# 2. Then, start the Gradio interface (in another terminal)
python api/gradio_app.py
```

The Gradio interface will be available at **`http://localhost:7860`**

### Features

The Gradio interface provides:
- 📤 **Drag-and-drop image upload** - Simple image selection
- 🎚️ **Interactive sliders** - Adjust confidence and NMS thresholds in real-time
- 🎨 **Visual results** - See bounding boxes drawn directly on your images
- 📊 **Detection summary** - View detection count, confidence scores, and inference time
- 🖼️ **Example images** - Pre-loaded sample images to test
- 💾 **Download results** - Save annotated images with one click

### Using the Interface

1. **Upload an image**: Click or drag-and-drop an image into the upload area
2. **Adjust parameters** (optional):
   - **Confidence Threshold** (0.0-1.0): Minimum confidence for detections
   - **NMS Threshold** (0.0-1.0): IoU threshold for duplicate removal
3. **Click "Detect Barcodes"** to run inference
4. **View results**:
   - Left panel: Original image with bounding boxes
   - Right panel: Detection summary with confidence scores
5. **Download**: Click the download button to save the annotated image

### Example Workflow

```bash
# Terminal 1: Start API
cd /path/to/barcodetask
uvicorn api.app:app --reload

# Terminal 2: Start Gradio UI
python api/gradio_app.py
```

Then open **`http://localhost:7860`** in your browser and start detecting!

### Troubleshooting Gradio

**"Cannot connect to API" error:**
- Make sure the API is running on `http://localhost:8000`
- Check API health: `curl http://localhost:8000/health`

**Port already in use:**
```bash
# Change the Gradio port by editing gradio_app.py
# Look for: demo.launch(server_port=7860)
# Change to: demo.launch(server_port=7861)
```

## 📡 API Endpoints

### 1. **POST /predict** - Detect Barcodes

Upload an image and get barcode detections.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict?conf_threshold=0.5&nms_threshold=0.4" \
  -F "file=@path/to/image.jpg"
```

**Query Parameters:**
- `conf_threshold` (float, 0.0-1.0): Confidence threshold for detections (default: 0.5)
- `nms_threshold` (float, 0.0-1.0): IoU threshold for Non-Maximum Suppression (default: 0.4)

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "bbox": [120, 45, 350, 180],
      "label": "barcode",
      "confidence": 0.92,
      "class_probability": 0.88
    }
  ],
  "num_detections": 1,
  "image_shape": {
    "width": 640,
    "height": 480
  },
  "parameters": {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4
  },
  "inference_time_seconds": 0.145
}
```

### 2. **GET /health** - Health Check

Check if the API is running and model is loaded.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### 3. **GET /** - API Information

Get API metadata and available endpoints.

**Request:**
```bash
curl http://localhost:8000/
```

### 4. **GET /docs** - Interactive Documentation

FastAPI automatically generates interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing the API

A test script is provided to easily test the API with sample images.

### Basic Usage

```bash
# Test with a specific image
python api/test_api.py --image path/to/image.jpg

# Test with auto-detected sample image
python api/test_api.py

# Custom thresholds
python api/test_api.py --image image.jpg --conf 0.6 --nms 0.3

# Only test health endpoint
python api/test_api.py --health-only
```

### Test Script Features

The test script will:
1. ✅ Check API health
2. 📤 Send image to `/predict` endpoint
3. 📊 Display predictions in terminal
4. 🎨 Draw bounding boxes on image
5. 💾 Save annotated image to `prediction_result.jpg`

### Example Output

```
============================================================
🔍 BARCODE DETECTION API TEST
============================================================
🏥 Testing health endpoint...
   ✅ API is healthy
   Device: cpu

📸 Loading image: barcode_dataset/images/valid/IMG_001.jpg
   Image size: (640, 480)

🚀 Sending request to http://localhost:8000/predict
   Confidence threshold: 0.5
   NMS threshold: 0.4

✅ Prediction successful!
   Inference time: 0.145s
   Detections: 2

📊 Detected barcodes:
   1. bbox: [120, 45, 350, 180], confidence: 0.920
   2. bbox: [400, 200, 580, 320], confidence: 0.856

🎨 Drawing bounding boxes...
   ✓ Visualization saved
   Saved result to: prediction_result.jpg

============================================================
✅ TEST COMPLETED SUCCESSFULLY
============================================================
```

## 📦 Project Structure

```
barcodetask/
├── api/
│   ├── app.py              # FastAPI application
│   ├── gradio_app.py       # Gradio web interface
│   ├── model_loader.py     # Model loading singleton
│   ├── utils.py            # Utility functions (NMS, preprocessing)
│   ├── test_api.py         # Test script with visualization
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # This file
├── src/
│   └── detector_v2.py      # Model architecture
├── checkpoints_v2/
│   └── latest_model.pt     # Trained model weights
├── Dockerfile             # Container definition
├── docker-compose.yml     # Docker Compose configuration
└── .dockerignore         # Files to exclude from build
```

## 🔧 Configuration

### Model Parameters

The model uses these default parameters:
- **Grid size:** 7x7
- **Boxes per cell:** 2
- **Image size:** 448x448 (automatically resized)
- **Device:** CPU (configurable)

### Inference Parameters

Adjust detection behavior via query parameters:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `conf_threshold` | 0.5 | 0.0-1.0 | Minimum confidence to keep detection |
| `nms_threshold` | 0.4 | 0.0-1.0 | IoU threshold for duplicate removal |

**Lower confidence threshold** = More detections (higher recall, more false positives)  
**Higher confidence threshold** = Fewer detections (higher precision, fewer false positives)

## 🐳 Docker Details

### Image Size

The Docker image is optimized for size:
- Base: `python:3.12-slim`
- PyTorch: CPU-only version
- Multi-stage build for minimal final image
- Expected size: ~2-3 GB

### Build Arguments

```bash
# Build with custom tag
docker build -t my-barcode-api:v1.0 .

# Build without cache
docker build --no-cache -t barcode-api .
```

### Container Management

```bash
# Stop the container
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after code changes
docker-compose up --build

# Remove everything
docker-compose down -v
```

## 🌐 Production Deployment

### Environment Variables

You can configure the API using environment variables:

```yaml
# docker-compose.yml
environment:
  - PYTHONUNBUFFERED=1
  - MODEL_PATH=checkpoints_v2/latest_model.pt
  - GRID_SIZE=7
  - NUM_BOXES=2
```

### Scaling

To run multiple instances:

```bash
# Scale to 3 instances
docker-compose up --scale barcode-api=3
```

### Reverse Proxy

For production, use a reverse proxy like Nginx:

```nginx
location /api {
    proxy_pass http://localhost:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

## 📊 Performance

Typical inference times (on CPU):
- **Single image:** ~0.1-0.3 seconds
- **Batch processing:** Not currently supported
- **Concurrent requests:** Limited by CPU cores

## 🐛 Troubleshooting

### API won't start

```bash
# Check if port 8000 is in use
lsof -i :8000

# View container logs
docker-compose logs
```

### Model not found error

Make sure `checkpoints_v2/latest_model.pt` exists:
```bash
ls -lh checkpoints_v2/latest_model.pt
```

### Connection refused

Ensure the API is running:
```bash
curl http://localhost:8000/health
```

### Low detection accuracy

Try adjusting thresholds:
- Lower `conf_threshold` to get more detections
- Adjust `nms_threshold` to control duplicate removal

## 📝 Example Usage in Python

```python
import requests

# Test image
image_path = "barcode_image.jpg"

# Send request
with open(image_path, "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        params={"conf_threshold": 0.5, "nms_threshold": 0.4}
    )

# Parse response
result = response.json()
print(f"Found {result['num_detections']} barcodes")

for pred in result['predictions']:
    bbox = pred['bbox']
    conf = pred['confidence']
    print(f"Barcode at {bbox} with confidence {conf:.2f}")
```
## 🤝 Support

For issues or questions:
1. Check the `/docs` endpoint for API documentation
2. Run `python api/test_api.py --health-only` to test connectivity
3. Check Docker logs: `docker-compose logs`

