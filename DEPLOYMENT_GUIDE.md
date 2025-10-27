# 🚀 Docker & API Deployment - Quick Start Guide

This guide covers how to build, run, and test the Barcode Detection API.

## 📦 What's Included

✅ FastAPI REST API with automatic documentation  
✅ Docker containerization (CPU-optimized)  
✅ Docker Compose for easy deployment  
✅ Test script with visualization  
✅ Health check endpoints  
✅ Non-Maximum Suppression (NMS)  
✅ Configurable confidence thresholds  

## 🎯 One-Command Deployment

```bash
# Build and run everything (recommended)
docker-compose up --build
```

That's it! The API is now running at `http://localhost:8000`

## 📡 Test the API

### Option 1: Interactive Documentation
Visit `http://localhost:8000/docs` in your browser for interactive Swagger UI

### Option 2: Test Script (Recommended)

```bash
# Install requests if not already installed
pip install requests

# Run the test script (auto-detects sample image)
python api/test_api.py

# Or specify your own image
python api/test_api.py --image path/to/barcode.jpg
```

The test script will:
1. Check API health
2. Send image for prediction
3. Display results in terminal
4. Save annotated image with bounding boxes

### Option 3: cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict barcodes
curl -X POST "http://localhost:8000/predict" \
  -F "file=@barcode_dataset/images/valid/your_image.jpg"
```

## 🛠️ Alternative: Docker Without Compose

```bash
# Build
docker build -t barcode-api .

# Run
docker run -p 8000:8000 barcode-api
```

## 📊 API Response Format

```json
{
  "success": true,
  "predictions": [
    {
      "bbox": [x1, y1, x2, y2],
      "label": "barcode",
      "confidence": 0.92,
      "class_probability": 0.88
    }
  ],
  "num_detections": 1,
  "image_shape": {"width": 640, "height": 480},
  "inference_time_seconds": 0.145
}
```

## 🎛️ Configuration

Adjust detection parameters via query parameters:

```bash
# Higher confidence threshold (fewer false positives)
curl -X POST "http://localhost:8000/predict?conf_threshold=0.7" \
  -F "file=@image.jpg"

# Adjust NMS threshold
curl -X POST "http://localhost:8000/predict?nms_threshold=0.3" \
  -F "file=@image.jpg"
```

## 🐛 Troubleshooting

**API won't start?**
```bash
# Check if port 8000 is in use
lsof -i :8000

# View logs
docker-compose logs
```

**Model not found?**
```bash
# Verify checkpoint exists
ls -lh checkpoints/detector_best.pt
```

**Connection refused?**
```bash
# Wait ~30 seconds for model loading, then test
curl http://localhost:8000/health
```

## 📂 Project Structure

```
api/
├── app.py              # FastAPI application
├── model_loader.py     # Model singleton
├── utils.py            # NMS, preprocessing utilities
├── test_api.py         # Test script with visualization
├── requirements.txt    # Dependencies
└── README.md          # Full documentation

Dockerfile              # Container definition
docker-compose.yml      # Compose configuration
.dockerignore          # Build exclusions
```

## 📚 Full Documentation

See `api/README.md` for comprehensive documentation including:
- Detailed API endpoints
- Production deployment guide
- Performance benchmarks
- Advanced configuration
- Python integration examples

## 🎉 Summary

You now have a production-ready barcode detection API that:
- ✅ Accepts images via HTTP POST
- ✅ Returns JSON predictions with bounding boxes
- ✅ Runs in a Docker container
- ✅ Includes health monitoring
- ✅ Has automatic API documentation
- ✅ Includes test script with visualization

**Build:** `docker-compose up --build`  
**Test:** `python api/test_api.py`  
**Docs:** `http://localhost:8000/docs`  

---

Built with FastAPI + PyTorch + Docker
