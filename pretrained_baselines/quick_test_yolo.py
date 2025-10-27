from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
# Use relative path from workspace root, not from src/ directory
results = model.train(data="barcode_dataset/data.yaml", epochs=100, imgsz=640)