import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

def load_yolov8_model():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Using the smallest version for speed; adjust as needed
    return model

def run_object_detection(model, image_path):
    # Load the image
    #image = Image.open(image_path).convert('RGB')
    image_np = np.array(image_path)
    
    # Run inference
    results = model(image_np)
    
    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = model.names[cls]  # Class name

            detections.append({
                'box': [x1, y1, x2, y2],
                'confidence': conf,
                'label': label
            })
    
    return detections