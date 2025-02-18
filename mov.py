import torch
import requests
import cv2
import numpy as np
from ultralytics import YOLO

# Function to get approximate geolocation from IP
def get_geolocation():
    try:
        response = requests.get('http://ipinfo.io/json')
        data = response.json()
        location = data['loc'].split(',')
        latitude = location[0]
        longitude = location[1]
        return latitude, longitude
    except Exception as e:
        print(f"Error getting geolocation: {e}")
        return None, None

# Function to load your custom YOLOv8 model (replace with the path to your model weights)
def load_yolov8_model(model_path='bestklaggleyolo8.pt'):
    try:
        # Load your custom YOLOv8 model
        model = YOLO(model_path)  # Make sure the model path is correct
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to process image and detect traffic signs using YOLOv8
def detect_traffic_signs(image, model):
    try:
        # Perform inference using YOLOv8
        results = model.predict(image)
        
        # Get detected objects (boxes, labels, and confidences)
        detections = results.pandas().xywh[0]
        
        # Filter detections for traffic signs (replace 'traffic_sign_class_id' with your model's class ID)
        traffic_sign_class_id = 9  # Example class ID for traffic signs in COCO dataset, adjust as necessary
        
        traffic_signs = detections[detections['class'] == traffic_sign_class_id]
        return traffic_signs
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

# Function to simulate the full detection and location process
def detect_and_get_location(image_path):
    # Load your custom YOLOv8 model
    model = load_yolov8_model()
    
    if model is None:
        print("Model loading failed, cannot proceed.")
        return
    
    # Load the image from file (ensure the image path is correct)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return
    
    # Detect traffic signs in the image
    traffic_signs = detect_traffic_signs(image, model)
    
    if traffic_signs is not None and not traffic_signs.empty:
        # For each detected traffic sign, get the location
        latitude, longitude = get_geolocation()
        if latitude and longitude:
            print(f"Detected traffic sign at Latitude: {latitude}, Longitude: {longitude}")
        else:
            print("Unable to retrieve geolocation.")
        
        # Print or store the detected sign's coordinates (bbox)
        for index, row in traffic_signs.iterrows():
            print(f"Traffic Sign Detected at BBox: {row['xmin']}, {row['ymin']}, {row['xmax']}, {row['ymax']}")
    else:
        print("No traffic sign detected.")

# Example usage (replace '30.jpg' with your actual image or video feed path)
detect_and_get_location('30.jpg')  # Replace with your actual image file path

