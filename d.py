import torch
import requests
import cv2
import numpy as np
import webbrowser  # Import webbrowser to open URLs automatically
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
        detections = results[0].boxes  # Use the boxes attribute to get the detections
        
        # Print out the detected classes and their IDs to help debug
        print("Detected Classes and IDs:")
        for detection in detections:
            class_id = int(detection.cls)
            print(f"Class ID: {class_id}, Class Name: {model.names[class_id]}")
        
        # Filter detections for traffic signs (use the correct class ID for "Speed Limit 30")
        traffic_sign_class_id = 7  # Corrected class ID for "Speed Limit 30"
        
        traffic_signs = []
        for detection in detections:
            class_id = int(detection.cls)
            if class_id == traffic_sign_class_id:
                # Get the bounding box (xmin, ymin, xmax, ymax)
                bbox = detection.xywh[0]
                traffic_signs.append(bbox)
        
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
    
    if traffic_signs:
        # For each detected traffic sign, get the location
        latitude, longitude = get_geolocation()
        if latitude and longitude:
            print(f"Detected traffic sign at Latitude: {latitude}, Longitude: {longitude}")
            
            # Generate the OpenStreetMap URL
            osm_url = f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}#map=16/{latitude}/{longitude}"
            print(f"Opening map in browser: {osm_url}")
            
            # Open the URL in the default web browser
            webbrowser.open(osm_url)  # This will open the link in the default browser
        else:
            print("Unable to retrieve geolocation.")
        
        # Print or store the detected sign's coordinates (bbox)
        for bbox in traffic_signs:
            xmin, ymin, xmax, ymax = bbox
            print(f"Traffic Sign Detected at BBox: {xmin}, {ymin}, {xmax}, {ymax}")
            print(f"Location of Detected Traffic Sign: Latitude: {latitude}, Longitude: {longitude}")
            print(f"Opening map in browser: {osm_url}")
            webbrowser.open(osm_url)  # Open the map for each detected sign
    else:
        print("No traffic sign detected.")

# Example usage (replace '30.jpg' with your actual image or video feed path)
detect_and_get_location('30.jpg')  # Replace with your actual image file path

