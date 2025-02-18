import torch
import requests
import cv2
import numpy as np
from ultralytics import YOLO

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

def load_yolov8_model(model_path='bestklaggleyolo8.pt'):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_traffic_signs(image, model):
    try:
        results = model.predict(image)
        
        detections = results[0].boxes
        
        print("Detected Classes and IDs:")
        for detection in detections:
            class_id = int(detection.cls)
            print(f"Class ID: {class_id}, Class Name: {model.names[class_id]}")
        
        traffic_sign_class_id = 7
        
        traffic_signs = []
        for detection in detections:
            class_id = int(detection.cls)
            if class_id == traffic_sign_class_id:
                bbox = detection.xywh[0]
                traffic_signs.append(bbox)
        
        return traffic_signs
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

def detect_and_get_location(image_path):
    model = load_yolov8_model()
    
    if model is None:
        print("Model loading failed, cannot proceed.")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return
    
    traffic_signs = detect_traffic_signs(image, model)
    
    if traffic_signs:
        latitude, longitude = get_geolocation()
        if latitude and longitude:
            print(f"Detected traffic sign at Latitude: {latitude}, Longitude: {longitude}")
        else:
            print("Unable to retrieve geolocation.")
        
        for bbox in traffic_signs:
            xmin, ymin, xmax, ymax = bbox
            print(f"Traffic Sign Detected at BBox: {xmin}, {ymin}, {xmax}, {ymax}")
            print(f"Location of Detected Traffic Sign: Latitude: {latitude}, Longitude: {longitude}")
    else:
        print("No traffic sign detected.")

detect_and_get_location('30.jpg')
