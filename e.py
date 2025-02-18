import torch
import requests
import cv2
import numpy as np
import webbrowser
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

def detect_traffic_signs(frame, model):
    try:
        results = model.predict(frame)
        
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

def detect_and_get_location_from_camera():
    model = load_yolov8_model()
    
    if model is None:
        print("Model loading failed, cannot proceed.")
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        traffic_signs = detect_traffic_signs(frame, model)
        
        if traffic_signs:
            latitude, longitude = get_geolocation()
            if latitude and longitude:
                print(f"Detected traffic sign at Latitude: {latitude}, Longitude: {longitude}")
                
                osm_url = f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}#map=16/{latitude}/{longitude}"
                print(f"Opening map in browser: {osm_url}")
                
                webbrowser.open(osm_url)
            else:
                print("Unable to retrieve geolocation.")
        
        cv2.imshow('Live Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

detect_and_get_location_from_camera()

