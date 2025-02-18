import torch
import requests
import cv2
import numpy as np

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

# Function to load your custom YOLOv5 model (replace with the path to your model weights)
def load_yolov5_model():
    # Load your custom YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    return model

# Function to process image and detect traffic signs using YOLOv5
def detect_traffic_signs(image, model):
    # Perform inference using YOLOv5
    results = model(image)
    
    # Get detected objects (boxes, labels, and confidences)
    detections = results.pandas().xywh[0]
    
    # Filter detections for traffic signs (replace 'traffic_sign_class_id' with your model's class ID)
    traffic_sign_class_id = 9  # Example class ID for traffic signs in COCO dataset, adjust as necessary
    
    traffic_signs = detections[detections['class'] == traffic_sign_class_id]
    
    return traffic_signs

# Function to simulate the full detection and location process
def detect_and_get_location(image):
    # Load your custom YOLOv5 model
    model = load_yolov5_model()
    
    # Detect traffic signs
    traffic_signs = detect_traffic_signs(image, model)
    
    if not traffic_signs.empty:
        # For each detected traffic sign, get the location
        latitude, longitude = get_geolocation()
        print(f"Detected traffic sign at Latitude: {latitude}, Longitude: {longitude}")
        
        # You can also print or store the detected sign's coordinates (bbox)
        for index, row in traffic_signs.iterrows():
            print(f"Traffic Sign Detected at BBox: {row['xmin']}, {row['ymin']}, {row['xmax']}, {row['ymax']}")
    else:
        print("No traffic sign detected.")

# Example usage (replace 'input_image.jpg' with your actual image or video feed)
image = cv2.imread('30.jpg')  # You can replace this with a frame from your video feed
detect_and_get_location(image)

