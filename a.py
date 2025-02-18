import torch
from ultralytics import YOLO

# Function to load your custom YOLOv8 model (replace with the path to your model weights)
def load_yolov8_model(model_path='bestklaggleyolo8.pt'):
    try:
        # Load your custom YOLOv8 model
        model = YOLO(model_path)  # Make sure the model path is correct
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to get all class names in order from the YOLOv8 model
def get_class_names(model):
    try:
        # Get all class names from the model
        class_names = model.names
        return class_names
    except Exception as e:
        print(f"Error getting class names: {e}")
        return None

# Example usage
model = load_yolov8_model('bestklaggleyolo8.pt')  # Load the model (use your model path)
if model:
    class_names = get_class_names(model)
    if class_names:
        print("Class Names in Order:")
        for class_id, class_name in enumerate(class_names):
            print(f"{class_id}: {class_name}")

