from flask import Flask, request, jsonify
import numpy as np
import pickle
from ultralytics import YOLO

app = Flask(__name__)

# Load the trained YOLOv8 model
model = YOLO('./best_seg_yolov8l.pt')

# Load the pickled linear regression model
with open('./Pickle_RL_Model.pkl', 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

# Dictionary to map gender
gender_dict = {'F': 1, 'M': 0}

def get_predictions(image_path, model):
    results = model(image_path)
    return results

def calculate_polygon_area(polygon):
    x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
    y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
    area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))
    return area

def find_area_from_predictions(results):
    class_areas = {
        0: 0.0,  # Sticker
        1: 0.0,  # Cattle
    }
    
    for result in results:
        boxes = result.boxes  # Bounding boxes
        masks = result.masks  # Segmentation masks
        
        for box, mask in zip(boxes, masks):
            class_id = int(box.cls)  # Class ID
            if mask.xy is not None:
                for polygon in mask.xyn:
                    flat_polygon = [coord for point in polygon for coord in point]
                    area = calculate_polygon_area(flat_polygon)
                    class_areas[class_id] += area
            
    return [area for class_id, area in class_areas.items()]

def get_features(gender, side_image_path, rear_image_path):
    predictions = get_predictions(side_image_path, model)
    side_area = find_area_from_predictions(predictions)
    
    predictions = get_predictions(rear_image_path, model)
    rear_area = find_area_from_predictions(predictions)
    
    features = [
        gender_dict[gender],
        side_area[0] / side_area[1],
        side_area[0] / rear_area[1],
        rear_area[1] / side_area[1],
        side_area[0],
        side_area[1],
        rear_area[1]
    ]
    
    return features
@app.route('/', methods=['GET'])
def home():
    return "hello its nafi"


@app.route('/predict_weight', methods=['POST'])
def predict_weight():
    data = request.json
    
    gender = data.get('gender')
    side_image_link = data.get('side_image_link')
    rear_image_link = data.get('rear_image_link')
    
    if not all([gender, side_image_link, rear_image_link]):
        return jsonify({"error": "Missing data in request"}), 400
    
    try:
        features = get_features(gender, side_image_link, rear_image_link)
        weight_prediction = Pickled_LR_Model.predict([features])[0]
        
        return jsonify({"predicted_weight": weight_prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
