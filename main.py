from fastapi import FastAPI, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import Base, engine, SessionLocal
from pydantic import BaseModel
from models import User, Cattle, WeightPredict, CattleWithWeightPredictions, WeightPredictionResponse
from typing import List
from crud import create_user, create_cattle, create_weight_prediction, get_all_users, get_all_cattle, get_all_weight_predictions
from utils import generate_unique_userid, generate_unique_cattle_id, generate_unique_weight_predict_id
import numpy as np
import pickle
import requests
import os
import uuid
import time

from fastapi.middleware.cors import CORSMiddleware



class UserCreate(BaseModel):
    full_name: str
    email: str
    password: str
    cattle_farm_name: str
    location: str
    phone_number: str

class CattleCreate(BaseModel):
    userid: str
    color: str
    name: str
    age: int
    teeth_number: int
    foods: str
    price: float
    gender: str

class WeightPredictionCreate(BaseModel):
    cattle_id: str
    cattle_side_url: str
    cattle_rear_url: str
    weight: float
    date: str

app = FastAPI()

# Create tables
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/")
def add_user(user: UserCreate, db: Session = Depends(get_db)):
    # Generate a unique 6-character user ID
    userid = generate_unique_userid(db)
    
    return create_user(
        db,
        userid,
        user.full_name,
        user.email,
        user.password,
        user.cattle_farm_name,
        user.location,
        user.phone_number,
    )

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/login/")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    # Query the user by email
    user = db.query(User).filter(User.email == request.email).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Match plain password directly
    if user.password != request.password:
        raise HTTPException(status_code=401, detail="Invalid password")
    
    # Return all user information
    return {
        "userid": user.userid,
        "full_name": user.full_name,
        "email": user.email,
        "cattle_farm_name": user.cattle_farm_name,
        "location": user.location,
        "phone_number": user.phone_number,
    }


@app.get("/users/")
def list_users(db: Session = Depends(get_db)):
    return get_all_users(db)

@app.post("/cattle/")
def add_cattle(cattle: CattleCreate, db: Session = Depends(get_db)):
    # Generate a unique 6-character cattle ID
    cattle_id = generate_unique_cattle_id(db)

    new_cattle = Cattle(
        cattle_id=cattle_id,
        userid=cattle.userid,
        color=cattle.color,
        name=cattle.name,
        age=cattle.age,
        teeth_number=cattle.teeth_number,
        foods=cattle.foods,  # Store list as comma-separated string
        price=cattle.price,
        gender=cattle.gender,
    )
    db.add(new_cattle)
    db.commit()
    db.refresh(new_cattle)
    return new_cattle


@app.get("/cattle/")
def list_cattle(db: Session = Depends(get_db)):
    return get_all_cattle(db)

@app.post("/weight-prediction/")
def add_weight_prediction(
    weight_prediction: WeightPredictionCreate, db: Session = Depends(get_db)
):
    # Generate a unique 6-character weight prediction ID
    weight_predict_id = generate_unique_weight_predict_id(db)

    new_weight_prediction = WeightPredict(
        weight_predict_id=weight_predict_id,
        cattle_id=weight_prediction.cattle_id,
        cattle_side_url=weight_prediction.cattle_side_url,
        cattle_rear_url=weight_prediction.cattle_rear_url,
        weight=weight_prediction.weight,
        date=weight_prediction.date,
    )
    db.add(new_weight_prediction)
    db.commit()
    db.refresh(new_weight_prediction)
    return new_weight_prediction



@app.get("/weight-predictions/")
def list_weight_predictions(db: Session = Depends(get_db)):
    return get_all_weight_predictions(db)



@app.get("/cattle/{cattle_id}/info")
def get_cattle_with_weight_predictions(cattle_id: str, db: Session = Depends(get_db)):
    # Fetch cattle information from the database
    cattle = db.query(Cattle).filter(Cattle.cattle_id == cattle_id).first()
    
    if not cattle:
        raise HTTPException(status_code=404, detail="Cattle not found")
    
    # Fetch all weight predictions related to the given cattle_id
    weight_predictions = db.query(WeightPredict).filter(WeightPredict.cattle_id == cattle_id).all()

    # Convert weight predictions to Pydantic model list
    weight_predictions_response = [
        WeightPredictionResponse(
            weight_predict_id=wp.weight_predict_id,
            cattle_id=wp.cattle_id,
            cattle_side_url=wp.cattle_side_url,
            cattle_rear_url=wp.cattle_rear_url,
            weight=wp.weight,
            date=wp.date,
        )
        for wp in weight_predictions
    ]
    
    # Create the response model combining cattle data and weight predictions
    response = CattleWithWeightPredictions(
        cattle_id=cattle.cattle_id,
        userid=cattle.userid,
        color=cattle.color,
        name=cattle.name,
        age=cattle.age,
        teeth_number=cattle.teeth_number,
        foods=cattle.foods,
        price=cattle.price,
        gender=cattle.gender,
        weight_predictions=weight_predictions_response
    )
    
    return response


@app.get("/cattles/{userid}/")
def get_cattles_by_userid(userid: str, db: Session = Depends(get_db)):
    # Fetch all cattles by the given user ID
    cattles = db.query(Cattle).filter(Cattle.userid == userid).all()
    
    if not cattles:
        raise HTTPException(status_code=404, detail="No cattle found for this user")

    # Prepare response
    response = []
    for cattle in cattles:
        # Fetch related weight predictions for each cattle
        weight_predictions = db.query(WeightPredict).filter(WeightPredict.cattle_id == cattle.cattle_id).all()

        # Convert weight prediction images into a list of responses
        weight_predictions_response = [
            WeightPredictionResponse(
                weight_predict_id=wp.weight_predict_id,
                cattle_id=wp.cattle_id,
                cattle_side_url=wp.cattle_side_url,
                cattle_rear_url=wp.cattle_rear_url,
                weight=wp.weight,
                date=wp.date,
            )
            for wp in weight_predictions
        ]
        
        # Append cattle with its related weight prediction images to the response
        response.append(
            CattleWithWeightPredictions(
                cattle_id=cattle.cattle_id,
                name=cattle.name,
                age=cattle.age,
                color=cattle.color,
                teeth_number=cattle.teeth_number,
                foods=cattle.foods,  # Assuming foods is a comma-separated string
                price=cattle.price,
                gender=cattle.gender,
                weight_predictions=weight_predictions_response
            )
        )

    return response




# Output directory for downloaded images
output_directory = './output_images'
os.makedirs(output_directory, exist_ok=True)

# Load the trained YOLOv8 model
from ultralytics import YOLO
model = YOLO('./best_seg_yolov8l.pt')

# Load the pickled linear regression model
with open('./Pickle_RL_Model.pkl', 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

# Dictionary to map gender
gender_dict = {'F': 1, 'M': 0}

# Function to download image from URL
def download_image(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download image from {url}")

# Function to get predictions from YOLO model
def get_predictions(image_path, model):
    results = model(image_path)
    return results

# Function to calculate polygon area for segmentation
def calculate_polygon_area(polygon):
    x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
    y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
    area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))
    return area

# Function to find areas for different classes (cattle, etc.)
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

# Function to extract features for prediction
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

# Pydantic model to receive the request body for prediction
class PredictRequest(BaseModel):
    cattle_id: str
    gender: str
    cattle_side_url: str
    cattle_rear_url: str

# FastAPI route to predict weight
@app.post("/predict_weight")
def predict_weight(request: PredictRequest, db: Session = Depends(get_db)):
    try:
        gender = request.gender  # Replace with actual gender logic based on your use case
        
        # Generate unique filenames for images
        timestamp = int(time.time())
        unique_id = uuid.uuid4()
        side_image_filename = f'side_image_{timestamp}_{unique_id}.jpg'
        rear_image_filename = f'rear_image_{timestamp}_{unique_id}.jpg'
        
        side_image_path = os.path.join(output_directory, side_image_filename)
        rear_image_path = os.path.join(output_directory, rear_image_filename)
        
        # Download the images
        download_image(request.cattle_side_url, side_image_path)
        download_image(request.cattle_rear_url, rear_image_path)
        
        # Get features and predict weight
        features = get_features(gender, side_image_path, rear_image_path)
        weight_prediction = Pickled_LR_Model.predict([features])[0]

        # Save to database (assuming you have weight_predict model set up)
        weight_predict_id = generate_unique_weight_predict_id(db)  # You need to define this ID generation logic
        new_weight_predict = WeightPredict(
            weight_predict_id=weight_predict_id,
            cattle_id=request.cattle_id,
            cattle_side_url=request.cattle_side_url,
            cattle_rear_url=request.cattle_rear_url,
            weight=weight_prediction,
            date=str(time.strftime("%Y-%m-%d"))
        )
        
        db.add(new_weight_predict)
        db.commit()

        return {"predicted_weight": weight_prediction, "message": "Weight prediction saved successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))