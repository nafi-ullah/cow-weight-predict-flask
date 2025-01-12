from sqlalchemy import Column, String, Integer, Float, ForeignKey, Date
from database import Base
from pydantic import BaseModel
from typing import List

class User(Base):
    __tablename__ = "user"
    userid = Column(String(6), primary_key=True, unique=True, nullable=False)  # 6-character unique ID
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, unique=True, nullable=False)
    cattle_farm_name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)

class Cattle(Base):
    __tablename__ = "cattle"
    cattle_id = Column(String(6), primary_key=True, unique=True, nullable=False)  # 6-character unique ID
    userid = Column(String(6), ForeignKey("user.userid"), nullable=False)
    color = Column(String, nullable=False)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    teeth_number = Column(Integer, nullable=False)
    foods = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    gender = Column(String, nullable=False)


class WeightPredict(Base):
    __tablename__ = "weight_predict"
    weight_predict_id = Column(String(6), primary_key=True, unique=True, nullable=False)  # 6-character unique ID
    cattle_id = Column(String(6), ForeignKey("cattle.cattle_id"), nullable=False)
    cattle_side_url = Column(String, nullable=False)
    cattle_rear_url = Column(String, nullable=False)
    weight = Column(Float, nullable=False)
    date = Column(String, nullable=False)





class WeightPredictionResponse(BaseModel):
    weight_predict_id: str
    cattle_id: str
    cattle_side_url: str
    cattle_rear_url: str
    weight: float
    date: str


class CattleWithWeightPredictions(BaseModel):
    cattle_id: str
    userid: str
    color: str
    name: str
    age: int
    teeth_number: int
    foods: str
    price: float
    gender: str
    weight_predictions: List[WeightPredictionResponse]

# class WeightPredictionResponseImages(BaseModel):
#     cattle_side_url: str
#     cattle_rear_url: str

class CattleWithWeightPredictions(BaseModel):
    cattle_id: str
    name: str
    age: int
    color: str
    teeth_number: int
    foods: str
    price: float
    gender: str
    weight_predictions: List[WeightPredictionResponse]

