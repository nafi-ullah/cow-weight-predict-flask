import random
import string
from sqlalchemy.orm import Session
from models import User, Cattle, WeightPredict

def generate_unique_userid(db: Session) -> str:
    while True:
        userid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))  # Generate a random 6-character ID
        existing_user = db.query(User).filter(User.userid == userid).first()
        if not existing_user:
            return userid


def generate_unique_cattle_id(db: Session) -> str:
    while True:
        cattle_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        existing_cattle = db.query(Cattle).filter(Cattle.cattle_id == cattle_id).first()
        if not existing_cattle:
            return cattle_id


def generate_unique_weight_predict_id(db: Session):
    while True:
        # Generate a random 6-character ID
        weight_predict_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

        # Check if the generated ID already exists in the database
        existing_entry = db.query(WeightPredict).filter(WeightPredict.weight_predict_id == weight_predict_id).first()
        if not existing_entry:
            return weight_predict_id 
