from sqlalchemy.orm import Session
from models import User, Cattle, WeightPredict

# CRUD for User
def create_user(db: Session, userid: str, full_name: str, email: str, password: str, cattle_farm_name: str, location: str, phone_number: str):
    user = User(
        userid=userid,
        full_name=full_name,
        email=email,
        password=password,
        cattle_farm_name=cattle_farm_name,
        location=location,
        phone_number=phone_number,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_all_users(db: Session):
    return db.query(User).all()

# CRUD for Cattle
def create_cattle(db: Session, userid: str, color: str, name: str, age: int, teeth_number: int, foods: str, price: float, gender: str):
    cattle = Cattle(userid=userid, color=color, name=name, age=age, teeth_number=teeth_number, foods=foods, price=price, gender=gender)
    db.add(cattle)
    db.commit()
    db.refresh(cattle)
    return cattle

def get_all_cattle(db: Session):
    return db.query(Cattle).all()

# CRUD for Weight Prediction
def create_weight_prediction(db: Session, cattle_id: int, cattle_side_url: str, cattle_rear_url: str, weight: float, date: str):
    weight_predict = WeightPredict(cattle_id=cattle_id, cattle_side_url=cattle_side_url, cattle_rear_url=cattle_rear_url, weight=weight, date=date)
    db.add(weight_predict)
    db.commit()
    db.refresh(weight_predict)
    return weight_predict

def get_all_weight_predictions(db: Session):
    return db.query(WeightPredict).all()
