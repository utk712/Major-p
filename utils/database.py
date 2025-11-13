import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    policies = relationship("Policy", back_populates="user")
    claims = relationship("Claim", back_populates="user")

class Policy(Base):
    __tablename__ = 'policies'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    name = Column(String(255))
    probability = Column(Float)
    user = relationship("User", back_populates="policies")

class Claim(Base):
    __tablename__ = 'claims'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    prediction = Column(Boolean)
    probability = Column(Float)
    user = relationship("User", back_populates="claims")

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    premium = Column(Float)
    policies = Column(Text)  # JSON string of policies list
    claim = Column(Boolean)
    probability = Column(Float)
    personal_details = Column(Text)  # JSON string
    user_data = Column(Text)  # JSON string
    advice = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="predictions")

# Add predictions relationship to User
User.predictions = relationship("Prediction", back_populates="user")



# Database setup
DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+mysqlconnector://root:Utkarsh%402312@localhost:3306/insurencedb')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)

def save_user(email):
    db = get_db()
    try:
        # Check if user exists
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(email=email)
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    finally:
        db.close()

def get_user_data(email):
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = save_user(email)

        # Build the expected dict structure
        policies = [{"name": p.name, "prob": p.probability} for p in user.policies]
        claims = [{"prediction": c.prediction, "probability": c.probability} for c in user.claims]
        predictions = []
        for pred in user.predictions:
            predictions.append({
                "id": pred.id,
                "premium": pred.premium,
                "policies": json.loads(pred.policies),
                "claim": pred.claim,
                "probability": pred.probability,
                "personal_details": json.loads(pred.personal_details),
                "user_data": json.loads(pred.user_data),
                "advice": json.loads(pred.advice),
                "created_at": pred.created_at.isoformat()
            })
        history = []  # No history since History table is removed

        return {
            "email": user.email,
            "policies": policies,
            "claims": claims,
            "predictions": predictions,
            "history": history
        }
    finally:
        db.close()

def save_prediction_data(email, prediction_data):
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = save_user(email)

        # Save policies
        for policy_name, prob in prediction_data["policies"]:
            policy = Policy(user_id=user.id, name=policy_name, probability=prob)
            db.add(policy)

        # Save claim
        claim = Claim(user_id=user.id, prediction=prediction_data["claim"], probability=prediction_data["probability"])
        db.add(claim)

        # Save full prediction details
        prediction = Prediction(
            user_id=user.id,
            premium=prediction_data["premium"],
            policies=json.dumps(prediction_data["policies"]),
            claim=prediction_data["claim"],
            probability=prediction_data["probability"],
            personal_details=json.dumps(prediction_data["personal_details"]),
            user_data=json.dumps(prediction_data["user_data"]),
            advice=json.dumps(prediction_data["advice"])
        )
        db.add(prediction)

        db.commit()
    finally:
        db.close()