import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    address = Column(String(255), nullable=True)
    blood_group = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    policies = relationship("Policy", back_populates="user")
    claims = relationship("Claim", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")

class Policy(Base):
    __tablename__ = 'policies'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    name = Column(String(255))
    probability = Column(Float)
    user = relationship("User", back_populates="policies")

class Claim(Base):
    __tablename__ = 'claims'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    prediction = Column(Boolean)
    probability = Column(Float)
    user = relationship("User", back_populates="claims")

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    premium = Column(Float)
    policies = Column(Text)  # JSON string of policies list
    claim = Column(Boolean)
    probability = Column(Float)
    personal_details = Column(Text)  # JSON string
    user_data = Column(Text)  # JSON string
    advice = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="predictions")


# Database setup with robust fallback
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    DATABASE_URL = 'sqlite:///insurance.db'

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        pass
except Exception as e:
    print(f"[WARN] Primary DB connection failed ({e}). Falling back to SQLite insurance.db.")
    DATABASE_URL = 'sqlite:///insurance.db'
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    return SessionLocal()

def create_tables():
    Base.metadata.create_all(bind=engine)
    
    # Safe schema migration for SQLite column updates
    try:
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(users)")).fetchall()
            existing_cols = [row[1] for row in result]
            
            if 'password_hash' not in existing_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)"))
            if 'name' not in existing_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN name VARCHAR(255)"))
            if 'address' not in existing_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN address VARCHAR(255)"))
            if 'blood_group' not in existing_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN blood_group VARCHAR(50)"))
            conn.commit()
    except Exception as e:
        print(f"[WARN] Schema migration note: {e}")

def save_user(email, password=None):
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            p_hash = generate_password_hash(password) if password else None
            user = User(email=email, password_hash=p_hash)
            db.add(user)
            db.commit()
            db.refresh(user)
        elif password and not user.password_hash:
            user.password_hash = generate_password_hash(password)
            db.commit()
            db.refresh(user)
        return user
    finally:
        db.close()

def set_user_password(email, password):
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user and password:
            user.password_hash = generate_password_hash(password)
            db.commit()
            db.refresh(user)
            return True
        return False
    finally:
        db.close()

def verify_user_password(email, password):
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.password_hash or not password:
            return False
        return check_password_hash(user.password_hash, password)
    finally:
        db.close()

def update_user_profile(email, name=None, address=None, blood_group=None):
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(email=email)
            db.add(user)
        
        if name is not None:
            user.name = name
        if address is not None:
            user.address = address
        if blood_group is not None:
            user.blood_group = blood_group

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

        user_policies = [{"name": p.name, "prob": p.probability} for p in user.policies]
        claims = [{"prediction": c.prediction, "probability": c.probability} for c in user.claims]
        predictions = []

        for pred in user.predictions:
            try:
                p_list = json.loads(pred.policies) if pred.policies else []
            except Exception:
                p_list = []
            try:
                personal_details = json.loads(pred.personal_details) if pred.personal_details else {}
            except Exception:
                personal_details = {}
            try:
                user_data = json.loads(pred.user_data) if pred.user_data else {}
            except Exception:
                user_data = {}
            try:
                advice = json.loads(pred.advice) if pred.advice else {}
            except Exception:
                advice = {}

            predictions.append({
                "id": pred.id,
                "premium": pred.premium,
                "monthly_premium": round(pred.premium / 12, 2) if pred.premium else 0.0,
                "policies": p_list,
                "claim": pred.claim,
                "probability": pred.probability,
                "personal_details": personal_details,
                "user_data": user_data,
                "advice": advice,
                "created_at": pred.created_at.strftime("%Y-%m-%d %H:%M") if pred.created_at else ""
            })

        return {
            "email": user.email,
            "has_password": bool(user.password_hash),
            "name": user.name or "",
            "address": user.address or "",
            "blood_group": user.blood_group or "",
            "policies": user_policies,
            "claims": claims,
            "predictions": predictions
        }
    finally:
        db.close()

def save_prediction_data(email, prediction_data):
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = save_user(email)

        pd_details = prediction_data.get("personal_details", {})
        if pd_details.get("name"):
            user.name = pd_details.get("name")
        if pd_details.get("address"):
            user.address = pd_details.get("address")
        if pd_details.get("blood_group"):
            user.blood_group = pd_details.get("blood_group")

        for policy_name, prob in prediction_data.get("policies", []):
            policy = Policy(user_id=user.id, name=policy_name, probability=prob)
            db.add(policy)

        claim = Claim(
            user_id=user.id,
            prediction=prediction_data.get("claim", False),
            probability=prediction_data.get("probability", 0.0)
        )
        db.add(claim)

        prediction = Prediction(
            user_id=user.id,
            premium=prediction_data.get("premium", 0.0),
            policies=json.dumps(prediction_data.get("policies", [])),
            claim=prediction_data.get("claim", False),
            probability=prediction_data.get("probability", 0.0),
            personal_details=json.dumps(prediction_data.get("personal_details", {})),
            user_data=json.dumps(prediction_data.get("user_data", {})),
            advice=json.dumps(prediction_data.get("advice", {}))
        )
        db.add(prediction)

        db.commit()
    finally:
        db.close()

def get_prediction_by_id(pred_id, email):
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return None

        pred = db.query(Prediction).filter(Prediction.id == pred_id, Prediction.user_id == user.id).first()
        if not pred:
            return None

        try:
            p_list = json.loads(pred.policies) if pred.policies else []
        except Exception:
            p_list = []
        try:
            personal_details = json.loads(pred.personal_details) if pred.personal_details else {}
        except Exception:
            personal_details = {}
        try:
            user_data = json.loads(pred.user_data) if pred.user_data else {}
        except Exception:
            user_data = {}
        try:
            advice = json.loads(pred.advice) if pred.advice else {}
        except Exception:
            advice = {}

        return {
            "id": pred.id,
            "premium": pred.premium,
            "monthly_premium": round(pred.premium / 12, 2) if pred.premium else 0.0,
            "policies": p_list,
            "claim": pred.claim,
            "probability": pred.probability,
            "personal_details": personal_details,
            "user_data": user_data,
            "advice": advice,
            "created_at": pred.created_at.strftime("%Y-%m-%d %H:%M") if pred.created_at else ""
        }
    finally:
        db.close()