import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import numpy as np
import os

MODEL_DIR = 'models'
DATA_PATH = 'data/data.csv'

os.makedirs(MODEL_DIR, exist_ok=True)

try:
    df = pd.read_csv(DATA_PATH, delimiter=',')
    print(f"[OK] Data loaded successfully from {DATA_PATH} ({len(df)} rows)")
except FileNotFoundError:
    print(f"[ERROR] '{DATA_PATH}' not found. Please ensure training data exists.")
    exit(1)

# Preprocess features
df['sex_encoded'] = df['sex'].apply(lambda x: 1 if str(x).lower() == 'male' else 0)
df['smoker_encoded'] = df['smoker'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

region_map = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
df['region_encoded'] = df['region'].str.lower().map(region_map).fillna(0).astype(int)

# Domain-accurate logic for Claim Status (Binary Classification)
def generate_claim_status(row):
    risk_score = 0.1
    if row['smoker_encoded'] == 1:
        risk_score += 0.45
    if row['bmi'] >= 30:
        risk_score += 0.25
    elif row['bmi'] >= 25:
        risk_score += 0.10
    if row['age'] >= 50:
        risk_score += 0.20
    elif row['age'] >= 35:
        risk_score += 0.10
    if row['children'] >= 2:
        risk_score += 0.10
    
    np.random.seed(int(row['age'] * 100 + row['bmi'] * 10))
    return 1 if np.random.rand() < min(risk_score, 0.95) else 0

df['claim_status'] = df.apply(generate_claim_status, axis=1)

# Domain-accurate logic for Policy Type Recommendations (Multi-Class)
def assign_policy_type(row):
    charges = row['charges']
    if charges >= 20000 or (row['smoker_encoded'] == 1 and row['bmi'] >= 28):
        return 'Gold Comprehensive'
    elif charges >= 10000 or row['age'] >= 40:
        return 'Standard Care'
    else:
        return 'Basic Saver'

df['policy_type'] = df.apply(assign_policy_type, axis=1)

policy_label_encoder = LabelEncoder()
df['policy_type_encoded'] = policy_label_encoder.fit_transform(df['policy_type'])

# Feature sets
X_premium = df[['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']]
y_premium = df['charges']

X_claim = df[['age', 'bmi', 'smoker_encoded', 'region_encoded', 'children']]
y_claim = df['claim_status']

X_policy = df[['age', 'bmi', 'smoker_encoded', 'children']]
y_policy = df['policy_type_encoded']

# Model training functions
def train_and_save_regression_model(X, y, model_path):
    print(f"Training Premium Regression Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"  Validation RMSE: {rmse:.2f}")
    joblib.dump(model, model_path)

def train_and_save_classification_model(X, y, model_path):
    num_classes = len(y.unique())
    print(f"Training Classification Model ({num_classes} classes)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if num_classes > 2:
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            n_estimators=500,
            learning_rate=0.05,
            eval_metric='mlogloss',
            random_state=42
        )
    else:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            learning_rate=0.05,
            eval_metric='logloss',
            random_state=42
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Validation Accuracy: {acc:.4f}")
    joblib.dump(model, model_path)

# Execute training
train_and_save_regression_model(X_premium, y_premium, os.path.join(MODEL_DIR, 'premium_model.joblib'))
train_and_save_classification_model(X_claim, y_claim, os.path.join(MODEL_DIR, 'claim_model.joblib'))
train_and_save_classification_model(X_policy, y_policy, os.path.join(MODEL_DIR, 'policy_model.joblib'))

joblib.dump(policy_label_encoder, os.path.join(MODEL_DIR, 'policy_label_encoder.joblib'))
print(f"[OK] Saved Policy Label Encoder with classes: {list(policy_label_encoder.classes_)}")
print("[OK] All models trained and updated successfully!")
