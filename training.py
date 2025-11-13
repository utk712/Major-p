import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import os

# --- 1. Setup and Data Loading ---
MODEL_DIR = 'models'
DATA_PATH = 'data/data.csv'  # UPDATED PATH TO LOOK INSIDE THE 'data' FOLDER

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    # Loading data from the new path: 'data/data.csv'
    df = pd.read_csv(DATA_PATH, delimiter=',')
    print(f"Data loaded successfully from {DATA_PATH}")
except FileNotFoundError:
    print(f"Error: '{DATA_PATH}' not found. Please ensure your training data is available.")
    exit()

# --- 2. Data Preprocessing (Mapping based on app.py logic) ---

# Initialize LabelEncoder for policy target
policy_label_encoder = LabelEncoder()

# Preprocess features used in the models (assuming columns 'sex', 'region', 'policy_type' exist)
# NOTE: The feature columns must be created/encoded exactly as they are expected by the final models.
df['sex_encoded'] = df['sex'].apply(lambda x: 1 if x.lower() == 'male' else 0)
df['smoker_encoded'] = df['smoker'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

# Manually map regions based on the order in app.py's dictionary for consistency (0:southwest, 1:southeast, etc.)
region_map = {"southwest":0, "southeast":1, "northwest":2, "northeast":3}
df['region_encoded'] = df['region'].str.lower().map(region_map)

# Add dummy claim_status (binary: 0 or 1) based on some logic, e.g., random or based on charges
import numpy as np
np.random.seed(42)
df['claim_status'] = np.random.choice([0, 1], size=len(df))

# Add dummy policy_type (multi-class: e.g., 'Basic', 'Premium', 'Gold')
policy_types = ['Basic', 'Premium', 'Gold']
df['policy_type'] = np.random.choice(policy_types, size=len(df))

# Encode policy type for the multi-class classification target
df['policy_type_encoded'] = policy_label_encoder.fit_transform(df['policy_type'])


# --- 3. Define Feature Sets and Targets ---

# Target 1: Premium (Regression) - Target variable must be 'charges'
X_premium = df[['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']]
y_premium = df['charges']

# Target 2: Claim Probability (Binary Classification) - Target variable must be 'claim_status' (1 or 0)
# NOTE: Ensure you have a binary column representing claim status.
X_claim = df[['age', 'bmi', 'smoker_encoded', 'region_encoded', 'children']]
y_claim = df['claim_status']

# Target 3: Policy Recommendation (Multi-Class Classification)
X_policy = df[['age', 'bmi', 'smoker_encoded', 'children']]
y_policy = df['policy_type_encoded']


# --- 4. Training and Saving Functions ---

def train_and_save_regression_model(X, y, model_path):
    """Trains an XGBoost Regressor for premium prediction."""
    print(f"Training Regression Model: {model_path}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    print(f"  Validation RMSE: {rmse:.2f}")

    joblib.dump(model, model_path)
    print(f"  Successfully saved model to {model_path}")

def train_and_save_classification_model(X, y, model_path):
    """Trains an XGBoost Classifier for binary or multi-class prediction."""
    print(f"Training Classification Model: {model_path}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_classes = len(y.unique())
    if num_classes > 2:
        # Multi-class for Policy Model
        model = xgb.XGBClassifier(objective='multi:softprob', num_class=num_classes, n_estimators=1000, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    else:
        # Binary for Claim Model
        model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Validation Accuracy: {accuracy:.4f}")

    joblib.dump(model, model_path)
    print(f"  Successfully saved model to {model_path}")


# --- 5. Execution ---

# 5.1 Train Premium Model (Regression)
train_and_save_regression_model(X_premium, y_premium, os.path.join(MODEL_DIR, 'premium_model.joblib'))

# 5.2 Train Claim Model (Binary Classification)
train_and_save_classification_model(X_claim, y_claim, os.path.join(MODEL_DIR, 'claim_model.joblib'))

# 5.3 Train Policy Model (Multi-Class Classification)
train_and_save_classification_model(X_policy, y_policy, os.path.join(MODEL_DIR, 'policy_model.joblib'))

# 5.4 Save Policy Label Encoder
joblib.dump(policy_label_encoder, os.path.join(MODEL_DIR, 'policy_label_encoder.joblib'))
print(f"\nSuccessfully saved Label Encoder to {os.path.join(MODEL_DIR, 'policy_label_encoder.joblib')}")

print("\nAll models and encoders have been trained and saved successfully.")
