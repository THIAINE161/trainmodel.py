# train_model.py
"""
Student Enrollment Prediction (Logistic Regression)
Save this file as train_model.py and run: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

DATA_FILE = "student_data.csv"
MODEL_FILE = "logistic_enrollment_model.pkl"
SCALER_FILE = "scaler.pkl"

def load_data(path=DATA_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)

def prepare_features(df):
    # minimal preprocessing: choose numeric columns expected in sample CSV
    features = ['GPA', 'attendance_rate', 'study_hours', 'age', 'previous_enrollment']
    for c in features:
        if c not in df.columns:
            raise KeyError(f"Missing feature column in CSV: {c}")
    X = df[features].astype(float)
    y = df['will_enroll'].astype(int)  # target column
    return X, y

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and scaler for later use
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved model to: {MODEL_FILE}")
    print(f"Saved scaler to: {SCALER_FILE}")

    return model, scaler

def predict_example(model, scaler):
    # Example: [GPA, attendance_rate, study_hours, age, previous_enrollment]
    example = np.array([[3.2, 85, 4, 20, 1]])
    example_scaled = scaler.transform(example)
    pred = model.predict(example_scaled)[0]
    prob = model.predict_proba(example_scaled)[0,1]
    print(f"\nExample prediction -> Will enroll?: {'Yes' if pred==1 else 'No'} (probability {prob:.2f})")

def main():
    print("Loading data...")
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} rows.")

    X, y = prepare_features(df)
    model, scaler = train_and_evaluate(X, y)
    predict_example(model, scaler)

if __name__ == "__main__":
    main()
