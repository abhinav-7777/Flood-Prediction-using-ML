import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Correct paths
model_path = BASE_DIR / "models" / "flood_model.pkl"
data_path = BASE_DIR / "data" / "processed" / "cleaned_flood.csv"

print("📦 Looking for model at:", model_path)

# Load model
model = joblib.load(model_path)

# Load data
data = pd.read_csv(data_path)

# Auto-detect target column
target_column = data.columns[-1]

X = data.drop(columns=[target_column])
y = data[target_column]

y_pred = model.predict(X)

print("\nAccuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
