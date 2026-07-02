import pandas as pd
import joblib
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load model and data
model = joblib.load(BASE_DIR / "models" / "flood_model.pkl")
data = pd.read_csv(BASE_DIR / "data" / "processed" / "cleaned_flood.csv")

# Separate features
X = data.drop(columns=["flood"])

# Predict
predictions = model.predict(X)

# Add predictions to dataset
data["Predicted_Flood"] = predictions

# Extract only Flood predictions (1)
flood_cases = data[data["Predicted_Flood"] == 1]

print("🌊 TOTAL FLOOD PREDICTIONS:", len(flood_cases))
print("\n🔹 Showing first 15 Flood predictions:\n")

print(flood_cases.head(15))
