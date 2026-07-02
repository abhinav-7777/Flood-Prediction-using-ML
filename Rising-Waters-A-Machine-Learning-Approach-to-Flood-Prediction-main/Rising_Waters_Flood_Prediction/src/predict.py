import pandas as pd
import joblib
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
model_path = BASE_DIR / "models" / "flood_model.pkl"
data_path = BASE_DIR / "data" / "processed" / "cleaned_flood.csv"

print("📦 Loading model from:", model_path)
model = joblib.load(model_path)

# Load training data to get feature structure
data = pd.read_csv(data_path)

# Target column = last column
target_column = data.columns[-1]

# Feature columns used during training
feature_columns = data.drop(columns=[target_column]).columns
print("🧩 Model expects features:", list(feature_columns))

# Create ONE new sample with SAME columns
new_sample = pd.DataFrame([{
    col: data[col].mean() for col in feature_columns
}])

# (Optional) Change some values manually
if "Rainfall" in new_sample.columns:
    new_sample["Rainfall"] = 140
if "RiverLevel" in new_sample.columns:
    new_sample["RiverLevel"] = 9.2

# Predict
prediction = model.predict(new_sample)

if prediction[0] == 1:
    print("⚠️ Flood Expected")
else:
    print("✅ No Flood")
