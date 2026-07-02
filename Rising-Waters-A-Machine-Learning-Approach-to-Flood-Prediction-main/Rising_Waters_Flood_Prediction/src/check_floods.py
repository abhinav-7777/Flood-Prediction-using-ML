# src/check_floods.py
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "flood_model.pkl")
data = pd.read_csv(BASE_DIR / "data" / "processed" / "cleaned_flood.csv")

X = data.drop(columns=["flood"])
preds = model.predict(X)

print("Total Flood predictions:", preds.sum())
