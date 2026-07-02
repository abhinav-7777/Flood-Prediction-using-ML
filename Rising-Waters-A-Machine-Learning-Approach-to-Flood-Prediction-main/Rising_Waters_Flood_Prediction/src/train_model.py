import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned data
data = pd.read_csv(BASE_DIR / "data" / "processed" / "cleaned_flood.csv")

# Target = last column
target_column = data.columns[-1]

X = data.drop(columns=[target_column])
y = data[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ⭐ KEY FIX: stronger class weight
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight={0: 1, 1: 6}
)

model.fit(X_train, y_train)

# Save model
model_dir = BASE_DIR / "models"
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / "flood_model.pkl")

print("✅ Model trained and saved successfully")
