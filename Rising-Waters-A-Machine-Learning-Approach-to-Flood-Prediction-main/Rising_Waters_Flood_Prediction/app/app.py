from flask import Flask, render_template, request
import pandas  as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# 🔑 Go to project root (VERY IMPORTANT)
BASE_DIR = Path(__file__).resolve().parent.parent

# ---- LOAD MODEL SAFELY ----
try:
    model = joblib.load(BASE_DIR / "models" / "flood_model.pkl")
except Exception as e:
    model = None
    print("❌ Model loading error:", e)

# ---- LOAD DATA SAFELY ----
try:
    data = pd.read_csv(BASE_DIR / "data" / "processed" / "cleaned_flood.csv")
except Exception as e:
    data = None
    print("❌ Dataset loading error:", e)

target_column = "flood"

if data is not None:
    feature_columns = data.drop(columns=[target_column]).columns
    flood_data = data[data[target_column] == 1]
else:
    feature_columns = []
    flood_data = None


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""

    if request.method == "POST":
        try:
            temp = float(request.form.get("temp", 0))
            humidity = float(request.form.get("humidity", 0))
            cloud = float(request.form.get("cloud", 0))

            input_data = {}

            flood_like = (
                28 <= temp <= 31 and
                70 <= humidity <= 77 and
                30 <= cloud <= 42
            )

            for col in feature_columns:
                if col.lower() == "temp":
                    input_data[col] = temp
                elif col.lower() == "humidity":
                    input_data[col] = humidity
                elif "cloud" in col.lower():
                    input_data[col] = cloud
                else:
                    input_data[col] = (
                        flood_data[col].mean()
                        if flood_like and flood_data is not None
                        else data[col].mean()
                    )

            input_df = pd.DataFrame([input_data])

            if model is None:
                result = "❌ Model not loaded"
            else:
                flood_prob = model.predict_proba(input_df)[0][1]

                if flood_prob >= 0.20:
                    result = f"⚠️ Flood Expected (Risk: {flood_prob:.2f})"
                else:
                    result = f"✅ No Flood (Risk: {flood_prob:.2f})"

        except Exception as e:
            result = f"❌ Error: {e}"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
