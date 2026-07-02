import pandas as pd
from pathlib import Path

def preprocess_data(input_path, output_path):
    print("🔄 Loading data from:", input_path)
    data = pd.read_csv(input_path)

    print("🧹 Cleaning data...")
    # Fill numeric missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)

    print("💾 Saving cleaned data to:", output_path)
    data.to_csv(output_path, index=False)

    print("✅ Data preprocessing completed")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_file = BASE_DIR / "data" / "raw" / "flood.csv"
    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "cleaned_flood.csv"

    preprocess_data(input_file, output_file)
