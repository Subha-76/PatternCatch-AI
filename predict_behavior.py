"""Load a trained model and predict a future study-duration estimate."""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_preprocessing import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def predict_next_study_minutes(
    recent_records: pd.DataFrame,
    model_type: str = "lstm",
) -> float:
    """Predict next-day study minutes from the most recent sequence.

    `recent_records` must contain exactly the most recent sequence_length rows,
    with columns matching FEATURE_COLUMNS. The model and scaler must already exist.
    """
    if model_type not in {"lstm", "gru"}:
        raise ValueError("model_type must be 'lstm' or 'gru'.")
    model_path = MODEL_DIR / f"thinkpattern_{model_type}.keras"
    scaler_path = MODEL_DIR / f"thinkpattern_{model_type}_scaler.npz"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Trained model files are missing. Run the optional training script first."
        )

    scaler_data = np.load(scaler_path, allow_pickle=False)
    expected_features = scaler_data["feature_columns"].tolist()
    sequence_length = int(scaler_data["sequence_length"][0])
    missing = [col for col in expected_features if col not in recent_records.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if len(recent_records) != sequence_length:
        raise ValueError(f"Expected exactly {sequence_length} recent rows.")

    values = recent_records[expected_features].astype("float32").to_numpy()
    mean = scaler_data["mean"]
    scale = np.where(scaler_data["scale"] == 0, 1, scaler_data["scale"])
    scaled = (values - mean) / scale
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(scaled[np.newaxis, ...], verbose=0)
    return max(0.0, float(prediction.reshape(-1)[0]))
