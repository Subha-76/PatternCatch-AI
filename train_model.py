"""Train an LSTM or GRU regression model (optional user-run step).

Example from project root:
    python -m src.train_model --model lstm --epochs 20
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import load_behavior_data, chronological_split, FEATURE_COLUMNS
from src.sequence_builder import build_sequences

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_behavior.csv"
MODEL_DIR = ROOT / "models"


def make_model(model_type: str, sequence_length: int, feature_count: int) -> tf.keras.Model:
    """Create the requested sequential regression architecture."""
    if model_type not in {"lstm", "gru"}:
        raise ValueError("model_type must be 'lstm' or 'gru'.")
    recurrent = tf.keras.layers.LSTM if model_type == "lstm" else tf.keras.layers.GRU
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, feature_count)),
            recurrent(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train(model_type: str = "lstm", epochs: int = 20, sequence_length: int = 5) -> Path:
    """Train a model and save the model plus feature scaler."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    frame = load_behavior_data(DATA_PATH)
    train_frame, test_frame = chronological_split(frame)

    scaler = StandardScaler()
    train_frame = train_frame.copy()
    test_frame = test_frame.copy()
    train_frame[FEATURE_COLUMNS] = scaler.fit_transform(train_frame[FEATURE_COLUMNS])
    test_frame[FEATURE_COLUMNS] = scaler.transform(test_frame[FEATURE_COLUMNS])

    X_train, y_train = build_sequences(train_frame, sequence_length)
    X_test, y_test = build_sequences(test_frame, sequence_length)
    if len(X_train) < 2:
        raise ValueError(
            "Not enough training sequences. Add more chronological records or reduce "
            "the sequence length."
        )
    if len(X_test) == 0:
        raise ValueError(
            "Not enough test sequences. Add more data or reduce the sequence length."
        )

    model = make_model(model_type, sequence_length, len(FEATURE_COLUMNS))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=8,
        callbacks=[early_stop],
        verbose=1,
    )
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {test_loss:.3f}")
    print(f"Test MAE: {test_mae:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"thinkpattern_{model_type}.keras"
    model.save(model_path)
    np.savez(
        MODEL_DIR / f"thinkpattern_{model_type}_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        feature_columns=np.asarray(FEATURE_COLUMNS),
        sequence_length=np.asarray([sequence_length]),
    )
    print(f"Saved model to: {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=5)
    args = parser.parse_args()
    train(args.model, args.epochs, args.sequence_length)


if __name__ == "__main__":
    main()
