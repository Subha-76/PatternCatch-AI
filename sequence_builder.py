"""Utilities for converting ordered records into fixed-length sequences."""
import numpy as np
import pandas as pd

from src.data_preprocessing import FEATURE_COLUMNS


def build_sequences(
    frame: pd.DataFrame,
    sequence_length: int = 5,
    target_column: str = "next_day_study_minutes",
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-user sliding windows without crossing user boundaries.

    Returns:
        X: float32 array shaped (samples, sequence_length, features)
        y: float32 array shaped (samples,)
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")
    required = ["user_id", "timestamp", target_column, *FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    ordered = ordered.sort_values(["user_id", "timestamp"])

    for _, user_rows in ordered.groupby("user_id", sort=False):
        values = user_rows[FEATURE_COLUMNS].astype("float32").to_numpy()
        labels = pd.to_numeric(user_rows[target_column], errors="raise").to_numpy(dtype="float32")
        for end in range(sequence_length, len(user_rows)):
            sequences.append(values[end - sequence_length : end])
            targets.append(float(labels[end]))

    if not sequences:
        return (
            np.empty((0, sequence_length, len(FEATURE_COLUMNS)), dtype="float32"),
            np.empty((0,), dtype="float32"),
        )
    return np.stack(sequences).astype("float32"), np.asarray(targets, dtype="float32")
