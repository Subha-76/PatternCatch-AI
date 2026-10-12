"""Loading, validation, and feature preparation for behavioral records."""
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "user_id",
    "timestamp",
    "study_minutes",
    "app_sessions",
    "tasks_completed",
    "sleep_hours",
    "next_day_study_minutes",
]
FEATURE_COLUMNS = [
    "study_minutes",
    "app_sessions",
    "tasks_completed",
    "sleep_hours",
]


def load_behavior_data(path: str | Path) -> pd.DataFrame:
    """Load CSV data and normalize its timestamp column."""
    frame = pd.read_csv(path)
    issues = validate_behavior_data(frame)
    if issues:
        raise ValueError("Invalid behavioral dataset: " + "; ".join(issues))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return frame


def validate_behavior_data(frame: pd.DataFrame) -> list[str]:
    """Return validation issues; an empty list means the schema is acceptable."""
    issues: list[str] = []
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        issues.append(f"Missing required columns: {', '.join(missing)}")
        return issues
    if frame.empty:
        issues.append("Dataset contains no rows.")
        return issues
    for col in FEATURE_COLUMNS + ["next_day_study_minutes"]:
        converted = pd.to_numeric(frame[col], errors="coerce")
        if converted.isna().any():
            issues.append(f"Column '{col}' must contain numeric values without blanks.")
    if frame["user_id"].isna().any():
        issues.append("Column 'user_id' contains missing values.")
    parsed_time = pd.to_datetime(frame["timestamp"], errors="coerce")
    if parsed_time.isna().any():
        issues.append("Column 'timestamp' contains invalid or missing timestamps.")
    return issues


def prepare_numeric_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric feature matrix X and next-day study-minutes target y."""
    missing = [col for col in FEATURE_COLUMNS + ["next_day_study_minutes"] if col not in frame]
    if missing:
        raise ValueError(f"Missing columns required for modelling: {missing}")
    ordered = frame.sort_values(["user_id", "timestamp"])
    X = ordered[FEATURE_COLUMNS].astype("float32")
    y = ordered["next_day_study_minutes"].astype("float32")
    return X, y


def chronological_split(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split globally by time to reduce leakage from future records."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5.")
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    split_at = int(len(ordered) * (1 - test_fraction))
    if split_at < 1 or split_at >= len(ordered):
        raise ValueError("Not enough rows to create chronological train/test splits.")
    return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()
