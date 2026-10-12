"""Regression metrics for behavioral outcome predictions."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE and RMSE after validating aligned, non-empty inputs."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size == 0:
        raise ValueError("Cannot evaluate empty arrays.")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same number of values.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs must contain only finite values.")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
