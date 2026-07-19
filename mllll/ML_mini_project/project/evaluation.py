"""Evaluation helpers shared by analysis scripts."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
)


def classification_metrics(y_true, y_pred):
    return {
        'accuracy': round(accuracy_score(y_true, y_pred) * 100, 2),
        'precision': round(precision_score(y_true, y_pred, average='weighted') * 100, 2),
        'recall': round(recall_score(y_true, y_pred, average='weighted') * 100, 2),
        'f1': round(f1_score(y_true, y_pred, average='weighted') * 100, 2),
    }


def regression_metrics(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'r2': r2_score(y_true, y_pred),
    }


def within_tolerance_rate(error_series, tolerance_percent):
    within = int((error_series <= tolerance_percent).sum())
    total = len(error_series)
    outside = total - within
    return {
        'total': total,
        'within': within,
        'outside': outside,
        'within_rate': round((within / total) * 100, 2) if total else 0,
        'error_rate': round((outside / total) * 100, 2) if total else 0,
    }
