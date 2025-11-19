"""
Energy-Demand-Forecasting-DevMLOps
Centralized Metric Engine (L5/L6 Production Quality)

Author: Corey Leath (Trojan3877)

Provides:
✔ RMSE, MAE, MAPE, MSE
✔ R² Score
✔ Horizon-wise error breakdown
✔ Aggregation helpers
✔ JSON-safe metric packaging for MLFlow / Airflow
✔ Consistent metrics across training/evaluation/inference
"""

import numpy as np
from sklearn.metrics import r2_score


# -----------------------------------------------------------
# Core Metrics
# -----------------------------------------------------------
def mse(true, pred):
    return np.mean((true - pred) ** 2)


def rmse(true, pred):
    return np.sqrt(mse(true, pred))


def mae(true, pred):
    return np.mean(np.abs(true - pred))


def mape(true, pred, epsilon=1e-7):
    return np.mean(np.abs((true - pred) / (true + epsilon))) * 100


def r2(true, pred):
    return r2_score(true, pred)


# -----------------------------------------------------------
# Horizon-Wise Evaluation
# -----------------------------------------------------------
def horizon_metrics(trues, preds):
    """
    Computes:
    ✔ RMSE(h)
    ✔ MAE(h)
    ✔ MAPE(h)
    for each forecast horizon step.

    Parameters
    ----------
    trues : np.array, shape (samples, horizon)
    preds : np.array, shape (samples, horizon)
    """
    horizon = trues.shape[1]
    metrics_list = []

    for h in range(horizon):
        t = trues[:, h]
        p = preds[:, h]

        metrics_list.append(
            {
                "horizon": h + 1,
                "rmse": rmse(t, p),
                "mae": mae(t, p),
                "mape": mape(t, p),
            }
        )

    return metrics_list


# -----------------------------------------------------------
# Aggregate All Metrics for a Model
# -----------------------------------------------------------
def aggregate_metrics(trues, preds):
    """
    Creates a full evaluation dictionary
    used by:
    - evaluate.py
    - train.py
    - dashboards
    - MLflow
    - Airflow DAGs
    - Monitoring systems
    """

    return {
        "rmse": rmse(trues, preds).item() if not np.isscalar(rmse(trues, preds)) else rmse(trues, preds),
        "mae": mae(trues, preds).item() if not np.isscalar(mae(trues, preds)) else mae(trues, preds),
        "mape": mape(trues, preds).item() if not np.isscalar(mape(trues, preds)) else mape(trues, preds),
        "r2": r2(trues.flatten(), preds.flatten()).item()
        if not np.isscalar(r2(trues.flatten(), preds.flatten()))
        else r2(trues.flatten(), preds.flatten()),
        "horizon_breakdown": horizon_metrics(trues, preds),
        "samples": len(trues),
    }


# -----------------------------------------------------------
# Pretty Print Helper
# -----------------------------------------------------------
def print_metrics(metrics):
    print("\n[ METRIC REPORT ]")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"MAE  : {metrics['mae']:.4f}")
    print(f"MAPE : {metrics['mape']:.2f}%")
    print(f"R²   : {metrics['r2']:.4f}")
    print("PER-HORIZON BREAKDOWN:")
    for h in metrics["horizon_breakdown"]:
        print(
            f"  Horizon {h['horizon']}: "
            f"RMSE={h['rmse']:.4f}, "
            f"MAE={h['mae']:.4f}, "
            f"MAPE={h['mape']:.2f}%"
        )
