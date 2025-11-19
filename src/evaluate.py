"""
Energy-Demand-Forecasting-DevMLOps
Evaluation Pipeline

Author: Corey Leath (Trojan3877)

Provides:
✔ Full test set evaluation
✔ RMSE / MAE / MAPE / R² metrics
✔ Horizon-wise error analysis
✔ Visual evaluation plots
✔ Exported JSON evaluation report
"""

import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from src.model import build_model, get_device
from src.utils import (
    load_scaler,
    load_data,
    ensure_dir,
    save_json
)
from src.predict import prepare_sequence


# -----------------------------------------------------------
# Metrics
# -----------------------------------------------------------
def RMSE(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))


def MAE(true, pred):
    return np.mean(np.abs(true - pred))


def MAPE(true, pred, epsilon=1e-7):
    return np.mean(np.abs((true - pred) / (true + epsilon))) * 100


# -----------------------------------------------------------
# Load Model + Scaler
# -----------------------------------------------------------
def load_model_for_eval(config_path="config/config.yaml", checkpoint_path=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df = load_data(config["data"]["path"])
    input_dim = df.shape[1]

    model = build_model(
        config=config,
        input_size=input_dim,
        output_size=config["data"]["forecast_horizon"]
    )

    if checkpoint_path is None:
        checkpoints = sorted(
            [ckpt for ckpt in os.listdir("checkpoints") if ckpt.endswith(".pth")]
        )
        if not checkpoints:
            raise FileNotFoundError("No model checkpoints found in /checkpoints")
        checkpoint_path = os.path.join("checkpoints", checkpoints[-1])

    device = get_device()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    scaler = load_scaler("models/scaler.pkl")

    print(f"[INFO] Loaded model from: {checkpoint_path}")
    return model, scaler, config, df, device



# -----------------------------------------------------------
# Evaluate on Test Set
# -----------------------------------------------------------
def evaluate_full_testset(model, scaler, config, df, device):
    """
    Evaluates model performance on full test dataset
    using sliding windows for multi-step predictions.
    """
    seq_len = config["data"]["sequence_length"]
    horizon = config["data"]["forecast_horizon"]
    test_ratio = config["data"]["test_ratio"]

    test_size = int(len(df) * test_ratio)
    test_df = df[-test_size:]

    preds = []
    trues = []

    for i in range(len(test_df) - seq_len - horizon):
        seq = test_df[i : i + seq_len].values
        true = test_df[i + seq_len : i + seq_len + horizon].values.flatten()

        inp = prepare_sequence(seq, scaler, device)

        with torch.no_grad():
            pred = model(inp).cpu().numpy().flatten()

        pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

        preds.append(pred)
        trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)

    return preds, trues



# -----------------------------------------------------------
# Plot Prediction vs Actual
# -----------------------------------------------------------
def plot_evaluation(trues, preds, save_path="artifacts/evaluation_plot.png"):
    ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(12, 5))
    plt.plot(trues[:, 0], label="Actual (t+1)", linewidth=2)
    plt.plot(preds[:, 0], label="Predicted (t+1)", linestyle="--")
    plt.title("Model Evaluation - Next Step Forecast")
    plt.xlabel("Time Index")
    plt.ylabel("Energy Demand")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved evaluation plot → {save_path}")



# -----------------------------------------------------------
# Main Evaluation Script
# -----------------------------------------------------------
def run_evaluation(config_path="config/config.yaml"):
    model, scaler, config, df, device = load_model_for_eval(config_path)

    preds, trues = evaluate_full_testset(model, scaler, config, df, device)

    # -----------------------------
    # Aggregate Metrics
    # -----------------------------
    rmse = RMSE(trues, preds)
    mae = MAE(trues, preds)
    mape = MAPE(trues, preds)
    r2 = r2_score(trues.flatten(), preds.flatten())

    # Horizon-wise metrics
    horizon_metrics = []
    horizon = config["data"]["forecast_horizon"]

    for h in range(horizon):
        horizon_metrics.append({
            "horizon": h + 1,
            "rmse": RMSE(trues[:, h], preds[:, h]),
            "mae": MAE(trues[:, h], preds[:, h]),
            "mape": MAPE(trues[:, h], preds[:, h]),
        })

    # -----------------------------
    # Save Evaluation Report
    # -----------------------------
    report = {
        "rmse_total": rmse,
        "mae_total": mae,
        "mape_total": mape,
        "r2_score": r2,
        "horizon_metrics": horizon_metrics,
        "samples_evaluated": len(preds),
    }

    ensure_dir("artifacts")
    save_json(report, "artifacts/evaluation_report.json")

    # -----------------------------
    # Save Evaluation Plot
    # -----------------------------
    plot_evaluation(trues, preds)

    print("\n[ EVALUATION REPORT ]")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")
    print(f"Saved evaluation report → artifacts/evaluation_report.json\n")

    return report



# -----------------------------------------------------------
# Script Entry
# -----------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()
