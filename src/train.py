"""
Energy-Demand-Forecasting-DevMLOps
Inference Pipeline

Author: Corey Leath (Trojan3877)

Provides:
✔ Load trained model + scaler
✔ Single prediction
✔ Batch prediction
✔ Multi-step forecasting
✔ Plotting predicted vs actual
"""

import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.model import build_model, get_device
from src.utils import load_scaler, load_data, ensure_dir


# -----------------------------------------------------------
# Load Model & Scaler
# -----------------------------------------------------------
def load_trained_model(config_path="config/config.yaml", checkpoint_path=None):
    """
    Loads the trained PyTorch model and scaler.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load scaler
    scaler = load_scaler("models/scaler.pkl")

    # Load data to detect input shape
    df = load_data(config["data"]["path"])
    input_dim = df.shape[1]

    model = build_model(
        config=config,
        input_size=input_dim,
        output_size=config["data"]["forecast_horizon"]
    )

    # Checkpoint handling
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

    print(f"[INFO] Loaded model from: {checkpoint_path}")
    return model, scaler, config



# -----------------------------------------------------------
# Prepare Input Sequence
# -----------------------------------------------------------
def prepare_sequence(sequence, scaler, device):
    """
    Scales and reshapes an input sequence for model inference.
    """
    sequence = scaler.transform(sequence)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    sequence = sequence.unsqueeze(0)  # shape: (1, seq_len, features)
    return sequence.to(device)



# -----------------------------------------------------------
# Single-Step Forecast
# -----------------------------------------------------------
def predict_single(model, sequence, scaler, device):
    """
    Predicts a single time step ahead.
    """
    inp = prepare_sequence(sequence, scaler, device)

    with torch.no_grad():
        pred = model(inp).cpu().numpy().flatten()

    # Reverse scaling (forecast horizon steps)
    pred_full = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    return pred_full



# -----------------------------------------------------------
# Multi-Step Forecast
# -----------------------------------------------------------
def predict_multi(model, initial_seq, steps, scaler, device):
    """
    Recursive multi-step forecasting.
    """
    seq = initial_seq.copy()
    predictions = []

    for _ in range(steps):
        pred = predict_single(model, seq, scaler, device)
        predictions.append(pred[0])

        # Append prediction to sequence for next step
        seq = np.vstack([seq[1:], [[pred[0]]]])

    return np.array(predictions)



# -----------------------------------------------------------
# Batch Forecast
# -----------------------------------------------------------
def batch_predict(model, sequences, scaler, device):
    """
    Predicts for multiple sequences at once.
    """
    preds = []
    for seq in sequences:
        pred = predict_single(model, seq, scaler, device)
        preds.append(pred[0])

    return np.array(preds)



# -----------------------------------------------------------
# Plot Predictions vs Actual
# -----------------------------------------------------------
def plot_predictions(true, predicted, save_path="artifacts/prediction_plot.png"):
    ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(10, 5))
    plt.plot(true, label="Actual", linewidth=2)
    plt.plot(predicted, label="Predicted", linestyle="--")
    plt.title("Energy Demand Forecast vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Energy Demand")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved prediction plot → {save_path}")



# -----------------------------------------------------------
# CLI Inference Entry
# -----------------------------------------------------------
def run_inference(steps=24):
    """
    Runs a complete forecasting sequence:
    ✔ Loads model + scaler
    ✔ Takes last N observations
    ✔ Produces multi-step forecast
    ✔ Saves plot
    """
    model, scaler, config = load_trained_model()

    df = load_data(config["data"]["path"])
    seq_len = config["data"]["sequence_length"]

    # Last N points used as initial seed
    last_seq = df[-seq_len:].values

    device = get_device()

    print("[INFO] Running multi-step forecast...")
    preds = predict_multi(
        model=model,
        initial_seq=last_seq,
        steps=steps,
        scaler=scaler,
        device=device,
    )

    true_future = df[-(steps+1):-1].values.flatten()

    plot_predictions(true_future, preds)

    return preds



# -----------------------------------------------------------
# Script Entry
# -----------------------------------------------------------
if __name__ == "__main__":
    run_inference()
