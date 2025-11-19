"""
Energy-Demand-Forecasting-DevMLOps
Training Pipeline (LSTM / GRU / Transformer)

Author: Corey Leath (Trojan3877)

This file provides:
✔ Full training loop + evaluation
✔ Early stopping
✔ Model checkpointing
✔ Learning rate scheduling
✔ tqdm progress bars
✔ RMSE / MAE / MAPE metrics
✔ Config-driven training
✔ GPU/MPS/CPU support
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model import build_model, get_device
from src.utils import (
    load_data,
    split_data,
    create_sequences,
    save_metrics,
)

import yaml
import numpy as np
from datetime import datetime


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def RMSE(pred, true):
    return torch.sqrt(nn.MSELoss()(pred, true))


def MAE(pred, true):
    return nn.L1Loss()(pred, true)


def MAPE(pred, true, epsilon=1e-7):
    return torch.mean(torch.abs((true - pred) / (true + epsilon))) * 100


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []

    loop = tqdm(loader, desc="Training", leave=False)

    for seq, target in loop:
        seq, target = seq.to(device), target.to(device)

        optimizer.zero_grad()
        pred = model(seq)
        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        loop.set_postfix({"loss": loss.item()})

    return np.mean(losses)


# ---------------------------------------------------------
# Validation Loop
# ---------------------------------------------------------
def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    trues = []

    with torch.no_grad():
        for seq, target in loader:
            seq, target = seq.to(device), target.to(device)

            pred = model(seq)
            loss = criterion(pred, target)

            preds.append(pred.cpu())
            trues.append(target.cpu())
            losses.append(loss.item())

    preds = torch.cat(preds)
    trues = torch.cat(trues)

    return (
        np.mean(losses),
        RMSE(preds, trues).item(),
        MAE(preds, trues).item(),
        MAPE(preds, trues).item(),
    )


# ---------------------------------------------------------
# Early Stopping Class
# ---------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[INFO] EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.stop = True


# ---------------------------------------------------------
# Training Entry
# ---------------------------------------------------------
def train_model(config_path="config/config.yaml"):
    # -------------------------------
    # Load Config File
    # -------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # -------------------------------
    # Load Data
    # -------------------------------
    df = load_data(config["data"]["path"])

    train_df, val_df, test_df = split_data(
        df,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
    )

    seq_len = config["data"]["sequence_length"]
    horizon = config["data"]["forecast_horizon"]

    train_seq, train_targets = create_sequences(train_df, seq_len, horizon)
    val_seq, val_targets = create_sequences(val_df, seq_len, horizon)

    train_loader = DataLoader(
        list(zip(train_seq, train_targets)),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        list(zip(val_seq, val_targets)),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    input_size = train_seq.shape[-1]
    output_size = horizon

    # -------------------------------
    # Build Model
    # -------------------------------
    device = get_device()
    model = build_model(config, input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["training"]["patience"], verbose=True
    )

    # -------------------------------
    # Logging + Checkpoints
    # -------------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = f"checkpoints/model_{timestamp}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "rmse": [], "mae": [], "mape": []}

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(config["training"]["epochs"]):
        print(f"\n[ Epoch {epoch+1}/{config['training']['epochs']} ]")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        (
            val_loss,
            val_rmse,
            val_mae,
            val_mape,
        ) = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Logging
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["rmse"].append(val_rmse)
        history["mae"].append(val_mae)
        history["mape"].append(val_mape)

        print(
            f"[INFO] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"RMSE={val_rmse:.4f} | MAE={val_mae:.4f} | MAPE={val_mape:.2f}%"
        )

        # Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.stop:
            print("[INFO] Early stopping triggered — stopping training.")
            break

    # -------------------------------
    # Save Metrics + Curve Plot
    # -------------------------------
    save_metrics(history, "artifacts/metrics.json")

    os.makedirs("artifacts", exist_ok=True)
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("artifacts/training_curve.png")

    print(f"[INFO] Training complete. Model saved at: {checkpoint_path}")

    return model, history


# ---------------------------------------------------------
# Script Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    train_model()
