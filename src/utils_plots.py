"""
Energy-Demand-Forecasting-DevMLOps
Advanced Plotting Utilities (L5/L6 Quality)

Author: Corey Leath (Trojan3877)

Provides:
✔ Training loss curves
✔ Actual vs predicted forecast plots
✔ Residual diagnostic plots
✔ Error histograms
✔ Correlation heatmaps
✔ Multi-horizon forecast visualization
✔ Saved PNG plots for GitHub & LinkedIn

All outputs saved automatically into:
artifacts/plots/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import ensure_dir


# -----------------------------------------------------------
# Setup Plot Style
# -----------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")


# -----------------------------------------------------------
# Save Helper
# -----------------------------------------------------------
def save_plot(fig, filename):
    ensure_dir("artifacts/plots")
    path = f"artifacts/plots/{filename}"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved plot → {path}")


# -----------------------------------------------------------
# Training Curve Plot
# -----------------------------------------------------------
def plot_training_curves(history):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")

    ax.set_title("Training & Validation Loss Curve", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    save_plot(fig, "training_validation_curve.png")


# -----------------------------------------------------------
# Actual vs Predicted (Single Horizon)
# -----------------------------------------------------------
def plot_actual_vs_pred(true, predicted, title="Forecast vs Actual"):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(true, label="Actual", linewidth=2)
    ax.plot(predicted, label="Predicted", linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Energy Demand")
    ax.legend()

    save_plot(fig, "actual_vs_predicted.png")


# -----------------------------------------------------------
# Multi-horizon Predictions
# -----------------------------------------------------------
def plot_multi_horizon(true_matrix, pred_matrix):
    """
    true_matrix : (samples, horizon)
    pred_matrix : (samples, horizon)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    horizon = true_matrix.shape[1]
    colors = sns.color_palette("magma", horizon)

    for h in range(horizon):
        ax.plot(
            true_matrix[:, h],
            color=colors[h],
            label=f"Actual t+{h+1}",
            linewidth=2,
        )
        ax.plot(
            pred_matrix[:, h],
            color=colors[h],
            linestyle="--",
            label=f"Predicted t+{h+1}",
        )

    ax.set_title("Multi-Horizon Forecast Evaluation")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Energy Demand")
    ax.legend(ncol=2)

    save_plot(fig, "multi_horizon_evaluation.png")


# -----------------------------------------------------------
# Residual Plot
# -----------------------------------------------------------
def plot_residuals(true, predicted):
    residuals = true - predicted

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(residuals, label="Residuals", color="red")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Residual Plot (Prediction Errors)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Residual")
    ax.legend()

    save_plot(fig, "residual_plot.png")


# -----------------------------------------------------------
# Error Distribution Histogram
# -----------------------------------------------------------
def plot_error_distribution(true, predicted):
    errors = true - predicted

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=40, color="steelblue", edgecolor="black")

    ax.set_title("Error Distribution (Histogram)")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")

    save_plot(fig, "error_histogram.png")


# -----------------------------------------------------------
# Correlation Matrix Heatmap
# -----------------------------------------------------------
def plot_correlation_matrix(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr()

    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix")

    save_plot(fig, "correlation_heatmap.png")
