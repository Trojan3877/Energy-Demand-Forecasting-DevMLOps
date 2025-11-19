"""
Energy-Demand-Forecasting-DevMLOps
Core Model Architecture — LSTM / GRU / Transformer

Author: Corey Leath (Trojan3877)

This module provides:
✔ Hybrid forecasting architectures (LSTM / GRU / Transformer)
✔ Multi-step forecasting support
✔ Dropout regularization
✔ Weight initialization
✔ GPU/MPS/CPU device auto-detection
✔ Config-driven architecture
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------
# Device Resolver (GPU → MPS → CPU)
# ---------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------
# Weight Initialization
# ---------------------------------------------------------
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    if isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


# ---------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

        self.apply(init_weights)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]   # last hidden state
        x = self.fc(x)
        return x


# ---------------------------------------------------------
# GRU Model
# ---------------------------------------------------------
class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

        self.apply(init_weights)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]  # final hidden state
        x = self.fc(x)
        return x


# ---------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------
class TransformerForecaster(nn.Module):
    def __init__(self, input_size, num_heads, hidden_dim, num_layers, dropout, output_size):
        super().__init__()

        self.input_projection = nn.Linear(input_size, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 4
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, output_size)

        self.apply(init_weights)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# ---------------------------------------------------------
# Model Factory (LSTM / GRU / Transformer)
# ---------------------------------------------------------
def build_model(config, input_size, output_size):
    model_type = config["model"]["type"]
    hidden_dim = config["model"]["hidden_dim"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]

    if model_type == "lstm":
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )

    elif model_type == "gru":
        model = GRUForecaster(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )

    elif model_type == "transformer":
        model = TransformerForecaster(
            input_size=input_size,
            num_heads=config["model"]["num_heads"],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    device = get_device()
    model.to(device)

    print(f"[INFO] Model '{model_type}' built on: {device}")
    print(model)

    return model
