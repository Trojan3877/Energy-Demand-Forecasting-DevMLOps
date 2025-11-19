
---

# ✅ **2. FastAPI Server (api.py)**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch

from src.utils import load_data
from src.predict import predict_single, prepare_sequence
from src.model import build_model, get_device
from src.utils import load_scaler
import yaml

app = FastAPI(title="Energy Demand Forecasting API")

# Load Config, Model, Scaler
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

df = load_data(config["data"]["path"])
input_dim = df.shape[1]

device = get_device()
model = build_model(config, input_dim, config["data"]["forecast_horizon"])
ckpt = sorted([f"checkpoints/{f}" for f in os.listdir("checkpoints")])[-1]
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

scaler = load_scaler("models/scaler.pkl")

class ForecastRequest(BaseModel):
    sequence: list  # list of values

@app.post("/predict")
def predict(req: ForecastRequest):
    seq = np.array(req.sequence).reshape(-1, input_dim)
    pred = predict_single(model, seq, scaler, device)
    return {"forecast": pred.tolist()}
