# Energy Demand Forecasting - DevMLOps Pipeline 🚀⚡

# ⚡ Energy-Demand-Forecasting-DevMLOps  
**Advanced Time-Series Forecasting | MLOps-Ready | L5/L6 Software Engineering Quality**  
Author: **Corey Leath (Trojan3877)**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
[![MLOps](https://img.shields.io/badge/MLOps-Production_GRADE-green)]()
[![Docker](https://img.shields.io/badge/Docker-ready-informational)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-brightgreen)]()
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-active-blue)]()

---

## 🚀 **Project Overview**
This project is a **full MLOps-ready energy demand forecasting system** using advanced time-series deep learning models (LSTM, GRU, Transformer).  

It is built to meet **L5/L6-level engineering hiring standards** from top AI companies (OpenAI, Anthropic, Meta, Netflix, AWS, Nvidia).

Includes:

- Advanced preprocessing (lags, rolling windows, holiday encoding)
- Config-driven architecture
- Fully modular pipeline
- Multi-horizon deep learning models
- CI/CD automation
- Docker deployment
- FastAPI inference server
- Streamlit visualization dashboard
- Full experiment tracking & artifact logging

---

## 📁 **Project Structure**

Energy-Demand-Forecasting-DevMLOps
│
├── config/
│ └── config.yaml
│
├── data/
│ ├── raw.csv
│ └── processed.csv
│
├── models/
│ └── scaler.pkl
│
├── checkpoints/
│ └── model_timestamp.pth
│
├── artifacts/
│ ├── metrics.json
│ ├── evaluation_report.json
│ └── plots/
│ ├── training_validation_curve.png
│ ├── actual_vs_predicted.png
│ ├── multi_horizon_evaluation.png
│ ├── residual_plot.png
│ ├── error_histogram.png
│ └── correlation_heatmap.png
│
├── src/
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ ├── data_loader.py
│ ├── data_preprocess.py
│ ├── model.py
│ ├── utils.py
│ ├── utils_metrics.py
│ └── utils_plots.py
│
├── api.py
├── streamlit_app.py
├── Dockerfile
└── requirements.txt


<img width="404" alt="image" src="https://github.com/user-attachments/assets/2c375d72-6b7c-4a6e-babb-a6f8e20432de" />


                 ┌──────────────────────────────────────┐
                 │          RAW ENERGY DATA             │
                 │        (CSV, API, Streaming)         │
                 └──────────────────────────────────────┘
                                   │
                                   ▼
                     ┌────────────────────────┐
                     │   Data Preprocessing    │
                     │  - Lag Features         │
                     │  - Rolling Stats        │
                     │  - Holiday Encoding     │
                     │  - Timestamp Features   │
                     └────────────────────────┘
                                   │
                          Processed CSV
                                   │
                                   ▼
               ┌─────────────────────────────────────────────┐
               │                 Training                     │
               │  - LSTM / GRU / Transformer                 │
               │  - Checkpoints                              │
               │  - Early Stopping                           │
               │  - LR Scheduler                             │
               │  - Metrics + Plots                          │
               └─────────────────────────────────────────────┘
                                   │
                             Saved Model
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │                    Evaluation                         │
        │  - RMSE / MAE / MAPE / R²                            │
        │  - Multi-Horizon Analysis                            │
        │  - Diagnostic Plots                                  │
        └──────────────────────────────────────────────────────┘
                                   │
                        Artifacts, Reports, Plots
                                   │
                                   ▼
     ┌─────────────────────────────────────────────────────────────┐
     │                            Serving                          │
     │            (FastAPI + Docker + Streamlit UI)                │
     │                                                             │
     │    User Input Sequence  ────────────────▶  FastAPI Predict │
     │                                                             │
     │    Interactive Forecasting Dashboard ─────▶ Streamlit UI    │
     └─────────────────────────────────────────────────────────────┘



graph TD

subgraph Data Layer
A1[Raw Data] --> A2[Preprocessing]
A2 --> A3[Feature Engineering]
A3 --> A4[Processed Dataset]
end

subgraph Model Layer
B1[Build Model (LSTM/GRU/Transformer)]
B1 --> B2[Training Loop]
B2 --> B3[Checkpoints + Early Stopping]
B3 --> B4[Evaluation + Metrics]
end

subgraph Serving Layer
C1[FastAPI Inference Server]
C2[Streamlit UI Dashboard]
end

A4 --> B1
B4 --> C1
B4 --> C2


classDiagram
    class DataPreprocessor {
        +load_raw_data()
        +add_time_features()
        +add_lag_features()
        +add_rolling_features()
        +normalize()
        +run_preprocessing()
    }

    class ModelBuilder {
        +build_model()
        +get_device()
    }

    class Trainer {
        +train_model()
        +early_stopping
        +lr_scheduler
        +save_checkpoints()
    }

    class Evaluator {
        +run_evaluation()
        +calculate_metrics()
        +save_report()
    }

    class Predictor {
        +predict_single()
        +predict_multi()
        +batch_predict()
    }

    DataPreprocessor --> Trainer
    Trainer --> Evaluator
    Trainer --> Predictor
    Predictor --> FastAPI
    Predictor --> Streamlit



Production-grade **DevMLOps** pipeline for forecasting **smart grid energy demand**.  
Built with **Airflow, DVC, Docker, Kubernetes, Helm, Ansible, MLflow, Prometheus & Grafana**, with end-to-end automation.

## 🌎 Real-World Impact
- Helps optimize energy production  
- Reduces waste & carbon footprint  
- Supports smart grid resiliency  

## ⚙️ Pipeline Architecture
1. Data Ingestion (Airflow)  
2. Feature Engineering  
3. Model Training (XGBoost)  
4. Model Evaluation  
5. Model Registry (MLflow)  
6. Model Serving (FastAPI + Docker + K8s)  
7. Monitoring & Alerting (Prometheus & Grafana)  
8. CI/CD & IaC (GitHub Actions, Terraform, Ansible)  

## 🛠 Tech Stack
- **Language:** Python 3.x  
- **Orchestration:** Apache Airflow  
- **Data & Experiments:** DVC, MLflow  
- **Modeling:** scikit-learn, XGBoost, Prophet  
- **API:** FastAPI + Uvicorn  
- **Deploy:** Docker, Kubernetes, Helm, Ansible, Terraform  
- **Monitoring:** Prometheus + Grafana  
- **CI/CD:** GitHub Actions  

## 🚀 Getting Started
```bash
git clone https://github.com/YourUsername/Energy-Demand-Forecasting-DevMLOps.git
cd Energy-Demand-Forecasting-DevMLOps
make deps
make airflow-up
dvc pull
docker-compose up -d         # for API
cd mlflow && docker-compose up -d
cd monitoring && docker-compose up -d
ansible-playbook ansible/playbook.yaml -i ansible/inventories/dev.ini









