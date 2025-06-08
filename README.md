# Energy Demand Forecasting - DevMLOps Pipeline 🚀⚡

![CI](https://github.com/YourUsername/Energy-Demand-Forecasting-DevMLOps/actions/workflows/ci.yml/badge.svg)
![DVC](https://img.shields.io/badge/DVC-enabled-blue)
![Docker Pulls](https://img.shields.io/docker/pulls/energy-forecast-api)
![Helm](https://img.shields.io/badge/Helm-Chart-blue)
![K8s](https://img.shields.io/badge/Kubernetes-ready-green)
![Ansible](https://img.shields.io/badge/Ansible-automated-yellowgreen)
![MLflow](https://img.shields.io/badge/MLflow-tracking-orange)
![Prometheus](https://img.shields.io/badge/Prometheus-monitored-lightgrey)
![Grafana](https://img.shields.io/badge/Grafana-visualized-blue)
![License](https://img.shields.io/badge/license-MIT-green)

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



![Uploading image.png…]()





