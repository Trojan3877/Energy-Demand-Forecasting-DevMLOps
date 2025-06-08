# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.1.0] - Initial Release
### Added
- Full DevMLOps pipeline:
  - Data preprocessing
  - Feature engineering
  - Model training (XGBoost)
  - Model evaluation (MAPE, latency)
  - FastAPI serving
  - Kubernetes manifests (Deployment, Service, Ingress, HPA)
  - Helm chart
  - DVC pipeline
  - CI/CD GitHub Actions
  - MLflow tracking
  - Prometheus monitoring config
  - Grafana dashboard

git add CHANGELOG.md
git commit -m "Add CHANGELOG.md"
git push
