# mlflow/Dockerfile
# MLflow tracking server
# Author: Corey Leath

FROM python:3.9-slim

WORKDIR /mlflow

# Install MLflow + dependencies
RUN pip install --no-cache-dir mlflow[extras] boto3

# Expose MLflow default port
EXPOSE 5000

# Run MLflow server (local file store)
CMD mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5000

mkdir -p mlflow
git add mlflow/Dockerfile
git commit -m "Add MLflow Dockerfile for tracking server"
git push

docker-compose up -d

http://localhost:5000/
