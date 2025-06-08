# Dockerfile for Energy Demand Forecasting API
# Author: Corey Leath

# Base image
FROM python:3.9-slim

# Set workdir
WORKDIR /app

# Copy requirements and install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY models/ models/

# Expose port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Build Docker image
docker build -t energy-forecast-api:latest .

# Run container
docker run -d -p 8000:8000 energy-forecast-api:latest

# Test API
http://localhost:8000/docs

git add Dockerfile
git commit -m "Add Dockerfile for API"
git push
