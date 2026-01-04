# Heart Disease Risk Prediction – MLOps Pipeline

This repository contains an end-to-end MLOps pipeline for predicting the presence of heart disease using patient health data. The project demonstrates modern MLOps practices including experiment tracking, CI/CD automation, containerization, Kubernetes deployment, and monitoring.

---

## Project Overview

- Dataset: Heart Disease UCI Dataset
- Problem: Binary classification (presence/absence of heart disease)
- Models: Logistic Regression, Random Forest
- Experiment Tracking: MLflow
- CI/CD: GitHub Actions
- API: FastAPI
- Containerization: Docker
- Deployment: Kubernetes (Minikube)
- UI (Optional): Streamlit

---

## Repository Structure

heart-disease-mlops/
├── data/ # Cleaned dataset
├── src/ # Source code (EDA, training, API, Streamlit)
├── tests/ # Unit tests
├── artifacts/plots/ # EDA visualizations
├── mlruns/ # MLflow experiments (local)
├── manifests/ # Kubernetes deployment & service YAMLs
├── .github/workflows/ # CI/CD pipeline definitions
├── requirements.txt
├── Dockerfile
└── REPORT.md # Detailed project report

---
## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Krithika-Madhavan-5421/MLOPS-ASSIGNMENT1-GROUP44.git
cd heart-disease-mlops
```
---
### 2. Create and Activate a Conda Environment

A Conda environment is used to isolate project dependencies and ensure reproducibility across different systems.

Create a new Conda environment:

```bash
conda create -n heart-disease-mlops python=3.10 -y
```

--- 

### 3. Install Project Dependencies

Install all required Python dependencies using the provided requirements.txt file.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Data Acquisition and Exploratory Data Analysis

The dataset is automatically fetched from the UCI Machine Learning Repository.

Run the EDA script:

```bash
python src/eda.py
```
---

### 5. Model Training and Experiment Tracking

Train the machine learning models using the training script:

```bash
python src/train.py
```

To view experiments and metrics:
```bash
mlflow ui
```
Access the MLflow dashboard at:
```bash
http://localhost:5000
```

---

### 6. Running the FastAPI Inference Service Locally

The trained model is served through a FastAPI-based REST API.
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Swagger API documentation is available at:
```bash
http://localhost:8000/docs
```

---

### 7. Docker Image Build and Local Execution
Build the Docker image for the inference service:
```bash
docker build -t heart-disease-api:latest .
```
Run the Docker container locally:
```bash
docker run -p 8000:8000 heart-disease-api:latest
```

---

## Documentation

Detailed Report: REPORT.md
MLflow screenshots, CI/CD logs, deployment screenshots are included in the report