# MLOps (S1-25_AIMLCZG523)
## ASSIGNMENT – I  
**Total Marks: 50**

## MLOps Experimental Learning Assignment  
**End-to-End ML Model Development, CI/CD, and Production Deployment**

---

## Objective
Design, develop, and deploy a scalable and reproducible machine learning solution using modern **MLOps best practices**.  
This assignment emphasizes practical automation, experiment tracking, CI/CD pipelines, containerization, cloud deployment, and monitoring—mirroring real-world production scenarios.

---

## Dataset
**Title:** Heart Disease UCI Dataset  
**Source:** UCI Machine Learning Repository  

- CSV dataset containing **14+ features** (age, sex, blood pressure, cholesterol, etc.)  
- **Binary target:** Presence / absence of heart disease  

---

## Problem Statement
Build a **machine learning classifier** to predict the **risk of heart disease** based on patient health data and deploy the solution as a **cloud-ready, monitored API**.

---

## Assignment Tasks

### 1. Data Acquisition & Exploratory Data Analysis (EDA) – *5 Marks*
- Obtain the dataset (provide download script or instructions).
- Clean and preprocess data:
  - Handle missing values
  - Encode categorical features
- Perform EDA with professional visualizations:
  - Histograms
  - Correlation heatmaps
  - Class balance analysis

---

### 2. Feature Engineering & Model Development – *8 Marks*
- Prepare final ML features:
  - Scaling
  - Encoding
- Train at least **two classification models** (e.g., Logistic Regression, Random Forest).
- Document:
  - Model selection
  - Hyperparameter tuning
- Evaluate using:
  - Cross-validation
  - Accuracy, Precision, Recall, ROC-AUC

---

### 3. Experiment Tracking – *5 Marks*
- Integrate **MLflow** (or similar).
- Log:
  - Parameters
  - Metrics
  - Artifacts
  - Plots
- Track all experiment runs.

---

### 4. Model Packaging & Reproducibility – *7 Marks*
- Save final model in a reusable format:
  - MLflow / Pickle / ONNX
- Provide:
  - `requirements.txt` or Conda environment file
  - Preprocessing pipeline / transformers
- Ensure full reproducibility.

---

### 5. CI/CD Pipeline & Automated Testing – *8 Marks*
- Write unit tests using **Pytest** or `unittest`:
  - Data processing
  - Model logic
- Create **GitHub Actions** (or Jenkins) pipeline including:
  - Linting
  - Unit testing
  - Model training
- Maintain artifacts and logs for each workflow run.

---

### 6. Model Containerization – *5 Marks*
- Build a **Docker container** for the model-serving API.
- Use **Flask** or **FastAPI**.
- Expose `/predict` endpoint:
  - Accept JSON input
  - Return prediction and confidence score
- Container must build and run locally with sample input.

---

### 7. Production Deployment – *7 Marks*
- Deploy Dockerized API to:
  - Public cloud (GKE / EKS / AKS), or
  - Local Kubernetes (Minikube / Docker Desktop)
- Use:
  - Deployment manifest or Helm chart
- Expose service via:
  - LoadBalancer or Ingress
- Verify endpoints and include deployment screenshots.

---

### 8. Monitoring & Logging – *3 Marks*
- Integrate API request logging.
- Demonstrate monitoring using:
  - Prometheus + Grafana, or
  - API metrics / logs dashboard

---

### 9. Documentation & Reporting – *2 Marks*
Submit a professional **Markdown or PDF report** including:
- Setup and installation instructions
- EDA and modeling decisions
- Experiment tracking summary
- Architecture diagram
- CI/CD and deployment workflow screenshots
- Link to the code repository

---

## Deliverables

### a) GitHub Repository
Must include:
- Source code
- `Dockerfile(s)`
- `requirements.txt` / `environment.yml`
- Dataset download script or instructions
- Jupyter notebooks / scripts (EDA, training, inference)
- `tests/` folder with unit tests
- GitHub Actions workflow YAML (or Jenkinsfile)
- Deployment manifests / Helm charts
- `screenshots/` folder
- Final written report (**10 pages**, `.doc` / `.docx`)

### b) Short Video
- Demonstrating the **end-to-end MLOps pipeline**

### c) Deployment Access
- Deployed API URL (if public), **or**
- Local access instructions for testing

---

## Production-Readiness Requirements
- All scripts must run from a **clean environment** using the provided requirements file.
- Model must serve correctly in an **isolated Docker environment** (container build/test proof required).
- CI/CD pipeline must:
  - Fail on code or test errors
  - Provide clear and meaningful logs

---
