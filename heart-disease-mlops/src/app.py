from pathlib import Path
from typing import Optional
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.utils import get_project_root


# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

EXPERIMENT_NAME = "Heart Disease Prediction - Inference"

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="Heart Disease Prediction API")


# ------------------------------------------------------------------
# Monitoring middleware (Task 8)
# ------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = round(time.time() - start_time, 4)

    print(
        f"[MONITORING] method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"latency={latency}s"
    )

    return response


# ------------------------------------------------------------------
# MLflow setup (SAFE for Docker/K8s)
# ------------------------------------------------------------------
mlflow.set_tracking_uri("file:///app/mlruns")

# Create experiment if it does not exist (IMPORTANT FIX)
mlflow.set_experiment(EXPERIMENT_NAME)

# ------------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------------
root = get_project_root()

model_path = (
    Path(root)
    / "artifacts"
    / "model"
)

# Expect the model to be packaged/copied during Docker build
model = mlflow.sklearn.load_model(model_path)


# ------------------------------------------------------------------
# Input schema
# ------------------------------------------------------------------
class PatientData(BaseModel):
    age: float
    sex: int
    trestbps: float
    chol: float
    thalach: float

    cp: Optional[int] = 0
    fbs: Optional[int] = 0
    restecg: Optional[int] = 0
    exang: Optional[int] = 0
    oldpeak: Optional[float] = 0.0
    slope: Optional[int] = 0
    ca: Optional[int] = 0
    thal: Optional[int] = 0


# ------------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------------
@app.post("/predict")
def predict(data: PatientData):
    input_dict = data.model_dump()

    # Build full feature row
    row = {col: input_dict.get(col, 0) for col in FEATURE_COLUMNS}
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    prob = model.predict_proba(X)[0][1]
    prediction = int(prob >= 0.5)
    confidence = round(float(prob), 4)

    # Optional inference logging to MLflow (non-blocking)
    with mlflow.start_run(run_name="inference"):
        mlflow.log_param("endpoint", "/predict")
        mlflow.log_metric("confidence", confidence)
        mlflow.log_metric("prediction", prediction)

    return {
        "prediction": prediction,
        "confidence": confidence,
    }


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
