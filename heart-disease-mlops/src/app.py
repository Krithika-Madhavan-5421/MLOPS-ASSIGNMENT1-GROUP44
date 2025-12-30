from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
from src.utils import get_project_root
import pandas as pd

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

app = FastAPI(title="Heart Disease Prediction API")

root = get_project_root()

# Set local MLflow tracking directory (important for Docker)
mlflow.set_tracking_uri(f"file://{root}/mlruns")

# Load best model automatically
experiment = mlflow.get_experiment_by_name("Heart Disease Prediction")

if experiment is None:
    raise RuntimeError("MLflow experiment not found")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"],
    max_results=1,
)

if runs.empty:
    raise RuntimeError("No MLflow runs found")

run_id = runs.iloc[0]["run_id"]

model_path = (
        Path(root)
        / "mlruns"
        / str(experiment.experiment_id)
        / run_id
        / "artifacts"
        / "model"
)

model = mlflow.sklearn.load_model(model_path)


# Input schema
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


@app.post("/predict")
def predict(data: PatientData):
    input_dict = data.model_dump()

    # Build FULL feature row
    row = {col: input_dict.get(col, 0) for col in FEATURE_COLUMNS}

    # MUST be DataFrame (not NumPy!)
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    prob = model.predict_proba(X)[0][1]
    prediction = int(prob >= 0.5)

    return {
        "prediction": prediction,
        "confidence": round(float(prob), 4),
    }

