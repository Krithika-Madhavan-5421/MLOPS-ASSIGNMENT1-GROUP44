import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import build_preprocessor


# --------------------------------------------------
# MLflow setup (CI-safe)
# --------------------------------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart Disease Prediction")


# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
preprocessor = build_preprocessor(X.columns.tolist())


# --------------------------------------------------
# Models
# --------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
    ),
}


# --------------------------------------------------
# Training loop
# --------------------------------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        mlflow.log_metrics(
            {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, probs),
            }
        )

        mlflow.sklearn.log_model(pipeline, "model")

print("Training completed")