import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.utils import build_preprocessor

def test_model_training_and_prediction():
    df = pd.DataFrame({
        "age": [45, 54, 65, 50],
        "chol": [210, 250, 230, 240],
        "trestbps": [130, 140, 150, 135],
        "target": [0, 1, 1, 0]
    })

    X = df.drop("target", axis=1)
    y = df["target"]

    preprocessor = build_preprocessor(X.columns.tolist())

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression())
    ])

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
