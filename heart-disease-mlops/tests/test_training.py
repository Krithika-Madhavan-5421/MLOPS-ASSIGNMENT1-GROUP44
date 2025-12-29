import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.utils import build_preprocessor


def test_model_pipeline_fit_and_predict():
    df = pd.DataFrame({
        "age": [50, 60, 55, 45],
        "sex": [1, 0, 1, 0],
        "chol": [200, 180, 190, 170],
        "target": [1, 0, 1, 0]
    })

    X = df.drop("target", axis=1)
    y = df["target"]

    pipeline = Pipeline([
        ("preprocess", build_preprocessor(X.columns.tolist())),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
