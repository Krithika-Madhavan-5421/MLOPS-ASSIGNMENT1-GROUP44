import pandas as pd
from src.utils import build_preprocessor


def test_preprocessor_fit_transform():
    df = pd.DataFrame({
        "age": [50, 60],
        "sex": [1, 0],
        "chol": [200, 180]
    })

    preprocessor = build_preprocessor(df.columns.tolist())
    transformed = preprocessor.fit_transform(df)

    assert transformed is not None
    assert transformed.shape[0] == df.shape[0]
