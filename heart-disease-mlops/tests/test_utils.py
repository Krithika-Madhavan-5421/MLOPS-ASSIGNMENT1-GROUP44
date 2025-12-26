
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from src.utils import build_preprocessor

def test_build_preprocessor():
    features = ["age", "chol", "trestbps"]
    preprocessor = build_preprocessor(features)

    df = pd.DataFrame({
        "age": [50, 60],
        "chol": [200, 240],
        "trestbps": [120, 140]
    })

    transformed = preprocessor.fit_transform(df)

    # Check shape: rows preserved, columns transformed
    assert transformed.shape[0] == df.shape[0]
    assert transformed.shape[1] == len(features)
