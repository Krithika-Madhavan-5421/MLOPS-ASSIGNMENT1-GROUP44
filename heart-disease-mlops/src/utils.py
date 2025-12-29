from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def build_preprocessor(numerical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features)
        ]
    )


def get_project_root():
    return Path(__file__).resolve().parents[1]
