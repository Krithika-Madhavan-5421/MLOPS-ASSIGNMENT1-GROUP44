from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def build_preprocessor(numerical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features)
        ]
    )
