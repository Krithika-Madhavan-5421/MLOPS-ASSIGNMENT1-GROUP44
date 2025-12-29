import pandas as pd
from src.utils import get_project_root


def test_dataset_loads():
    root = get_project_root()
    df = pd.read_csv(root / "data" / "heart.csv")
    assert not df.empty


def test_target_is_binary():
    root = get_project_root()
    df = pd.read_csv(root / "data" / "heart.csv")
    assert set(df["target"].unique()).issubset({0, 1})


def test_required_columns_exist():
    root = get_project_root()
    df = pd.read_csv(root / "data" / "heart.csv")

    required_columns = {"age", "sex", "chol", "target"}
    assert required_columns.issubset(set(df.columns))
