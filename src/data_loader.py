"""
Data loading, validation, and train/target split logic.
"""
import os
import pandas as pd
from src.config import TRAIN_PATH, TEST_PATH, TARGET_COL


def load_train(path=None):
    """Load training CSV and return the full DataFrame."""
    path = path or TRAIN_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at: {path}")
    df = pd.read_csv(path)
    print(f"[DataLoader] Train loaded  — shape {df.shape}")
    return df


def load_test(path=None):
    """Load test CSV and return the full DataFrame."""
    path = path or TEST_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test data not found at: {path}")
    df = pd.read_csv(path)
    print(f"[DataLoader] Test loaded   — shape {df.shape}")
    return df


def split_target(df, target_col=None):
    """
    Split a DataFrame into features (X) and target (y).
    Returns (X, y).
    """
    target_col = target_col or TARGET_COL
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. "
                         f"Available columns: {list(df.columns)}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"[DataLoader] Split target  — X {X.shape}, y {y.shape}")
    return X, y


def summarize(df, name="DataFrame"):
    """Print a quick summary of the DataFrame for sanity checking."""
    print(f"\n{'='*50}")
    print(f" Summary: {name}")
    print(f"{'='*50}")
    print(f" Shape       : {df.shape}")
    print(f" Columns     : {list(df.columns)}")
    print(f" Dtypes      :\n{df.dtypes.to_string()}")
    print(f" Null counts :\n{df.isnull().sum().to_string()}")
    print(f"{'='*50}\n")
