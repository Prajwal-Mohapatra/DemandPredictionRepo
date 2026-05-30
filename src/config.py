"""
Central configuration for the Demand Prediction pipeline.
All paths, hyperparameters, and constants live here.
"""
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_PATH = os.path.join(SUBMISSIONS_DIR, "baseline_sub.csv")

# ──────────────────────────────────────────────
# Target
# ──────────────────────────────────────────────
TARGET_COL = "demand"
ID_COL = "Index"

# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
USE_OHE = True  # Use One-Hot Encoding for categoricals
USE_SVD = False # Set to True to compress OHE features
SVD_N_COMPONENTS = 5
MISSING_FLAG_COLS = ["Temperature", "NumberofLanes"]
CATEGORICAL_COLS = ["RoadType", "Weather", "Landmarks", "LargeVehicles"]
DROP_AFTER_ENCODE = ["geohash", "timestamp", ID_COL]

# ──────────────────────────────────────────────
# Model Hyperparameters
# ──────────────────────────────────────────────
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "random_state": 42,
    "verbose": -1,
}

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}

RIDGE_ALPHA = 1.0

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
LOG_EVAL_PERIOD = 50

# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────
N_SPLITS = 5
RANDOM_STATE = 42
