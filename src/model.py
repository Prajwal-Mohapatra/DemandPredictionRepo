"""
Model wrappers for the Stacking Ensemble.
Supports LightGBM, XGBoost, and Ridge Regression.
"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.config import (
    LGB_PARAMS, XGB_PARAMS, RIDGE_ALPHA, NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS, LOG_EVAL_PERIOD
)


def evaluate_metrics(y_true, y_pred):
    """Return RMSE and MAE dict."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": round(rmse, 6), "MAE": round(mae, 6)}


class LightGBMWrapper:
    def __init__(self, params=None):
        self.params = params or LGB_PARAMS.copy()
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0), # Silence evaluation logs to keep output clean
            ],
        )

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.0)


class XGBoostWrapper:
    def __init__(self, params=None):
        self.params = params or XGB_PARAMS.copy()
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'eval'))
            
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=evals,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False
        )

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        return np.clip(preds, 0.0, 1.0)


class RidgeWrapper:
    def __init__(self, alpha=None):
        self.alpha = alpha or RIDGE_ALPHA
        self.model = Ridge(alpha=self.alpha, random_state=42)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Ridge doesn't use early stopping/validation sets natively in sklearn
        # Just train on the train set. NaNs must be filled first.
        # We fill NaNs with 0 for Ridge to prevent errors.
        X_train_filled = X_train.fillna(0) if hasattr(X_train, 'fillna') else np.nan_to_num(X_train, nan=0.0)
        self.model.fit(X_train_filled, y_train)

    def predict(self, X):
        X_filled = X.fillna(0) if hasattr(X, 'fillna') else np.nan_to_num(X, nan=0.0)
        preds = self.model.predict(X_filled)
        return np.clip(preds, 0.0, 1.0)
