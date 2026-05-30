import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class DemandModel:
    def __init__(self, random_state=42):
        # Baseline parameters tuned for continuous demand [0,1]
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'random_state': random_state,
            'verbose': -1
        }
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            
        # LightGBM handles NaN values natively.
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
        )

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        preds = self.model.predict(X)
        # Demand is strictly between 0.0 and 1.0, clip predictions
        return np.clip(preds, 0.0, 1.0)
        
    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae}
