import numpy as np
from sklearn.model_selection import KFold
from src.features import FeatureEngineer
from src.model import DemandModel

class ValidationPipeline:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def run_cv(self, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(X))
        metrics = []
        models = []
        
        print(f"Starting {self.n_splits}-Fold Cross Validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"--- Fold {fold + 1} ---")
            
            X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
            X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
            
            # Feature Engineering
            fe = FeatureEngineer()
            X_train_fe = fe.fit_transform(X_train, y_train)
            X_val_fe = fe.transform(X_val)
            
            # Model Training
            model = DemandModel()
            model.train(X_train_fe, y_train, X_val_fe, y_val)
            models.append(model)
            
            # Validation
            val_preds = model.predict(X_val_fe)
            oof_preds[val_idx] = val_preds
            fold_metric = model.evaluate(y_val, val_preds)
            metrics.append(fold_metric)
            print(f"Fold {fold + 1} Metrics: {fold_metric}")
            
        overall_metric = DemandModel().evaluate(y, oof_preds)
        print(f"Overall OOF Metrics: {overall_metric}")
        
        return models, metrics, oof_preds
