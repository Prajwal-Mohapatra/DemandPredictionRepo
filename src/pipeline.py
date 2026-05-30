"""
Stacking Ensemble Validation pipeline.
Trains Base Level models, extracts Meta Features, and trains a Level 2 Meta-Model.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.config import N_SPLITS, RANDOM_STATE, USE_SVD, SVD_N_COMPONENTS
from src.features import FeatureEngineer
from src.model import LightGBMWrapper, XGBoostWrapper, RidgeWrapper, evaluate_metrics


class ValidationPipeline:
    def __init__(self, n_splits=None):
        self.n_splits = n_splits or N_SPLITS

    def run_cv(self, X, y):
        """
        Run Stacking CV.
        Returns:
            dict with base_models, meta_model, metrics
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)
        
        # Arrays to store Out-Of-Fold predictions for our 3 base models
        oof_lgb = np.zeros(len(X))
        oof_xgb = np.zeros(len(X))
        oof_ridge = np.zeros(len(X))
        
        base_models = {'lgb': [], 'xgb': [], 'ridge': []}
        
        print(f"[Pipeline] Starting {self.n_splits}-Fold Stacking CV...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n--- Fold {fold + 1}/{self.n_splits} ---")

            X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
            X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()

            # Feature Engineering
            fe = FeatureEngineer(use_svd=USE_SVD, n_components=SVD_N_COMPONENTS)
            X_train_fe = fe.fit_transform(X_train, y_train)
            X_val_fe = fe.transform(X_val)

            # 1. Train LightGBM
            lgb_model = LightGBMWrapper()
            lgb_model.train(X_train_fe, y_train, X_val_fe, y_val)
            lgb_preds = lgb_model.predict(X_val_fe)
            oof_lgb[val_idx] = lgb_preds
            base_models['lgb'].append(lgb_model)
            
            # 2. Train XGBoost
            xgb_model = XGBoostWrapper()
            xgb_model.train(X_train_fe, y_train, X_val_fe, y_val)
            xgb_preds = xgb_model.predict(X_val_fe)
            oof_xgb[val_idx] = xgb_preds
            base_models['xgb'].append(xgb_model)
            
            # 3. Train Ridge
            ridge_model = RidgeWrapper()
            ridge_model.train(X_train_fe, y_train, X_val_fe, y_val) # No early stopping
            ridge_preds = ridge_model.predict(X_val_fe)
            oof_ridge[val_idx] = ridge_preds
            base_models['ridge'].append(ridge_model)
            
            print(f"Fold {fold + 1} - LGB: {evaluate_metrics(y_val, lgb_preds)['RMSE']} | XGB: {evaluate_metrics(y_val, xgb_preds)['RMSE']} | Ridge: {evaluate_metrics(y_val, ridge_preds)['RMSE']}")

        print("\n[Pipeline] Training Level 2 Meta-Model (XGBoost)...")
        # Construct Meta Features dataset from OOF predictions
        meta_features = pd.DataFrame({
            'lgb_meta': oof_lgb,
            'xgb_meta': oof_xgb,
            'ridge_meta': oof_ridge
        })
        
        # Train Meta-Model on the entire OOF dataset
        meta_model = XGBoostWrapper()
        # Since it's trained on the whole OOF, we don't use early stopping here. 
        # But XGBoostWrapper expects val sets if provided, so we just provide train.
        meta_model.train(meta_features, y)
        
        final_oof_preds = meta_model.predict(meta_features)
        overall_metric = evaluate_metrics(y, final_oof_preds)
        print(f"\n[Pipeline] Overall Stacking Ensemble OOF Metrics: {overall_metric}")

        return {
            'base_models': base_models,
            'meta_model': meta_model,
            'metrics': overall_metric
        }
