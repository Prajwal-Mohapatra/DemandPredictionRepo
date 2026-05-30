import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from src.config import (
    USE_OHE, USE_SVD, SVD_N_COMPONENTS, 
    MISSING_FLAG_COLS, CATEGORICAL_COLS, DROP_AFTER_ENCODE
)

class FeatureEngineer:
    def __init__(self, use_ohe=None, use_svd=None, n_components=None):
        self.use_ohe = use_ohe if use_ohe is not None else USE_OHE
        self.use_svd = use_svd if use_svd is not None else USE_SVD
        self.n_components = n_components if n_components is not None else SVD_N_COMPONENTS
        
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.geohash_target_mean = {}
        self.train_dummy_cols = None
        
    def _extract_time_features(self, df):
        # Using string split which is much faster and robust for 'HH:MM'
        if 'timestamp' in df.columns:
            time_split = df['timestamp'].str.split(':', expand=True)
            df['hour'] = pd.to_numeric(time_split[0], errors='coerce')
            df['minute'] = pd.to_numeric(time_split[1], errors='coerce')
        return df
        
    def fit_transform(self, X, y=None):
        X = X.copy()
        
        # 1. Temporal Features
        X = self._extract_time_features(X)
        
        # 2. Missing Data Strategy
        for col in MISSING_FLAG_COLS:
            if col in X.columns:
                X[f'{col}_isna'] = X[col].isna().astype(int)
                
        # 3. Target Encoding for High Cardinality 'geohash'
        if y is not None and 'geohash' in X.columns:
            self.geohash_target_mean = pd.concat([X, y], axis=1).groupby('geohash')[y.name].mean().to_dict()
            X['geohash_encoded'] = X['geohash'].map(self.geohash_target_mean)
        
        # 4. One-Hot Encoding and/or Dimensionality Reduction
        cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
        if len(cat_cols) > 0 and self.use_ohe:
            # One-Hot Encoding
            dummies = pd.get_dummies(X[cat_cols], dummy_na=True, drop_first=False)
            self.train_dummy_cols = dummies.columns
            
            if self.use_svd:
                # SVD on top of OHE
                reduced_features = self.svd.fit_transform(dummies)
                for i in range(self.n_components):
                    X[f'svd_cat_{i}'] = reduced_features[:, i]
            else:
                # Just keep OHE
                X = pd.concat([X, dummies], axis=1)
                
            # Drop original categorical columns
            X = X.drop(columns=cat_cols)
            
        # Drop columns no longer needed
        cols_to_drop = [c for c in DROP_AFTER_ENCODE if c in X.columns]
        X = X.drop(columns=cols_to_drop)
        
        return X

    def transform(self, X):
        X = X.copy()
        
        # 1. Temporal Features
        X = self._extract_time_features(X)
        
        # 2. Missing Data Tracking
        for col in MISSING_FLAG_COLS:
            if col in X.columns:
                X[f'{col}_isna'] = X[col].isna().astype(int)
        
        # 3. Apply Geohash Target Encoding
        if 'geohash' in X.columns:
            global_mean = np.mean(list(self.geohash_target_mean.values())) if self.geohash_target_mean else 0
            X['geohash_encoded'] = X['geohash'].map(self.geohash_target_mean).fillna(global_mean)
            
        # 4. Apply OHE / Dimensionality Reduction
        cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
        if len(cat_cols) > 0 and self.use_ohe:
            dummies = pd.get_dummies(X[cat_cols], dummy_na=True, drop_first=False)
            
            # Align test dummies with train dummies perfectly
            if self.train_dummy_cols is not None:
                dummies = dummies.reindex(columns=self.train_dummy_cols, fill_value=0)
            
            if self.use_svd:
                reduced_features = self.svd.transform(dummies)
                for i in range(self.n_components):
                    X[f'svd_cat_{i}'] = reduced_features[:, i]
            else:
                X = pd.concat([X, dummies], axis=1)
                
            X = X.drop(columns=cat_cols)
            
        cols_to_drop = [c for c in DROP_AFTER_ENCODE if c in X.columns]
        X = X.drop(columns=cols_to_drop)
        
        return X
