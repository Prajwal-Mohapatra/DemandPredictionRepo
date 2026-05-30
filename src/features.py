import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, use_svd=True, n_components=5):
        self.use_svd = use_svd
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.scaler = StandardScaler()
        self.geohash_target_mean = {}
        
    def _extract_time_features(self, df):
        # Assuming 'timestamp' is something like '13:45'
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp'], format='%H:%M', errors='coerce').dt.hour
            df['minute'] = pd.to_datetime(df['timestamp'], format='%H:%M', errors='coerce').dt.minute
        return df
        
    def fit_transform(self, X, y=None):
        X = X.copy()
        
        # 1. Temporal Features
        X = self._extract_time_features(X)
        
        # 2. Missing Data Strategy
        # Tree models can handle NaNs natively, but explicitly flag missingness for specific columns
        for col in ['Temperature']:
            if col in X.columns:
                X[f'{col}_isna'] = X[col].isna().astype(int)
                
        # 3. High Cardinality 'geohash' -> Target Encoding (Dimensionality Reduction vs OHE)
        if y is not None and 'geohash' in X.columns:
            self.geohash_target_mean = pd.concat([X, y], axis=1).groupby('geohash')[y.name].mean().to_dict()
            X['geohash_encoded'] = X['geohash'].map(self.geohash_target_mean)
        
        # 4. Dimensionality Reduction (SVD on Categorical interactions)
        cat_cols = [c for c in ['RoadType', 'Weather', 'Landmarks'] if c in X.columns]
        if self.use_svd and len(cat_cols) > 0:
            dummies = pd.get_dummies(X[cat_cols], dummy_na=True)
            reduced_features = self.svd.fit_transform(dummies)
            for i in range(self.n_components):
                X[f'svd_cat_{i}'] = reduced_features[:, i]
            # Drop original categoricals
            X = X.drop(columns=cat_cols)
            
        # Drop raw 'geohash' and 'timestamp'
        cols_to_drop = [c for c in ['geohash', 'timestamp'] if c in X.columns]
        X = X.drop(columns=cols_to_drop)
        
        return X

    def transform(self, X):
        X = X.copy()
        
        # 1. Temporal Features
        X = self._extract_time_features(X)
        
        # 2. Missing Data Tracking
        for col in ['Temperature']:
            if col in X.columns:
                X[f'{col}_isna'] = X[col].isna().astype(int)
        
        # 3. Apply Geohash Target Encoding
        if 'geohash' in X.columns:
            global_mean = np.mean(list(self.geohash_target_mean.values())) if self.geohash_target_mean else 0
            X['geohash_encoded'] = X['geohash'].map(self.geohash_target_mean).fillna(global_mean)
            
        # 4. Dimensionality Reduction Transform
        cat_cols = [c for c in ['RoadType', 'Weather', 'Landmarks'] if c in X.columns]
        if self.use_svd and len(cat_cols) > 0:
            dummies = pd.get_dummies(X[cat_cols], dummy_na=True)
            train_dummy_cols = self.svd.feature_names_in_ if hasattr(self.svd, 'feature_names_in_') else dummies.columns
            dummies = dummies.reindex(columns=train_dummy_cols, fill_value=0)
            reduced_features = self.svd.transform(dummies)
            for i in range(self.n_components):
                X[f'svd_cat_{i}'] = reduced_features[:, i]
            X = X.drop(columns=cat_cols)
            
        cols_to_drop = [c for c in ['geohash', 'timestamp'] if c in X.columns]
        X = X.drop(columns=cols_to_drop)
        
        return X
