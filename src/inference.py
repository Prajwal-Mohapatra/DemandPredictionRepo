"""
Test-set inference and submission file generation using Stacking Ensemble.
"""
import os
import numpy as np
import pandas as pd
from src.features import FeatureEngineer
from src.config import (
    SAMPLE_SUB_PATH, OUTPUT_PATH,
    USE_OHE, USE_SVD, SVD_N_COMPONENTS, ID_COL
)


def run_inference(ensemble_dict, X_train, y_train, test_df,
                  output_path=None, sub_path=None):
    """
    1. Re-fit a FeatureEngineer on the full training set.
    2. Transform the test set.
    3. Generate Meta Features using Base Models.
    4. Predict final scores using Meta-Model.
    5. Save submission CSV.
    """
    output_path = output_path or OUTPUT_PATH
    sub_path = sub_path or SAMPLE_SUB_PATH

    print("[Inference] Fitting feature engineer on full training data...")
    fe = FeatureEngineer(use_ohe=USE_OHE, use_svd=USE_SVD, n_components=SVD_N_COMPONENTS)
    fe.fit_transform(X_train, y_train)

    print("[Inference] Transforming test set...")
    test_fe = fe.transform(test_df)

    base_models = ensemble_dict['base_models']
    meta_model = ensemble_dict['meta_model']
    
    print("[Inference] Generating Meta Features from Level 1 Models...")
    
    # 1. Average predictions of LGBM across all folds
    lgb_preds = np.zeros(len(test_df))
    for model in base_models['lgb']:
        lgb_preds += model.predict(test_fe) / len(base_models['lgb'])
        
    # 2. Average predictions of XGB across all folds
    xgb_preds = np.zeros(len(test_df))
    for model in base_models['xgb']:
        xgb_preds += model.predict(test_fe) / len(base_models['xgb'])
        
    # 3. Average predictions of Ridge across all folds
    ridge_preds = np.zeros(len(test_df))
    for model in base_models['ridge']:
        ridge_preds += model.predict(test_fe) / len(base_models['ridge'])
        
    test_meta_features = pd.DataFrame({
        'lgb_meta': lgb_preds,
        'xgb_meta': xgb_preds,
        'ridge_meta': ridge_preds
    })

    print("[Inference] Predicting final scores using Level 2 Meta-Model...")
    final_preds = meta_model.predict(test_meta_features)

    # Build submission DataFrame explicitly to match test_df length
    sub = pd.DataFrame()
        
    # Use exact IDs from test_df to perfectly map the predictions
    if ID_COL in test_df.columns:
        sub[ID_COL] = test_df[ID_COL]
    else:
        sub[ID_COL] = range(len(test_df))

    sub["demand"] = final_preds

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"[Inference] Submission saved to {output_path}")

    return sub
