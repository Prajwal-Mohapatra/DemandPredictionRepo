import os
import argparse
import pandas as pd
import numpy as np
from src.features import FeatureEngineer
from src.pipeline import ValidationPipeline

def main():
    parser = argparse.ArgumentParser(description="Demand Prediction ML Pipeline")
    parser.add_argument('--train_path', type=str, default='data/train.csv')
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--sub_path', type=str, default='data/sample_submission.csv')
    parser.add_argument('--output_path', type=str, default='submissions/baseline_sub.csv')
    args = parser.parse_args()

    # Wait for datasets
    if not os.path.exists(args.train_path) or not os.path.exists(args.test_path):
        print(f"Dataset not found at {args.train_path} or {args.test_path}")
        print("Please upload the dataset files into the 'data/' directory.")
        return

    print("Loading data...")
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    
    target_col = 'demand'
    if target_col not in train.columns:
        print(f"Error: Target column '{target_col}' not found in train data.")
        return

    X = train.drop(columns=[target_col])
    y = train[target_col]
    
    print("Running Pipeline...")
    pipeline = ValidationPipeline(n_splits=5)
    models, metrics, oof_preds = pipeline.run_cv(X, y)
    
    print("Refitting feature engineer on full train set for final inference...")
    fe = FeatureEngineer()
    X_fe = fe.fit_transform(X, y)
    test_fe = fe.transform(test)
    
    print("Averaging model predictions over folds for the test set...")
    test_preds = np.zeros(len(test))
    for model in models:
        test_preds += model.predict(test_fe) / len(models)
        
    print("Saving submission...")
    sub = pd.read_csv(args.sub_path) if os.path.exists(args.sub_path) else pd.DataFrame({'id': test.index})
    sub['demand'] = test_preds
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    sub.to_csv(args.output_path, index=False)
    print(f"Submission saved to {args.output_path}")

if __name__ == "__main__":
    main()
