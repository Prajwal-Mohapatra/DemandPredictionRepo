# Demand Prediction Hackathon Repo

## Overview
This repository contains an end-to-end Machine Learning pipeline for predicting continuous spatial-temporal demand.

## Directory Structure
- `data/`: Directory for placing `train.csv`, `test.csv`, and `sample_submission.csv`
- `src/`: Core Python modules for feature engineering, model architecture, and training pipeline
- `submissions/`: Output directory for generated prediction CSV files
- `main.py`: Entry point for running the pipeline

## Dimensionality Reduction Techniques Explored
To make the pipeline highly efficient given spatial-temporal features, we have incorporated:
1. **Target Encoding on `geohash`**: High-cardinality categorical spatial signals (geohash) are reduced to a 1D continuous representation based on target mean correlation. This is vastly more dimension-efficient than One-Hot Encoding and works wonderfully with tree models.
2. **Truncated SVD (PCA for Sparse Data)**: For categorical attributes like `RoadType`, `Weather`, and `Landmarks`, dummy encoding can blow up the feature space. We apply SVD to compress these sparse dummy variables into dense latent principal components, reducing dimensionality noise and improving training time.

## Usage
1. Place your dataset files in the `data/` folder.
2. Ensure you have the dependencies installed (`pandas`, `numpy`, `scikit-learn`, `lightgbm`).
3. Run `python main.py` to execute the Cross-Validation pipeline and generate inference on the test set.
