# Demand Prediction Hackathon Repo

## Overview
End-to-end ML pipeline for predicting continuous spatial-temporal **demand** (0.0 – 1.0) using LightGBM.

## Directory Structure
```
DemandPredictionRepo/
├── data/                     # Place train.csv, test.csv, sample_submission.csv here
├── submissions/              # Generated submission CSVs
├── src/
│   ├── config.py             # Central config — paths, hyperparams, constants
│   ├── data_loader.py        # CSV loading, validation, target split
│   ├── features.py           # Feature engineering & dimensionality reduction
│   ├── model.py              # LightGBM model wrapper
│   ├── pipeline.py           # K-Fold cross-validation loop
│   └── inference.py          # Test prediction & submission generation
├── main.py                   # Thin orchestrator (importable + runnable)
├── run_pipeline.ipynb         # Notebook — calls scripts, holds no logic
└── environment.yml           # Conda environment definition
```

## Modular Design
Every module owns **one responsibility**. The notebook is a thin caller.

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for all paths, hyperparameters, and constants |
| `data_loader.py` | Load CSVs, validate columns, split target |
| `features.py` | Temporal extraction, target encoding, SVD dimensionality reduction |
| `model.py` | LightGBM training, prediction, evaluation |
| `pipeline.py` | K-Fold cross-validation orchestration |
| `inference.py` | Full-data refit, test prediction, submission CSV export |
| `main.py` | Chains all modules into a single `run_pipeline()` call |

## Dimensionality Reduction Techniques
1. **Target Encoding on `geohash`** — Replaces high-cardinality categorical with target mean (1D continuous).
2. **Truncated SVD** — Compresses dummy-encoded categoricals (`RoadType`, `Weather`, `Landmarks`) into dense latent components.

## Usage

### Option A — From Terminal
```bash
conda activate demand_pred
python main.py
```

### Option B — From Jupyter Notebook
Open `run_pipeline.ipynb` and run cells. Two modes:
- **One-call**: `from main import run_pipeline; results = run_pipeline()`
- **Step-by-step**: Import individual modules and inspect outputs between steps.

## Setup
```bash
conda env create -f environment.yml
conda activate demand_pred
```
