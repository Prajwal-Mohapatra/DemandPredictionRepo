"""
Main orchestrator — callable from the terminal OR importable from a notebook.

Usage (terminal):
    python main.py

Usage (notebook):
    from main import run_pipeline
    results = run_pipeline()
"""
from src.data_loader import load_train, load_test, split_target, summarize
from src.pipeline import ValidationPipeline
from src.inference import run_inference


def run_pipeline():
    """
    End-to-end pipeline: load → summarize → cross-validate → infer → save.
    Returns a dict with models, metrics, oof_preds, and the submission df.
    """
    # Step 1: Load data
    print("=" * 60)
    print(" STEP 1 / 4  —  Loading Data")
    print("=" * 60)
    train_df = load_train()
    test_df = load_test()

    # Step 2: Summarize & split
    print("\n" + "=" * 60)
    print(" STEP 2 / 4  —  Data Summary & Target Split")
    print("=" * 60)
    summarize(train_df, name="Train")
    X, y = split_target(train_df)

    # Step 3: Cross-validation
    print("\n" + "=" * 60)
    print(" STEP 3 / 4  —  Cross-Validation Training")
    print("=" * 60)
    pipeline = ValidationPipeline()
    models, metrics, oof_preds = pipeline.run_cv(X, y)

    # Step 4: Test inference & submission
    print("\n" + "=" * 60)
    print(" STEP 4 / 4  —  Test Inference & Submission")
    print("=" * 60)
    submission = run_inference(models, X, y, test_df)

    print("\n✅ Pipeline complete.")
    return {
        "models": models,
        "metrics": metrics,
        "oof_preds": oof_preds,
        "submission": submission,
    }


if __name__ == "__main__":
    run_pipeline()
