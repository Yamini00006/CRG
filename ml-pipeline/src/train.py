from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    validate_feature_configuration,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_DATASET = (
    PROJECT_ROOT
    / "ml-pipeline"
    / "data"
    / "processed"
    / "synthetic"
    / "synthetic_credit_risk_dataset.csv"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "ml-pipeline"
    / "models"
)

EXPERIMENTS_DIR = (
    PROJECT_ROOT
    / "ml-pipeline"
    / "experiments"
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------------------

def build_models():
    """
    Create the three required baseline models.

    class_weight='balanced' gives additional importance to the
    relatively smaller risk classes during training.
    """

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_training_data():
    """Load and validate the synthetic training dataset."""

    if not SYNTHETIC_DATASET.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found:\n{SYNTHETIC_DATASET}"
        )

    df = pd.read_csv(SYNTHETIC_DATASET)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )

    # Ensure leakage columns are never accidentally included.
    leakage_columns = [
        "PD_1y_pct",
        "LGD_pct",
        "EAD_usd_m",
        "implied_rating",
    ]

    leakage_present = [
        column
        for column in leakage_columns
        if column in FEATURE_COLUMNS
    ]

    if leakage_present:
        raise ValueError(
            "Leakage columns detected in FEATURE_COLUMNS:\n"
            + "\n".join(f"- {column}" for column in leakage_present)
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


# ---------------------------------------------------------------------
# PIPELINE CREATION
# ---------------------------------------------------------------------

def build_model_pipeline(model):
    """
    Combine the existing preprocessing logic with the ML model.

    The saved artifact therefore contains both preprocessing and
    the trained estimator.
    """

    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_models(X_train, y_train):
    """Train all configured models."""

    models = build_models()
    trained_models = {}

    for model_name, model in models.items():

        print(f"\n{'=' * 70}")
        print(f"TRAINING: {model_name}")
        print(f"{'=' * 70}")

        pipeline = build_model_pipeline(model)

        pipeline.fit(X_train, y_train)

        trained_models[model_name] = pipeline

        print("Training completed.")

    return trained_models


# ---------------------------------------------------------------------
# SAVE MODELS
# ---------------------------------------------------------------------

def save_models(trained_models):
    """Save each complete preprocessing + model pipeline."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, pipeline in trained_models.items():

        model_directory = MODELS_DIR / model_name
        model_directory.mkdir(parents=True, exist_ok=True)

        model_path = model_directory / "model.pkl"

        joblib.dump(pipeline, model_path)

        print(f"Saved: {model_path}")


# ---------------------------------------------------------------------
# SAVE EXPERIMENT INFORMATION
# ---------------------------------------------------------------------

def save_training_metadata(X_train, X_test, y_train, y_test):
    """
    Save basic reproducibility information.

    Evaluation metrics are deliberately handled by evaluate.py,
    not train.py.
    """

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "total_rows": len(X_train) + len(X_test),
        "training_rows": len(X_train),
        "internal_test_rows": len(X_test),
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "classes": sorted(y_train.unique().tolist()),
    }

    metadata_df = pd.DataFrame(
        [metadata]
    )

    metadata_path = (
        EXPERIMENTS_DIR
        / "training_metadata.csv"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False,
    )

    print(f"Saved: {metadata_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CREDIT RISK MODEL TRAINING")
    print("=" * 70)

    # Validate the feature configuration before training.
    validate_feature_configuration()

    # Load synthetic training data.
    X, y = load_training_data()

    print(f"\nDataset: {SYNTHETIC_DATASET}")
    print(f"Total rows: {len(X)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    # Stratified split keeps the Low / Medium / High proportions
    # approximately consistent between train and internal test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nData split:")
    print(f"- Training rows: {len(X_train)}")
    print(f"- Internal test rows: {len(X_test)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nInternal test target distribution:")
    print(y_test.value_counts().sort_index())

    # Train models.
    trained_models = train_models(
        X_train,
        y_train,
    )

    # Save complete pipelines.
    save_models(trained_models)

    # Save reproducibility metadata.
    save_training_metadata(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETED")
    print("=" * 70)

    print("\nTrained models:")

    for model_name in trained_models:
        print(f"- {model_name}")

    print("\nModels contain:")
    print("- preprocessing pipeline")
    print("- trained estimator")

    print("\nEvaluation is intentionally handled separately by evaluate.py.")


if __name__ == "__main__":
    main()