from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_configuration,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_DATASET = (
    PROJECT_ROOT
    / "ml-pipeline"
    / "data"
    / "raw"
    / "official"
    / "credit_risk_dataset_50_entities.csv"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "ml-pipeline"
    / "models"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "ml-pipeline"
    / "reports"
    / "validation"
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

LABELS = ["Low", "Medium", "High"]


# ---------------------------------------------------------------------
# LOAD OFFICIAL DATA
# ---------------------------------------------------------------------

def load_official_data():
    """
    Load the untouched official dataset.

    Only the approved FEATURE_COLUMNS are passed to the model.
    """

    if not OFFICIAL_DATASET.exists():
        raise FileNotFoundError(
            f"Official dataset not found:\n{OFFICIAL_DATASET}"
        )

    df = pd.read_csv(OFFICIAL_DATASET)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Official dataset is missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return df, X, y


# ---------------------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------------------

def get_model_paths():
    """Return all trained model artifacts."""

    model_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
    ]

    paths = {}

    for model_name in model_names:

        path = (
            MODELS_DIR
            / model_name
            / "model.pkl"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{path}"
            )

        paths[model_name] = path

    return paths


# ---------------------------------------------------------------------
# EVALUATE ONE MODEL
# ---------------------------------------------------------------------

def evaluate_model(
    model_name,
    model,
    X,
    y,
):
    """Evaluate one trained model against the official dataset."""

    print("\n" + "=" * 70)
    print(f"OFFICIAL VALIDATION: {model_name}")
    print("=" * 70)

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    macro_precision = precision_score(
        y,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    high_recall = report["High"]["recall"]

    print(f"\nAccuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"High Risk Recall:  {high_recall:.4f}")

    print("\nPer-class performance:")

    for label in LABELS:

        print(
            f"{label:8s} "
            f"Precision={report[label]['precision']:.4f} "
            f"Recall={report[label]['recall']:.4f} "
            f"F1={report[label]['f1-score']:.4f}"
        )

    # -----------------------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------------------

    matrix = confusion_matrix(
        y,
        predictions,
        labels=LABELS,
    )

    print("\nConfusion Matrix:")

    print(
        pd.DataFrame(
            matrix,
            index=[
                f"Actual {label}"
                for label in LABELS
            ],
            columns=[
                f"Predicted {label}"
                for label in LABELS
            ],
        )
    )

    # -----------------------------------------------------------------
    # SAVE CONFUSION MATRIX
    # -----------------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS,
    )

    display.plot()

    plt.title(
        f"Official Validation - {model_name}"
    )

    matrix_path = (
        REPORTS_DIR
        / f"{model_name}_official_confusion_matrix.png"
    )

    plt.savefig(
        matrix_path,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved confusion matrix:\n{matrix_path}"
    )

    # -----------------------------------------------------------------
    # ENTITY-LEVEL PREDICTIONS
    # -----------------------------------------------------------------

    result_df = pd.DataFrame(
        {
            "entity_id": y.index,
            "actual_risk": y.values,
            "predicted_risk": predictions,
        }
    )

    # Keep entity IDs/names from the original dataset when available.
    # This is only reporting information and is never passed to the model.
    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "high_risk_recall": high_recall,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": predictions,
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------
# SAVE ENTITY PREDICTIONS
# ---------------------------------------------------------------------

def save_entity_predictions(
    official_df,
    model_name,
    result,
):
    """Save actual vs predicted risk for every official entity."""

    output = pd.DataFrame(
        {
            "entity_id": official_df["entity_id"],
            "entity_name": official_df["entity_name"],
            "actual_risk": official_df[TARGET_COLUMN],
            "predicted_risk": result["predictions"],
        }
    )

    # Add class probabilities.
    probability_columns = [
        f"probability_{label.lower()}"
        for label in LABELS
    ]

    probability_df = pd.DataFrame(
        result["probabilities"],
        columns=probability_columns,
    )

    output = pd.concat(
        [
            output.reset_index(drop=True),
            probability_df.reset_index(drop=True),
        ],
        axis=1,
    )

    output["prediction_correct"] = (
        output["actual_risk"]
        == output["predicted_risk"]
    )

    output_path = (
        REPORTS_DIR
        / f"{model_name}_official_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved entity predictions:\n{output_path}"
    )


# ---------------------------------------------------------------------
# SAVE SUMMARY
# ---------------------------------------------------------------------

def save_summary(results):
    """Save official validation model comparison."""

    summary = []

    for result in results:

        summary.append(
            {
                "model": result["model"],
                "accuracy": result["accuracy"],
                "macro_precision": result["macro_precision"],
                "macro_recall": result["macro_recall"],
                "macro_f1": result["macro_f1"],
                "high_risk_recall": result["high_risk_recall"],
            }
        )

    summary_df = pd.DataFrame(summary)

    summary_path = (
        REPORTS_DIR
        / "official_model_comparison.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nSaved official comparison:\n{summary_path}"
    )

    return summary_df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CREDIT RISK — OFFICIAL DATASET VALIDATION")
    print("=" * 70)

    # Validate feature configuration.
    validate_feature_configuration()

    # Load official dataset.
    official_df, X, y = load_official_data()

    print(
        f"\nOfficial dataset:\n{OFFICIAL_DATASET}"
    )

    print(
        f"\nOfficial rows: {len(X)}"
    )

    print(
        f"Model features used: {len(FEATURE_COLUMNS)}"
    )

    print("\nOfficial target distribution:")

    print(
        y.value_counts()
        .sort_index()
    )

    # Explicitly confirm that downstream fields are not model inputs.
    leakage_columns = [
        "PD_1y_pct",
        "LGD_pct",
        "EAD_usd_m",
        "implied_rating",
    ]

    used_leakage = [
        column
        for column in leakage_columns
        if column in FEATURE_COLUMNS
    ]

    if used_leakage:
        raise ValueError(
            "Leakage detected in validation features:\n"
            + "\n".join(
                f"- {column}"
                for column in used_leakage
            )
        )

    print(
        "\nLeakage protection: PASS"
    )

    print(
        "Official data is used only for validation."
    )

    # Load trained models.
    model_paths = get_model_paths()

    results = []

    # Evaluate every model.
    for model_name, model_path in model_paths.items():

        print(
            f"\nLoading model:\n{model_path}"
        )

        model = joblib.load(model_path)

        result = evaluate_model(
            model_name,
            model,
            X,
            y,
        )

        save_entity_predictions(
            official_df,
            model_name,
            result,
        )

        results.append(result)

    # Save comparison.
    summary_df = save_summary(results)

    # -----------------------------------------------------------------
    # RANKING
    # -----------------------------------------------------------------

    ranked = sorted(
        results,
        key=lambda result: (
            result["macro_f1"],
            result["high_risk_recall"],
        ),
        reverse=True,
    )

    best = ranked[0]

    selection_path = (
        REPORTS_DIR
        / "official_selected_model.txt"
    )

    selection_path.write_text(
        (
            "Official validation model ranking\n"
            "=================================\n\n"
            f"Selected model: {best['model']}\n"
            f"Macro F1: {best['macro_f1']:.4f}\n"
            f"High-risk recall: "
            f"{best['high_risk_recall']:.4f}\n\n"
            "Important:\n"
            "This validation uses the official 50-entity "
            "dataset and is not a production performance estimate.\n"
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("OFFICIAL MODEL COMPARISON")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print("OFFICIAL VALIDATION WINNER")
    print("=" * 70)

    print(
        f"Model: {best['model']}"
    )

    print(
        f"Macro F1: {best['macro_f1']:.4f}"
    )

    print(
        f"High-risk recall: "
        f"{best['high_risk_recall']:.4f}"
    )

    print(
        f"\nSaved selection:\n{selection_path}"
    )

    print("\nOfficial validation completed.")


if __name__ == "__main__":
    main()