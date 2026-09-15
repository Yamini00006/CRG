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

REPORTS_DIR = (
    PROJECT_ROOT
    / "ml-pipeline"
    / "reports"
    / "model_comparison"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------------------
# DATA SPLIT
# ---------------------------------------------------------------------

def load_evaluation_data():
    """
    Recreate the exact train/test split used by train.py.

    Because train.py uses:
        random_state=42
        test_size=0.20
        stratify=y

    we can reproduce the same internal test set here.
    """

    if not SYNTHETIC_DATASET.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found:\n{SYNTHETIC_DATASET}"
        )

    df = pd.read_csv(SYNTHETIC_DATASET)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_test, y_test


# ---------------------------------------------------------------------
# MODEL DISCOVERY
# ---------------------------------------------------------------------

def get_model_paths():
    """Return the three trained model artifact paths."""

    model_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
    ]

    model_paths = {}

    for model_name in model_names:

        model_path = (
            MODELS_DIR
            / model_name
            / "model.pkl"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found:\n{model_path}"
            )

        model_paths[model_name] = model_path

    return model_paths


# ---------------------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------------------

def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate one trained model."""

    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    labels = ["Low", "Medium", "High"]

    # Overall metrics.
    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    # Per-class metrics.
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    high_risk_recall = report["High"]["recall"]

    print(f"\nAccuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"High Risk Recall:  {high_risk_recall:.4f}")

    print("\nPer-class performance:")

    for label in labels:

        print(
            f"{label:8s} "
            f"Precision={report[label]['precision']:.4f} "
            f"Recall={report[label]['recall']:.4f} "
            f"F1={report[label]['f1-score']:.4f}"
        )

    # Confusion matrix.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("\nConfusion Matrix:")

    print(
        pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels],
        )
    )

    # Print probability information.
    probability_df = pd.DataFrame(
        probabilities,
        columns=[
            f"probability_{label.lower()}"
            for label in model.classes_
        ],
    )

    print("\nProbability output:")
    print(
        probability_df.head().to_string(
            index=False
        )
    )

    # Generate confusion matrix image.
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    confusion_path = (
        REPORTS_DIR
        / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved confusion matrix: {confusion_path}"
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "high_risk_recall": high_risk_recall,
    }


# ---------------------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------------------

def select_best_model(results):
    """
    Select the model using macro F1 as the primary metric.

    High-risk recall is used as the first tie-breaker.
    """

    ranked = sorted(
        results,
        key=lambda result: (
            result["macro_f1"],
            result["high_risk_recall"],
        ),
        reverse=True,
    )

    return ranked[0]


# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------

def save_results(results, best_model):
    """Save model comparison results."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(results)

    results_path = (
        REPORTS_DIR
        / "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    selection_path = (
        REPORTS_DIR
        / "selected_model.txt"
    )

    selection_path.write_text(
        (
            f"Selected model: {best_model['model']}\n"
            f"Primary metric: Macro F1\n"
            f"Macro F1: {best_model['macro_f1']:.4f}\n"
            f"High-risk recall: "
            f"{best_model['high_risk_recall']:.4f}\n"
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved comparison: {results_path}"
    )

    print(
        f"Saved selection: {selection_path}"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CREDIT RISK MODEL EVALUATION")
    print("=" * 70)

    validate_feature_configuration()

    X_test, y_test = load_evaluation_data()

    print(
        f"\nInternal evaluation rows: {len(X_test)}"
    )

    print("\nEvaluation target distribution:")
    print(
        y_test.value_counts()
        .sort_index()
    )

    model_paths = get_model_paths()

    results = []

    for model_name, model_path in model_paths.items():

        print(
            f"\nLoading: {model_path}"
        )

        model = joblib.load(model_path)

        result = evaluate_model(
            model_name,
            model,
            X_test,
            y_test,
        )

        results.append(result)

    # Select the best model based on actual results.
    best_model = select_best_model(results)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print("SELECTED MODEL")
    print("=" * 70)

    print(
        f"Model: {best_model['model']}"
    )

    print(
        f"Macro F1: {best_model['macro_f1']:.4f}"
    )

    print(
        f"High-risk recall: "
        f"{best_model['high_risk_recall']:.4f}"
    )

    save_results(
        results,
        best_model,
    )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()