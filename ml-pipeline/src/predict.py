from pathlib import Path

import joblib
import pandas as pd

from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_configuration,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml-pipeline"
    / "models"
    / "logistic_regression"
    / "model.pkl"
)


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------

def load_model():
    """
    Load the complete trained Logistic Regression pipeline.

    The artifact contains:
        1. preprocessing
        2. trained Logistic Regression model
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ---------------------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------------------

def validate_input(application):
    """
    Validate that an application contains exactly the model features.

    Extra fields are rejected so downstream/leakage fields cannot
    accidentally be passed into the model.
    """

    if not isinstance(application, dict):
        raise TypeError(
            "Application input must be a dictionary."
        )

    provided_features = set(application.keys())
    expected_features = set(FEATURE_COLUMNS)

    missing_features = (
        expected_features
        - provided_features
    )

    extra_features = (
        provided_features
        - expected_features
    )

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(
                f"- {feature}"
                for feature in sorted(
                    missing_features
                )
            )
        )

    if extra_features:
        raise ValueError(
            "Unexpected input fields:\n"
            + "\n".join(
                f"- {feature}"
                for feature in sorted(
                    extra_features
                )
            )
        )


# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------

def predict_risk(application):
    """
    Predict credit risk for one application.

    Returns:
        risk_bucket
        probability_low
        probability_medium
        probability_high
    """

    validate_input(application)

    model = load_model()

    # Construct a DataFrame using the exact training feature order.
    input_df = pd.DataFrame(
        [
            {
                feature: application[feature]
                for feature in FEATURE_COLUMNS
            }
        ]
    )

    prediction = model.predict(
        input_df
    )[0]

    probabilities = model.predict_proba(
        input_df
    )[0]

    # Map probabilities using the model's actual class ordering.
    class_probabilities = dict(
        zip(
            model.classes_,
            probabilities,
        )
    )

    result = {
        "risk_bucket": prediction,
        "probability_low": float(
            class_probabilities.get(
                "Low",
                0.0,
            )
        ),
        "probability_medium": float(
            class_probabilities.get(
                "Medium",
                0.0,
            )
        ),
        "probability_high": float(
            class_probabilities.get(
                "High",
                0.0,
            )
        ),
    }

    return result


# ---------------------------------------------------------------------
# TEST WITH OFFICIAL DATA
# ---------------------------------------------------------------------

def run_sample_prediction():
    """
    Run one prediction using the first official entity.

    This is only a smoke test of the prediction pipeline.
    """

    official_dataset = (
        PROJECT_ROOT
        / "ml-pipeline"
        / "data"
        / "raw"
        / "official"
        / "credit_risk_dataset_50_entities.csv"
    )

    if not official_dataset.exists():
        raise FileNotFoundError(
            f"Official dataset not found:\n{official_dataset}"
        )

    df = pd.read_csv(
        official_dataset
    )

    sample = df.iloc[0]

    application = {
        feature: sample[feature]
        for feature in FEATURE_COLUMNS
    }

    result = predict_risk(
        application
    )

    print("\n" + "=" * 70)
    print("SAMPLE CREDIT RISK PREDICTION")
    print("=" * 70)

    print(
        f"\nEntity: {sample['entity_name']}"
    )

    print(
        f"Actual risk: {sample[TARGET_COLUMN]}"
    )

    print(
        f"Predicted risk: {result['risk_bucket']}"
    )

    print(
        f"\nLow probability: "
        f"{result['probability_low']:.6f}"
    )

    print(
        f"Medium probability: "
        f"{result['probability_medium']:.6f}"
    )

    print(
        f"High probability: "
        f"{result['probability_high']:.6f}"
    )

    print(
        "\nPrediction pipeline working successfully."
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CREDIT RISK PREDICTION SERVICE")
    print("=" * 70)

    validate_feature_configuration()

    print(
        f"\nModel:\n{MODEL_PATH}"
    )

    print(
        f"\nExpected input features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "\nLeakage protection: PASS"
    )

    print(
        "Prediction uses only approved model features."
    )

    run_sample_prediction()


if __name__ == "__main__":
    main()