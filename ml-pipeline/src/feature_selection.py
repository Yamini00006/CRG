from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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
    / "feature_analysis"
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

CORRELATION_THRESHOLD = 0.85

LEAKAGE_COLUMNS = {
    "entity_id",
    "entity_name",
    "PD_1y_pct",
    "LGD_pct",
    "EAD_usd_m",
    "implied_rating",
    "risk_bucket",
}


# ---------------------------------------------------------------------
# FEATURE SELECTION ANALYSIS
# ---------------------------------------------------------------------

def analyze_correlations(df):
    """
    Analyze correlations among legitimate numeric predictors.

    This does not automatically remove highly correlated features.
    Correlation is treated as a review signal; model performance and
    interpretability are considered before making removals.
    """

    numeric_features = [
        feature
        for feature in FEATURE_COLUMNS
        if pd.api.types.is_numeric_dtype(df[feature])
    ]

    correlation_matrix = df[
        numeric_features
    ].corr()

    pairs = []

    for i, feature_a in enumerate(numeric_features):

        for feature_b in numeric_features[i + 1:]:

            correlation = correlation_matrix.loc[
                feature_a,
                feature_b,
            ]

            if pd.notna(correlation):

                pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "pearson_correlation": correlation,
                        "absolute_correlation": abs(correlation),
                        "high_correlation": (
                            abs(correlation)
                            >= CORRELATION_THRESHOLD
                        ),
                    }
                )

    correlation_df = pd.DataFrame(pairs)

    if not correlation_df.empty:
        correlation_df = correlation_df.sort_values(
            "absolute_correlation",
            ascending=False,
        )

    return correlation_df


# ---------------------------------------------------------------------
# LEAKAGE REVIEW
# ---------------------------------------------------------------------

def check_synthetic_leakage():

    synthetic_dataset = (
        PROJECT_ROOT
        / "ml-pipeline"
        / "data"
        / "processed"
        / "synthetic"
        / "synthetic_credit_risk_dataset.csv"
    )

    if not synthetic_dataset.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found:\n{synthetic_dataset}"
        )

    df = pd.read_csv(synthetic_dataset)

    leakage_results = []

    for column in [
        "PD_1y_pct",
        "LGD_pct",
        "EAD_usd_m",
        "implied_rating",
    ]:

        if column not in df.columns:
            leakage_results.append(
                {
                    "column": column,
                    "present": False,
                    "non_null_values": 0,
                    "status": "NOT PRESENT",
                }
            )
            continue

        non_null_count = int(
            df[column].notna().sum()
        )

        leakage_results.append(
            {
                "column": column,
                "present": True,
                "non_null_values": non_null_count,
                "status": (
                    "PASS"
                    if non_null_count == 0
                    else "LEAKAGE DETECTED"
                ),
            }
        )

    return pd.DataFrame(leakage_results)


# ---------------------------------------------------------------------
# LOGISTIC REGRESSION FEATURE IMPORTANCE
# ---------------------------------------------------------------------

def extract_logistic_feature_importance():

    model_path = (
        MODELS_DIR
        / "logistic_regression"
        / "model.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Logistic Regression model not found:\n{model_path}"
        )

    pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    # Get the exact transformed feature names generated during training.
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = model.coef_

    classes = model.classes_

    if coefficients.shape[1] != len(feature_names):
        raise ValueError(
            "Coefficient count does not match "
            "transformed feature count."
        )

    rows = []

    for class_index, class_name in enumerate(classes):

        for feature_index, feature_name in enumerate(
            feature_names
        ):

            coefficient = coefficients[
                class_index,
                feature_index,
            ]

            rows.append(
                {
                    "class": class_name,
                    "feature": feature_name,
                    "coefficient": coefficient,
                    "absolute_coefficient": abs(
                        coefficient
                    ),
                }
            )

    coefficient_df = pd.DataFrame(rows)

    # -------------------------------------------------------------
    # Overall importance across classes.
    #
    # For each transformed feature, take the largest absolute
    # coefficient across the three classes.
    # -------------------------------------------------------------

    overall = (
        coefficient_df
        .groupby("feature", as_index=False)
        ["absolute_coefficient"]
        .max()
        .rename(
            columns={
                "absolute_coefficient":
                "overall_importance"
            }
        )
    )

    # Find the class associated with the strongest coefficient.
    strongest_class = (
        coefficient_df
        .loc[
            coefficient_df
            .groupby("feature")
            ["absolute_coefficient"]
            .idxmax()
        ]
        [
            [
                "feature",
                "class",
                "coefficient",
            ]
        ]
        .rename(
            columns={
                "class": "strongest_class",
                "coefficient":
                "strongest_class_coefficient",
            }
        )
    )

    overall = overall.merge(
        strongest_class,
        on="feature",
        how="left",
    )

    overall = overall.sort_values(
        "overall_importance",
        ascending=False,
    )

    return coefficient_df, overall


# ---------------------------------------------------------------------
# SAVE REPORTS
# ---------------------------------------------------------------------

def save_reports(
    correlation_df,
    leakage_df,
    coefficient_df,
    overall_importance_df,
):

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    correlation_path = (
        REPORTS_DIR
        / "numeric_feature_correlations.csv"
    )

    correlation_df.to_csv(
        correlation_path,
        index=False,
    )

    leakage_path = (
        REPORTS_DIR
        / "synthetic_leakage_check.csv"
    )

    leakage_df.to_csv(
        leakage_path,
        index=False,
    )

    coefficient_path = (
        REPORTS_DIR
        / "logistic_regression_coefficients.csv"
    )

    coefficient_df.to_csv(
        coefficient_path,
        index=False,
    )

    importance_path = (
        REPORTS_DIR
        / "logistic_regression_feature_importance.csv"
    )

    overall_importance_df.to_csv(
        importance_path,
        index=False,
    )

    print(
        f"\nSaved correlation report:\n{correlation_path}"
    )

    print(
        f"Saved leakage report:\n{leakage_path}"
    )

    print(
        f"Saved coefficient report:\n{coefficient_path}"
    )

    print(
        f"Saved feature importance report:\n{importance_path}"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CREDIT RISK — FEATURE SELECTION & MODEL INTERPRETABILITY")
    print("=" * 70)

    validate_feature_configuration()

    # -------------------------------------------------------------
    # LOAD OFFICIAL DATASET FOR CORRELATION ANALYSIS
    # -------------------------------------------------------------

    if not OFFICIAL_DATASET.exists():
        raise FileNotFoundError(
            f"Official dataset not found:\n{OFFICIAL_DATASET}"
        )

    df = pd.read_csv(
        OFFICIAL_DATASET
    )

    print(
        f"\nOfficial dataset rows: {len(df)}"
    )

    print(
        f"Legitimate model features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    # -------------------------------------------------------------
    # VERIFY LEAKAGE EXCLUSIONS
    # -------------------------------------------------------------

    leakage_in_features = (
        set(FEATURE_COLUMNS)
        & LEAKAGE_COLUMNS
    )

    if leakage_in_features:
        raise ValueError(
            "Leakage columns found in FEATURE_COLUMNS:\n"
            + "\n".join(
                f"- {column}"
                for column in sorted(
                    leakage_in_features
                )
            )
        )

    print(
        "\nFeature leakage exclusion: PASS"
    )

    print(
        "Excluded downstream fields:"
    )

    for column in sorted(
        LEAKAGE_COLUMNS
        - {"entity_id", "entity_name", "risk_bucket"}
    ):
        print(f"- {column}")

    # -------------------------------------------------------------
    # CORRELATION ANALYSIS
    # -------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "HIGH-CORRELATION FEATURE PAIRS"
    )

    correlation_df = analyze_correlations(
        df
    )

    high_corr = correlation_df[
        correlation_df[
            "high_correlation"
        ]
    ]

    if high_corr.empty:

        print(
            "No feature pairs exceeded "
            f"|correlation| >= {CORRELATION_THRESHOLD}."
        )

    else:

        for _, row in high_corr.iterrows():

            print(
                f"- {row['feature_a']} ↔ "
                f"{row['feature_b']}: "
                f"{row['pearson_correlation']:.4f}"
            )

    # -------------------------------------------------------------
    # SYNTHETIC LEAKAGE
    # -------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "SYNTHETIC DATA LEAKAGE CHECK"
    )

    leakage_df = (
        check_synthetic_leakage()
    )

    for _, row in leakage_df.iterrows():

        print(
            f"- {row['column']}: "
            f"{row['status']} "
            f"({row['non_null_values']} "
            f"non-null values)"
        )

    if (
        leakage_df["status"]
        == "LEAKAGE DETECTED"
    ).any():

        raise ValueError(
            "Synthetic-data leakage detected."
        )

    print(
        "Result: PASS — no usable downstream "
        "leakage values are present."
    )

    # -------------------------------------------------------------
    # LOGISTIC REGRESSION IMPORTANCE
    # -------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "LOGISTIC REGRESSION FEATURE IMPORTANCE"
    )

    coefficient_df, importance_df = (
        extract_logistic_feature_importance()
    )

    print(
        "\nTop 20 transformed features "
        "by absolute coefficient:"
    )

    print(
        importance_df
        .head(20)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # -------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------

    save_reports(
        correlation_df,
        leakage_df,
        coefficient_df,
        importance_df,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()