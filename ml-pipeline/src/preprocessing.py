from typing import Sequence

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# These are the 37 legitimate candidate predictors established
# during leakage analysis and feature selection.
FEATURE_COLUMNS = [
    "sector",
    "country",
    "revenue_usd_m",
    "ebitda_margin_pct",
    "ebit_margin_pct",
    "cash_usd_m",
    "total_assets_usd_m",
    "equity_usd_m",
    "net_debt_usd_m",
    "debt_to_equity",
    "interest_expense_usd_m",
    "interest_coverage",
    "operating_cf_usd_m",
    "capex_usd_m",
    "fcf_usd_m",
    "dscr",
    "current_ratio",
    "quick_ratio",
    "dso_days",
    "dpo_days",
    "dio_days",
    "revenue_cagr_3y_pct",
    "years_in_operation",
    "ownership_type",
    "auditor_tier",
    "governance_score_0_100",
    "esg_controversies_3y",
    "country_risk_0_100",
    "industry_cyclicality",
    "fx_revenue_pct",
    "hedging_policy",
    "collateral_coverage_pct",
    "covenant_quality",
    "payment_incidents_12m",
    "legal_disputes_open",
    "sanctions_exposure",
    "financials_audited",
]

TARGET_COLUMN = "risk_bucket"

NUMERIC_FEATURES = [
    "revenue_usd_m",
    "ebitda_margin_pct",
    "ebit_margin_pct",
    "cash_usd_m",
    "total_assets_usd_m",
    "equity_usd_m",
    "net_debt_usd_m",
    "debt_to_equity",
    "interest_expense_usd_m",
    "interest_coverage",
    "operating_cf_usd_m",
    "capex_usd_m",
    "fcf_usd_m",
    "dscr",
    "current_ratio",
    "quick_ratio",
    "dso_days",
    "dpo_days",
    "dio_days",
    "revenue_cagr_3y_pct",
    "years_in_operation",
    "governance_score_0_100",
    "esg_controversies_3y",
    "country_risk_0_100",
    "fx_revenue_pct",
    "collateral_coverage_pct",
    "payment_incidents_12m",
    "legal_disputes_open",
]

CATEGORICAL_FEATURES = [
    "sector",
    "country",
    "ownership_type",
    "auditor_tier",
    "industry_cyclicality",
    "hedging_policy",
    "covenant_quality",
    "sanctions_exposure",
    "financials_audited",
]


def validate_feature_configuration() -> None:
    """Verify that the feature groups cover the approved feature set."""

    grouped_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    if set(grouped_features) != set(FEATURE_COLUMNS):
        missing = sorted(set(FEATURE_COLUMNS) - set(grouped_features))
        extra = sorted(set(grouped_features) - set(FEATURE_COLUMNS))

        raise ValueError(
            "Feature configuration mismatch. "
            f"Missing: {missing}; Extra: {extra}"
        )

    if len(grouped_features) != len(FEATURE_COLUMNS):
        raise ValueError("Duplicate feature found in preprocessing configuration.")


def build_preprocessor(
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """
    Build the reusable preprocessing transformer.

    Numeric:
        - median imputation
        - standardization

    Categorical:
        - explicit '<MISSING>' category
        - one-hot encoding
        - unknown categories ignored at inference time
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="<MISSING>",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(numeric_features),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate approved predictors from the target."""

    validate_feature_configuration()

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required feature columns: {missing_columns}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset is missing target column: {TARGET_COLUMN}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


if __name__ == "__main__":
    from data_loader import load_official_dataset

    dataset = load_official_dataset()

    X, y = prepare_features(dataset)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)

    print("=" * 70)
    print("PREPROCESSING VALIDATION")
    print("=" * 70)

    print(f"\nRaw feature count: {X.shape[1]}")
    print(f"Transformed feature count: {transformed.shape[1]}")
    print(f"Target count: {y.shape[0]}")

    print("\nNumeric features:")
    print(f"- {len(NUMERIC_FEATURES)}")

    print("\nCategorical features:")
    print(f"- {len(CATEGORICAL_FEATURES)}")

    print("\nMissing values before preprocessing:")
    missing = X.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("- None")
    else:
        for column, count in missing.items():
            print(f"- {column}: {count}")

    print("\nPreprocessing pipeline:")
    print("- Numeric: median imputation -> StandardScaler")
    print("- Categorical: <MISSING> imputation -> OneHotEncoder")
    print("- Unknown categories: ignored")
    print("- Unapproved columns: dropped")