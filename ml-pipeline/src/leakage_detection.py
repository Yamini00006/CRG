from pathlib import Path

import pandas as pd

from data_loader import load_official_dataset


TARGET_COLUMN = "risk_bucket"

# Columns that identify the entity rather than describe its credit risk.
IDENTIFIER_COLUMNS = [
    "entity_id",
    "entity_name",
]

# Columns that are downstream credit-risk outputs and must not be used
# as predictors when predicting risk_bucket.
KNOWN_LEAKAGE_COLUMNS = [
    "PD_1y_pct",
    "LGD_pct",
    "EAD_usd_m",
    "implied_rating",
]

# Columns that are not predictors because they are the target itself.
TARGET_COLUMNS = [
    TARGET_COLUMN,
]


def calculate_numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman correlation between numeric features and target."""

    risk_order = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    encoded_target = df[TARGET_COLUMN].map(risk_order)

    results = []

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        correlation = df[column].corr(encoded_target, method="spearman")

        results.append(
            {
                "feature": column,
                "spearman_correlation": correlation,
                "absolute_correlation": abs(correlation)
                if pd.notna(correlation)
                else None,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("absolute_correlation", ascending=False)
        .reset_index(drop=True)
    )


def calculate_categorical_risk_tables(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Show risk distribution for each categorical feature."""

    categorical_columns = [
        column
        for column in df.select_dtypes(include=["object", "str"]).columns
        if column != TARGET_COLUMN
    ]

    tables = {}

    for column in categorical_columns:
        table = pd.crosstab(
            df[column].fillna("<MISSING>"),
            df[TARGET_COLUMN],
            normalize="index",
        )

        tables[column] = table.round(3)

    return tables


def generate_leakage_report(df: pd.DataFrame) -> None:
    """Print the leakage analysis report."""

    print("=" * 70)
    print("LEAKAGE ANALYSIS")
    print("=" * 70)

    print("\nTarget:")
    print(f"- {TARGET_COLUMN}")

    print("\nIdentifier columns — exclude from ML:")
    for column in IDENTIFIER_COLUMNS:
        print(f"- {column}")

    print("\nKnown downstream/leakage columns — exclude from ML:")
    for column in KNOWN_LEAKAGE_COLUMNS:
        print(f"- {column}")

    print("\nTarget column — exclude from features:")
    for column in TARGET_COLUMNS:
        print(f"- {column}")

    excluded_columns = (
        IDENTIFIER_COLUMNS
        + KNOWN_LEAKAGE_COLUMNS
        + TARGET_COLUMNS
    )

    candidate_features = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    print(
        f"\nCandidate predictor columns before feature selection: "
        f"{len(candidate_features)}"
    )

    print("\nCandidate predictors:")
    for column in candidate_features:
        print(f"- {column}")

    print("\n" + "-" * 70)
    print("NUMERIC FEATURE / TARGET CORRELATIONS")
    print("-" * 70)

    correlation_table = calculate_numeric_correlations(df)

    if correlation_table.empty:
        print("No numeric columns found.")
    else:
        print(correlation_table.to_string(index=False))

    print("\n" + "-" * 70)
    print("CATEGORICAL FEATURE / TARGET DISTRIBUTIONS")
    print("-" * 70)

    categorical_tables = calculate_categorical_risk_tables(df)

    for column, table in categorical_tables.items():
        print(f"\n{column}:")
        print(table.to_string())

    print("\n" + "=" * 70)
    print("LEAKAGE ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nExclude these from model training:")
    for column in excluded_columns:
        print(f"- {column}")

    print(
        "\nImportant: correlation alone does not determine whether a "
        "feature is valid or invalid."
    )
    print(
        "A feature is considered leakage when it contains information "
        "that would not legitimately be available at prediction time."
    )


if __name__ == "__main__":
    dataset = load_official_dataset()
    generate_leakage_report(dataset)
    