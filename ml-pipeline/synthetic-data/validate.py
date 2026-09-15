from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "official"
    / "credit_risk_dataset_50_entities.csv"
)

SYNTHETIC_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic"
    / "synthetic_credit_risk_dataset.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "risk_bucket"

RISK_CLASSES = [
    "Low",
    "Medium",
    "High",
]

LEAKAGE_COLUMNS = [
    "PD_1y_pct",
    "LGD_pct",
    "EAD_usd_m",
    "implied_rating",
]

CATEGORICAL_COLUMNS = [
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

NUMERIC_COLUMNS = [
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


# ============================================================
# VALIDATION THRESHOLDS
# ============================================================

MISSINGNESS_TOLERANCE = 0.05

MEAN_TOLERANCE = 0.10

MEDIAN_TOLERANCE = 0.10

STD_TOLERANCE = 0.20

CATEGORICAL_TOLERANCE = 0.05


# ============================================================
# SEMANTIC RANGES
# ============================================================

RANGES = {
    "ebitda_margin_pct": (-50.0, 100.0),
    "ebit_margin_pct": (-50.0, 100.0),
    "debt_to_equity": (0.0, 20.0),
    "interest_coverage": (0.0, 100.0),
    "dscr": (0.0, 20.0),
    "current_ratio": (0.0, 20.0),
    "quick_ratio": (0.0, 20.0),
    "dso_days": (0.0, 365.0),
    "dpo_days": (0.0, 365.0),
    "dio_days": (0.0, 365.0),
    "revenue_cagr_3y_pct": (-100.0, 100.0),
    "years_in_operation": (0.0, 200.0),
    "governance_score_0_100": (0.0, 100.0),
    "esg_controversies_3y": (0.0, 50.0),
    "country_risk_0_100": (0.0, 100.0),
    "fx_revenue_pct": (0.0, 100.0),
    "collateral_coverage_pct": (0.0, 500.0),
    "payment_incidents_12m": (0.0, 50.0),
    "legal_disputes_open": (0.0, 50.0),
}

NON_NEGATIVE_COLUMNS = [
    "revenue_usd_m",
    "cash_usd_m",
    "total_assets_usd_m",
    "equity_usd_m",
    "net_debt_usd_m",
    "interest_expense_usd_m",
    "operating_cf_usd_m",
    "capex_usd_m",
]


# ============================================================
# RESULT TRACKING
# ============================================================

failures = []
warnings = []


def add_failure(message: str):
    failures.append(message)


def add_warning(message: str):
    warnings.append(message)


# ============================================================
# BASIC STRUCTURE
# ============================================================

def validate_basic_structure(
    official_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
):
    print()
    print("=" * 70)
    print("BASIC STRUCTURE")
    print("=" * 70)

    print(f"Official rows: {len(official_df):,}")
    print(f"Synthetic rows: {len(synthetic_df):,}")

    if synthetic_df.empty:
        add_failure("Synthetic dataset is empty.")

    if TARGET_COLUMN not in synthetic_df.columns:
        add_failure(
            f"Missing target column: {TARGET_COLUMN}"
        )

    expected_columns = set(official_df.columns)
    actual_columns = set(synthetic_df.columns)

    missing_columns = expected_columns - actual_columns

    if missing_columns:
        add_failure(
            f"Missing columns: {sorted(missing_columns)}"
        )
    else:
        print("PASS — expected schema is present.")


# ============================================================
# LEAKAGE VALIDATION
# ============================================================

def validate_leakage(
    synthetic_df: pd.DataFrame,
):
    print()
    print("-" * 70)
    print("LEAKAGE CHECK")
    print("-" * 70)

    leakage_failure = False

    for column in LEAKAGE_COLUMNS:

        if column not in synthetic_df.columns:
            continue

        non_null_count = (
            synthetic_df[column]
            .notna()
            .sum()
        )

        if non_null_count > 0:
            leakage_failure = True

            add_failure(
                f"Leakage column '{column}' contains "
                f"{non_null_count:,} non-null values."
            )

            print(
                f"FAIL — {column}: "
                f"{non_null_count:,} non-null"
            )
        else:
            print(
                f"PASS — {column}: all values are null"
            )

    if not leakage_failure:
        print(
            "PASS — no usable leakage values are present."
        )


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

def validate_duplicates(
    synthetic_df: pd.DataFrame,
):
    print()
    print("-" * 70)
    print("DUPLICATE CHECK")
    print("-" * 70)

    duplicate_rows = synthetic_df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_rows}")

    if duplicate_rows > 0:
        add_failure(
            f"{duplicate_rows} duplicate rows found."
        )
    else:
        print("PASS — no duplicate rows.")

    if "entity_id" in synthetic_df.columns:

        duplicate_ids = (
            synthetic_df["entity_id"]
            .duplicated()
            .sum()
        )

        print(
            f"Duplicate entity IDs: {duplicate_ids}"
        )

        if duplicate_ids > 0:
            add_failure(
                f"{duplicate_ids} duplicate entity IDs found."
            )
        else:
            print(
                "PASS — no duplicate entity IDs."
            )


# ============================================================
# MISSINGNESS COMPARISON
# ============================================================

def validate_missingness(
    official_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
):
    print()
    print("-" * 70)
    print("MISSINGNESS COMPARISON")
    print("-" * 70)

    for column in CATEGORICAL_COLUMNS:

        if (
            column not in official_df.columns
            or column not in synthetic_df.columns
        ):
            continue

        official_rate = (
            official_df[column]
            .isna()
            .mean()
        )

        synthetic_rate = (
            synthetic_df[column]
            .isna()
            .mean()
        )

        difference = abs(
            official_rate - synthetic_rate
        )

        status = (
            "PASS"
            if difference <= MISSINGNESS_TOLERANCE
            else "WARN"
        )

        print(
            f"{status} — {column}: "
            f"official={official_rate:.2%}, "
            f"synthetic={synthetic_rate:.2%}, "
            f"difference={difference:.2%}"
        )

        if difference > MISSINGNESS_TOLERANCE:
            add_warning(
                f"Missingness difference for "
                f"{column} is {difference:.2%}."
            )


# ============================================================
# SAFE RELATIVE DIFFERENCE
# ============================================================

def relative_difference(
    official_value: float,
    synthetic_value: float,
):
    """
    Calculate relative difference safely.

    Important:
    If the official value is zero, percentage difference
    is mathematically unsuitable. In that case:

        official = 0, synthetic = 0
            -> 0 difference

        official = 0, synthetic != 0
            -> infinity

    This prevents meaningless billions-of-percent results
    when comparing zero medians.
    """

    official_value = float(official_value)
    synthetic_value = float(synthetic_value)

    if np.isclose(
        official_value,
        0.0,
        atol=1e-12,
    ):

        if np.isclose(
            synthetic_value,
            0.0,
            atol=1e-12,
        ):
            return 0.0

        return np.inf

    return abs(
        synthetic_value - official_value
    ) / abs(official_value)


def format_difference(value: float):
    if np.isinf(value):
        return "INF"

    return f"{value:.2%}"


# ============================================================
# CLASS-CONDITIONAL NUMERIC DISTRIBUTIONS
# ============================================================

def validate_numeric_distributions_by_class(
    official_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
):
    print()
    print("=" * 70)
    print("CLASS-CONDITIONAL NUMERIC DISTRIBUTION COMPARISON")
    print("=" * 70)

    print(
        "Comparing Official Low ↔ Synthetic Low,"
        " Official Medium ↔ Synthetic Medium,"
        " Official High ↔ Synthetic High."
    )

    for risk_class in RISK_CLASSES:

        official_class = official_df[
            official_df[TARGET_COLUMN] == risk_class
        ]

        synthetic_class = synthetic_df[
            synthetic_df[TARGET_COLUMN] == risk_class
        ]

        print()
        print(
            f"[{risk_class}] "
            f"Official={len(official_class):,}, "
            f"Synthetic={len(synthetic_class):,}"
        )

        if official_class.empty:
            add_failure(
                f"No official rows exist for "
                f"risk class '{risk_class}'."
            )
            continue

        if synthetic_class.empty:
            add_failure(
                f"No synthetic rows exist for "
                f"risk class '{risk_class}'."
            )
            continue

        for column in NUMERIC_COLUMNS:

            if (
                column not in official_class.columns
                or column not in synthetic_class.columns
            ):
                continue

            official = (
                official_class[column]
                .dropna()
            )

            synthetic = (
                synthetic_class[column]
                .dropna()
            )

            if official.empty or synthetic.empty:
                add_warning(
                    f"{risk_class}/{column}: "
                    f"insufficient non-null values."
                )
                continue

            official_mean = official.mean()
            synthetic_mean = synthetic.mean()

            official_median = official.median()
            synthetic_median = synthetic.median()

            official_std = official.std()
            synthetic_std = synthetic.std()

            mean_diff = relative_difference(
                official_mean,
                synthetic_mean,
            )

            median_diff = relative_difference(
                official_median,
                synthetic_median,
            )

            std_diff = relative_difference(
                official_std,
                synthetic_std,
            )

            if mean_diff > MEAN_TOLERANCE:
                add_warning(
                    f"{risk_class}/{column}: "
                    f"mean difference "
                    f"{format_difference(mean_diff)}"
                )

            if median_diff > MEDIAN_TOLERANCE:
                add_warning(
                    f"{risk_class}/{column}: "
                    f"median difference "
                    f"{format_difference(median_diff)}"
                )

            if std_diff > STD_TOLERANCE:
                add_warning(
                    f"{risk_class}/{column}: "
                    f"std difference "
                    f"{format_difference(std_diff)}"
                )

            print(
                f"{column:30s} "
                f"mean={format_difference(mean_diff):>8s} "
                f"median={format_difference(median_diff):>8s} "
                f"std={format_difference(std_diff):>8s}"
            )


# ============================================================
# CLASS-CONDITIONAL CATEGORICAL DISTRIBUTIONS
# ============================================================

def validate_categorical_distributions_by_class(
    official_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
):
    print()
    print("=" * 70)
    print(
        "CLASS-CONDITIONAL CATEGORICAL DISTRIBUTION COMPARISON"
    )
    print("=" * 70)

    for risk_class in RISK_CLASSES:

        official_class = official_df[
            official_df[TARGET_COLUMN] == risk_class
        ]

        synthetic_class = synthetic_df[
            synthetic_df[TARGET_COLUMN] == risk_class
        ]

        print()
        print(
            f"[{risk_class}] "
            f"Official={len(official_class):,}, "
            f"Synthetic={len(synthetic_class):,}"
        )

        if official_class.empty or synthetic_class.empty:
            continue

        for column in CATEGORICAL_COLUMNS:

            if (
                column not in official_class.columns
                or column not in synthetic_class.columns
            ):
                continue

            official_distribution = (
                official_class[column]
                .fillna("<MISSING>")
                .value_counts(
                    normalize=True
                )
            )

            synthetic_distribution = (
                synthetic_class[column]
                .fillna("<MISSING>")
                .value_counts(
                    normalize=True
                )
            )

            categories = (
                set(official_distribution.index)
                | set(synthetic_distribution.index)
            )

            maximum_difference = 0.0

            for category in categories:

                official_rate = (
                    official_distribution.get(
                        category,
                        0.0,
                    )
                )

                synthetic_rate = (
                    synthetic_distribution.get(
                        category,
                        0.0,
                    )
                )

                difference = abs(
                    official_rate
                    - synthetic_rate
                )

                maximum_difference = max(
                    maximum_difference,
                    difference,
                )

            status = (
                "PASS"
                if maximum_difference
                <= CATEGORICAL_TOLERANCE
                else "WARN"
            )

            print(
                f"{status} — {column}: "
                f"maximum category difference="
                f"{maximum_difference:.2%}"
            )

            if (
                maximum_difference
                > CATEGORICAL_TOLERANCE
            ):
                add_warning(
                    f"{risk_class}/{column}: maximum "
                    f"category difference is "
                    f"{maximum_difference:.2%}."
                )


# ============================================================
# SEMANTIC RANGE VALIDATION
# ============================================================

def validate_semantic_ranges(
    synthetic_df: pd.DataFrame,
):
    print()
    print("-" * 70)
    print("SEMANTIC RANGE CHECK")
    print("-" * 70)

    invalid_count = 0

    for column, (
        minimum,
        maximum,
    ) in RANGES.items():

        if column not in synthetic_df.columns:
            continue

        values = synthetic_df[column].dropna()

        invalid = (
            (values < minimum)
            | (values > maximum)
        )

        count = invalid.sum()

        if count > 0:
            invalid_count += count

            add_failure(
                f"{column}: {count} values "
                f"outside [{minimum}, {maximum}]."
            )

            print(
                f"FAIL — {column}: "
                f"{count} invalid values"
            )

    for column in NON_NEGATIVE_COLUMNS:

        if column not in synthetic_df.columns:
            continue

        values = synthetic_df[column].dropna()

        count = (values < 0).sum()

        if count > 0:
            invalid_count += count

            add_failure(
                f"{column}: {count} negative values."
            )

            print(
                f"FAIL — {column}: "
                f"{count} negative values"
            )

    if invalid_count == 0:
        print(
            "PASS — no semantic range violations."
        )


# ============================================================
# FINANCIAL RELATIONSHIP CHECK
# ============================================================

def validate_financial_relationships(
    synthetic_df: pd.DataFrame,
):
    print()
    print("-" * 70)
    print("FINANCIAL RELATIONSHIP CHECK")
    print("-" * 70)

    required = [
        "operating_cf_usd_m",
        "capex_usd_m",
        "fcf_usd_m",
    ]

    if not all(
        column in synthetic_df.columns
        for column in required
    ):
        add_failure(
            "Required FCF columns are missing."
        )
        return

    expected_fcf = (
        synthetic_df["operating_cf_usd_m"]
        - synthetic_df["capex_usd_m"]
    )

    actual_fcf = synthetic_df["fcf_usd_m"]

    valid_rows = (
        expected_fcf.notna()
        & actual_fcf.notna()
    )

    mismatches = ~np.isclose(
        expected_fcf[valid_rows],
        actual_fcf[valid_rows],
        atol=1e-6,
    )

    mismatch_count = mismatches.sum()

    if mismatch_count == 0:
        print(
            "PASS — FCF = Operating CF - CapEx "
            "for all valid rows."
        )
    else:
        add_failure(
            f"{mismatch_count} FCF relationship "
            f"violations found."
        )

        print(
            f"FAIL — {mismatch_count} "
            f"FCF mismatches."
        )


# ============================================================
# TARGET VALIDATION
# ============================================================

def validate_target(
    synthetic_df: pd.DataFrame,
):
    print()
    print("-" * 70)
    print("TARGET VALIDATION")
    print("-" * 70)

    allowed = set(RISK_CLASSES)

    actual = set(
        synthetic_df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    unexpected = actual - allowed

    if unexpected:
        add_failure(
            f"Unexpected target values: "
            f"{sorted(unexpected)}"
        )

        print(
            f"FAIL — unexpected target values: "
            f"{sorted(unexpected)}"
        )
    else:
        print(
            "PASS — target contains only "
            "Low / Medium / High."
        )

    print()
    print("Synthetic target distribution:")

    distribution = (
        synthetic_df[TARGET_COLUMN]
        .value_counts()
    )

    for risk_class in RISK_CLASSES:

        count = distribution.get(
            risk_class,
            0,
        )

        percentage = (
            count / len(synthetic_df) * 100
        )

        print(
            f"- {risk_class}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# FINAL REPORT
# ============================================================

def print_final_result():

    print()
    print("=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    print()

    if failures:

        print(
            f"FAIL — {len(failures)} "
            f"critical validation issue(s)."
        )

        for failure in failures:
            print(f"- {failure}")

    else:

        print(
            "PASS — no critical validation failures detected."
        )

    print()

    if warnings:

        print(
            f"WARN — {len(warnings)} "
            f"distribution-quality warning(s)."
        )

        for warning in warnings:
            print(f"- {warning}")

    else:

        print(
            "PASS — no distribution warnings."
        )

    print()
    print(
        "Important: passing these checks does not prove "
        "that synthetic data is equivalent to real-world data."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not OFFICIAL_DATASET.exists():
        raise FileNotFoundError(
            f"Official dataset not found:\n"
            f"{OFFICIAL_DATASET}"
        )

    if not SYNTHETIC_DATASET.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found:\n"
            f"{SYNTHETIC_DATASET}"
        )

    official_df = pd.read_csv(
        OFFICIAL_DATASET
    )

    synthetic_df = pd.read_csv(
        SYNTHETIC_DATASET
    )

    validate_basic_structure(
        official_df,
        synthetic_df,
    )

    validate_leakage(
        synthetic_df
    )

    validate_duplicates(
        synthetic_df
    )

    validate_missingness(
        official_df,
        synthetic_df,
    )

    validate_numeric_distributions_by_class(
        official_df,
        synthetic_df,
    )

    validate_categorical_distributions_by_class(
        official_df,
        synthetic_df,
    )

    validate_semantic_ranges(
        synthetic_df
    )

    validate_financial_relationships(
        synthetic_df
    )

    validate_target(
        synthetic_df
    )

    print_final_result()


if __name__ == "__main__":
    main()