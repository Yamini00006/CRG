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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic"
)

OUTPUT_FILE = OUTPUT_DIR / "synthetic_credit_risk_dataset.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
N_SYNTHETIC_ROWS = 10_000

TARGET_COUNTS = {
    "Low": 4_000,
    "Medium": 4_500,
    "High": 1_500,
}


# ============================================================
# COLUMNS
# ============================================================

TARGET_COLUMN = "risk_bucket"

IDENTIFIER_COLUMNS = [
    "entity_id",
    "entity_name",
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


EXPECTED_COLUMNS = (
    IDENTIFIER_COLUMNS
    + CATEGORICAL_COLUMNS
    + NUMERIC_COLUMNS
    + LEAKAGE_COLUMNS
    + [TARGET_COLUMN]
)


# ============================================================
# INTEGER / COUNT FEATURES
# ============================================================

COUNT_COLUMNS = [
    "esg_controversies_3y",
    "payment_incidents_12m",
    "legal_disputes_open",
]


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


NON_NEGATIVE_FINANCIAL_COLUMNS = [
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
# CORRELATED FEATURE GROUPS
# ============================================================

FINANCIAL_SCALE_GROUP = [
    "revenue_usd_m",
    "cash_usd_m",
    "total_assets_usd_m",
    "equity_usd_m",
    "net_debt_usd_m",
    "interest_expense_usd_m",
    "operating_cf_usd_m",
    "capex_usd_m",
]

PROFITABILITY_GROUP = [
    "ebitda_margin_pct",
    "ebit_margin_pct",
]

LIQUIDITY_GROUP = [
    "current_ratio",
    "quick_ratio",
]

WORKING_CAPITAL_GROUP = [
    "dso_days",
    "dpo_days",
    "dio_days",
]

RISK_SCORE_GROUP = [
    "governance_score_0_100",
    "country_risk_0_100",
]


# ============================================================
# VALUE UTILITIES
# ============================================================

def clip_value(
    column: str,
    value,
):
    if pd.isna(value):
        return value

    if column in RANGES:
        minimum, maximum = RANGES[column]

        return float(
            np.clip(
                value,
                minimum,
                maximum,
            )
        )

    if column in NON_NEGATIVE_FINANCIAL_COLUMNS:
        return max(
            0.0,
            float(value),
        )

    return value


# ============================================================
# NUMERIC PERTURBATION
# ============================================================

def perturb_count(
    column: str,
    value,
    rng: np.random.Generator,
    scale_factor: float,
):
    """
    Perturb a count-valued feature while preserving its
    integer/count semantics.

    Zero counts are not converted into fractional values.
    """

    if pd.isna(value):
        return np.nan

    value = int(round(float(value)))

    # For an observed zero, retain zero.
    #
    # This is important for:
    # - payment incidents
    # - legal disputes
    # - ESG controversies
    #
    # We do not invent an incident simply because Gaussian
    # noise happened to be positive.
    if value == 0:
        return 0

    noise = rng.normal(
        0.0,
        0.20 * scale_factor,
    )

    result = int(
        round(
            value + noise
        )
    )

    result = max(
        0,
        result,
    )

    minimum, maximum = RANGES[column]

    result = int(
        np.clip(
            result,
            minimum,
            maximum,
        )
    )

    return result


def perturb_value(
    column: str,
    value,
    rng: np.random.Generator,
    scale_factor: float = 1.0,
):
    """
    Generate a moderate perturbation around an observed value.
    """

    if pd.isna(value):
        return np.nan

    # Count features require special treatment.
    if column in COUNT_COLUMNS:
        return perturb_count(
            column,
            value,
            rng,
            scale_factor,
        )

    value = float(value)

    # Financial amounts.
    if column in NON_NEGATIVE_FINANCIAL_COLUMNS:

        noise = rng.normal(
            0.0,
            0.05 * scale_factor,
        )

        result = value * (
            1.0 + noise
        )

    # Percentage features.
    elif column.endswith("_pct"):

        result = (
            value
            + rng.normal(
                0.0,
                1.5 * scale_factor,
            )
        )

    # Ratio features.
    elif column in [
        "debt_to_equity",
        "interest_coverage",
        "dscr",
        "current_ratio",
        "quick_ratio",
    ]:

        noise = rng.normal(
            0.0,
            0.04 * scale_factor,
        )

        result = value * (
            1.0 + noise
        )

    # Operational day metrics.
    elif column in [
        "dso_days",
        "dpo_days",
        "dio_days",
    ]:

        result = (
            value
            + rng.normal(
                0.0,
                3.0 * scale_factor,
            )
        )

    # Governance / country scores.
    elif column in [
        "governance_score_0_100",
        "country_risk_0_100",
    ]:

        result = (
            value
            + rng.normal(
                0.0,
                1.5 * scale_factor,
            )
        )

    # Years in operation.
    elif column == "years_in_operation":

        result = (
            value
            + rng.normal(
                0.0,
                0.5 * scale_factor,
            )
        )

    else:
        result = value

    return clip_value(
        column,
        result,
    )


# ============================================================
# CORRELATED GROUP PERTURBATION
# ============================================================

def perturb_correlated_group(
    row: pd.Series,
    columns: list[str],
    rng: np.random.Generator,
    scale_factor: float,
):
    """
    Apply a shared movement to correlated features followed
    by small feature-specific noise.
    """

    shared_shock = rng.normal(
        0.0,
        0.025 * scale_factor,
    )

    for column in columns:

        value = row[column]

        if pd.isna(value):
            continue

        # Count variables should never pass through this
        # function, but keep the guard for safety.
        if column in COUNT_COLUMNS:
            continue

        value = float(value)

        if column in NON_NEGATIVE_FINANCIAL_COLUMNS:

            specific_noise = rng.normal(
                0.0,
                0.02 * scale_factor,
            )

            new_value = value * (
                1.0
                + shared_shock
                + specific_noise
            )

        elif column.endswith("_pct"):

            specific_noise = rng.normal(
                0.0,
                0.6 * scale_factor,
            )

            new_value = (
                value
                + (shared_shock * 5.0)
                + specific_noise
            )

        elif column in [
            "current_ratio",
            "quick_ratio",
            "debt_to_equity",
            "interest_coverage",
            "dscr",
        ]:

            specific_noise = rng.normal(
                0.0,
                0.015 * scale_factor,
            )

            new_value = value * (
                1.0
                + shared_shock
                + specific_noise
            )

        elif column in [
            "dso_days",
            "dpo_days",
            "dio_days",
        ]:

            specific_noise = rng.normal(
                0.0,
                1.0 * scale_factor,
            )

            new_value = (
                value
                + (shared_shock * 10.0)
                + specific_noise
            )

        elif column in [
            "governance_score_0_100",
            "country_risk_0_100",
        ]:

            specific_noise = rng.normal(
                0.0,
                0.7 * scale_factor,
            )

            new_value = (
                value
                + (shared_shock * 5.0)
                + specific_noise
            )

        else:
            new_value = value

        row[column] = clip_value(
            column,
            new_value,
        )


# ============================================================
# FINANCIAL RELATIONSHIP
# ============================================================

def preserve_financial_relationships(
    row: pd.Series,
):
    """
    Preserve:
        FCF = Operating CF - CapEx
    """

    operating_cf = row[
        "operating_cf_usd_m"
    ]

    capex = row[
        "capex_usd_m"
    ]

    if (
        not pd.isna(operating_cf)
        and not pd.isna(capex)
    ):
        row["fcf_usd_m"] = (
            operating_cf
            - capex
        )

    return row


# ============================================================
# CATEGORICAL SAMPLING
# ============================================================

def sample_categorical_value(
    official_df: pd.DataFrame,
    risk_class: str,
    column: str,
    rng: np.random.Generator,
):
    """
    Sample a categorical value from the same risk class.

    Critical rule:
    If the selected risk class contains only missing values
    for a column, preserve that missingness.

    We do NOT fall back to the global distribution because that
    can create artificial class-specific relationships.
    """

    class_values = official_df.loc[
        official_df[TARGET_COLUMN] == risk_class,
        column,
    ]

    if class_values.empty:
        return np.nan

    non_null_values = (
        class_values
        .dropna()
    )

    # --------------------------------------------------------
    # Entire class is missing for this feature.
    # Preserve missingness.
    # --------------------------------------------------------

    if non_null_values.empty:
        return np.nan

    # --------------------------------------------------------
    # Preserve observed class-specific missingness.
    # --------------------------------------------------------

    missing_rate = (
        class_values
        .isna()
        .mean()
    )

    if rng.random() < missing_rate:
        return np.nan

    # --------------------------------------------------------
    # Sample only from observed values in this risk class.
    # --------------------------------------------------------

    return non_null_values.iloc[
        rng.integers(
            0,
            len(non_null_values),
        )
    ]


# ============================================================
# SOURCE PROFILE
# ============================================================

def select_source_profile(
    official_df: pd.DataFrame,
    risk_class: str,
    rng: np.random.Generator,
):
    candidates = official_df[
        official_df[TARGET_COLUMN]
        == risk_class
    ]

    if candidates.empty:
        raise ValueError(
            f"No official rows found for "
            f"risk class: {risk_class}"
        )

    index = rng.integers(
        0,
        len(candidates),
    )

    return candidates.iloc[index].copy()


# ============================================================
# ROW GENERATION
# ============================================================

def generate_row(
    official_df: pd.DataFrame,
    risk_class: str,
    synthetic_id: int,
    rng: np.random.Generator,
):
    """
    Generate one synthetic record around an observed profile
    belonging to the requested risk class.
    """

    source = select_source_profile(
        official_df,
        risk_class,
        rng,
    )

    row = {}

    # --------------------------------------------------------
    # Identifiers
    # --------------------------------------------------------

    row["entity_id"] = (
        f"SYN{synthetic_id:06d}"
    )

    row["entity_name"] = (
        f"Synthetic Entity {synthetic_id:06d}"
    )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    for column in CATEGORICAL_COLUMNS:

        row[column] = sample_categorical_value(
            official_df,
            risk_class,
            column,
            rng,
        )

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:
        row[column] = source[column]

    temp_row = pd.Series(row)

    # High-risk class has only three source profiles.
    # Use slightly smaller perturbations to avoid creating
    # unrealistic extremes around those few profiles.

    scale_factor = 1.0

    if risk_class == "High":
        scale_factor = 0.75

    # Correlated groups.
    perturb_correlated_group(
        temp_row,
        FINANCIAL_SCALE_GROUP,
        rng,
        scale_factor,
    )

    perturb_correlated_group(
        temp_row,
        PROFITABILITY_GROUP,
        rng,
        scale_factor,
    )

    perturb_correlated_group(
        temp_row,
        LIQUIDITY_GROUP,
        rng,
        scale_factor,
    )

    perturb_correlated_group(
        temp_row,
        WORKING_CAPITAL_GROUP,
        rng,
        scale_factor,
    )

    perturb_correlated_group(
        temp_row,
        RISK_SCORE_GROUP,
        rng,
        scale_factor,
    )

    grouped_columns = set(
        FINANCIAL_SCALE_GROUP
        + PROFITABILITY_GROUP
        + LIQUIDITY_GROUP
        + WORKING_CAPITAL_GROUP
        + RISK_SCORE_GROUP
    )

    # Remaining numeric columns.
    for column in NUMERIC_COLUMNS:

        if column in grouped_columns:
            continue

        temp_row[column] = perturb_value(
            column,
            temp_row[column],
            rng,
            scale_factor,
        )

    row.update(
        temp_row.to_dict()
    )

    # --------------------------------------------------------
    # Restore deterministic financial relationship.
    # --------------------------------------------------------

    row_series = pd.Series(row)

    row_series = (
        preserve_financial_relationships(
            row_series
        )
    )

    row.update(
        row_series.to_dict()
    )

    # --------------------------------------------------------
    # Target
    #
    # IMPORTANT:
    # The target is inherited from the official source class.
    # We do not invent an overall-risk formula.
    # --------------------------------------------------------

    row[TARGET_COLUMN] = risk_class

    # --------------------------------------------------------
    # Leakage columns
    #
    # They remain present for schema compatibility but contain
    # no usable values.
    # --------------------------------------------------------

    for column in LEAKAGE_COLUMNS:
        row[column] = np.nan

    return row


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_synthetic_dataset():

    if not OFFICIAL_DATASET.exists():
        raise FileNotFoundError(
            f"Official dataset not found:\n"
            f"{OFFICIAL_DATASET}"
        )

    official_df = pd.read_csv(
        OFFICIAL_DATASET
    )

    if TARGET_COLUMN not in official_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found."
        )

    expected_classes = set(
        TARGET_COUNTS.keys()
    )

    actual_classes = set(
        official_df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not expected_classes.issubset(
        actual_classes
    ):
        raise ValueError(
            "Official dataset does not contain "
            f"all required classes. Found: "
            f"{actual_classes}"
        )

    missing_leakage = [
        column
        for column in LEAKAGE_COLUMNS
        if column not in official_df.columns
    ]

    if missing_leakage:
        raise ValueError(
            "Expected leakage columns are missing: "
            f"{missing_leakage}"
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []

    synthetic_id = 1

    for risk_class, count in TARGET_COUNTS.items():

        print(
            f"Generating {count:,} "
            f"{risk_class} synthetic records..."
        )

        for _ in range(count):

            row = generate_row(
                official_df=official_df,
                risk_class=risk_class,
                synthetic_id=synthetic_id,
                rng=rng,
            )

            rows.append(row)

            synthetic_id += 1

    synthetic_df = pd.DataFrame(rows)

    # Ensure expected schema.
    for column in EXPECTED_COLUMNS:

        if column not in synthetic_df.columns:
            synthetic_df[column] = np.nan

    synthetic_df = synthetic_df[
        EXPECTED_COLUMNS
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    synthetic_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return (
        official_df,
        synthetic_df,
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    official_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 70)

    print()
    print(
        f"Source rows: {len(official_df):,}"
    )

    print(
        f"Synthetic rows: {len(synthetic_df):,}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print("Official target distribution:")

    official_distribution = (
        official_df[TARGET_COLUMN]
        .value_counts()
    )

    for risk_class in [
        "Low",
        "Medium",
        "High",
    ]:

        count = official_distribution.get(
            risk_class,
            0,
        )

        percentage = (
            count
            / len(official_df)
            * 100
        )

        print(
            f"- {risk_class}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print()
    print("Synthetic target distribution:")

    synthetic_distribution = (
        synthetic_df[TARGET_COLUMN]
        .value_counts()
    )

    for risk_class in [
        "Low",
        "Medium",
        "High",
    ]:

        count = synthetic_distribution.get(
            risk_class,
            0,
        )

        percentage = (
            count
            / len(synthetic_df)
            * 100
        )

        print(
            f"- {risk_class}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print()
    print("Leakage columns in synthetic dataset:")

    leakage_found = []

    for column in LEAKAGE_COLUMNS:

        non_null_count = (
            synthetic_df[column]
            .notna()
            .sum()
        )

        if non_null_count > 0:
            leakage_found.append(
                f"{column}: "
                f"{non_null_count:,} non-null"
            )

    if leakage_found:

        for item in leakage_found:
            print(f"- {item}")

    else:

        print(
            "- None "
            "(all intentionally blank)"
        )

    print()
    print(
        "Duplicate rows:",
        synthetic_df.duplicated().sum(),
    )

    print(
        "Duplicate synthetic IDs:",
        synthetic_df[
            "entity_id"
        ].duplicated().sum(),
    )

    print()
    print("Generation completed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    official_df, synthetic_df = (
        generate_synthetic_dataset()
    )

    print_summary(
        official_df,
        synthetic_df,
    )