import pandas as pd


# ---------------------------------------------------------------------
# OFFICIAL FACTOR THRESHOLDS
# ---------------------------------------------------------------------

THRESHOLDS = {
    "auditor_tier": [
        (1.0, 1.0, "Low"),
        (0.0, 0.0, "High"),
    ],

    "collateral_coverage_pct": [
        (100.0, float("inf"), "Low"),
        (50.0, 99.9999, "Medium"),
        (float("-inf"), 49.9999, "High"),
    ],

    "country_risk_0_100": [
        (float("-inf"), 20.0, "Low"),
        (21.0, 40.0, "Medium"),
        (41.0, float("inf"), "High"),
    ],

    "covenant_quality": [
        (2.0, 2.0, "Low"),
        (1.0, 1.0, "Medium"),
        (0.0, 0.0, "High"),
    ],

    "current_ratio": [
        (1.5, float("inf"), "Low"),
        (1.0, 1.4999, "Medium"),
        (float("-inf"), 0.9999, "High"),
    ],

    "debt_to_equity": [
        (float("-inf"), 0.9999, "Low"),
        (1.0, 2.5, "Medium"),
        (2.5001, float("inf"), "High"),
    ],

    "dscr": [
        (1.5001, float("inf"), "Low"),
        (1.0, 1.5, "Medium"),
        (float("-inf"), 0.9999, "High"),
    ],

    "ebit_margin_pct": [
        (12.0, float("inf"), "Low"),
        (6.0, 11.9999, "Medium"),
        (float("-inf"), 5.9999, "High"),
    ],

    "ebitda_margin_pct": [
        (20.0, float("inf"), "Low"),
        (10.0, 19.9999, "Medium"),
        (float("-inf"), 9.9999, "High"),
    ],

    "esg_controversies_3y": [
        (float("-inf"), 1.0, "Low"),
        (2.0, 3.0, "Medium"),
        (4.0, float("inf"), "High"),
    ],

    "financials_audited": [
        (1.0, 1.0, "Low"),
        (0.0, 0.0, "High"),
    ],

    "fx_revenue_pct": [
        (float("-inf"), 20.0, "Low"),
        (20.0001, 50.0, "Medium"),
        (50.0001, float("inf"), "High"),
    ],

    "governance_score_0_100": [
        (75.0, 100.0, "Low"),
        (60.0, 74.9999, "Medium"),
        (float("-inf"), 59.9999, "High"),
    ],

    "hedging_policy": [
        (2.0, 2.0, "Low"),
        (1.0, 1.0, "Medium"),
        (0.0, 0.0, "High"),
    ],

    "industry_cyclicality": [
        (0.0, 0.0, "Low"),
        (1.0, 1.0, "Medium"),
        (2.0, 2.0, "High"),
    ],

    "interest_coverage": [
        (6.0001, float("inf"), "Low"),
        (2.0, 6.0, "Medium"),
        (float("-inf"), 1.9999, "High"),
    ],

    "legal_disputes_open": [
        (float("-inf"), 1.0, "Low"),
        (2.0, 3.0, "Medium"),
        (4.0, float("inf"), "High"),
    ],

    "payment_incidents_12m": [
        (0.0, 0.0, "Low"),
        (1.0, 2.0, "Medium"),
        (3.0, float("inf"), "High"),
    ],

    "quick_ratio": [
        (1.0, float("inf"), "Low"),
        (0.8, 0.9999, "Medium"),
        (float("-inf"), 0.7999, "High"),
    ],

    "revenue_cagr_3y_pct": [
        (5.0, float("inf"), "Low"),
        (0.0, 4.9999, "Medium"),
        (float("-inf"), -0.0001, "High"),
    ],

    "revenue_usd_m": [
        (2000.0, float("inf"), "Low"),
        (500.0, 1999.99, "Medium"),
        (float("-inf"), 499.99, "High"),
    ],

    "sanctions_exposure": [
        (0.0, 0.0, "Low"),
        (1.0, 1.0, "Medium"),
        (2.0, 2.0, "High"),
    ],

    "years_in_operation": [
        (10.0, float("inf"), "Low"),
        (3.0, 9.9999, "Medium"),
        (float("-inf"), 2.9999, "High"),
    ],
}


# ---------------------------------------------------------------------
# CATEGORY ENCODING
# ---------------------------------------------------------------------

CATEGORY_CODES = {
    "auditor_tier": {
        "Other": 0.0,
        "Big4": 1.0,
    },

    "covenant_quality": {
        "Weak": 0.0,
        "Standard": 1.0,
        "Strong": 2.0,
    },

    "financials_audited": {
        "No": 0.0,
        "Yes": 1.0,
    },

    "industry_cyclicality": {
        "Low": 0.0,
        "Medium": 1.0,
        "High": 2.0,
    },

    "hedging_policy": {
        "None": 0.0,
        "Partial": 1.0,
        "Comprehensive": 2.0,
    },

    "sanctions_exposure": {
        "None": 0.0,
        "Indirect": 1.0,
        "Direct": 2.0,
    },
}


# ---------------------------------------------------------------------
# CRITICAL FACTORS
# ---------------------------------------------------------------------

CRITICAL_FACTORS = {
    "financials_audited": "Unaudited financials detected",
    "sanctions_exposure": "Direct sanctions exposure detected",
    "payment_incidents_12m": "High volume of recent payment incidents",
    "legal_disputes_open": "Excessive open legal disputes",
}


# ---------------------------------------------------------------------
# RISK DIMENSIONS
#
# These are application-level aggregation groups.
# They do NOT replace or modify the official factor thresholds.
# ---------------------------------------------------------------------

RISK_DIMENSIONS = {
    "Repayment Capacity": {
        "dscr",
        "interest_coverage",
        "operating_cf_usd_m",
        "fcf_usd_m",
        "payment_incidents_12m",
    },

    "Leverage & Liquidity": {
        "debt_to_equity",
        "current_ratio",
        "quick_ratio",
        "cash_usd_m",
        "collateral_coverage_pct",
        "covenant_quality",
    },

    "Financial Strength": {
        "revenue_usd_m",
        "ebitda_margin_pct",
        "ebit_margin_pct",
        "revenue_cagr_3y_pct",
        "years_in_operation",
    },

    "Governance & Operational": {
        "auditor_tier",
        "governance_score_0_100",
        "esg_controversies_3y",
        "financials_audited",
        "legal_disputes_open",
    },

    "External Risk": {
        "country_risk_0_100",
        "industry_cyclicality",
        "fx_revenue_pct",
        "hedging_policy",
        "sanctions_exposure",
    },
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _numeric_value(factor_name, value):
    """Convert categorical/numeric input into the value used by rules."""

    if value is None or pd.isna(value):
        return None

    if factor_name in CATEGORY_CODES:
        if isinstance(value, str):
            return CATEGORY_CODES[factor_name].get(value)

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def _aggregate_dimension(risk_levels):
    """
    Aggregate factor-level risks inside one credit-risk dimension.

    Policy:
    - 2 or more High factors -> High
    - 1 High + at least 1 Medium -> High
    - 1 isolated High -> Medium
    - 2 or more Medium factors -> Medium
    - 1 Medium -> Medium
    - all Low -> Low
    """

    high_count = risk_levels.count("High")
    medium_count = risk_levels.count("Medium")

    if high_count >= 2:
        return "High"

    if high_count == 1 and medium_count >= 1:
        return "High"

    if high_count == 1:
        return "Medium"

    if medium_count >= 1:
        return "Medium"

    if risk_levels:
        return "Low"

    return "Unknown"


def _aggregate_overall_dimension_risk(dimension_results):
    """
    Determine overall non-critical rule risk.

    Policy:
    - 2 or more High-risk dimensions -> High
    - 1 High-risk dimension -> High
    - otherwise 1 or more Medium dimensions -> Medium
    - otherwise Low
    """

    risks = [
        result["risk_level"]
        for result in dimension_results.values()
        if result["risk_level"] != "Unknown"
    ]

    high_dimensions = risks.count("High")
    medium_dimensions = risks.count("Medium")

    if high_dimensions >= 1:
        return "High"

    if medium_dimensions >= 1:
        return "Medium"

    if risks:
        return "Low"

    return "Unknown"


# ---------------------------------------------------------------------
# MAIN RULE ENGINE
# ---------------------------------------------------------------------

def evaluate(features: dict) -> dict:
    """
    Evaluate the supplied features against the official deterministic
    thresholds and aggregate them using credit-risk dimensions.

    Important:
    - Official factor thresholds are unchanged.
    - Critical factors remain hard overrides.
    - A single non-critical High factor does not automatically make
      the entire application High.
    """

    factors = []
    critical_flags = []

    counts = {
        "Low": 0,
        "Medium": 0,
        "High": 0,
    }

    # -------------------------------------------------------------
    # STEP 1: Evaluate every official factor
    # -------------------------------------------------------------

    for factor_name, rules in THRESHOLDS.items():

        if factor_name not in features:
            continue

        value = _numeric_value(
            factor_name,
            features[factor_name],
        )

        if value is None:
            continue

        assigned_risk = "Unknown"

        for min_v, max_v, risk in rules:

            if min_v <= value <= max_v:
                assigned_risk = risk
                break

        if assigned_risk == "Unknown":
            continue

        factors.append(
            {
                "factor": factor_name,
                "value": value,
                "risk_level": assigned_risk,
                "threshold_ref": str(rules),
            }
        )

        counts[assigned_risk] += 1

        # ---------------------------------------------------------
        # Critical override detection
        # ---------------------------------------------------------

        if (
            assigned_risk == "High"
            and factor_name in CRITICAL_FACTORS
        ):
            critical_flags.append(
                f"{CRITICAL_FACTORS[factor_name]} "
                f"({factor_name}={value})"
            )

    # -------------------------------------------------------------
    # STEP 2: Build dimension-level results
    # -------------------------------------------------------------

    factor_lookup = {
        factor["factor"]: factor
        for factor in factors
    }

    dimension_results = {}

    for dimension_name, dimension_factors in RISK_DIMENSIONS.items():

        dimension_factor_results = []

        for factor_name in dimension_factors:

            if factor_name in factor_lookup:

                dimension_factor_results.append(
                    factor_lookup[factor_name]
                )

        risk_levels = [
            result["risk_level"]
            for result in dimension_factor_results
        ]

        dimension_risk = _aggregate_dimension(
            risk_levels
        )

        dimension_results[dimension_name] = {
            "risk_level": dimension_risk,
            "factor_count": len(dimension_factor_results),
            "low_count": risk_levels.count("Low"),
            "medium_count": risk_levels.count("Medium"),
            "high_count": risk_levels.count("High"),
            "factors": dimension_factor_results,
        }

    # -------------------------------------------------------------
    # STEP 3: Critical override
    # -------------------------------------------------------------

    if critical_flags:

        overall_risk = "High"
        aggregation_reason = (
            "Critical deterministic factor(s) triggered. "
            "Critical conditions override normal dimension aggregation."
        )

    else:

        overall_risk = _aggregate_overall_dimension_risk(
            dimension_results
        )

        if overall_risk == "High":
            aggregation_reason = (
                "One or more credit-risk dimensions show "
                "significant concentrated deterioration."
            )

        elif overall_risk == "Medium":
            aggregation_reason = (
                "The assessment contains material non-critical "
                "risk factors, but no critical override was triggered."
            )

        elif overall_risk == "Low":
            aggregation_reason = (
                "Risk factors are predominantly Low and no critical "
                "condition was detected."
            )

        else:
            aggregation_reason = (
                "Insufficient deterministic rule results were available."
            )

    # -------------------------------------------------------------
    # STEP 4: Return complete assessment
    # -------------------------------------------------------------

    return {
        "overall_risk": overall_risk,

        "factor_results": factors,

        "critical_flags": critical_flags,

        "counts": {
            "low_risk": counts["Low"],
            "medium_risk": counts["Medium"],
            "high_risk": counts["High"],
            "total_factors": len(factors),
        },

        "dimension_results": dimension_results,

        "aggregation_reason": aggregation_reason,
    }