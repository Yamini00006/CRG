from pathlib import Path
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "model" / "trained_model.pkl"

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

LEAKAGE_COLUMNS = {
    "entity_id",
    "entity_name",
    "PD_1y_pct",
    "LGD_pct",
    "EAD_usd_m",
    "risk_bucket",
    "implied_rating",
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model artifact not found: {MODEL_PATH}")

    # The model artifact was trained with a newer sklearn release than the
    # current runtime in this development environment. Keep the warning visible
    # rather than silently hiding a model/runtime compatibility issue.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", InconsistentVersionWarning)
        loaded = joblib.load(MODEL_PATH)

    for warning in captured:
        print(f"MODEL COMPATIBILITY WARNING: {warning.message}")

    expected = list(FEATURE_COLUMNS)
    actual = list(getattr(loaded, "feature_names_in_", []))
    if actual != expected:
        raise ValueError(
            "Trained model feature contract does not match the application contract. "
            f"Expected {len(expected)} ordered features, received {len(actual)}."
        )

    return loaded


model = load_model()


def predict(features: dict) -> dict:
    print("--- ML SERVICE REQUEST RECEIVED ---")

    provided = set(features)
    expected = set(FEATURE_COLUMNS)

    missing = sorted(expected - provided)
    extra = sorted(provided - expected)

    if missing:
        raise ValueError(f"Model contract violation: missing features: {missing}")
    if extra:
        raise ValueError(f"Model contract violation: unexpected features: {extra}")
    if provided & LEAKAGE_COLUMNS:
        raise ValueError("Leakage/identifier fields detected in ML input.")

    # Preserve categorical strings exactly as they were used during training.
    # Missing values remain None and are handled by the fitted preprocessing pipeline.
    df = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    pred_class = model.predict(df)[0]
    pred_probs = model.predict_proba(df)[0]

    probabilities = {
        str(cls): float(prob)
        for cls, prob in zip(model.classes_, pred_probs)
    }

    if not all(pd.notna(value) and pd.api.types.is_number(value) for value in probabilities.values()):
        raise ValueError("Model returned a non-finite probability.")

    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(f"Prediction: {pred_class}")
    print(f"Probabilities: {probabilities}")
    print("-----------------------------------")

    return {
        "predicted_risk": str(pred_class),
        "probabilities": {
            "Low": probabilities.get("Low", 0.0),
            "Medium": probabilities.get("Medium", 0.0),
            "High": probabilities.get("High", 0.0),
        },
        "model_version": "v1.0-official",
    }
