from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="Risk Assessment",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "assessments" not in st.session_state:
    st.session_state["assessments"] = []


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "official_dataset.csv"
)

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://backend:8000"
)


# ============================================================
# MODEL FEATURE CONTRACT
# ============================================================

FEATURE_COLUMNS = [
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
    "sector",
    "country",
]


# ============================================================
# LEAKAGE PROTECTION
# ============================================================

LEAKAGE_COLUMNS = {
    "entity_id",
    "entity_name",
    "PD_1y_pct",
    "LGD_pct",
    "EAD_usd_m",
    "risk_bucket",
    "implied_rating",
}


# ============================================================
# LOAD OFFICIAL DATASET
# ============================================================

@st.cache_data
def load_official_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


df = load_official_data()


# ============================================================
# JSON-SAFE VALUE CONVERSION
# ============================================================

def safe_value(value):
    """
    Convert pandas/numpy values to JSON-safe Python values
    without inventing data.
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        value = float(value)

        return value if math.isfinite(value) else None

    return str(value)


# ============================================================
# BUILD MODEL FEATURE PAYLOAD
# ============================================================

def build_feature_payload(row: pd.Series) -> dict:

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in row.index
    ]

    if missing:
        raise ValueError(
            f"Official dataset is missing model features: {missing}"
        )

    features = {
        feature: safe_value(row[feature])
        for feature in FEATURE_COLUMNS
    }

    # --------------------------------------------------------
    # Verify exact feature contract
    # --------------------------------------------------------

    if set(features) != set(FEATURE_COLUMNS):
        raise ValueError(
            "ML feature contract mismatch."
        )

    # --------------------------------------------------------
    # Verify no leakage fields
    # --------------------------------------------------------

    if set(features) & LEAKAGE_COLUMNS:
        raise ValueError(
            "Leakage field detected in ML payload."
        )

    # --------------------------------------------------------
    # Verify JSON serialization
    # --------------------------------------------------------

    json.dumps(
        features,
        allow_nan=False
    )

    return features


# ============================================================
# POST ASSESSMENT TO BACKEND
# ============================================================

def post_assessment(
    entity_id: str,
    entity_name: str,
    features: dict
):

    payload = {
        "entity_id": str(entity_id),
        "entity_name": str(entity_name),
        "features": features,
    }

    # --------------------------------------------------------
    # Verify complete payload is JSON safe
    # --------------------------------------------------------

    json.dumps(
        payload,
        allow_nan=False
    )

    response = requests.post(
        f"{BACKEND_URL}/api/assess",
        json=payload,
        timeout=45,
    )

    if response.status_code != 200:

        try:
            detail = response.json().get(
                "detail",
                response.text
            )

        except Exception:
            detail = response.text

        raise RuntimeError(
            f"Backend returned HTTP "
            f"{response.status_code}: {detail}"
        )

    return response.json()


# ============================================================
# RENDER CLIENT PROFILE
# ============================================================

def render_profile(row: pd.Series):

    st.markdown(
        "### Client Profile Review"
    )

    # ========================================================
    # A. ENTITY INFORMATION
    # ========================================================

    with st.expander(
        "A. Entity Information",
        expanded=True
    ):

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Entity ID",
            row["entity_id"]
        )

        c2.metric(
            "Entity Name",
            row["entity_name"]
        )

        c3.metric(
            "Sector",
            row["sector"]
        )

        c4.metric(
            "Country",
            row["country"]
        )

        c1.write(
            f"**Ownership Type:** "
            f"{row['ownership_type']}"
        )

        c2.write(
            f"**Years in Operation:** "
            f"{row['years_in_operation']}"
        )

    # ========================================================
    # B. FINANCIAL STRENGTH
    # ========================================================

    with st.expander(
        "B. Financial Strength",
        expanded=True
    ):

        items = [
            ("Revenue ($M)", "revenue_usd_m"),
            ("EBITDA Margin (%)", "ebitda_margin_pct"),
            ("EBIT Margin (%)", "ebit_margin_pct"),
            ("Cash ($M)", "cash_usd_m"),
            ("Total Assets ($M)", "total_assets_usd_m"),
            ("Equity ($M)", "equity_usd_m"),
            ("Net Debt ($M)", "net_debt_usd_m"),
            ("Operating Cash Flow ($M)", "operating_cf_usd_m"),
            ("Capex ($M)", "capex_usd_m"),
            ("Free Cash Flow ($M)", "fcf_usd_m"),
        ]

        cols = st.columns(4)

        for i, (label, key) in enumerate(items):

            cols[i % 4].metric(
                label,
                "N/A"
                if pd.isna(row[key])
                else row[key]
            )

    # ========================================================
    # C. DEBT & LIQUIDITY
    # ========================================================

    with st.expander(
        "C. Debt & Liquidity",
        expanded=True
    ):

        items = [
            ("Debt to Equity", "debt_to_equity"),
            ("Interest Expense ($M)", "interest_expense_usd_m"),
            ("Interest Coverage", "interest_coverage"),
            ("DSCR", "dscr"),
            ("Current Ratio", "current_ratio"),
            ("Quick Ratio", "quick_ratio"),
            ("DSO (days)", "dso_days"),
            ("DPO (days)", "dpo_days"),
            ("DIO (days)", "dio_days"),
        ]

        cols = st.columns(4)

        for i, (label, key) in enumerate(items):

            cols[i % 4].metric(
                label,
                "N/A"
                if pd.isna(row[key])
                else row[key]
            )

    # ========================================================
    # D. GROWTH & OPERATING PERFORMANCE
    # ========================================================

    with st.expander(
        "D. Growth & Operating Performance",
        expanded=True
    ):

        c1, c2 = st.columns(2)

        c1.metric(
            "Revenue CAGR (3Y) (%)",
            row["revenue_cagr_3y_pct"]
        )

        c2.metric(
            "Years in Operation",
            row["years_in_operation"]
        )

    # ========================================================
    # E. RISK & GOVERNANCE
    # ========================================================

    with st.expander(
        "E. Risk & Governance",
        expanded=True
    ):

        items = [
            ("Auditor Tier", "auditor_tier"),
            ("Governance Score (0–100)", "governance_score_0_100"),
            ("ESG Controversies (3Y)", "esg_controversies_3y"),
            ("Country Risk (0–100)", "country_risk_0_100"),
            ("Industry Cyclicality", "industry_cyclicality"),
            ("FX Revenue (%)", "fx_revenue_pct"),
            ("Hedging Policy", "hedging_policy"),
            ("Collateral Coverage (%)", "collateral_coverage_pct"),
            ("Covenant Quality", "covenant_quality"),
        ]

        cols = st.columns(3)

        for i, (label, key) in enumerate(items):

            value = row[key]

            cols[i % 3].write(
                f"**{label}:** "
                f"{'N/A' if pd.isna(value) else value}"
            )

    # ========================================================
    # F. CREDIT HISTORY & COMPLIANCE
    # ========================================================

    with st.expander(
        "F. Credit History & Compliance",
        expanded=True
    ):

        items = [
            ("Payment Incidents (12M)", "payment_incidents_12m"),
            ("Open Legal Disputes", "legal_disputes_open"),
            ("Sanctions Exposure", "sanctions_exposure"),
            ("Financials Audited", "financials_audited"),
        ]

        cols = st.columns(4)

        for i, (label, key) in enumerate(items):

            value = row[key]

            cols[i].write(
                f"**{label}:** "
                f"{'N/A' if pd.isna(value) else value}"
            )

    # ========================================================
    # REFERENCE / GROUND TRUTH
    # ========================================================

    with st.expander(
        "Reference / Ground-Truth Fields (Not ML Inputs)",
        expanded=False
    ):

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.write(
            f"**PD 1Y (%):** "
            f"{row['PD_1y_pct']}"
        )

        c2.write(
            f"**LGD (%):** "
            f"{row['LGD_pct']}"
        )

        c3.write(
            f"**EAD ($M):** "
            f"{row['EAD_usd_m']}"
        )

        c4.write(
            f"**Reference Risk:** "
            f"{row['risk_bucket']}"
        )

        c5.write(
            f"**Implied Rating:** "
            f"{row['implied_rating']}"
        )


# ============================================================
# RENDER ASSESSMENT RESULTS
# ============================================================

def render_results(data: dict):

    st.markdown("---")

    st.header(
        "Assessment Results"
    )

    # ========================================================
    # FINAL DECISION SUMMARY
    # ========================================================

    decision = data["final_decision"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Final Risk",
        decision["final_risk"]
    )

    c2.metric(
        "Application Status",
        decision["application_status"]
    )

    c3.metric(
        "Review Requirement",
        decision["review_requirement"]
    )

    st.info(
        decision.get(
            "rationale",
            ""
        )
    )

    # ========================================================
    # RESULT TABS
    # ========================================================

    tab_ml, tab_rules, tab_genai = st.tabs(
        [
            "ML Insights",
            "Rule Engine Drill-down",
            "GenAI Rationale",
        ]
    )

    # ========================================================
    # ML TAB
    # ========================================================

    with tab_ml:

        ml = data["ml_assessment"]

        st.subheader(
            "Machine Learning Assessment"
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Predicted Risk",
            ml["predicted_risk"]
        )

        c1.caption(
            f"Model: "
            f"{ml.get('model_version', 'Unknown')}"
        )

        probs = {
            "Low": float(
                ml["probabilities"].get(
                    "Low",
                    0.0
                )
            ),
            "Medium": float(
                ml["probabilities"].get(
                    "Medium",
                    0.0
                )
            ),
            "High": float(
                ml["probabilities"].get(
                    "High",
                    0.0
                )
            ),
        }

        prob_df = pd.DataFrame(
            {
                "Risk": list(
                    probs.keys()
                ),
                "Probability": list(
                    probs.values()
                ),
            }
        )

        fig = px.bar(
            prob_df,
            x="Risk",
            y="Probability",
            text=prob_df["Probability"].map(
                lambda x: f"{x:.2%}"
            ),
            height=360,
        )

        fig.update_yaxes(
            range=[0, 1],
            tickformat=".0%"
        )

        fig.update_layout(
            showlegend=False
        )

        c2.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # RULE ENGINE TAB
    # ========================================================

    with tab_rules:

        rule = data["rule_assessment"]

        # ----------------------------------------------------
        # RULE ENGINE SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Deterministic Rule Evaluation"
        )

        rule_risk = rule.get(
            "overall_risk",
            "Unknown"
        )

        counts = rule.get(
            "counts",
            {}
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Overall Rule Risk",
            rule_risk
        )

        c2.metric(
            "Low Factors",
            counts.get(
                "low_risk",
                0
            )
        )

        c3.metric(
            "Medium Factors",
            counts.get(
                "medium_risk",
                0
            )
        )

        c4.metric(
            "High Factors",
            counts.get(
                "high_risk",
                0
            )
        )

        # ----------------------------------------------------
        # RAG RISK INDICATOR
        # ----------------------------------------------------

        risk_upper = str(
            rule_risk
        ).upper()

        if risk_upper == "LOW":

            st.success(
                "🟢 LOW RISK — "
                "No High-risk factor is driving "
                "the rule assessment."
            )

        elif risk_upper == "MEDIUM":

            st.warning(
                "🟠 MEDIUM RISK — "
                "One or more risk dimensions "
                "require attention."
            )

        elif risk_upper == "HIGH":

            st.error(
                "🔴 HIGH RISK — "
                "At least one risk dimension has "
                "significant deterioration."
            )

        else:

            st.info(
                "Risk level is not available."
            )

        # ----------------------------------------------------
        # AGGREGATION EXPLANATION
        # ----------------------------------------------------

        aggregation_reason = rule.get(
            "aggregation_reason"
        )

        if aggregation_reason:

            st.info(
                f"**How the rule engine reached "
                f"this result:** "
                f"{aggregation_reason}"
            )

        # ----------------------------------------------------
        # CRITICAL FLAGS
        # ----------------------------------------------------

        critical_flags = rule.get(
            "critical_flags",
            []
        )

        if critical_flags:

            st.error(
                "Critical Flags Detected"
            )

            for flag in critical_flags:

                st.write(
                    f"- {flag}"
                )

            st.warning(
                "Critical deterministic conditions "
                "require mandatory review."
            )

        else:

            st.success(
                "No critical deterministic flags "
                "were triggered."
            )

        # ----------------------------------------------------
        # RISK DIMENSIONS
        # ----------------------------------------------------

        st.markdown(
            "#### Risk Dimension Assessment"
        )

        dimension_results = rule.get(
            "dimension_results",
            {}
        )

        if dimension_results:

            dimension_rows = []

            for (
                dimension_name,
                dimension
            ) in dimension_results.items():

                dimension_rows.append(
                    {
                        "Risk Dimension": dimension_name,

                        "Overall Risk": dimension.get(
                            "risk_level",
                            dimension.get(
                                "overall_risk",
                                "Unknown"
                            )
                        ),

                        "Low": dimension.get(
                            "low_count",
                            0
                        ),

                        "Medium": dimension.get(
                            "medium_count",
                            0
                        ),

                        "High": dimension.get(
                            "high_count",
                            0
                        ),

                        "Factors Evaluated": dimension.get(
                            "factor_count",
                            dimension.get(
                                "total_factor_count",
                                0
                            )
                        ),
                    }
                )

            dimension_df = pd.DataFrame(
                dimension_rows
            )

            st.dataframe(
                dimension_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.warning(
                "Dimension-level rule results "
                "are not available in this response."
            )

        # ----------------------------------------------------
        # KEY RISK DRIVERS
        # ----------------------------------------------------

        st.markdown(
            "#### Key Risk Drivers"
        )

        factor_results = rule.get(
            "factor_results",
            []
        )

        high_factors = [
            factor
            for factor in factor_results
            if str(
                factor.get(
                    "risk_level",
                    ""
                )
            ).lower() == "high"
        ]

        medium_factors = [
            factor
            for factor in factor_results
            if str(
                factor.get(
                    "risk_level",
                    ""
                )
            ).lower() == "medium"
        ]

        if high_factors:

            st.markdown(
                "**High-risk factors**"
            )

            for factor in high_factors:

                st.write(
                    f"- **"
                    f"{factor.get('factor', 'Unknown')}"
                    f"**: "
                    f"{factor.get('value', 'N/A')}"
                )

        if medium_factors:

            st.markdown(
                "**Medium-risk factors**"
            )

            for factor in medium_factors:

                st.write(
                    f"- **"
                    f"{factor.get('factor', 'Unknown')}"
                    f"**: "
                    f"{factor.get('value', 'N/A')}"
                )

        if (
            not high_factors
            and not medium_factors
        ):

            st.success(
                "No High or Medium risk factors "
                "were identified."
            )

        # ----------------------------------------------------
        # COMPLETE FACTOR-LEVEL EVALUATION
        # ----------------------------------------------------

        with st.expander(
            "View Complete Factor-Level Evaluation",
            expanded=False
        ):

            factors_df = pd.DataFrame(
                factor_results
            )

            if not factors_df.empty:

                display_columns = [
                    "factor",
                    "value",
                    "risk_level",
                    "threshold_ref",
                ]

                available_columns = [
                    column
                    for column in display_columns
                    if column in factors_df.columns
                ]

                st.dataframe(
                    factors_df[
                        available_columns
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.warning(
                    "No deterministic factor results "
                    "are available."
                )

    # ========================================================
    # GENAI TAB
    # ========================================================

    with tab_genai:

        st.subheader(
            "Assessment Explanation"
        )

        st.info(
            data["explanation"]
        )

    # ========================================================
    # PROBABILITY DISCLAIMER
    # ========================================================

    st.caption(
        "The ML probabilities shown above are "
        "model class probabilities, not literal "
        "probabilities of default."
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "Credit Risk Assessment"
)

st.caption(
    "Evaluate an existing official entity or "
    "a new application using the same "
    "37-feature model contract."
)


# ============================================================
# MAIN TABS
# ============================================================

tab_existing, tab_new = st.tabs(
    [
        "Existing Client",
        "New Application",
    ]
)


# ============================================================
# EXISTING CLIENT
# ============================================================

with tab_existing:

    if df.empty:

        st.error(
            "Official dataset could not be loaded."
        )

    else:

        selected_name = st.selectbox(
            "Select Existing Entity",
            df["entity_name"].tolist(),
        )

        row = df.loc[
            df["entity_name"] == selected_name
        ].iloc[0]

        render_profile(row)

        if st.button(
            "Assess Existing Client",
            type="primary"
        ):

            try:

                features = build_feature_payload(
                    row
                )

                st.caption(
                    f"Sending exactly "
                    f"{len(features)} model features "
                    f"to the assessment service."
                )

                with st.spinner(
                    "Running ML model, rule engine "
                    "and decision engine..."
                ):

                    data = post_assessment(
                        row["entity_id"],
                        row["entity_name"],
                        features
                    )

                st.session_state[
                    "assessments"
                ].append(data)

                render_results(data)

            except (
                ValueError,
                RuntimeError,
                requests.exceptions.RequestException
            ) as exc:

                st.error(
                    f"Assessment failed: {exc}"
                )


# ============================================================
# NEW APPLICATION
# ============================================================

with tab_new:

    st.subheader(
        "New Credit Application"
    )

    st.caption(
        "Manual applications use the complete "
        "37-feature input contract."
    )

    with st.form(
        "new_application_form"
    ):

        # ====================================================
        # A. ENTITY INFORMATION
        # ====================================================

        st.markdown(
            "#### A. Entity Information"
        )

        c1, c2, c3 = st.columns(3)

        entity_name = c1.text_input(
            "Entity Name",
            "New Enterprise LLC"
        )

        sector = c2.selectbox(
            "Sector",
            sorted(
                df["sector"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        country = c3.selectbox(
            "Country",
            sorted(
                df["country"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        ownership_type = c1.selectbox(
            "Ownership Type",
            [
                "Public",
                "Private",
                "State"
            ]
        )

        years_in_operation = c2.number_input(
            "Years in Operation",
            min_value=0.0,
            value=5.0
        )

        c3.write("")

        # ====================================================
        # B. FINANCIAL STRENGTH
        # ====================================================

        st.markdown(
            "#### B. Financial Strength"
        )

        c1, c2, c3, c4 = st.columns(4)

        revenue_usd_m = c1.number_input(
            "Revenue ($M)",
            min_value=0.0,
            value=100.0
        )

        ebitda_margin_pct = c2.number_input(
            "EBITDA Margin (%)",
            value=15.0
        )

        ebit_margin_pct = c3.number_input(
            "EBIT Margin (%)",
            value=10.0
        )

        cash_usd_m = c4.number_input(
            "Cash ($M)",
            min_value=0.0,
            value=20.0
        )

        total_assets_usd_m = c1.number_input(
            "Total Assets ($M)",
            min_value=0.0,
            value=200.0
        )

        equity_usd_m = c2.number_input(
            "Equity ($M)",
            min_value=0.0,
            value=80.0
        )

        net_debt_usd_m = c3.number_input(
            "Net Debt ($M)",
            value=50.0
        )

        operating_cf_usd_m = c4.number_input(
            "Operating Cash Flow ($M)",
            value=15.0
        )

        capex_usd_m = c1.number_input(
            "Capex ($M)",
            min_value=0.0,
            value=5.0
        )

        fcf_usd_m = c2.number_input(
            "Free Cash Flow ($M)",
            value=10.0
        )

        # ====================================================
        # C. DEBT & LIQUIDITY
        # ====================================================

        st.markdown(
            "#### C. Debt & Liquidity"
        )

        c1, c2, c3, c4 = st.columns(4)

        debt_to_equity = c1.number_input(
            "Debt to Equity",
            min_value=0.0,
            value=0.62
        )

        interest_expense_usd_m = c2.number_input(
            "Interest Expense ($M)",
            min_value=0.0,
            value=2.0
        )

        interest_coverage = c3.number_input(
            "Interest Coverage",
            min_value=0.0,
            value=5.0
        )

        dscr = c4.number_input(
            "DSCR",
            min_value=0.0,
            value=1.5
        )

        current_ratio = c1.number_input(
            "Current Ratio",
            min_value=0.0,
            value=1.2
        )

        quick_ratio = c2.number_input(
            "Quick Ratio",
            min_value=0.0,
            value=0.9
        )

        dso_days = c3.number_input(
            "DSO (Days)",
            min_value=0.0,
            value=45.0
        )

        dpo_days = c4.number_input(
            "DPO (Days)",
            min_value=0.0,
            value=30.0
        )

        dio_days = c1.number_input(
            "DIO (Days)",
            min_value=0.0,
            value=60.0
        )

        # ====================================================
        # D. GROWTH & OPERATING PERFORMANCE
        # ====================================================

        st.markdown(
            "#### D. Growth & Operating Performance"
        )

        revenue_cagr_3y_pct = st.number_input(
            "Revenue CAGR 3Y (%)",
            value=8.0
        )

        # ====================================================
        # E. RISK & GOVERNANCE
        # ====================================================

        st.markdown(
            "#### E. Risk & Governance"
        )

        c1, c2, c3 = st.columns(3)

        auditor_tier = c1.selectbox(
            "Auditor Tier",
            [
                "Other",
                "Big4"
            ]
        )

        governance_score_0_100 = c2.number_input(
            "Governance Score (0–100)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        esg_controversies_3y = c3.number_input(
            "ESG Controversies (3Y)",
            min_value=0.0,
            step=1.0,
            value=0.0
        )

        country_risk_0_100 = c1.number_input(
            "Country Risk (0–100)",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

        industry_cyclicality = c2.selectbox(
            "Industry Cyclicality",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

        fx_revenue_pct = c3.number_input(
            "FX Revenue (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )

        hedging_policy = c1.selectbox(
            "Hedging Policy",
            [
                "None",
                "Partial",
                "Comprehensive"
            ]
        )

        collateral_coverage_pct = c2.number_input(
            "Collateral Coverage (%)",
            min_value=0.0,
            value=100.0
        )

        covenant_quality = c3.selectbox(
            "Covenant Quality",
            [
                "Weak",
                "Standard",
                "Strong"
            ]
        )

        # ====================================================
        # F. CREDIT HISTORY & COMPLIANCE
        # ====================================================

        st.markdown(
            "#### F. Credit History & Compliance"
        )

        c1, c2, c3, c4 = st.columns(4)

        payment_incidents_12m = c1.number_input(
            "Payment Incidents (12M)",
            min_value=0.0,
            step=1.0,
            value=0.0
        )

        legal_disputes_open = c2.number_input(
            "Open Legal Disputes",
            min_value=0.0,
            step=1.0,
            value=0.0
        )

        sanctions_exposure = c3.selectbox(
            "Sanctions Exposure",
            [
                "None",
                "Indirect",
                "Direct"
            ]
        )

        financials_audited = c4.selectbox(
            "Financials Audited",
            [
                "No",
                "Yes"
            ]
        )

        submitted = st.form_submit_button(
            "Assess New Application",
            type="primary"
        )

    # ========================================================
    # PROCESS NEW APPLICATION
    # ========================================================

    if submitted:

        features = {

            "revenue_usd_m":
                revenue_usd_m,

            "ebitda_margin_pct":
                ebitda_margin_pct,

            "ebit_margin_pct":
                ebit_margin_pct,

            "cash_usd_m":
                cash_usd_m,

            "total_assets_usd_m":
                total_assets_usd_m,

            "equity_usd_m":
                equity_usd_m,

            "net_debt_usd_m":
                net_debt_usd_m,

            "debt_to_equity":
                debt_to_equity,

            "interest_expense_usd_m":
                interest_expense_usd_m,

            "interest_coverage":
                interest_coverage,

            "operating_cf_usd_m":
                operating_cf_usd_m,

            "capex_usd_m":
                capex_usd_m,

            "fcf_usd_m":
                fcf_usd_m,

            "dscr":
                dscr,

            "current_ratio":
                current_ratio,

            "quick_ratio":
                quick_ratio,

            "dso_days":
                dso_days,

            "dpo_days":
                dpo_days,

            "dio_days":
                dio_days,

            "revenue_cagr_3y_pct":
                revenue_cagr_3y_pct,

            "years_in_operation":
                years_in_operation,

            "ownership_type":
                ownership_type,

            "auditor_tier":
                auditor_tier,

            "governance_score_0_100":
                governance_score_0_100,

            "esg_controversies_3y":
                esg_controversies_3y,

            "country_risk_0_100":
                country_risk_0_100,

            "industry_cyclicality":
                industry_cyclicality,

            "fx_revenue_pct":
                fx_revenue_pct,

            "hedging_policy":
                hedging_policy,

            "collateral_coverage_pct":
                collateral_coverage_pct,

            "covenant_quality":
                covenant_quality,

            "payment_incidents_12m":
                payment_incidents_12m,

            "legal_disputes_open":
                legal_disputes_open,

            "sanctions_exposure":
                sanctions_exposure,

            "financials_audited":
                financials_audited,

            "sector":
                sector,

            "country":
                country,
        }

        # ----------------------------------------------------
        # EXACT 37 FEATURE CHECK
        # ----------------------------------------------------

        if set(features) != set(
            FEATURE_COLUMNS
        ):

            st.error(
                "Internal error: new application "
                "does not contain exactly "
                "37 model features."
            )

        else:

            try:

                # ------------------------------------------------
                # JSON SAFETY CHECK
                # ------------------------------------------------

                json.dumps(
                    features,
                    allow_nan=False
                )

                # ------------------------------------------------
                # SEND TO BACKEND
                # ------------------------------------------------

                with st.spinner(
                    "Running ML model, rule engine "
                    "and decision engine..."
                ):

                    data = post_assessment(
                        "NEW-001",
                        entity_name,
                        features
                    )

                # ------------------------------------------------
                # STORE ASSESSMENT
                # ------------------------------------------------

                st.session_state[
                    "assessments"
                ].append(data)

                # ------------------------------------------------
                # DISPLAY RESULTS
                # ------------------------------------------------

                render_results(data)

            except (
                ValueError,
                RuntimeError,
                requests.exceptions.RequestException
            ) as exc:

                st.error(
                    f"Assessment failed: {exc}"
                )