from typing import Optional, Any
import math
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator

import ml_service
import rule_engine
import decision_engine
import genai_service
from database import get_connection


app = FastAPI(
    title="Credit Risk Gauge API",
    version="1.0.0"
)


FEATURE_COLUMNS = ml_service.FEATURE_COLUMNS

NUMERIC_FEATURES = {
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
}


class FeaturesContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    revenue_usd_m: Optional[float] = None
    ebitda_margin_pct: Optional[float] = None
    ebit_margin_pct: Optional[float] = None
    cash_usd_m: Optional[float] = None
    total_assets_usd_m: Optional[float] = None
    equity_usd_m: Optional[float] = None
    net_debt_usd_m: Optional[float] = None
    debt_to_equity: Optional[float] = None
    interest_expense_usd_m: Optional[float] = None
    interest_coverage: Optional[float] = None
    operating_cf_usd_m: Optional[float] = None
    capex_usd_m: Optional[float] = None
    fcf_usd_m: Optional[float] = None
    dscr: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    dso_days: Optional[float] = None
    dpo_days: Optional[float] = None
    dio_days: Optional[float] = None
    revenue_cagr_3y_pct: Optional[float] = None
    years_in_operation: Optional[float] = None

    ownership_type: Optional[str] = None
    auditor_tier: Optional[str] = None

    governance_score_0_100: Optional[float] = None
    esg_controversies_3y: Optional[float] = None
    country_risk_0_100: Optional[float] = None

    industry_cyclicality: Optional[str] = None
    fx_revenue_pct: Optional[float] = None
    hedging_policy: Optional[str] = None
    collateral_coverage_pct: Optional[float] = None
    covenant_quality: Optional[str] = None

    payment_incidents_12m: Optional[float] = None
    legal_disputes_open: Optional[float] = None

    sanctions_exposure: Optional[str] = None
    financials_audited: Optional[str] = None

    sector: Optional[str] = None
    country: Optional[str] = None

    @model_validator(mode="after")
    def validate_feature_values(self):
        values = self.model_dump()

        for name in NUMERIC_FEATURES:
            value = values.get(name)

            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        categorical_allowed = {
            "ownership_type": {"Public", "Private", "State"},
            "auditor_tier": {"Other", "Big4"},
            "industry_cyclicality": {"Low", "Medium", "High"},
            "hedging_policy": {"None", "Partial", "Comprehensive"},
            "covenant_quality": {"Weak", "Standard", "Strong"},
            "sanctions_exposure": {"None", "Indirect", "Direct"},
            "financials_audited": {"No", "Yes"},
        }

        for name, allowed in categorical_allowed.items():
            value = values.get(name)

            if value is not None and value not in allowed:
                raise ValueError(
                    f"{name} must be one of {sorted(allowed)}."
                )

        return self


class AssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    entity_name: str
    features: FeaturesContract

    @model_validator(mode="before")
    @classmethod
    def validate_exact_feature_contract(cls, values: Any):

        if not isinstance(values, dict):
            return values

        raw_features = values.get("features")

        if not isinstance(raw_features, dict):
            raise ValueError(
                "features must be an object containing exactly "
                "37 model features."
            )

        provided = set(raw_features)
        expected = set(FEATURE_COLUMNS)

        missing = sorted(expected - provided)
        extra = sorted(provided - expected)

        if missing:
            raise ValueError(
                f"Missing model features: {missing}"
            )

        if extra:
            raise ValueError(
                f"Unexpected model features: {extra}"
            )

        return values


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ml_service.model is not None,
        "model_features": len(FEATURE_COLUMNS),
        "model_path": str(ml_service.MODEL_PATH),
    }


@app.post("/api/assess")
def assess_credit_risk(request: AssessmentRequest):

    feature_dict = request.features.model_dump()

    # -------------------------------------------------------------
    # STEP 1: ML prediction
    # -------------------------------------------------------------

    try:
        ml_result = ml_service.predict(feature_dict)

    except Exception as exc:
        print(f"ML Service Error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"ML Service Error: {exc}"
        ) from exc

    # -------------------------------------------------------------
    # STEP 2: Rule evaluation
    # STEP 3: Decision
    # STEP 4: GenAI explanation
    # -------------------------------------------------------------

    try:
        rule_result = rule_engine.evaluate(feature_dict)

        decision_result = decision_engine.decide(
            ml_risk=ml_result["predicted_risk"],
            rule_risk=rule_result["overall_risk"],
            critical_flags=rule_result["critical_flags"],
        )

        explanation = genai_service.generate_explanation(
            client_name=request.entity_name,
            features=feature_dict,
            ml_result=ml_result,
            rule_result=rule_result,
            decision_result=decision_result,
        )

    except Exception as exc:
        print(f"Assessment decision error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"Assessment decision error: {exc}"
        ) from exc

    # -------------------------------------------------------------
    # STEP 5: Persist assessment + factor results
    # -------------------------------------------------------------

    conn = None

    try:
        conn = get_connection()

        with conn:
            with conn.cursor() as cursor:

                # Convert JSON-compatible values for PostgreSQL JSONB.
                probabilities = ml_result.get("probabilities", {})
                critical_flags = rule_result.get("critical_flags", [])

                model_version = ml_result.get("model_version")

                # -------------------------------------------------
                # Insert assessment
                # -------------------------------------------------

                cursor.execute(
                    """
                    INSERT INTO assessments (
                        entity_id,
                        entity_name,
                        ml_risk,
                        rule_risk,
                        final_risk,
                        application_status,
                        review_requirement,
                        explanation,
                        ml_probabilities,
                        critical_flags,
                        model_version
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb,
                        %s::jsonb,
                        %s
                    )
                    RETURNING assessment_id;
                    """,
                    (
                        request.entity_id,
                        request.entity_name,
                        ml_result["predicted_risk"],
                        rule_result["overall_risk"],
                        decision_result["final_risk"],
                        decision_result["application_status"],
                        decision_result["review_requirement"],
                        explanation,
                        json.dumps(probabilities),
                        json.dumps(critical_flags),
                        model_version,
                    ),
                )

                assessment_row = cursor.fetchone()

                if not assessment_row:
                    raise RuntimeError(
                        "Assessment insert did not return an assessment_id."
                    )

                assessment_id = assessment_row["assessment_id"]

                # -------------------------------------------------
                # Insert factor-level rule results
                # -------------------------------------------------

                factor_results = rule_result.get(
                    "factor_results",
                    []
                )

                for factor in factor_results:

                    factor_name = factor.get("factor")

                    # Find the dimension associated with this factor.
                    dimension = None

                    for (
                        dimension_name,
                        dimension_result
                    ) in rule_result.get(
                        "dimension_results",
                        {}
                    ).items():

                        dimension_factors = dimension_result.get(
                            "factors",
                            []
                        )

                        if any(
                            item.get("factor") == factor_name
                            for item in dimension_factors
                        ):
                            dimension = dimension_name
                            break

                    cursor.execute(
                        """
                        INSERT INTO assessment_factors (
                            assessment_id,
                            factor,
                            value,
                            risk_level,
                            threshold_ref,
                            dimension
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        );
                        """,
                        (
                            assessment_id,
                            factor_name,
                            str(factor.get("value")),
                            factor.get("risk_level"),
                            factor.get("threshold_ref"),
                            dimension,
                        ),
                    )

        # Leaving the `with conn:` block commits the transaction.

    except Exception as exc:

        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass

        print(f"Database Error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"Database Error: {exc}"
        ) from exc

    finally:

        if conn is not None:
            conn.close()

    # -------------------------------------------------------------
    # STEP 6: Return assessment
    # -------------------------------------------------------------

    return {
        "assessment_id": assessment_id,

        "entity_info": {
            "id": request.entity_id,
            "name": request.entity_name,
        },

        "ml_assessment": ml_result,

        "rule_assessment": rule_result,

        "final_decision": decision_result,

        "explanation": explanation,
    }


@app.get("/api/assessments")
def get_assessments():

    conn = None

    try:
        conn = get_connection()

        with conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT
                        assessment_id,
                        entity_id,
                        entity_name,
                        ml_risk,
                        rule_risk,
                        final_risk,
                        application_status,
                        review_requirement,
                        explanation,
                        ml_probabilities,
                        critical_flags,
                        model_version,
                        created_at
                    FROM assessments
                    ORDER BY created_at DESC, assessment_id DESC;
                    """
                )

                rows = cursor.fetchall()

                assessments = []

                for row in rows:

                    assessments.append({
                        "assessment_id": row["assessment_id"],
                        "entity_info": {
                            "id": row["entity_id"],
                            "name": row["entity_name"],
                        },
                        "ml_assessment": {
                            "predicted_risk": row["ml_risk"],
                            "probabilities": row["ml_probabilities"] or {},
                            "model_version": row["model_version"],
                        },
                        "rule_assessment": {
                            "overall_risk": row["rule_risk"],
                            "critical_flags": row["critical_flags"] or [],
                        },
                        "final_decision": {
                            "final_risk": row["final_risk"],
                            "application_status": row["application_status"],
                            "review_requirement": row["review_requirement"],
                        },
                        "explanation": row["explanation"],
                        "created_at": row["created_at"].isoformat()
                        if row["created_at"]
                        else None,
                    })

                return {
                    "count": len(assessments),
                    "assessments": assessments,
                }

    except Exception as exc:

        print(f"Database Error while fetching assessments: {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"Database Error: {exc}"
        ) from exc

    finally:

        if conn is not None:
            conn.close()