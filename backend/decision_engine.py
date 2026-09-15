RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def decide(ml_risk: str, rule_risk: str, critical_flags: list) -> dict:
    if critical_flags:
        return {
            "final_risk": "High",
            "application_status": "Pending",
            "review_requirement": "Mandatory Review",
            "rationale": (
                f"Critical deterministic rule(s) triggered. "
                f"ML predicted {ml_risk}; rule engine evaluated {rule_risk}. "
                "Mandatory review overrides the normal decision path."
            ),
        }

    if ml_risk not in RISK_ORDER or rule_risk not in RISK_ORDER:
        return {
            "final_risk": "High",
            "application_status": "Pending",
            "review_requirement": "Mandatory Review",
            "rationale": (
                f"Unable to reconcile ML risk '{ml_risk}' and rule risk "
                f"'{rule_risk}' deterministically; manual review is required."
            ),
        }

    final_risk = (
        ml_risk
        if RISK_ORDER[ml_risk] >= RISK_ORDER[rule_risk]
        else rule_risk
    )

    same = ml_risk == rule_risk

    if final_risk == "Low" and same:
        status = "Approved"
        review = "No Review"
    elif final_risk == "High" and same:
        status = "Pending"
        review = "Mandatory Review"
    elif final_risk == "High":
        status = "Pending"
        review = "Mandatory Review"
    else:
        status = "Pending"
        review = "Standard Review"

    return {
        "final_risk": final_risk,
        "application_status": status,
        "review_requirement": review,
        "rationale": (
            f"ML predicted {ml_risk}; deterministic rules evaluated {rule_risk}. "
            f"The final risk uses the more conservative outcome: {final_risk}."
        ),
    }
