import os
import json

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Reports",
    page_icon="📄",
    layout="wide",
)


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000"
)


@st.cache_data(ttl=10)
def load_assessments():

    try:
        response = requests.get(
            f"{BACKEND_URL}/api/assessments",
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("assessments", [])

    except requests.RequestException as exc:

        st.error(
            f"Unable to load assessment history: {exc}"
        )

        return []


assessments = load_assessments()


st.title("Reporting Center")

st.markdown(
    "Historical credit risk assessments persisted in PostgreSQL."
)


# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------

if assessments:

    col1, col2, col3, col4 = st.columns(4)

    total = len(assessments)

    high_count = sum(
        1
        for assessment in assessments
        if assessment.get("final_decision", {}).get(
            "final_risk"
        ) == "High"
    )

    medium_count = sum(
        1
        for assessment in assessments
        if assessment.get("final_decision", {}).get(
            "final_risk"
        ) == "Medium"
    )

    low_count = sum(
        1
        for assessment in assessments
        if assessment.get("final_decision", {}).get(
            "final_risk"
        ) == "Low"
    )

    col1.metric(
        "Total Assessments",
        total,
    )

    col2.metric(
        "High Risk",
        high_count,
    )

    col3.metric(
        "Medium Risk",
        medium_count,
    )

    col4.metric(
        "Low Risk",
        low_count,
    )

    st.markdown("---")


    # ---------------------------------------------------------
    # Assessment history table
    # ---------------------------------------------------------

    st.subheader("Assessment History")

    history = []

    for assessment in assessments:

        entity = assessment.get(
            "entity_info",
            {}
        )

        ml = assessment.get(
            "ml_assessment",
            {}
        )

        rule = assessment.get(
            "rule_assessment",
            {}
        )

        decision = assessment.get(
            "final_decision",
            {}
        )

        history.append(
            {
                "Assessment ID": assessment.get(
                    "assessment_id"
                ),

                "Entity ID": entity.get(
                    "id",
                    "Unknown"
                ),

                "Entity": entity.get(
                    "name",
                    "Unknown"
                ),

                "ML Risk": ml.get(
                    "predicted_risk",
                    "Unknown"
                ),

                "Rule Risk": rule.get(
                    "overall_risk",
                    "Unknown"
                ),

                "Final Risk": decision.get(
                    "final_risk",
                    "Unknown"
                ),

                "Status": decision.get(
                    "application_status",
                    "Unknown"
                ),

                "Review": decision.get(
                    "review_requirement",
                    "Unknown"
                ),

                "Created At": assessment.get(
                    "created_at",
                    ""
                ),
            }
        )

    history_df = pd.DataFrame(history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")


    # ---------------------------------------------------------
    # Individual reports
    # ---------------------------------------------------------

    st.subheader("Assessment Reports")

    for assessment in assessments:

        entity = assessment.get(
            "entity_info",
            {}
        )

        ml = assessment.get(
            "ml_assessment",
            {}
        )

        rule = assessment.get(
            "rule_assessment",
            {}
        )

        decision = assessment.get(
            "final_decision",
            {}
        )

        assessment_id = assessment.get(
            "assessment_id"
        )

        entity_name = entity.get(
            "name",
            "Unknown"
        )

        final_risk = decision.get(
            "final_risk",
            "Unknown"
        )

        created_at = assessment.get(
            "created_at",
            ""
        )

        with st.expander(
            f"Assessment #{assessment_id} — "
            f"{entity_name} — {final_risk}"
        ):

            # -------------------------------------------------
            # Basic information
            # -------------------------------------------------

            st.markdown("### Assessment Information")

            info_col1, info_col2, info_col3 = st.columns(3)

            info_col1.metric(
                "Assessment ID",
                assessment_id,
            )

            info_col2.metric(
                "Entity",
                entity_name,
            )

            info_col3.metric(
                "Created At",
                created_at,
            )


            st.markdown("---")


            # -------------------------------------------------
            # Final decision
            # -------------------------------------------------

            st.markdown("### Final Decision")

            decision_col1, decision_col2, decision_col3 = st.columns(3)

            decision_col1.metric(
                "Risk Level",
                decision.get(
                    "final_risk",
                    "Unknown"
                ),
            )

            decision_col2.metric(
                "Application Status",
                decision.get(
                    "application_status",
                    "Unknown"
                ),
            )

            decision_col3.metric(
                "Review Requirement",
                decision.get(
                    "review_requirement",
                    "Unknown"
                ),
            )


            # -------------------------------------------------
            # ML assessment
            # -------------------------------------------------

            st.markdown("### ML Assessment")

            st.write(
                "Predicted Risk:",
                ml.get(
                    "predicted_risk",
                    "Unknown"
                ),
            )

            probabilities = ml.get(
                "probabilities",
                {}
            )

            if probabilities:

                probability_df = pd.DataFrame(
                    [
                        {
                            "Risk": risk,
                            "Probability": probability,
                        }
                        for risk, probability
                        in probabilities.items()
                    ]
                )

                st.bar_chart(
                    probability_df.set_index("Risk")
                )

                st.caption(
                    "These are model class probabilities, "
                    "not literal default probabilities."
                )

            else:

                st.info(
                    "Probability data was not stored for "
                    "this older assessment."
                )

            if ml.get("model_version"):

                st.caption(
                    f"Model Version: {ml['model_version']}"
                )


            # -------------------------------------------------
            # Rule assessment
            # -------------------------------------------------

            st.markdown("### Rule Engine Assessment")

            st.write(
                "Overall Rule Risk:",
                rule.get(
                    "overall_risk",
                    "Unknown"
                ),
            )

            critical_flags = rule.get(
                "critical_flags",
                []
            )

            if critical_flags:

                st.error("Critical Flags")

                for flag in critical_flags:

                    st.write(
                        f"- {flag}"
                    )

            else:

                st.success(
                    "No critical flags."
                )


            # -------------------------------------------------
            # Explanation
            # -------------------------------------------------

            st.markdown("### GenAI Explanation")

            explanation = assessment.get(
                "explanation"
            )

            if explanation:

                st.info(explanation)

            else:

                st.info(
                    "No explanation was stored for "
                    "this assessment."
                )


            # -------------------------------------------------
            # Downloadable report
            # -------------------------------------------------

            report_text = f"""
CREDIT RISK ASSESSMENT REPORT
=============================

Assessment ID: {assessment_id}
Client: {entity_name}
Entity ID: {entity.get("id", "Unknown")}
Created At: {created_at}

FINAL STATUS
------------
Risk Level: {decision.get("final_risk", "Unknown")}
Status: {decision.get("application_status", "Unknown")}
Action: {decision.get("review_requirement", "Unknown")}

ML INSIGHTS
-----------
Predicted Risk: {ml.get("predicted_risk", "Unknown")}
Probabilities: {json.dumps(probabilities)}
Model Version: {ml.get("model_version", "Unknown")}

RULE ENGINE
-----------
Risk: {rule.get("overall_risk", "Unknown")}
Critical Flags: {
    ", ".join(critical_flags)
    if critical_flags
    else "None"
}

EXPLANATION
-----------
{explanation or "No explanation available."}
"""

            st.download_button(
                "Download Report (TXT)",
                data=report_text,
                file_name=f"Report_{assessment_id}.txt",
                mime="text/plain",
                key=f"download_report_{assessment_id}",
            )

else:

    st.info(
        "No assessments have been recorded yet. "
        "Go to Risk Assessment to evaluate an existing "
        "client or new application."
    )


st.markdown("---")

st.caption(
    "Assessment history is persisted in PostgreSQL "
    "and retrieved through the FastAPI backend."
)