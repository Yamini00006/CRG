import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Credit Risk Gauge",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000"
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "official_dataset.csv"


@st.cache_data
def load_official_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(DATA_PATH)


@st.cache_data(ttl=10)
def load_assessments() -> list:
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/assessments",
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("assessments", [])

    except requests.RequestException as exc:
        st.warning(
            f"Unable to load assessment history: {exc}"
        )
        return []


df = load_official_data()
assessments = load_assessments()


st.title("Credit Risk Gauge")
st.markdown("### Enterprise Credit Risk Assessment Platform")


with st.sidebar:
    st.markdown("## Credit Risk Gauge")
    st.caption("Enterprise Risk Assessment")

    st.markdown("---")

    st.info(
        "Use the navigation menu above to move between "
        "Dashboard, Clients, Risk Assessment and Reports."
    )

    st.markdown("---")


# -------------------------------------------------------------
# Dashboard metrics
# -------------------------------------------------------------

total_clients = len(df)

pending_reviews = sum(
    1
    for assessment in assessments
    if assessment.get("final_decision", {}).get(
        "review_requirement"
    )
    in {
        "Standard Review",
        "Mandatory Review",
    }
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Active Entities",
    total_clients,
)

col2.metric(
    "Total Assessments",
    len(assessments),
)

col3.metric(
    "Pending Reviews",
    pending_reviews,
)


st.markdown("---")


# -------------------------------------------------------------
# Recent assessments
# -------------------------------------------------------------

if not assessments:

    st.info(
        "No assessments have been recorded yet. "
        "Go to Risk Assessment to evaluate an existing "
        "client or new application."
    )

else:

    st.subheader("Recent Assessments")

    history = []

    for assessment in assessments:

        decision = assessment.get(
            "final_decision",
            {}
        )

        entity = assessment.get(
            "entity_info",
            {}
        )

        history.append(
            {
                "Assessment ID": assessment.get(
                    "assessment_id"
                ),

                "Entity": entity.get(
                    "name",
                    "Unknown"
                ),

                "Entity ID": entity.get(
                    "id",
                    "Unknown"
                ),

                "ML Risk": assessment.get(
                    "ml_assessment",
                    {}
                ).get(
                    "predicted_risk",
                    "Unknown"
                ),

                "Rule Risk": assessment.get(
                    "rule_assessment",
                    {}
                ).get(
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

st.caption(
    "Assessment history is persisted in PostgreSQL "
    "and retrieved through the FastAPI backend."
)