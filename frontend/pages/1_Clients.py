from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clients", page_icon="🏢", layout="wide")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "official_dataset.csv"


@st.cache_data
def load_official_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


df = load_official_data()

st.title("Client Directory")
st.caption(f"Official client population: {len(df)} entities")

search = st.text_input("Search by entity name or ID", placeholder="e.g. Nova Technologies or ENT048")

filtered = df.copy()
if search.strip():
    mask = (
        filtered["entity_name"].astype(str).str.contains(search.strip(), case=False, na=False)
        | filtered["entity_id"].astype(str).str.contains(search.strip(), case=False, na=False)
    )
    filtered = filtered.loc[mask]

display_cols = [
    "entity_id",
    "entity_name",
    "sector",
    "country",
    "revenue_usd_m",
    "years_in_operation",
    "risk_bucket",
    "implied_rating",
]
display = filtered[display_cols].rename(
    columns={
        "entity_id": "Entity ID",
        "entity_name": "Entity Name",
        "sector": "Sector",
        "country": "Country",
        "revenue_usd_m": "Revenue ($M)",
        "years_in_operation": "Years in Operation",
        "risk_bucket": "Reference Risk",
        "implied_rating": "Reference Rating",
    }
)

st.dataframe(display, use_container_width=True, hide_index=True)

if len(filtered) == 1:
    row = filtered.iloc[0]
    st.markdown("---")
    st.subheader("Selected Client")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entity ID", row["entity_id"])
    c2.metric("Entity Name", row["entity_name"])
    c3.metric("Sector", row["sector"])
    c4.metric("Country", row["country"])
    st.info("Open Risk Assessment to run the actual ML + rule assessment for this entity.")
elif len(filtered) == 0:
    st.warning("No matching entities found.")
