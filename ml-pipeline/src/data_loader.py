from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_DATASET = (
    PROJECT_ROOT
    / "ml-pipeline"
    / "data"
    / "raw"
    / "official"
    / "credit_risk_dataset_50_entities.csv"
)


def load_official_dataset() -> pd.DataFrame:
    """Load the official credit-risk dataset."""
    if not OFFICIAL_DATASET.exists():
        raise FileNotFoundError(
            f"Official dataset not found: {OFFICIAL_DATASET}"
        )

    return pd.read_csv(OFFICIAL_DATASET)


if __name__ == "__main__":
    df = load_official_dataset()

    print(f"Dataset shape: {df.shape}")
    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")