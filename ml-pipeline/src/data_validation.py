from pathlib import Path

import pandas as pd

from data_loader import load_official_dataset


TARGET_COLUMN = "risk_bucket"
ID_COLUMN = "entity_id"


def validate_dataset(df: pd.DataFrame) -> None:
    """Run basic quality checks on the official dataset."""

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    # 1. Shape
    print(f"\nDataset shape: {df.shape}")

    # 2. Duplicate rows
    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows}")

    # 3. Duplicate entity IDs
    duplicate_ids = df[ID_COLUMN].duplicated().sum()
    print(f"Duplicate entity IDs: {duplicate_ids}")

    # 4. Missing values
    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        for column, count in missing.items():
            percentage = (count / len(df)) * 100
            print(f"- {column}: {count} ({percentage:.1f}%)")

    # 5. Data types
    print("\nData types:")
    print(df.dtypes.to_string())

    # 6. Target distribution
    print(f"\nTarget distribution: {TARGET_COLUMN}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    target_counts = df[TARGET_COLUMN].value_counts(dropna=False)

    for value, count in target_counts.items():
        percentage = (count / len(df)) * 100
        print(f"- {value}: {count} ({percentage:.1f}%)")

    # 7. Numeric sanity checks
    print("\nNumeric columns containing negative values:")

    numeric_columns = df.select_dtypes(include="number").columns
    negative_found = False

    for column in numeric_columns:
        count = (df[column] < 0).sum()

        if count > 0:
            negative_found = True
            print(f"- {column}: {count}")

    if not negative_found:
        print("None.")

    # 8. Unique values for categorical columns
    categorical_columns = df.select_dtypes(include="object").columns

    print("\nCategorical columns and unique-value counts:")

    for column in categorical_columns:
        print(f"- {column}: {df[column].nunique(dropna=False)} unique values")


if __name__ == "__main__":
    dataset = load_official_dataset()
    validate_dataset(dataset)