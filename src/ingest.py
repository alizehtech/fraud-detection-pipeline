"""Load the raw Kaggle 'Credit Card Fraud Detection' CSV, clean it, and load
it into the `transactions` table in Postgres.

Usage:
    python src/ingest.py [path/to/creditcard.csv]
"""

import sys
from pathlib import Path

import pandas as pd

from db import get_engine

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"


def load_raw(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No CSV at {csv_path}. Download 'creditcard.csv' from Kaggle "
            "(Credit Card Fraud Detection dataset) and place it in data/."
        )
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # This dataset is known to contain exact-duplicate rows (same PCA
    # features, same class). Duplicates would let identical transactions
    # end up in both the train and test split later, inflating our
    # reported accuracy. Drop them here so the DB holds one row per
    # transaction.
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} exact-duplicate rows")

    null_counts = df.isnull().sum()
    if null_counts.any():
        print("Null counts per column:\n", null_counts[null_counts > 0])
        df = df.dropna()

    # Lowercase, SQL-friendly column names; Class -> is_fraud so the
    # column is self-documenting once it's sitting in a database table.
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"class": "is_fraud"})
    df["is_fraud"] = df["is_fraud"].astype(int)

    return df.reset_index(drop=True)


def write_to_postgres(df: pd.DataFrame, table_name: str = "transactions") -> None:
    engine = get_engine()
    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
        chunksize=5000,
    )
    print(f"Wrote {len(df)} rows to '{table_name}'")


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV_PATH

    raw = load_raw(csv_path)
    print(f"Loaded {len(raw)} raw rows from {csv_path}")

    cleaned = clean(raw)
    fraud_rate = cleaned["is_fraud"].mean()
    print(
        f"{len(cleaned)} rows after cleaning | "
        f"{cleaned['is_fraud'].sum()} fraud ({fraud_rate:.4%})"
    )

    write_to_postgres(cleaned)


if __name__ == "__main__":
    main()
