from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from analytics.config import ART_DIR, DATA_PATH


def read_data(data_path: Path | None = None) -> pd.DataFrame:
    path = data_path or DATA_PATH
    df = pd.read_csv(path)
    if "customer_id" not in df.columns:
        df.insert(0, "customer_id", np.arange(len(df), dtype=np.int64))
    return df


def validate_schema(df: pd.DataFrame) -> None:
    expected = [
        "customer_id",
        "recency",
        "history",
        "used_discount",
        "used_bogo",
        "zip_code",
        "is_referral",
        "channel",
        "offer",
        "conversion",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def basic_qc_tables(df: pd.DataFrame) -> Dict[str, object]:
    qc: Dict[str, object] = {}
    qc["n_rows"] = int(len(df))
    qc["missing_by_col"] = df.isna().sum().to_dict()
    qc["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    qc["unique_zip_code"] = sorted(df["zip_code"].astype(str).unique().tolist())
    qc["unique_channel"] = sorted(df["channel"].astype(str).unique().tolist())
    qc["unique_offer"] = sorted(df["offer"].astype(str).unique().tolist())
    qc["conversion_rate"] = float(df["conversion"].mean())
    return qc


def write_qc_summary(qc: Dict[str, object]) -> None:
    (ART_DIR / "qc_summary.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")


def export_processed(df: pd.DataFrame) -> None:
    df.to_csv(ART_DIR / "processed.csv", index=False)
