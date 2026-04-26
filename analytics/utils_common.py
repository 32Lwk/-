from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from analytics.constants import CHANNEL_JP, OFFER_JP, ZIP_JP


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["購入"] = out["conversion"].map({0: "未購入", 1: "購入"}).astype(str)
    out["オファー"] = out["offer"].map(OFFER_JP).fillna(out["offer"]).astype(str)
    out["チャネル"] = out["channel"].map(CHANNEL_JP).fillna(out["channel"]).astype(str)
    out["居住エリア"] = out["zip_code"].map(ZIP_JP).fillna(out["zip_code"]).astype(str)
    out["紹介流入"] = out["is_referral"].map({0: "なし", 1: "あり"}).astype(str)
    out["過去_割引利用"] = out["used_discount"].map({0: "なし", 1: "あり"}).astype(str)
    out["過去_BOGO利用"] = out["used_bogo"].map({0: "なし", 1: "あり"}).astype(str)
    return out


def topk_metrics(y_true: np.ndarray, scores: np.ndarray, ks: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(y_true)
    order = np.argsort(scores)[::-1]
    for k in ks:
        m = max(1, int(round(n * k)))
        idx = order[:m]
        out[f"top_{int(k * 100)}pct_cvr"] = float(y_true[idx].mean())
    return out


def topk_precision_recall(y_true: np.ndarray, scores: np.ndarray, ks: List[float]) -> pd.DataFrame:
    rows = []
    n = len(y_true)
    order = np.argsort(scores)[::-1]
    pos = y_true.sum()
    for k in ks:
        m = max(1, int(round(n * k)))
        idx = order[:m]
        tp = float(y_true[idx].sum())
        prec = tp / m
        rec = tp / pos if pos > 0 else float("nan")
        rows.append({"k_pct": int(k * 100), "n": m, "precision": prec, "recall": rec})
    return pd.DataFrame(rows)
