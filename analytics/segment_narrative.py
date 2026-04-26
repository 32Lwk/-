from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def segment_feature_zprofiles(
    df: pd.DataFrame,
    segment_labels: np.ndarray,
    feature_cols: List[str],
) -> Dict[int, str]:
    """各セグメントについて、全体平均からの z スコアが大きい特徴を1行で要約。"""
    tmp = df[feature_cols].copy()
    tmp["segment"] = segment_labels
    overall_mean = tmp[feature_cols].mean()
    overall_std = tmp[feature_cols].std().replace(0, np.nan)

    narratives: Dict[int, str] = {}
    for seg in sorted(np.unique(segment_labels)):
        sub = tmp.loc[tmp["segment"] == seg, feature_cols]
        m = sub.mean()
        z = (m - overall_mean) / overall_std
        z = z.dropna().abs().sort_values(ascending=False)
        top = z.head(3)
        parts = [f"{idx}: z={float(top[idx]):+.2f}" for idx in top.index]
        narratives[int(seg)] = "主な特徴（全体比z）: " + ", ".join(parts)
    return narratives


def build_segment_narratives(df: pd.DataFrame, segment_labels: np.ndarray) -> pd.DataFrame:
    feats = ["recency", "history", "used_discount", "used_bogo", "is_referral"]
    ztext = segment_feature_zprofiles(df, segment_labels, feats)
    rows = [{"segment": k, "narrative_z": v} for k, v in ztext.items()]
    return pd.DataFrame(rows)
