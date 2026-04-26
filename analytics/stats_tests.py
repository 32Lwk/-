from __future__ import annotations

import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from analytics.config import ART_DIR, FIG_IMP
from analytics.figures_jp import save_fig


def chi2_tests(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    y = df["conversion"].astype(int)
    for col in cols:
        tab = pd.crosstab(df[col], y)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p, dof, _expected = chi2_contingency(tab.values)
        n = tab.values.sum()
        r, k = tab.shape
        denom = n * (min(r - 1, k - 1))
        cramer_v = float(math.sqrt(chi2 / denom)) if denom > 0 else float("nan")
        rows.append(
            {
                "feature": col,
                "chi2": float(chi2),
                "dof": int(dof),
                "p_value": float(p),
                "cramers_v": cramer_v,
                "n": int(n),
            }
        )
    out = pd.DataFrame(rows).sort_values("p_value")
    if out.empty:
        return out
    rej, qvals, _, _ = multipletests(out["p_value"].values, method="fdr_bh")
    out["reject_bh_fdr"] = rej
    out["q_value_bh"] = qvals
    return out


def plot_cramers_v_bar(chi2_df: pd.DataFrame, path=None) -> None:
    if chi2_df.empty:
        return
    path = path or (FIG_IMP / "chi2_cramers_v_bar.png")
    rename_feat = {
        "offer": "オファー",
        "channel": "チャネル",
        "zip_code": "居住エリア",
        "used_discount": "過去の割引利用",
        "used_bogo": "過去のBOGO利用",
        "is_referral": "紹介流入",
    }
    d = chi2_df.copy()
    d["label"] = d["feature"].map(lambda x: rename_feat.get(x, x))
    plt.figure(figsize=(8, 4))
    sns.barplot(data=d, x="label", y="cramers_v")
    plt.ylabel("Cramér's V（効果量）")
    plt.xlabel("特徴量")
    plt.title("カイ二乗：Cramér's V（conversion との関連の強さ）")
    plt.xticks(rotation=30, ha="right")
    save_fig(path)
