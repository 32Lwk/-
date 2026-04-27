"""セグメント×処置の探索的プロファイル（因果主張なし・記述統計）。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from analytics.config import ART_DIR, FIG_IMP
from analytics.figures_jp import init_plot_style, save_fig


def write_segment_treatment_exploratory(df: pd.DataFrame, segment_labels: pd.Series | list[int]) -> pd.DataFrame:
    """各セグメント・処置セルでの観測CVRとn。DRではなく記述。"""
    d = df.copy()
    d["segment"] = segment_labels
    d["treatment"] = d["offer"].astype(str) + " | " + d["channel"].astype(str)
    g = d.groupby(["segment", "treatment"], observed=True).agg(n=("conversion", "size"), cvr=("conversion", "mean")).reset_index()
    g.to_csv(ART_DIR / "segment_treatment_cvr_exploratory.csv", index=False)

    init_plot_style()
    top_t = d["treatment"].value_counts().head(6).index.tolist()
    seg_ids = sorted(d["segment"].unique())[:12]
    sub = g[g["treatment"].isin(top_t) & g["segment"].isin(seg_ids)]
    if sub.empty:
        return g
    pivot = sub.pivot(index="segment", columns="treatment", values="cvr").fillna(0.0)
    plt.figure(figsize=(max(8, len(top_t) * 1.2), max(4, len(seg_ids) * 0.4)))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=0.35)
    plt.colorbar(im, label="観測CVR（記述）")
    plt.yticks(range(len(pivot.index)), [f"seg {int(s)}" for s in pivot.index])
    plt.xticks(range(len(pivot.columns)), [c.replace(" | ", "\n") for c in pivot.columns], rotation=45, ha="right", fontsize=7)
    plt.xlabel("処置（オファー | チャネル）")
    plt.ylabel("UMAP セグメント")
    plt.title(
        "探索的ヒートマップ：セグメント×処置の観測CVR\n"
        "※交絡のため効果量の解釈は慎重に（RCTで検証）",
        fontsize=11,
    )
    save_fig(FIG_IMP / "segment_treatment_cvr_heatmap_exploratory.png")
    return g
