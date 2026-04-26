"""レポート用の追加サマリー図（final_report.md / LaTeX 向け）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics.config import FIG_DIR, FIG_IMP
from analytics.figures_jp import save_fig, init_plot_style


def plot_dr_full_vs_holdout(
    dr_full: pd.DataFrame,
    dr_holdout: pd.DataFrame,
    path: Path | None = None,
    max_treatments: int = 9,
) -> Path:
    """全件 DR とホールドアウト DR の棒比較。"""
    init_plot_style()
    path = path or (FIG_IMP / "report_dr_full_vs_holdout.png")
    m = dr_full[["treatment", "mu_dr"]].merge(
        dr_holdout[["treatment", "mu_dr"]],
        on="treatment",
        suffixes=("_full", "_ho"),
    ).sort_values("mu_dr_full", ascending=False).head(max_treatments)

    x = np.arange(len(m))
    w = 0.35
    plt.figure(figsize=(max(8, len(m) * 0.9), 5))
    plt.bar(x - w / 2, m["mu_dr_full"], width=w, label="全件学習（参考）", color="#4c72b0")
    plt.bar(x + w / 2, m["mu_dr_ho"], width=w, label="ホールドアウト", color="#dd8452")
    plt.xticks(x, [t.replace(" | ", "\n") for t in m["treatment"]], rotation=0, fontsize=8)
    plt.ylabel("DR 推定 μ（購入確率）")
    plt.title("処置別 DR：全件 vs ホールドアウト（上位処置）")
    plt.legend()
    plt.tight_layout()
    save_fig(path)
    return path


def plot_policy_eval_means(compare: pd.DataFrame, path: Path | None = None) -> Path:
    """フル / OOF / ホールドアウトの平均期待利益（シナリオ別）。"""
    init_plot_style()
    path = path or (FIG_IMP / "report_policy_eval_means.png")
    scenarios = compare["scenario"].tolist()
    x = np.arange(len(scenarios))
    w = 0.25
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - w, compare["mean_profit_full_fit"], width=w, label="フル学習", color="#4c72b0")
    plt.bar(x, compare["mean_profit_oof"], width=w, label="K折OOF", color="#55a868")
    plt.bar(x + w, compare["mean_profit_holdout"], width=w, label="ホールドアウト", color="#c44e52")
    plt.xticks(x, scenarios, rotation=15, ha="right")
    plt.ylabel("平均期待利益（proxy）")
    plt.title("コストシナリオ別：評価モード比較")
    plt.legend()
    plt.tight_layout()
    save_fig(path)
    return path


def plot_segment_mean_profit(segment_summary: pd.DataFrame, path: Path | None = None) -> Path:
    """セグメント別平均期待利益（件数降順で上位）。"""
    init_plot_style()
    path = path or (FIG_IMP / "report_segment_mean_profit.png")
    g = segment_summary.sort_values("n", ascending=False).head(12).copy()
    g = g.sort_values("mean_expected_profit", ascending=True)
    plt.figure(figsize=(7, max(4, len(g) * 0.35)))
    plt.barh(
        [f"seg {int(r.segment)}" for _, r in g.iterrows()],
        g["mean_expected_profit"].values,
        color="#8172b3",
    )
    plt.xlabel("セグメント平均 期待利益（mid_cost）")
    plt.title("セグメント別 期待利益（上位セグメント）")
    plt.tight_layout()
    save_fig(path)
    return path


def plot_propensity_ess_bars(diag: pd.DataFrame, path: Path | None = None) -> Path:
    """傾向スコア診断：ESS（IPW）を処置別に可視化。"""
    init_plot_style()
    path = path or (FIG_IMP / "report_propensity_ess.png")
    d = diag.sort_values("ess_ipw", ascending=True).tail(12)
    plt.figure(figsize=(7, max(4, len(d) * 0.35)))
    plt.barh(
        [t.replace(" | ", " / ") for t in d["treatment"]],
        d["ess_ipw"].values,
        color="#937860",
    )
    plt.xlabel("ESS（IPW 重み）")
    plt.title("処置別 有効標本量（ESS）※大きいほど重みが分散しにくい")
    plt.tight_layout()
    save_fig(path)
    return path


def plot_profit_summary_scenarios(profit_summary: pd.DataFrame, path: Path | None = None) -> Path:
    """シナリオ別 平均・中央値期待利益。"""
    init_plot_style()
    path = path or (FIG_IMP / "report_profit_summary_by_scenario.png")
    ps = profit_summary.set_index("scenario")
    x = np.arange(len(ps))
    w = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - w / 2, ps["mean_expected_profit"], width=w, label="平均", color="#4c72b0")
    plt.bar(x + w / 2, ps["median_expected_profit"], width=w, label="中央値", color="#8da0cb")
    plt.xticks(x, ps.index, rotation=15, ha="right")
    plt.ylabel("期待利益（proxy）")
    plt.title("コストシナリオ別 期待利益の要約")
    plt.legend()
    plt.tight_layout()
    save_fig(path)
    return path


def write_report_summary_figures(
    dr_table: pd.DataFrame,
    dr_holdout: pd.DataFrame,
    segment_summary: pd.DataFrame,
    policy_eval_compare: pd.DataFrame,
    propensity_diag: pd.DataFrame,
    profit_summary: pd.DataFrame,
) -> dict[str, str]:
    """パイプライン終盤で呼び出し、追加PNGの相対パス（figures/improvements/...）を返す。"""
    rel: dict[str, str] = {}
    if not dr_holdout.empty:
        p = plot_dr_full_vs_holdout(dr_table, dr_holdout)
        rel["dr_compare"] = str(p.relative_to(FIG_DIR.parent)).replace("\\", "/")
    if not policy_eval_compare.empty:
        p = plot_policy_eval_means(policy_eval_compare)
        rel["policy_eval"] = str(p.relative_to(FIG_DIR.parent)).replace("\\", "/")
    if not segment_summary.empty:
        p = plot_segment_mean_profit(segment_summary)
        rel["segment_profit"] = str(p.relative_to(FIG_DIR.parent)).replace("\\", "/")
    if not propensity_diag.empty:
        p = plot_propensity_ess_bars(propensity_diag)
        rel["propensity_ess"] = str(p.relative_to(FIG_DIR.parent)).replace("\\", "/")
    if not profit_summary.empty:
        p = plot_profit_summary_scenarios(profit_summary)
        rel["profit_summary_bar"] = str(p.relative_to(FIG_DIR.parent)).replace("\\", "/")
    return rel
