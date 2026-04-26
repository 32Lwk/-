from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from analytics.config import ART_DIR, ROOT
from analytics.constants import CHANNEL_JP, OFFER_JP, ZIP_JP


def write_final_report(
    qc: Dict[str, Any],
    model_eval: Dict[str, Dict[str, float]],
    best_model: str,
    dr_table: pd.DataFrame,
    dr_holdout: pd.DataFrame | None,
    profit_summary: pd.DataFrame,
    cvr_offer: pd.Series,
    n_offer: pd.Series,
    cvr_channel: pd.Series,
    n_channel: pd.Series,
    cvr_offer_channel: pd.DataFrame,
    n_offer_channel: pd.DataFrame,
    segment_summary: pd.DataFrame,
    chi2_table: pd.DataFrame,
    policy_mid_summary: Dict[str, object],
    segment_narrative: pd.DataFrame | None = None,
) -> None:
    cvr_offer_jp = cvr_offer.rename(index=lambda x: OFFER_JP.get(x, x))
    n_offer_jp = n_offer.rename(index=lambda x: OFFER_JP.get(x, x))
    cvr_channel_jp = cvr_channel.rename(index=lambda x: CHANNEL_JP.get(x, x))
    n_channel_jp = n_channel.rename(index=lambda x: CHANNEL_JP.get(x, x))
    cvr_offer_channel_jp = cvr_offer_channel.rename(index=lambda x: OFFER_JP.get(x, x)).rename(
        columns=lambda x: CHANNEL_JP.get(x, x)
    )
    n_offer_channel_jp = n_offer_channel.rename(index=lambda x: OFFER_JP.get(x, x)).rename(
        columns=lambda x: CHANNEL_JP.get(x, x)
    )

    md: list[str] = []
    md.append("# 期待利益最大化のための分析レポート（分析担当向け）\n")
    md.append("## 0. 要約（結論）\n")
    top_treatment = str(policy_mid_summary.get("top_treatment", ""))
    top_share = float(policy_mid_summary.get("top_share", 0.0))
    md.append(f"- **顧客別最適ポリシー（mid_cost）で最頻の推奨施策**: `{top_treatment.replace(' | ', ' / ')}`（{top_share:.1%}）\n")
    md.append("- **示唆**: CVRが高い施策（例: 割引）は存在するが、コスト仮定が大きいと期待利益では負ける可能性がある。\n")
    md.append("- **推奨アクション**: 低コストなインセンティブへ置換し、上位スコア層（TopK%）でA/BテストしてROIが合う範囲を探索。\n")

    md.append("\n## 1. 目的と前提\n")
    md.append("- **目的**: `conversion`（購入）を増やしつつ、オファー/チャネルを含む施策設計で**期待利益**を最大化する。\n")
    md.append("- **価値のproxy**: `history` を購入価値のproxyとし、期待価値を \\(P(購入|x)\\times history\\) と置く。\n")
    md.append("- **コスト仮定（感度分析）**: `analytics/config.py` の `DEFAULT_SCENARIOS` を参照。\n")
    md.append("- **注意**: 施策効果は観察データであるため、推定は因果を保証しない（後述）。\n")

    md.append("\n## 2. データ概要と品質\n")
    md.append(f"- **行数**: {qc['n_rows']}\n")
    md.append(f"- **全体CVR**: {qc['conversion_rate']:.4f}\n")
    md.append(f"- **居住エリア（zip_code）**: {[ZIP_JP.get(x, x) for x in qc['unique_zip_code']]}\n")
    md.append(f"- **チャネル**: {[CHANNEL_JP.get(x, x) for x in qc['unique_channel']]}\n")
    md.append(f"- **オファー**: {[OFFER_JP.get(x, x) for x in qc['unique_offer']]}\n")

    md.append("\n### 図（分布と購入差）\n")
    md.append("![前回購入からの月数の分布](figures/dist_recency.png)\n\n")
    md.append("![過去購入価値（history）の分布](figures/dist_history.png)\n\n")
    md.append("![前回購入からの月数（購入別）](figures/box_recency_by_conversion.png)\n\n")
    md.append("![過去購入価値（history）（購入別）](figures/box_history_by_conversion.png)\n\n")

    md.append("## 3. 基礎集計（CVR）と解釈\n")
    md.append("### 3.1 オファー別/チャネル別\n")
    md.append("\n**オファー別CVR**\n\n| オファー | n | CVR |\n| --- | ---: | ---: |\n")
    for k, v in cvr_offer_jp.items():
        md.append(f"| {k} | {int(n_offer_jp.get(k, 0))} | {v:.4f} |\n")
    md.append("\n**チャネル別CVR**\n\n| チャネル | n | CVR |\n| --- | ---: | ---: |\n")
    for k, v in cvr_channel_jp.items():
        md.append(f"| {k} | {int(n_channel_jp.get(k, 0))} | {v:.4f} |\n")

    md.append("\n### 3.2 オファー×チャネル\n")
    md.append("![CVRヒートマップ（オファー×チャネル）](figures/heat_cvr_offer_x_channel.png)\n\n")
    md.append("| オファー \\ チャネル | マルチチャネル（n/CVR） | 電話（n/CVR） | Web（n/CVR） |\n| --- | --- | --- | --- |\n")
    for offer in cvr_offer_channel_jp.index:
        row = cvr_offer_channel_jp.loc[offer]
        nrow = n_offer_channel_jp.loc[offer]

        def fmt(cell_n, cell_cvr):
            if pd.isna(cell_cvr):
                return "-"
            return f"{int(cell_n)}/{float(cell_cvr):.4f}"

        md.append(
            f"| {offer} | {fmt(nrow.get('マルチチャネル', 0), row.get('マルチチャネル', float('nan')))} | "
            f"{fmt(nrow.get('電話', 0), row.get('電話', float('nan')))} | "
            f"{fmt(nrow.get('Web', 0), row.get('Web', float('nan')))} |\n"
        )

    md.append("\n### 3.3 比率差検定（カイ二乗）と BH-FDR\n")
    md.append("- 詳細: `artifacts/chi2_tests_with_fdr.csv`\n\n")
    md.append("| 特徴量 | n | p値 | q値(BH) | Cramér's V |\n| --- | ---: | ---: | ---: | ---: |\n")
    rename_feat = {
        "offer": "オファー",
        "channel": "チャネル",
        "zip_code": "居住エリア",
        "used_discount": "過去の割引利用",
        "used_bogo": "過去のBOGO利用",
        "is_referral": "紹介流入",
    }
    for _, r in chi2_table.head(10).iterrows():
        qv = r.get("q_value_bh", float("nan"))
        md.append(
            f"| {rename_feat.get(r['feature'], r['feature'])} | {int(r['n'])} | {r['p_value']:.3e} | {float(qv):.3e} | {r['cramers_v']:.4f} |\n"
        )
    md.append("\n![効果量（Cramér's V）](figures/improvements/chi2_cramers_v_bar.png)\n")

    md.append("\n## 4. 予測モデル（ターゲティング）\n")
    md.append(f"- **採用モデル（AUC最大）**: {best_model}\n")
    md.append(f"- **学習/評価**: n_train={qc.get('n_train','?')}, n_test={qc.get('n_test','?')}\n")
    md.append("\n| model | AUC | Brier | AP | Top5% CVR | Top10% CVR |\n| --- | ---: | ---: | ---: | ---: | ---: |\n")
    for name, row in model_eval.items():
        md.append(
            f"| {name} | {row['auc']:.4f} | {row['brier']:.4f} | {row.get('average_precision', float('nan')):.4f} | "
            f"{row['top_5pct_cvr']:.4f} | {row['top_10pct_cvr']:.4f} |\n"
        )
    md.append("\n### PR曲線（改善）\n")
    md.append("![PR logreg](figures/improvements/pr_curve_logreg.png)\n\n")
    md.append("![PR hgb](figures/improvements/pr_curve_hgb.png)\n")

    md.append("\n## 5. 潜在空間とセグメント\n")
    md.append("![PCA 2D](figures/latent2d_pca2_color_conversion.png)\n\n")
    md.append("![UMAP 2D](figures/latent2d_umap2_color_conversion.png)\n\n")
    md.append("![UMAP セグメント3D PNG](figures/latent3d_umap3_color_segment.png)\n\n")
    md.append("- `artifacts/cluster_labels_auto.csv`（シルエット自動k） / `artifacts/cluster_labels_k4.csv`（参考k=4）\n")
    md.append("- シルエット: ![シルエット vs k](figures/improvements/silhouette_vs_k.png)\n")
    md.append("- 安定性: ![ARI](figures/improvements/segment_stability_ari.png)\n")

    md.append("\n### セグメント要約（mid_cost・主クラスタ）\n")
    md.append("| セグメント | 件数 | CVR | mean(history) | mean(recency) | mean(期待利益) | 推奨オファー | 推奨チャネル |\n")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
    for _, r in segment_summary.iterrows():
        md.append(
            f"| {int(r['segment'])} | {int(r['n'])} | {r['cvr']:.4f} | {r['mean_history']:.2f} | {r['mean_recency']:.2f} | "
            f"{r['mean_expected_profit']:.2f} | {r['segment_offer']} | {r['segment_channel']} |\n"
        )

    if segment_narrative is not None and not segment_narrative.empty:
        md.append("\n### セグメント別ナラティブ（zプロファイル）\n")
        for _, r in segment_narrative.sort_values("segment").iterrows():
            md.append(f"- **セグメント {int(r['segment'])}**: {r['narrative_z']}\n")

    md.append("\n## 6. DR（全件学習）とホールドアウトDR\n")
    md.append("### 6.1 全件（参考・楽観バイアスに注意）\n")
    md.append("| 施策 | mu_dr | mu_model | support |\n| --- | ---: | ---: | ---: |\n")
    for _, r in dr_table.head(12).iterrows():
        md.append(f"| `{r['treatment']}` | {r['mu_dr']:.4f} | {r['mu_model']:.4f} | {r['support']:.4f} |\n")
    if dr_holdout is not None and not dr_holdout.empty:
        md.append("\n### 6.2 ホールドアウト上のDR点推定\n")
        for _, r in dr_holdout.head(12).iterrows():
            md.append(f"- `{r['treatment']}`: {r['mu_dr']:.4f}\n")
    md.append("\n![DRブートストラップ区間](figures/improvements/dr_bootstrap_ci.png)\n")
    md.append("- 傾向スコア診断: `artifacts/propensity_diagnostics.csv`\n")

    md.append("\n## 7. 期待利益とOOF\n")
    md.append("- `artifacts/promo_targets.csv`（フル学習） / `artifacts/promo_targets_oof.csv`\n")
    md.append("- 比較: `artifacts/policy_eval_compare.csv`\n")
    md.append("- ベンチマーク: `artifacts/policy_benchmarks_mid_cost.csv`\n\n")
    md.append("![推奨分布](figures/improvements/policy_reco_dist_full.png)\n\n")
    md.append("![フルvsOOF](figures/improvements/profit_full_vs_oof_scatter.png)\n\n")
    md.append("![ベンチマーク](figures/improvements/policy_benchmarks_bar.png)\n")

    md.append("\n### 7.1 コストシナリオ別サマリ\n")
    md.append("| シナリオ | n | 利益>0割合 | 平均期待利益 | 中央値 |\n| --- | ---: | ---: | ---: | ---: |\n")
    for _, r in profit_summary.iterrows():
        md.append(
            f"| {r['scenario']} | {int(r['n_customers'])} | {r['share_profit_positive']:.4f} | "
            f"{r['mean_expected_profit']:.2f} | {r['median_expected_profit']:.2f} |\n"
        )

    md.append("\n## 8. 制約付き配信（貪欲近似）\n")
    md.append("- `artifacts/promo_targets_constrained_mid_cost.csv`\n")
    md.append("- 図: ![制約前後](figures/improvements/constrained_selection_summary.png)\n")

    md.append("\n## 9. 経営向けゲート・リスク・KPI感度\n")
    md.append("- `artifacts/kpi_bridge_sensitivity.csv` / `artifacts/risk_register.json` / `artifacts/decision_gate_README.md`\n")
    md.append("![意思決定フロー例](figures/improvements/decision_flow.png)\n")

    md.append("\n## 10. 実験設計（A/B）\n")
    md.append("- `artifacts/ab_design.csv` / LaTeX: `artifacts/tables/ab_design.tex`\n")

    md.append("\n## 11. ガバナンス・モニタリング\n")
    md.append("- `artifacts/data_dictionary.md`\n")
    md.append("- `artifacts/monitoring_spec.json` / `artifacts/tables/monitoring_summary.tex`\n")

    md.append("\n## 12. 限界と次の検証\n")
    md.append("- 観察データ・未観測交絡。DRブートストラップは部分標本再学習であり、完全な因果保証はしない。\n")
    md.append("- 詳細は `latex/final_report.tex`（LuaLaTeX）を参照。\n")

    md.append("\n## 付録：生成物\n")
    md.append("- `analytics/` パッケージ, `run_analysis.py`, `artifacts/`, `figures/`, `figures/improvements/`\n")

    (ROOT / "final_report.md").write_text("".join(md), encoding="utf-8")


def write_run_config(extra: Dict[str, Any] | None = None) -> None:
    import json
    import sys

    from analytics import __version__

    cfg = {"analytics_version": __version__, "python": sys.version, **(extra or {})}
    (ART_DIR / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
