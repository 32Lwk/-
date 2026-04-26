from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analytics.config import ART_DIR, FIG_IMP


def write_kpi_bridge_sensitivity(n_customers: int, mean_history: float, cvr: float) -> None:
    """history を売上proxyとし、粗利率の仮定レンジで「会計ブリッジ」を感度のみ表出（絶対額断定なし）。"""
    margin_low, margin_high = 0.15, 0.45
    rows = []
    for m in [margin_low, (margin_low + margin_high) / 2, margin_high]:
        approx_profit_per_buyer = mean_history * m
        exp_buyers = n_customers * cvr
        rows.append(
            {
                "assumed_gross_margin": m,
                "proxy_revenue_per_conversion": mean_history,
                "illustrative_profit_per_converter": approx_profit_per_buyer,
                "note": "※粗利率は仮定。実務では原価・返品・LTVで再定義すること。",
            }
        )
    pd.DataFrame(rows).to_csv(ART_DIR / "kpi_bridge_sensitivity.csv", index=False)
    md = [
        "# KPIブリッジ（仮定レンジ・感度のみ）\n\n",
        "本データは `history` を売上proxyとしており、**粗利・LTVの実数は持たない**。\n",
        "経営判断では、粗利率をレンジで置き、感度表（CSV）を必ず参照すること。\n",
    ]
    (ART_DIR / "kpi_bridge_README.md").write_text("".join(md), encoding="utf-8")


def write_risk_register() -> None:
    risks = [
        {
            "id": "R1",
            "risk": "未観測交絡（割当ルールがデータに含まれない要因に依存）",
            "mitigation": "A/Bで増分効果を推定、ドメイン知識で割当要因を列挙",
        },
        {
            "id": "R2",
            "risk": "SUTVA違反（顧客間干渉・在庫制約）",
            "mitigation": "セグメント単位デザイン、供給制約をモデル化",
        },
        {
            "id": "R3",
            "risk": "クーポン慣習化（長期LTV低下）",
            "mitigation": "長期アウトカム追跡、頻度キャップ",
        },
        {
            "id": "R4",
            "risk": "データ漏洩・ラベル定義変更",
            "mitigation": "特徴凍結、モニタリング仕様に従う",
        },
        {
            "id": "R5",
            "risk": "説明可能性・不利益（属性プロキシ）",
            "mitigation": "人のレビュー、差別リスク評価",
        },
    ]
    (ART_DIR / "risk_register.json").write_text(json.dumps(risks, ensure_ascii=False, indent=2), encoding="utf-8")


def write_decision_gate_readme() -> None:
    text = """# 意思決定ゲート（たたき台）

1. オフライン：OOF で期待利益>0 の割合、校正（Brier）、傾向ESSが閾値内
2. パイロット：RCT/A/B で主アウトカムが事前定義のMDEを満たすか
3. スケール：ドリフト・ESS監視、インシデント時は縮小運用

図は `figures/improvements/decision_flow.png` を生成（Matplotlibの簡易フロー）。
"""
    (ART_DIR / "decision_gate_README.md").write_text(text, encoding="utf-8")


def plot_decision_flow() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_axis_off()
    boxes = [
        (0.02, 0.2, "オフライン\n合格基準"),
        (0.28, 0.2, "パイロット\nA/B"),
        (0.54, 0.2, "スケール\n＋監視"),
    ]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.2, 0.55, boxstyle="round,pad=0.02", ec="black", fc="#e8f4f8"))
        ax.text(x + 0.1, y + 0.28, t, ha="center", va="center", fontsize=10)
    for i in range(2):
        ax.annotate("", xy=(0.02 + 0.2 + 0.08 * (i + 1), 0.48), xytext=(0.02 + 0.2 * (i + 1), 0.48), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("意思決定フロー（例）")
    from analytics.figures_jp import save_fig

    save_fig(FIG_IMP / "decision_flow.png")
