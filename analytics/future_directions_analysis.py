"""
「今後の方針」「追加で欲しいデータ」に対応する図解・感度分析（演習データベース）。

- 3 臂パイロット（低コスト訴求 / 現行オファー / ホールドアウト）の模式図
- 主アウトカムと 90 日副次の評価タイムライン（事前登録イメージ）
- 売上 proxy（history）に対する「実質マージン倍率」の感度：推奨処置の安定性
- 演習データで不足するデータ要素の可用性マトリクス

出力: figures/improvements/future_*.png, artifacts/future_*.csv, latex/future_directions_snippet.tex
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics.config import ART_DIR, FIG_IMP, LATEX_DIR
from analytics.figures_jp import init_plot_style, save_fig


def _plot_pilot_arm_schematic(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 0.85, 2.7, 1.35, "ホールドアウト\n（無接触／現行運用）", "#E8E8E8"),
        (3.55, 0.85, 2.7, 1.35, "低コスト訴求\n（例：No Offer / Web）", "#B8D4EC"),
        (6.8, 0.85, 2.7, 1.35, "現行オファー\n（割引・BOGO 等）", "#F5D5A9"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", fc=color, ec="#333333", lw=1.0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    ax.text(5.0, 2.55, "上位スコア層（例：Top 5–10%）を母集団に無作為割付", ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.annotate("", xy=(5.0, 0.55), xytext=(5.0, 0.15), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
    ax.text(5.0, 0.05, "KPI：増分購入率・期待利益proxy・接触コスト（KPI階層表と対応）", ha="center", va="top", fontsize=9)
    fig.tight_layout()
    save_fig(out)


def _plot_kpi_timeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.set_xlim(-5, 100)
    ax.set_ylim(0, 3.5)
    ax.axhline(2.35, color="#ccc", lw=0.8)
    ax.axhline(1.15, color="#ccc", lw=0.8)
    y_primary, y_sec = 2.35, 1.15
    ax.barh(y_primary, 85, left=0, height=0.55, color="#4C72B0", alpha=0.85, label="主評価期間（例：キャンペーン〜購入計測）")
    ax.barh(y_sec, 90, left=0, height=0.55, color="#55A868", alpha=0.85, label="副次：試験終了後90日窓")
    ax.plot([0, 0], [0.5, 2.8], "k--", lw=0.9)
    ax.text(0, 3.05, "割付開始（T0）", ha="center", fontsize=9)
    ax.text(85, 3.05, "主アウトカム締め（例）", ha="center", fontsize=9)
    ax.text(90, 0.35, "90日累積売上・再購入の事前登録", ha="center", fontsize=9)
    ax.set_yticks([y_primary, y_sec])
    ax.set_yticklabels(["Primary（短期）", "副次（長期proxy）"])
    ax.set_xlabel("経過日数（模式）")
    ax.set_title("評価タイムライン（事前登録イメージ）")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_fig(out)


def _plot_margin_sensitivity(profit_long: pd.DataFrame, out: Path, csv_path: Path) -> None:
    """推奨処置の安定性（mid_cost）。

    - 売上項のみ m 倍 … 粗利 proxy が楽観的な場合。順位が入れ替わりうる。
    - コスト項に m 倍 … インセンティブ・接触の実コストのストレス。順位が入れ替わりうる。

    注: ネット (EV - C) 全体に同一の正の m を掛けても argmax は変わらないため、図には含めない。
    """
    mid = profit_long.loc[profit_long["scenario"] == "mid_cost", ["customer_id", "treatment", "expected_value", "cost_offer", "cost_channel"]].copy()
    if len(mid) == 0:
        return
    c_tot = mid["cost_offer"] + mid["cost_channel"]
    mid["ref_profit"] = mid["expected_value"] - c_tot
    ref_idx = mid.groupby("customer_id")["ref_profit"].idxmax()
    ref_best = mid.loc[ref_idx].set_index("customer_id")["treatment"]

    ms = np.linspace(0.35, 1.45, 29)
    agree_rev: list[float] = []
    agree_cost: list[float] = []
    for m in ms:
        adj_r = mid["expected_value"] * m - c_tot
        mid["_adj"] = adj_r
        bm_idx = mid.groupby("customer_id")["_adj"].idxmax()
        bm = mid.loc[bm_idx].set_index("customer_id")["treatment"]
        agree_rev.append(float((bm == ref_best).mean()))

        adj_c = mid["expected_value"] - m * c_tot
        mid["_adj"] = adj_c
        bm_idx = mid.groupby("customer_id")["_adj"].idxmax()
        bm = mid.loc[bm_idx].set_index("customer_id")["treatment"]
        agree_cost.append(float((bm == ref_best).mean()))
    mid.drop(columns=["_adj"], errors="ignore")

    pd.DataFrame(
        {
            "m": ms,
            "agreement_revenue_only_scaled": agree_rev,
            "agreement_cost_scaled": agree_cost,
        }
    ).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6.8, 3.85))
    ax.plot(ms, np.array(agree_rev) * 100, color="#C44E52", lw=2.0, label=r"期待売上項のみ $m$ 倍（$\,m\,\mathrm{EV}-\mathrm{C}\,$）")
    ax.plot(ms, np.array(agree_cost) * 100, color="#4C72B0", lw=2.0, label=r"コスト項に $m$ 倍（$\,\mathrm{EV}-m\mathrm{C}\,$）")
    ax.axvline(1.0, color="gray", ls="--", lw=0.9)
    ax.set_xlabel(r"倍率 $m$（縦破線は $m=1$＝本稿の基準）")
    ax.set_ylabel(r"基準（$m=1$）と同一の推奨処置の割合（%）")
    ax.set_title("mid_cost：粗利proxy・コストのストレスと推奨処置の安定性")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    save_fig(out)


def _plot_data_gap_matrix(out: Path) -> None:
    rows = ["真の粗利・変動費・接触コスト", "割当・露出・配信ログ", "在庫・供給・クレーム", "複数期パネル（LTV 等）"]
    cols = ["演習データ", "本番で追加"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(rows))
    ax.axis("off")
    c_ng, c_ok = "#f8d7da", "#d4edda"
    labels = [
        ("proxy のみ\n（感度大）", "会計・オペ\n連携"),
        ("無し", "監査・増分\nに必須"),
        ("無し", "外部性\n評価に有用"),
        ("単期", "疲労・慣習化\nの分離に必須"),
    ]
    for i, (row_label, (t0, t1)) in enumerate(zip(rows, labels)):
        c0, c1 = c_ng, c_ok
        y = len(rows) - 1 - i
        ax.add_patch(mpatches.Rectangle((0, y), 1, 1, fc=c0, ec="#333", lw=0.8))
        ax.add_patch(mpatches.Rectangle((1, y), 1, 1, fc=c1, ec="#333", lw=0.8))
        ax.text(0.5, y + 0.5, t0, ha="center", va="center", fontsize=8.5)
        ax.text(1.5, y + 0.5, t1, ha="center", va="center", fontsize=8.5)
        ax.text(-0.02, y + 0.5, row_label, ha="right", va="center", fontsize=8.5)
    for j, cname in enumerate(cols):
        ax.text(j + 0.5, len(rows) + 0.35, cname, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("データギャップ（定性。色は利用のしやすさのイメージ）")
    fig.tight_layout()
    save_fig(out)


def _write_latex_snippet() -> None:
    lines = [
        r"% 本ファイルは \texttt{python -m analytics.future\_directions\_analysis} または \texttt{run\_analysis.py} 実行時に上書きされる。",
        r"% auto-generated by analytics/future_directions_analysis.py",
        r"\subsection{提案の図解と proxy 感度（Python 生成）}",
        r"\label{sec:future_figures}",
        r"「今後の方針」の箇条書きと「追加で欲しいデータ」を、パイロット割付・90日副次・粗利未観測時の感度・データギャップとして図示する。",
        r"いずれも\textbf{設計・監査用の補助}であり、因果効果の確定は無作為化試験に委ねる。",
        r"",
        r"\begin{figure}[!htbp]",
        r"  \centering",
        r"  \begin{subfigure}[t]{0.49\linewidth}",
        r"    \centering",
        r"    \includegraphics[width=\linewidth]{../figures/improvements/future_pilot_arm_schematic.png}",
        r"    \caption{上位スコア層を母集団とした\textbf{3臂}（低コスト訴求・現行オファー・ホールドアウト）。因子はチャネル等に拡張可。}",
        r"    \label{fig:future_pilot_arms}",
        r"  \end{subfigure}\hfill",
        r"  \begin{subfigure}[t]{0.49\linewidth}",
        r"    \centering",
        r"    \includegraphics[width=\linewidth]{../figures/improvements/future_kpi_timeline_90d.png}",
        r"    \caption{主アウトカム（短期）と事前登録した\textbf{90日}副次（累積売上・再購入等）。日付は組織で調整。}",
        r"    \label{fig:future_kpi_timeline}",
        r"  \end{subfigure}",
        r"  \caption{パイロット割付（左）と評価タイムライン（右）の模式}",
        r"  \label{fig:future_pilot_and_timeline}",
        r"\end{figure}",
        r"",
        r"\begin{figure}[!htbp]",
        r"  \centering",
        r"  \begin{subfigure}[t]{0.48\linewidth}",
        r"    \centering",
        r"    \includegraphics[width=\linewidth]{../figures/improvements/future_profit_margin_sensitivity.png}",
        r"    \caption{\textbf{真の粗利・返品・原価・実コスト}が未観測のとき、オフライン推奨は感度が大きい。",
        r"    \textbf{赤}は期待売上項だけ $m$ 倍（$m\,\mathrm{EV}-\mathrm{C}$）---{}粗利proxyが薄い・転換時の価値が過大評価のイメージ。",
        r"    \textbf{青}はコスト項に $m$ 倍（$\mathrm{EV}-m\mathrm{C}$）---{}インセンティブ・接触コストが想定より重い／軽いストレス。",
        r"    なお $(\mathrm{EV}-\mathrm{C})$ 全体に同一の $m>0$ を掛けても argmax は変わらない（本図には描かない）。",
        r"    数値は \texttt{artifacts/future\_margin\_sensitivity.csv}。}",
        r"    \label{fig:future_margin_sens}",
        r"  \end{subfigure}\hfill",
        r"  \begin{subfigure}[t]{0.48\linewidth}",
        r"    \centering",
        r"    \includegraphics[width=\linewidth]{../figures/improvements/future_data_gap_matrix.png}",
        r"    \caption{演習データに無い要素と、本番で揃えるとよいデータ（追加で欲しいデータの対応表・定性）。}",
        r"    \label{fig:future_data_gap}",
        r"  \end{subfigure}",
        r"  \caption{粗利・コスト未観測時のオフライン推奨感度（左）とデータギャップの定性マトリクス（右）}",
        r"\end{figure}",
        r"",
        r"\FloatBarrier",
    ]
    (LATEX_DIR / "future_directions_snippet.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_future_directions_analysis(profit_long: pd.DataFrame | None = None) -> None:
    init_plot_style()
    ART_DIR.mkdir(parents=True, exist_ok=True)
    FIG_IMP.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    _plot_pilot_arm_schematic(FIG_IMP / "future_pilot_arm_schematic.png")
    _plot_kpi_timeline(FIG_IMP / "future_kpi_timeline_90d.png")
    _plot_data_gap_matrix(FIG_IMP / "future_data_gap_matrix.png")

    pl = profit_long
    if pl is None:
        p = ART_DIR / "profit_long.csv"
        if p.is_file():
            pl = pd.read_csv(p)
    if pl is not None and len(pl) > 0:
        _plot_margin_sensitivity(pl, FIG_IMP / "future_profit_margin_sensitivity.png", ART_DIR / "future_margin_sensitivity.csv")
    else:
        placeholder = "# margin sensitivity skipped (profit_long missing)\n"
        (ART_DIR / "future_margin_sensitivity.csv").write_text(placeholder, encoding="utf-8")

    _write_latex_snippet()
    print("Wrote future_directions figures and latex/future_directions_snippet.tex")


def main() -> None:
    run_future_directions_analysis(None)


if __name__ == "__main__":
    main()
