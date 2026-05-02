"""
追加で欲しいデータ（粗利・接触コスト・露出・在庫・パネル等）に対し、
フェルミ推定ベースの仮定を置いて補完分析・図表を生成する。

出力:
  - artifacts/assumed_data_fermi.json
  - artifacts/assumed_economics_by_row.csv（サンプル用に最大5万行）
  - artifacts/appendix_exercise_sample.csv
  - artifacts/synthetic_exposure_log_sample.csv
  - artifacts/synthetic_panel_ltv.csv
  - figures/appendix/*.png
  - latex/appendix_assumed_supplementary.tex（図の includegraphics ブロック）
  - latex/appendix_fermi_numeric_snippet.tex（本文用の数値1段落）
  - latex/appendix_contact_cost_fermi.tex（接触コスト床のフェルミ推定の根拠）

再現: python -m analytics.assumed_supplementary_analysis
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics.config import ART_DIR, DATA_PATH, ROOT
from analytics.figures_jp import save_fig, set_japanese_font

FIG_APPENDIX = ROOT / "figures" / "appendix"
LATEX_DIR = ROOT / "latex"


def estimate_jpy_contact_costs_fermi() -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    チャネル別接触コスト（円）の床と history 比例項をフェルミ推定する。

    Web: 一斉メール等の限界費用は通数あたり約 0.3--2 円。ESP・創出の固定費按分を
    粗く 2--8 円/接触とみなし、幾何平均 √(1×12)≈3.5 に丸めて保守的に 5 円を床とする。

    Multichannel: Web 接触に SMS/プッシュ等を足す。SMS 辺際 3--12 円/通のオーダーで
    追加 20--35 円 → Web 床 +27 円 ≈ 32 円。

    Phone: 架電オペの時給負担（人件費+社保+場所+システム）を 2,000--2,800 円/時、
    実効接触（リストアウト除く）20--32 件/時と置く。中央 2,400/27≈89 → 90 円/接触。

    history 比例項: 高額客の優先ルーティング・短いスクリプト差し替え等を年間購買額の
    0.4--0.8% とし 0.006（0.6%）を採用。
    """
    # Web 床（円）: 限界費用帯と按分帯の幾何平均のオーダー → 5 円
    marginal_lo, marginal_hi = 0.5, 2.0
    alloc_lo, alloc_hi = 2.0, 10.0
    floor_web = round(math.sqrt(marginal_lo * marginal_hi) + math.sqrt(alloc_lo * alloc_hi) / 2)
    floor_web = float(max(5.0, floor_web))

    # Multichannel: Web + SMS/プッシュ等の追加分（18--30 円の中央）
    sms_band = (18.0, 30.0)
    sms_add = 0.5 * (sms_band[0] + sms_band[1])
    floor_mc = float(round(floor_web + sms_add))

    # Phone 床（円/接触）: 時給負担 ÷ 実効接触/時
    hourly_low, hourly_high = 2000.0, 2800.0
    hourly_mid = 0.5 * (hourly_low + hourly_high)
    tph_low, tph_high = 20.0, 32.0
    tph_mid = 0.5 * (tph_low + tph_high)
    floor_phone_raw = hourly_mid / tph_mid
    floor_phone = float(round(floor_phone_raw))

    # history に対する変動費率（年間購買プロキシに対する割合）
    rate_lo, rate_hi = 0.004, 0.008
    rate_vs_hist = round(0.5 * (rate_lo + rate_hi), 4)  # 0.006

    derivation: Dict[str, Any] = {
        "currency": "JPY",
        "web_floor_jpy": {
            "marginal_email_per_send_jpy_range": [marginal_lo, marginal_hi],
            "allocated_fixed_per_touch_jpy_range": [alloc_lo, alloc_hi],
            "formula_note": "geom_mean(marginal)+geom_mean(alloc)/2, floor_cap=5JPY",
            "chosen_floor_jpy": floor_web,
        },
        "multichannel_floor_jpy": {
            "web_component_jpy": floor_web,
            "sms_push_increment_jpy_range": list(sms_band),
            "sms_increment_mid_jpy": sms_add,
            "chosen_floor_jpy": floor_mc,
        },
        "phone_floor_jpy": {
            "loaded_hourly_cost_jpy_range": [hourly_low, hourly_high],
            "effective_touches_per_hour_range": [tph_low, tph_high],
            "mid_hourly_jpy": hourly_mid,
            "mid_touches_per_hour": tph_mid,
            "ratio_raw": round(floor_phone_raw, 2),
            "chosen_floor_jpy": float(floor_phone),
        },
        "contact_rate_vs_history": {
            "interpretation": "incremental_ops_vs_annual_spend_proxy",
            "range": [rate_lo, rate_hi],
            "chosen": rate_vs_hist,
        },
    }
    floors = {
        "contact_floor_web": float(floor_web),
        "contact_floor_multichannel": float(floor_mc),
        "contact_floor_phone": float(floor_phone),
        "contact_rate_vs_history": float(rate_vs_hist),
    }
    return derivation, floors


def init_plot_style_light() -> None:
    """seaborn 非依存（環境によっては site / seaborn が無い）。"""
    set_japanese_font()
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.28


@dataclass
class FermiAssumptions:
    """フェルミ推定の前提（オーダー整合性を優先。history は円建て年間購買額プロキシ）。"""

    # 小売粗利率: 25% と 40% の幾何平均 ≈ 32%
    gross_margin: float = 0.32
    # history を年間購買額プロキシとみなし、キャンペーン1回の購入売上 ≈ 月次相当
    revenue_fraction_of_history: float = 1.0 / 12.0
    # オファーが売上に与える割引負担（売上に対する比率の目安）
    discount_offer_redemption: float = 0.25
    bogo_effective_redemption: float = 0.38
    # 接触コスト（円）: 床 + history×率（estimate_jpy_contact_costs_fermi と整合）
    contact_floor_web: float = 5.0
    contact_floor_multichannel: float = 29.0
    contact_floor_phone: float = 92.0
    contact_rate_vs_history: float = 0.006
    # 露出: キャンペーン期間中の平均インプレッション（ポアソン期待値）
    impressions_lambda: float = 3.2
    # 在庫: 通常需要の変動係数、プロモ時の需要乗数
    demand_cv: float = 0.18
    promo_demand_multiplier_bogo: float = 1.35
    promo_demand_multiplier_discount: float = 1.22
    capacity_safety_factor: float = 1.12
    # クレーム率（購入ベース）
    claims_rate_baseline: float = 0.015
    claims_increment_discount: float = 0.004
    claims_increment_bogo: float = 0.007
    # 合成パネル: 接触累積に対する疲労（LTV 乗数の指数減衰）
    fatigue_decay_per_contact: float = 0.04
    panel_horizon: int = 24


def _offer_redemption(offer: str, f: FermiAssumptions) -> float:
    if offer == "No Offer":
        return 0.0
    if offer == "Discount":
        return f.discount_offer_redemption
    if offer == "Buy One Get One":
        return f.bogo_effective_redemption
    return 0.0


def _contact_cost_row(channel: str, history: float, f: FermiAssumptions) -> float:
    base = {"Web": f.contact_floor_web, "Multichannel": f.contact_floor_multichannel, "Phone": f.contact_floor_phone}.get(
        str(channel), f.contact_floor_multichannel
    )
    return float(base + f.contact_rate_vs_history * history)


def accounting_profit_row(
    history: float,
    conversion: int,
    offer: str,
    channel: str,
    fermi: FermiAssumptions,
) -> dict:
    """購入時のみ粗利計上。未購入は接触コストのみ。"""
    contact = _contact_cost_row(channel, history, fermi)
    if not conversion:
        return {
            "revenue": 0.0,
            "cogs": 0.0,
            "gross_profit_pre_promo": 0.0,
            "promo_cost": 0.0,
            "gross_profit_after_promo": -contact,
            "contact_cost": contact,
        }
    rev = fermi.revenue_fraction_of_history * history
    cogs = (1.0 - fermi.gross_margin) * rev
    gp_pre = rev - cogs
    red = _offer_redemption(offer, fermi)
    promo = red * rev
    gp = gp_pre - promo - contact
    return {
        "revenue": rev,
        "cogs": cogs,
        "gross_profit_pre_promo": gp_pre,
        "promo_cost": promo,
        "gross_profit_after_promo": gp,
        "contact_cost": contact,
    }


def simulate_exposure_assignment(df: pd.DataFrame, rng: np.random.Generator, fermi: FermiAssumptions) -> pd.DataFrame:
    """合成: インプレッション数・スコアに依存した割当の監査用ログ（一部列）。"""
    z = (
        0.35 * (df["recency"].to_numpy() / 12.0)
        + 0.45 * (df["history"].to_numpy() / (df["history"].median() + 1e-6))
        + 0.2 * df["is_referral"].to_numpy()
    )
    p_disc = 1.0 / (1.0 + np.exp(-(z - 0.55)))
    u = rng.uniform(size=len(df))
    assigned = np.where(u < p_disc * 0.45, "Discount", np.where(u < p_disc * 0.45 + 0.35, "Buy One Get One", "No Offer"))
    impressions = rng.poisson(fermi.impressions_lambda, size=len(df))
    return pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "impressions": impressions,
            "score_audit": z,
            "p_discount_propensity": p_disc,
            "synthetic_assigned_offer": assigned,
            "observed_offer": df["offer"],
        }
    )


def inventory_stockout_curve(fermi: FermiAssumptions, rng: np.random.Generator, n_mc: int = 8000) -> pd.DataFrame:
    """需要を正規近似し、キャパシティに対する欠品確率をオファー別にモンテカルロ。"""
    mu0 = 1.0
    sigma = fermi.demand_cv * mu0
    caps = np.linspace(0.85, 1.45, 35)
    rows = []
    for cap in caps:
        capacity = cap * mu0 * fermi.capacity_safety_factor
        for label, mult in [("No Offer", 1.0), ("Discount", fermi.promo_demand_multiplier_discount), ("BOGO", fermi.promo_demand_multiplier_bogo)]:
            demand = rng.normal(mu0 * mult, sigma, size=n_mc)
            p_stockout = float(np.mean(demand > capacity))
            rows.append({"capacity_ratio": cap, "offer": label, "p_stockout": p_stockout})
    return pd.DataFrame(rows)


def synthetic_ltv_panel_fixed(n_customers: int, rng: np.random.Generator, fermi: FermiAssumptions) -> pd.DataFrame:
    base = rng.lognormal(mean=5.0, sigma=0.35, size=n_customers)
    cum_c = np.zeros(n_customers)
    out = []
    for t in range(fermi.panel_horizon):
        lam = 0.35 + 0.05 * (t % 3)
        contacts = rng.poisson(lam, size=n_customers)
        cum_c = cum_c + contacts
        fatigue = np.exp(-fermi.fatigue_decay_per_contact * cum_c)
        ltv = base * fatigue * (1.0 + 0.008 * t)
        out.append(
            {
                "month": t + 1,
                "mean_ltv": float(ltv.mean()),
                "p10_ltv": float(np.percentile(ltv, 10)),
                "p90_ltv": float(np.percentile(ltv, 90)),
                "mean_cum_contacts": float(cum_c.mean()),
            }
        )
    return pd.DataFrame(out)


def fig_fermi_schematic(fermi: FermiAssumptions) -> None:
    init_plot_style_light()
    labels = [
        f"粗利率\n{fermi.gross_margin:.0%}",
        f"売上/history\n×{fermi.revenue_fraction_of_history:.3f}",
        f"割引負担\nDisc {fermi.discount_offer_redemption:.0%}\nBOGO~{fermi.bogo_effective_redemption:.0%}",
        f"接触(床)\nWeb {fermi.contact_floor_web}\nPh {fermi.contact_floor_phone}",
        f"露出λ\n{fermi.impressions_lambda}",
    ]
    y = [fermi.gross_margin * 100, fermi.revenue_fraction_of_history * 100, 30, 25, fermi.impressions_lambda * 10]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(range(len(labels)), y, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("相対スケール（示唆のみ）")
    ax.set_title("フェルミ仮定のオーダー（付録で定義）")
    save_fig(FIG_APPENDIX / "fig_fermi_schematic.png")


def fig_profit_by_offer(econ: pd.DataFrame) -> None:
    init_plot_style_light()
    g = econ.groupby("offer")["gross_profit_after_promo"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    g.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("仮定会計下の平均粗利−プロモ−接触（顧客行単位・観察データ）")
    ax.set_ylabel("平均（円）")
    ax.set_xlabel("観察オファー")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
    save_fig(FIG_APPENDIX / "fig_profit_by_observed_offer.png")


def fig_contact_cost_channels(fermi: FermiAssumptions) -> None:
    init_plot_style_light()
    h = np.linspace(50, 800, 50)
    fig, ax = plt.subplots(figsize=(7, 4))
    for ch, color in zip(
        ["Web", "Multichannel", "Phone"],
        ["#4C72B0", "#55A868", "#C44E52"],
    ):
        y = [_contact_cost_row(ch, float(x), fermi) for x in h]
        ax.plot(h, y, label=ch, color=color, lw=2)
    ax.set_xlabel("history（年間購買プロキシ、円）")
    ax.set_ylabel("接触コスト（円）")
    ax.set_title("チャネル別接触コスト（床＋history 比例）")
    ax.legend()
    save_fig(FIG_APPENDIX / "fig_contact_cost_by_channel.png")


def fig_exposure_audit(sample: pd.DataFrame) -> None:
    init_plot_style_light()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(
        sample["score_audit"],
        sample["impressions"],
        c=sample["p_discount_propensity"],
        cmap="viridis",
        alpha=0.35,
        s=8,
    )
    plt.colorbar(sc, ax=ax, label="Discount 傾向スコア")
    ax.set_xlabel("監査用スコア（合成）")
    ax.set_ylabel("インプレッション数（合成 Poisson）")
    ax.set_title("合成露出ログ: スコアと接触回数")
    save_fig(FIG_APPENDIX / "fig_exposure_scatter_audit.png")


def fig_conversion_by_impressions(merged: pd.DataFrame) -> None:
    init_plot_style_light()
    m = merged.copy()
    m["imp_bin"] = pd.cut(m["impressions"], bins=[0, 1, 2, 3, 5, 100], labels=["1", "2", "3", "4-5", "6+"])
    g = m.groupby("imp_bin", observed=False)["conversion"].mean()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    g.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_xlabel("合成インプレッション数ビン")
    ax.set_ylabel("観察CVR（同一行）")
    ax.set_title("露出回数ビン別のCVR（合成×観察の合流・記述）")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    save_fig(FIG_APPENDIX / "fig_cvr_by_impression_bin.png")


def fig_assignment_confusion(sample: pd.DataFrame) -> None:
    init_plot_style_light()
    cross = pd.crosstab(sample["synthetic_assigned_offer"], sample["observed_offer"])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cross.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cross.columns)))
    ax.set_xticklabels(cross.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(cross.index)))
    ax.set_yticklabels(cross.index)
    ax.set_xlabel("観察オファー")
    ax.set_ylabel("合成割当（監査用）")
    ax.set_title("割当×観察のクロス（合成・交絡の可視化）")
    for i in range(cross.shape[0]):
        for j in range(cross.shape[1]):
            ax.text(j, i, int(cross.iloc[i, j]), ha="center", va="center", color="black", fontsize=7)
    plt.colorbar(im, ax=ax, label="人数")
    save_fig(FIG_APPENDIX / "fig_assignment_cross_heatmap.png")


def fig_inventory(inv_df: pd.DataFrame) -> None:
    init_plot_style_light()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for offer, style in zip(["No Offer", "Discount", "BOGO"], ["-", "--", "-."]):
        sub = inv_df[inv_df["offer"] == offer]
        ax.plot(sub["capacity_ratio"], sub["p_stockout"], label=offer, lw=2, linestyle=style)
    ax.set_xlabel("キャパシティ係数（需要平均に対する比）")
    ax.set_ylabel("欠品確率（モンテカルロ）")
    ax.set_title("プロモ別需要ショックと欠品リスク（正規近似・フェルミ倍率）")
    ax.legend()
    save_fig(FIG_APPENDIX / "fig_inventory_stockout.png")


def fig_claims(fermi: FermiAssumptions) -> None:
    init_plot_style_light()
    offers = ["No Offer", "Discount", "BOGO"]
    rates = [
        fermi.claims_rate_baseline,
        fermi.claims_rate_baseline + fermi.claims_increment_discount,
        fermi.claims_rate_baseline + fermi.claims_increment_bogo,
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(offers, [100 * r for r in rates], color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("クレーム率（% , フェルミ）")
    ax.set_title("オファー別クレーム率の仮定（購入ベース）")
    save_fig(FIG_APPENDIX / "fig_claims_rate_by_offer.png")


def fig_ltv_panel(panel: pd.DataFrame) -> None:
    init_plot_style_light()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.fill_between(panel["month"], panel["p10_ltv"], panel["p90_ltv"], alpha=0.25, color="#4C72B0", label="10–90%")
    ax.plot(panel["month"], panel["mean_ltv"], color="#C44E52", lw=2, label="平均LTV（合成）")
    ax2 = ax.twinx()
    ax2.plot(panel["month"], panel["mean_cum_contacts"], color="#55A868", ls="--", alpha=0.8, label="平均累積接触")
    ax.set_xlabel("月")
    ax.set_ylabel("LTV 指数")
    ax2.set_ylabel("累積接触（平均）")
    ax.set_title("合成パネル: 疲労によるLTV低下と接触累積")
    ax.legend(loc="upper right")
    ax2.legend(loc="center right")
    save_fig(FIG_APPENDIX / "fig_panel_ltv_fatigue.png")


def fig_profit_components_grouped(econ: pd.DataFrame) -> None:
    init_plot_style_light()
    agg = (
        econ.groupby("offer")[["revenue", "cogs", "promo_cost", "contact_cost", "gross_profit_after_promo"]]
        .mean()
        .reindex(["No Offer", "Discount", "Buy One Get One"])
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    offers = agg.index.tolist()
    x = np.arange(len(offers))
    w = 0.18
    metrics = [
        ("revenue", "売上", "#4C72B0"),
        ("cogs", "COGS", "#C44E52"),
        ("promo_cost", "プロモ", "#8172B2"),
        ("contact_cost", "接触", "#CCB974"),
        ("gross_profit_after_promo", "粗利-接触等", "#55A868"),
    ]
    for k, (col, lab, color) in enumerate(metrics):
        ax.bar(x + (k - 2) * w, agg[col].to_numpy(), w, label=lab, color=color, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(offers, rotation=12, ha="right")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("顧客行あたり平均（円）")
    ax.set_title("仮定会計: オファー別の平均内訳（未購入は revenue=0）")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_fig(FIG_APPENDIX / "fig_profit_components_grouped.png")


def fig_violin_economics(econ: pd.DataFrame) -> None:
    init_plot_style_light()
    offers = ["No Offer", "Discount", "Buy One Get One"]
    data = [econ.loc[econ["offer"] == o, "gross_profit_after_promo"].to_numpy(dtype=float) for o in offers]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.violinplot(data, positions=range(1, len(offers) + 1), showmeans=True, showmedians=True)
    for b in parts["bodies"]:
        b.set_alpha(0.72)
    ax.set_xticks(range(1, len(offers) + 1))
    ax.set_xticklabels(offers, rotation=15, ha="right")
    ax.set_title("仮定会計粗利の分布（オファー別・観察）")
    ax.set_ylabel("gross_profit_after_promo")
    save_fig(FIG_APPENDIX / "fig_gp_violin_by_offer.png")


def write_latex_snippets(fermi: FermiAssumptions, mean_hist: float) -> None:
    rev1 = fermi.revenue_fraction_of_history * mean_hist
    gp_unit = fermi.gross_margin * rev1
    # f-string と LaTeX の {} が衝突するため、数値のみ f 補間で連結する
    snippet = (
        "% auto-generated by analytics/assumed_supplementary_analysis.py\n"
        "\\noindent\n"
        "\\textbf{数値仮定（フェルミ）.}\n"
        f"\\texttt{{history}} を\\textbf{{円建て}}の年間購買額プロキシとし、粗利率 {fermi.gross_margin:.2f}（$\\sqrt{{0.25\\times 0.40}}\\approx 0.32$ のオーダー）、\n"
        f"キャンペーン1回の売上をその {fermi.revenue_fraction_of_history:.3f} 倍（月次花弁相当・円）、\n"
        f"割引・BOGO の売上に対する負担をそれぞれ {fermi.discount_offer_redemption:.0%}・{fermi.bogo_effective_redemption:.0%} と置いた。\n"
        f"接触コストは Web/Multichannel/Phone で床 {fermi.contact_floor_web}/{fermi.contact_floor_multichannel}/{fermi.contact_floor_phone} 円"
        f"に加え history の {fermi.contact_rate_vs_history:.1%} を足す。\n"
        f"平均 history {mean_hist:.1f} 円の顧客が1回購入した場合の売上目安は {rev1:.2f} 円、粗利（プロモ前）は {gp_unit:.2f} 円。\n"
        f"露出は期間中 Poisson($\\lambda={fermi.impressions_lambda}$) の合成ログ、在庫は需要CV {fermi.demand_cv}・"
        f"BOGO需要×{fermi.promo_demand_multiplier_bogo} のショック、\n"
        f"クレームはベース {fermi.claims_rate_baseline:.1%} にオファー上乗せ、パネルは累積接触に "
        f"$\\exp(-{fermi.fatigue_decay_per_contact}\\times\\cdot)$ の疲労を仮定（詳細は付録\\ref{{app:assumed-data}}）。\n"
    )
    (LATEX_DIR / "appendix_fermi_numeric_snippet.tex").write_text(snippet, encoding="utf-8")

    figures_block = r"""% auto-generated — figures in ../figures/appendix/
\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_fermi_schematic.png}
  \caption{フェルミ仮定のオーダー（粗利・売上スケール・オファー負担・接触・露出）。}
  \label{fig:app-fermi-schematic}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_profit_by_observed_offer.png}
  \caption{仮定会計（粗利−プロモ−接触）下での観察オファー別の平均利益。因果効果ではない。}
  \label{fig:app-profit-offer}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_gp_violin_by_offer.png}
  \caption{同・仮定会計での利益分布（バイオリン）。}
  \label{fig:app-gp-violin}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_contact_cost_by_channel.png}
  \caption{チャネル別の接触コスト曲線（床＋history比例）。}
  \label{fig:app-contact}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_exposure_scatter_audit.png}
  \caption{合成露出ログ：監査スコアとインプレッション数。}
  \label{fig:app-exposure}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_cvr_by_impression_bin.png}
  \caption{合成インプレッション数のビン別・観察CVR（記述。因果の主張はしない）。}
  \label{fig:app-cvr-imp}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_assignment_cross_heatmap.png}
  \caption{合成割当と観察オファーのクロス（交絡のイメージ）。}
  \label{fig:app-assign-heat}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_inventory_stockout.png}
  \caption{需要ショックに対する欠品確率（モンテカルロ）。}
  \label{fig:app-inventory}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_claims_rate_by_offer.png}
  \caption{オファー別クレーム率のフェルミ仮定。}
  \label{fig:app-claims}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_panel_ltv_fatigue.png}
  \caption{合成パネルにおけるLTVと累積接触（疲労）。}
  \label{fig:app-panel}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/appendix/fig_profit_components_grouped.png}
  \caption{仮定会計におけるオファー別の平均内訳（売上・COGS・プロモ・接触・粗利）。}
  \label{fig:app-profit-grouped}
\end{figure}
"""
    (LATEX_DIR / "appendix_assumed_supplementary.tex").write_text(figures_block, encoding="utf-8")


def write_contact_cost_fermi_appendix_tex(derivation: Dict[str, Any], floors: Dict[str, float]) -> None:
    """接触コスト推定の根拠を付録用 TeX に書き出す。"""
    w = derivation["web_floor_jpy"]
    m = derivation["multichannel_floor_jpy"]
    p = derivation["phone_floor_jpy"]
    r = derivation["contact_rate_vs_history"]
    text = (
        "% auto-generated by analytics/assumed_supplementary_analysis.py\n"
        "\\paragraph{接触コスト（円）のフェルミ推定.}\n"
        "\\noindent\n"
        "\\textbf{Web.} 一斉メール等の限界費用を "
        f"{w['marginal_email_per_send_jpy_range'][0]:.1f}--{w['marginal_email_per_send_jpy_range'][1]:.1f} 円/通、"
        f"ESP・創出の按分を {w['allocated_fixed_per_touch_jpy_range'][0]:.0f}--{w['allocated_fixed_per_touch_jpy_range'][1]:.0f} 円/接触の帯とし、"
        "オーダーを足し合わせて床を \\textbf{"
        f"{floors['contact_floor_web']:.0f}"
        "}~円とした"
        "（限界費用帯と按分帯の幾何平均を合成し、最低 \\textbf{5}~円でキャップ）。\n\n"
        "\\noindent\n"
        "\\textbf{Multichannel.} Web 床に SMS/プッシュ等の追加分 "
        f"{m['sms_push_increment_jpy_range'][0]:.0f}--{m['sms_push_increment_jpy_range'][1]:.0f} 円の中央 "
        f"（{m['sms_increment_mid_jpy']:.0f} 円）を加え、床 \\textbf{{{floors['contact_floor_multichannel']:.0f}}}~円。\n\n"
        "\\noindent\n"
        "\\textbf{Phone.} 架電オペの時給負担（人件費・諸掛）を "
        f"{p['loaded_hourly_cost_jpy_range'][0]:.0f}--{p['loaded_hourly_cost_jpy_range'][1]:.0f} 円/時、"
        f"実効接触を {p['effective_touches_per_hour_range'][0]:.0f}--{p['effective_touches_per_hour_range'][1]:.0f} 件/時と置き、"
        f"中央 {p['mid_hourly_jpy']:.0f}/{p['mid_touches_per_hour']:.1f} $\\approx$ {p['ratio_raw']:.1f} 円/接触 $\\rightarrow$ 床 "
        f"\\textbf{{{floors['contact_floor_phone']:.0f}}}~円。\n\n"
        "\\noindent\n"
        "\\textbf{\\texttt{history} 比例項.} 年間購買額プロキシに対する変動オペ・差し替え負担を "
        f"{r['range'][0]:.1%}--{r['range'][1]:.1%} とし、\\textbf{{{floors['contact_rate_vs_history']:.1%}}} を採用。\n"
    )
    (LATEX_DIR / "appendix_contact_cost_fermi.tex").write_text(text, encoding="utf-8")


def main() -> None:
    FIG_APPENDIX.mkdir(parents=True, exist_ok=True)
    contact_derivation, contact_floors = estimate_jpy_contact_costs_fermi()
    fermi = FermiAssumptions(
        contact_floor_web=contact_floors["contact_floor_web"],
        contact_floor_multichannel=contact_floors["contact_floor_multichannel"],
        contact_floor_phone=contact_floors["contact_floor_phone"],
        contact_rate_vs_history=contact_floors["contact_rate_vs_history"],
    )
    rng = np.random.default_rng(42)

    meta = asdict(fermi)
    meta["notes"] = (
        "history は円建ての年間購買額プロキシとして扱う。BOGOの0.38は「第2品目の辺際原価＋オペ損失」の目安。"
        " 接触コストの床は estimate_jpy_contact_costs_fermi() のフェルミ推定に従う。"
    )
    meta["contact_cost_fermi_estimation"] = contact_derivation
    (ART_DIR / "assumed_data_fermi.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.read_csv(DATA_PATH)
    if "customer_id" not in df.columns:
        df.insert(0, "customer_id", np.arange(len(df), dtype=np.int64))

    econ_rows = []
    for _, r in df.iterrows():
        ac = accounting_profit_row(float(r["history"]), int(r["conversion"]), str(r["offer"]), str(r["channel"]), fermi)
        econ_rows.append(
            {
                "customer_id": r["customer_id"],
                "offer": r["offer"],
                "channel": r["channel"],
                "conversion": r["conversion"],
                **ac,
            }
        )
    econ = pd.DataFrame(econ_rows)
    econ.to_csv(ART_DIR / "assumed_economics_by_row.csv", index=False)

    df.head(500).to_csv(ART_DIR / "appendix_exercise_sample.csv", index=False)

    exposure = simulate_exposure_assignment(df, rng, fermi)
    exposure.head(8000).to_csv(ART_DIR / "synthetic_exposure_log_sample.csv", index=False)
    merged_exp = df[["customer_id", "conversion"]].merge(
        exposure[["customer_id", "impressions", "score_audit"]], on="customer_id", how="left"
    )
    fig_conversion_by_impressions(merged_exp)

    panel = synthetic_ltv_panel_fixed(4000, rng, fermi)
    panel.to_csv(ART_DIR / "synthetic_panel_ltv.csv", index=False)

    mean_hist = float(df["history"].mean())

    fig_fermi_schematic(fermi)
    fig_profit_by_offer(econ)
    fig_contact_cost_channels(fermi)
    sample = exposure.sample(n=min(4000, len(exposure)), random_state=7)
    fig_exposure_audit(sample)
    fig_assignment_confusion(sample)
    inv_df = inventory_stockout_curve(fermi, rng)
    inv_df.to_csv(ART_DIR / "inventory_stockout_mc.csv", index=False)
    fig_inventory(inv_df)
    fig_claims(fermi)
    fig_ltv_panel(panel)
    fig_violin_economics(econ)
    fig_profit_components_grouped(econ)

    write_latex_snippets(fermi, mean_hist)
    write_contact_cost_fermi_appendix_tex(contact_derivation, contact_floors)
    print("Wrote appendix figures and latex/appendix_assumed_supplementary.tex")


if __name__ == "__main__":
    main()
