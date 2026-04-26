from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from analytics.causal import (
    build_joint_treatment,
    dr_estimate_mu,
    fit_propensity_and_outcome_models,
    policy_recommendations,
)
from analytics.config import (
    ART_DIR,
    CONSTRAINT_BUDGET_HISTORY_UNITS,
    CONSTRAINT_CHANNEL_CAP,
    CONSTRAINT_MAX_CONTACTS,
    DEFAULT_SCENARIOS,
    FIG_IMP,
    K_FOLD,
    RANDOM_STATE,
)
from analytics.config import CostScenario
from analytics.constants import BASE_FEATURES
from analytics.figures_jp import save_fig


def offer_cost_multiplier(offer: str, scenario: CostScenario) -> float:
    if offer == "No Offer":
        return 0.0
    if offer == "Buy One Get One":
        return scenario.r_bogo
    if offer == "Discount":
        return scenario.r_disc
    raise ValueError(f"Unknown offer: {offer}")


def build_profit_tables(
    df: pd.DataFrame,
    p_by_treatment: pd.DataFrame,
    scenarios: List[CostScenario],
) -> pd.DataFrame:
    base = df[["customer_id", "history"]].copy()
    treatments = [c for c in p_by_treatment.columns if c != "customer_id"]
    base = base.merge(p_by_treatment, on="customer_id", how="left")

    all_rows = []
    for sc in scenarios:
        for t in treatments:
            offer, channel = t.split(" | ")
            p = base[t].to_numpy()
            value = base["history"].to_numpy()
            cost_offer = value * offer_cost_multiplier(offer, sc)
            cost_channel = value * sc.channel_rank_cost.get(channel, 0.0)
            profit = p * value - cost_offer - cost_channel
            all_rows.append(
                pd.DataFrame(
                    {
                        "scenario": sc.name,
                        "treatment": t,
                        "customer_id": base["customer_id"].to_numpy(),
                        "p_conv": p,
                        "expected_value": p * value,
                        "cost_offer": cost_offer,
                        "cost_channel": cost_channel,
                        "expected_profit": profit,
                    }
                )
            )
    return pd.concat(all_rows, axis=0, ignore_index=True)


def export_targets(
    profit_long: pd.DataFrame,
    p_by_treatment: pd.DataFrame,
    topks: List[float] | None = None,
    csv_path=None,
) -> pd.DataFrame:
    topks = topks or [0.05, 0.10, 0.20, 0.30]
    best = (
        profit_long.sort_values(["scenario", "customer_id", "expected_profit"], ascending=[True, True, False])
        .groupby(["scenario", "customer_id"], as_index=False)
        .first()
    )
    best[["recommend_offer", "recommend_channel"]] = best["treatment"].str.split(" \\| ", expand=True)

    treat_cols = [c for c in p_by_treatment.columns if c != "customer_id"]
    tmp = p_by_treatment.copy()
    tmp["best_p_treatment"] = tmp[treat_cols].idxmax(axis=1)
    tmp["best_p"] = tmp[treat_cols].max(axis=1)
    best = best.merge(tmp[["customer_id", "best_p_treatment", "best_p"]], on="customer_id", how="left")

    out_frames = []
    for sc, g in best.groupby("scenario", sort=False):
        g = g.copy()
        g["target_profit_positive"] = g["expected_profit"] > 0
        g = g.sort_values("expected_profit", ascending=False)
        n = len(g)
        for k in topks:
            m = max(1, int(round(n * k)))
            flag = np.zeros(n, dtype=bool)
            flag[:m] = True
            g[f"target_top_{int(k * 100)}pct"] = flag
        out_frames.append(g)

    out = pd.concat(out_frames, axis=0, ignore_index=True)
    if csv_path:
        out.to_csv(csv_path, index=False)
    return out


def summarize_profit(best_targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc, g in best_targets.groupby("scenario", sort=False):
        n = len(g)
        rows.append(
            {
                "scenario": sc,
                "n_customers": int(n),
                "share_profit_positive": float(g["target_profit_positive"].mean()),
                "mean_expected_profit": float(g["expected_profit"].mean()),
                "median_expected_profit": float(g["expected_profit"].median()),
                "mean_expected_profit_top5pct": float(g.loc[g["target_top_5pct"], "expected_profit"].mean()),
                "mean_expected_profit_top10pct": float(g.loc[g["target_top_10pct"], "expected_profit"].mean()),
                "mean_expected_profit_top20pct": float(g.loc[g["target_top_20pct"], "expected_profit"].mean()),
                "mean_expected_profit_top30pct": float(g.loc[g["target_top_30pct"], "expected_profit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def fit_full_models(df: pd.DataFrame, base_features: List[str]) -> Tuple[Pipeline, Pipeline, List[str]]:
    T = build_joint_treatment(df)
    return fit_propensity_and_outcome_models(df, base_features, T)


def oof_predictions_outcome(
    df: pd.DataFrame, base_features: List[str], treatments: List[str], k_fold: int = K_FOLD
) -> pd.DataFrame:
    X = df[base_features].copy()
    y = df["conversion"].astype(int).to_numpy()
    T = build_joint_treatment(df).astype(str).to_numpy()
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=RANDOM_STATE)
    pred_cols = {t: np.zeros(len(df), dtype=float) for t in treatments}

    for train_idx, val_idx in skf.split(X, y):
        df_tr = df.iloc[train_idx]
        T_tr = pd.Series(T[train_idx])
        prop, out_m, _ = fit_propensity_and_outcome_models(df_tr, base_features, T_tr)
        X_val = df.iloc[val_idx][base_features].copy()
        for t in treatments:
            Xo = X_val.copy()
            Xo["treatment"] = t
            pred_cols[t][val_idx] = out_m.predict_proba(Xo)[:, 1]

    out = pd.DataFrame(pred_cols)
    out.insert(0, "customer_id", df["customer_id"].to_numpy())
    return out


def scenario_agreement_rate(best_a: pd.DataFrame, best_b: pd.DataFrame, scenario: str) -> float:
    a = best_a.loc[best_a["scenario"] == scenario, ["customer_id", "treatment"]]
    b = best_b.loc[best_b["scenario"] == scenario, ["customer_id", "treatment"]]
    m = a.merge(b, on="customer_id", suffixes=("_a", "_b"))
    if m.empty:
        return float("nan")
    return float((m["treatment_a"] == m["treatment_b"]).mean())


def benchmark_policies(
    df: pd.DataFrame,
    p_oof: pd.DataFrame,
    scenario: CostScenario,
) -> pd.DataFrame:
    """OOF 予測に対する単純ルールの期待利益（mid 相当の1シナリオ）。"""
    treat_cols = [c for c in p_oof.columns if c != "customer_id"]
    history = df[["customer_id", "history"]].merge(p_oof, on="customer_id")
    rows = []

    def profit_for_treatment(t: str) -> float:
        offer, channel = t.split(" | ")
        p = history[t].to_numpy()
        v = history["history"].to_numpy()
        cost_o = v * offer_cost_multiplier(offer, scenario)
        cost_c = v * scenario.channel_rank_cost.get(channel, 0.0)
        return float(np.mean(p * v - cost_o - cost_c))

    # 常に各処置
    for t in treat_cols:
        rows.append({"policy": f"固定: {t}", "mean_profit": profit_for_treatment(t)})

    # ランダム処置（期待値：処置分布をデータの出現割合で）
    rng = np.random.RandomState(RANDOM_STATE)
    T = build_joint_treatment(df).to_numpy()
    mc = []
    for _ in range(200):
        pick = rng.choice(T, size=len(df), replace=True)
        prof = []
        for i, t in enumerate(pick):
            offer, ch = t.split(" | ")
            p = p_oof.iloc[i][t]
            v = df.iloc[i]["history"]
            co = v * offer_cost_multiplier(offer, scenario)
            cc = v * scenario.channel_rank_cost.get(ch, 0.0)
            prof.append(p * v - co - cc)
        mc.append(float(np.mean(prof)))
    rows.append({"policy": "ランダム処置（観測分布・モンテカルロ）", "mean_profit": float(np.mean(mc))})

    # 個別最適（OOF）
    mat = history[treat_cols].to_numpy()
    best_idx = np.argmax(mat, axis=1)
    prof_indiv = []
    for i, j in enumerate(best_idx):
        t = treat_cols[j]
        offer, ch = t.split(" | ")
        p = mat[i, j]
        v = df.iloc[i]["history"]
        co = v * offer_cost_multiplier(offer, scenario)
        cc = v * scenario.channel_rank_cost.get(ch, 0.0)
        prof_indiv.append(p * v - co - cc)
    rows.append({"policy": "個別最適（OOF予測）", "mean_profit": float(np.mean(prof_indiv))})

    return pd.DataFrame(rows).sort_values("mean_profit", ascending=False)


def greedy_constrained_targets(
    best_unconstrained: pd.DataFrame,
    df: pd.DataFrame,
    scenario: str = "mid_cost",
    max_contacts: int | None = CONSTRAINT_MAX_CONTACTS,
    budget_history_units: float | None = CONSTRAINT_BUDGET_HISTORY_UNITS,
    channel_cap: Dict[str, int] | None = CONSTRAINT_CHANNEL_CAP,
) -> pd.DataFrame:
    g = best_unconstrained.loc[best_unconstrained["scenario"] == scenario].copy()
    g = g.sort_values("expected_profit", ascending=False)
    g = g.merge(df[["customer_id", "history", "channel"]], on="customer_id", how="left")

    channel_cap = channel_cap or {}
    taken = 0
    spend = 0.0
    ch_counts: Dict[str, int] = {k: 0 for k in channel_cap}
    flags = []
    for _, row in g.iterrows():
        ch = str(row["recommend_channel"])
        ok = True
        if max_contacts is not None and taken >= max_contacts:
            ok = False
        if budget_history_units is not None and spend + row["history"] > budget_history_units:
            ok = False
        if ch in channel_cap and ch_counts.get(ch, 0) >= channel_cap[ch]:
            ok = False
        flags.append(ok)
        if ok:
            taken += 1
            spend += float(row["history"])
            if ch in ch_counts:
                ch_counts[ch] += 1
    g["selected_constrained"] = flags
    return g


def plot_policy_figures(
    best_full: pd.DataFrame,
    best_oof: pd.DataFrame,
    benchmarks: pd.DataFrame,
    scenario: str = "mid_cost",
) -> None:
    bf = best_full.loc[best_full["scenario"] == scenario]
    bo = best_oof.loc[best_oof["scenario"] == scenario]
    vc = bf["treatment"].value_counts(normalize=True)
    plt.figure(figsize=(8, 4))
    vc.plot(kind="bar", color="steelblue")
    plt.ylabel("割合")
    plt.xlabel("推奨施策（フル学習）")
    plt.title(f"推奨施策の分布（{scenario}）")
    plt.xticks(rotation=45, ha="right")
    save_fig(FIG_IMP / "policy_reco_dist_full.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(bf["expected_profit"].values[:8000], bo["expected_profit"].values[:8000], alpha=0.2, s=8)
    plt.xlabel("期待利益（フル学習）")
    plt.ylabel("期待利益（OOF）")
    plt.title("顧客別期待利益：フル学習 vs OOF（先頭8000点）")
    save_fig(FIG_IMP / "profit_full_vs_oof_scatter.png")

    plt.figure(figsize=(7, 4))
    plt.barh(benchmarks["policy"], benchmarks["mean_profit"], color="coral")
    plt.xlabel("平均期待利益（OOF・mid_cost 仮定）")
    plt.title("ベンチマーク政策の比較")
    save_fig(FIG_IMP / "policy_benchmarks_bar.png")


def summarize_segments(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    best_targets_mid: pd.DataFrame,
) -> pd.DataFrame:
    from analytics.constants import CHANNEL_JP, OFFER_JP

    tmp = df[
        ["customer_id", "conversion", "history", "recency", "zip_code", "is_referral", "used_discount", "used_bogo"]
    ].copy()
    tmp["segment"] = cluster_labels.to_numpy()
    tmp = tmp.merge(
        best_targets_mid[["customer_id", "recommend_offer", "recommend_channel", "expected_profit"]],
        on="customer_id",
        how="left",
    )
    seg = (
        tmp.groupby("segment")
        .agg(
            n=("customer_id", "count"),
            cvr=("conversion", "mean"),
            mean_history=("history", "mean"),
            mean_recency=("recency", "mean"),
            share_referral=("is_referral", "mean"),
            share_used_discount=("used_discount", "mean"),
            share_used_bogo=("used_bogo", "mean"),
            mean_expected_profit=("expected_profit", "mean"),
        )
        .reset_index()
        .sort_values("mean_expected_profit", ascending=False)
    )
    mode_offer = tmp.groupby("segment")["recommend_offer"].agg(lambda s: s.value_counts().index[0]).rename("segment_offer")
    mode_channel = tmp.groupby("segment")["recommend_channel"].agg(lambda s: s.value_counts().index[0]).rename("segment_channel")
    seg = seg.merge(mode_offer, on="segment").merge(mode_channel, on="segment")
    seg["segment_offer"] = seg["segment_offer"].map(OFFER_JP).fillna(seg["segment_offer"])
    seg["segment_channel"] = seg["segment_channel"].map(CHANNEL_JP).fillna(seg["segment_channel"])
    return seg


def segment_best_treatment_by_mean(
    df: pd.DataFrame,
    out_model: Pipeline,
    base_features: List[str],
    treatments: List[str],
    segment_labels: np.ndarray,
) -> pd.DataFrame:
    from analytics.constants import CHANNEL_JP, OFFER_JP

    p_by_treat = policy_recommendations(df, out_model, base_features, treatments)
    p_by_treat["segment"] = segment_labels.astype(int)
    treat_cols = [c for c in p_by_treat.columns if c not in ["customer_id", "segment"]]
    seg_mean = p_by_treat.groupby("segment")[treat_cols].mean()
    best = seg_mean.idxmax(axis=1).rename("best_treatment_cvr")
    best_p = seg_mean.max(axis=1).rename("mean_p_best_cvr")
    out = pd.concat([best, best_p], axis=1).reset_index()

    def to_jp(t: str) -> str:
        offer, channel = t.split(" | ")
        return f"{OFFER_JP.get(offer, offer)} / {CHANNEL_JP.get(channel, channel)}"

    out["best_treatment_cvr_jp"] = out["best_treatment_cvr"].map(to_jp)
    return out
