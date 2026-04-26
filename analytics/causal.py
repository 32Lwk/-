from __future__ import annotations

from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from analytics.config import (
    ART_DIR,
    BOOTSTRAP_B,
    BOOTSTRAP_N,
    FIG_IMP,
    HOLDOUT_RATIO,
    PROPENSITY_CLIP,
    RANDOM_STATE,
)
from analytics.figures_jp import save_fig
from analytics.preprocess import build_causal_preprocessor, build_outcome_preprocessor


def build_joint_treatment(df: pd.DataFrame) -> pd.Series:
    return (df["offer"].astype(str) + " | " + df["channel"].astype(str)).astype(str)


def fit_propensity_and_outcome_models(
    df: pd.DataFrame, base_features: List[str], treatment: pd.Series
) -> Tuple[Pipeline, Pipeline, List[str]]:
    X_base = df[base_features].copy()
    T = treatment.astype(str)
    y = df["conversion"].astype(int).to_numpy()

    pre = build_causal_preprocessor(base_features)
    prop = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs"),
            ),
        ]
    )
    prop.fit(X_base, T)

    X_out = X_base.copy()
    X_out["treatment"] = T
    pre_out = build_outcome_preprocessor(base_features)
    out = Pipeline(
        steps=[
            ("pre", pre_out),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    out.fit(X_out, y)

    classes = list(prop.named_steps["clf"].classes_)
    return prop, out, classes


def dr_estimate_mu(
    df: pd.DataFrame,
    prop_model: Pipeline,
    out_model: Pipeline,
    base_features: List[str],
    treatments: List[str],
    clip: float = PROPENSITY_CLIP,
) -> pd.DataFrame:
    X_base = df[base_features].copy()
    T = build_joint_treatment(df).astype(str).to_numpy()
    y = df["conversion"].astype(int).to_numpy()

    e = prop_model.predict_proba(X_base)
    class_order = list(prop_model.named_steps["clf"].classes_)
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    rows = []
    for t in treatments:
        X_out = X_base.copy()
        X_out["treatment"] = t
        mu = out_model.predict_proba(X_out)[:, 1]

        idx = class_to_idx[t]
        e_t = np.clip(e[:, idx], clip, 1.0)
        w = (T == t).astype(float) / e_t
        mu_dr = float(np.mean(mu + w * (y - mu)))
        rows.append(
            {
                "treatment": t,
                "mu_dr": mu_dr,
                "mu_model": float(np.mean(mu)),
                "support": float((T == t).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mu_dr", ascending=False)


def policy_recommendations(
    df: pd.DataFrame,
    out_model: Pipeline,
    base_features: List[str],
    treatments: List[str],
) -> pd.DataFrame:
    X_base = df[base_features].copy()
    preds = {}
    for t in treatments:
        X_out = X_base.copy()
        X_out["treatment"] = t
        preds[t] = out_model.predict_proba(X_out)[:, 1]
    P = pd.DataFrame(preds)
    P.insert(0, "customer_id", df["customer_id"].to_numpy())
    return P


def propensity_diagnostics(
    df: pd.DataFrame,
    prop_model: Pipeline,
    base_features: List[str],
    treatments: List[str],
    clip: float = PROPENSITY_CLIP,
) -> Tuple[pd.DataFrame, List[str]]:
    X_base = df[base_features].copy()
    T = build_joint_treatment(df).astype(str).to_numpy()
    e = prop_model.predict_proba(X_base)
    class_order = list(prop_model.named_steps["clf"].classes_)
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    rows = []
    plot_paths = []
    for t in treatments:
        idx = class_to_idx[t]
        e_t = e[:, idx]
        e_clipped = np.clip(e_t, clip, 1.0)
        mask = T == t
        w = np.zeros_like(e_t)
        w[mask] = 1.0 / e_clipped[mask]
        ess = float((w.sum() ** 2) / np.maximum((w ** 2).sum(), 1e-12))
        frac_small = float(np.mean(e_t < clip))
        rows.append(
            {
                "treatment": t,
                "ess_ipw": ess,
                "frac_e_below_clip": frac_small,
                "mean_e_among_assigned": float(e_t[mask].mean()) if mask.any() else float("nan"),
                "n_assigned": int(mask.sum()),
            }
        )

    diag_df = pd.DataFrame(rows)
    # 代表的な3処置のヒストグラム
    for t in treatments[: min(3, len(treatments))]:
        idx = class_to_idx[t]
        e_t = e[:, idx]
        plt.figure(figsize=(6, 4))
        plt.hist(e_t, bins=40, alpha=0.75, color="steelblue")
        plt.xlabel("傾向スコア（処置が選ばれる確率）")
        plt.ylabel("顧客数")
        plt.title(f"傾向スコア分布：{t.replace(' | ', ' / ')}")
        safe = t.replace(" | ", "_").replace(" ", "")[:40]
        pth = FIG_IMP / f"propensity_hist_{safe}.png"
        save_fig(pth)
        plot_paths.append(str(pth.name))

    return diag_df, plot_paths


def _bootstrap_one_dr(
    seed: int,
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    base_features: List[str],
    treatments: List[str],
    bootstrap_n: int,
) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    y = df_train["conversion"].astype(int).to_numpy()
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    half = max(1, bootstrap_n // 2)
    take0 = rng.choice(idx0, size=min(len(idx0), half), replace=True)
    take1 = rng.choice(idx1, size=min(len(idx1), bootstrap_n - len(take0)), replace=True)
    sub_idx = np.concatenate([take0, take1])
    sub = df_train.iloc[sub_idx].copy()
    T_sub = build_joint_treatment(sub)
    prop_b, out_b, _ = fit_propensity_and_outcome_models(sub, base_features, T_sub)
    dr = dr_estimate_mu(df_eval, prop_b, out_b, base_features, treatments)
    return {row["treatment"]: row["mu_dr"] for _, row in dr.iterrows()}


def bootstrap_dr_table(
    df: pd.DataFrame,
    base_features: List[str],
    treatments: List[str],
    n_boot: int = BOOTSTRAP_B,
    bootstrap_n: int = BOOTSTRAP_N,
    n_jobs: int = -1,
) -> pd.DataFrame:
    df_tr, df_te = train_test_split(
        df,
        test_size=HOLDOUT_RATIO,
        random_state=RANDOM_STATE,
        stratify=df["conversion"],
    )
    records = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(_bootstrap_one_dr)(RANDOM_STATE + 1000 + b, df_tr, df_te, base_features, treatments, bootstrap_n)
        for b in range(n_boot)
    )
    rows = []
    for t in treatments:
        vals = [r.get(t, float("nan")) for r in records]
        arr = np.array(vals, dtype=float)
        rows.append(
            {
                "treatment": t,
                "mu_dr_p025": float(np.nanpercentile(arr, 2.5)),
                "mu_dr_p50": float(np.nanpercentile(arr, 50)),
                "mu_dr_p975": float(np.nanpercentile(arr, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def plot_dr_bootstrap_forest(dr_point: pd.DataFrame, dr_boot: pd.DataFrame, path=None) -> None:
    path = path or (FIG_IMP / "dr_bootstrap_ci.png")
    m = dr_point.merge(dr_boot, on="treatment", how="left")
    m = m.sort_values("mu_dr", ascending=True)
    y_pos = np.arange(len(m))
    plt.figure(figsize=(8, max(4, len(m) * 0.35)))
    plt.errorbar(
        m["mu_dr"].values,
        y_pos,
        xerr=[
            m["mu_dr"].values - m["mu_dr_p025"].values,
            m["mu_dr_p975"].values - m["mu_dr"].values,
        ],
        fmt="o",
        capsize=3,
    )
    plt.yticks(y_pos, [t.replace(" | ", " / ") for t in m["treatment"]])
    plt.xlabel("推定購入確率（DR）とブートストラップ区間")
    plt.title("DR点推定とブートストラップ95%区間（ホールドアウト上・部分標本再学習）")
    save_fig(path)
