"""report.md チェックリスト向け：相関・重回帰（OLS）と分かりやすい図。"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import Patch

from analytics.config import ART_DIR, FIG_IMP
from analytics.figures_jp import init_plot_style, save_fig


def correlation_numeric_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """数値・二値列の Pearson 相関。日本語ラベル付き CSV とヒートマップを出力。"""
    init_plot_style()
    cols = {
        "recency": "前回からの月数\n(recency)",
        "history": "過去購入価値\n(history)",
        "used_discount": "過去割引利用",
        "used_bogo": "過去BOGO利用",
        "is_referral": "紹介流入",
        "conversion": "購入(0/1)",
    }
    sub = df[list(cols.keys())].astype(float)
    corr = sub.corr(method="pearson")
    corr.to_csv(ART_DIR / "correlation_numeric.csv", encoding="utf-8")

    plt.figure(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    labels = [cols[c] for c in corr.columns]
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Pearson r"},
    )
    plt.title(
        "数値・二値項目の相関行列（下三角のみ表示）\n"
        f"母集団 n={len(df):,}（Pearson・欠損なし）",
        fontsize=12,
    )
    plt.xticks(rotation=0, ha="center", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    save_fig(FIG_IMP / "corr_numeric_detailed.png")

    plt.figure(figsize=(8, 5.5))
    ax = plt.gca()
    conv = df["conversion"].astype(int)
    c_colors = np.where(conv == 1, "#c44e52", "#4c72b0")
    ax.scatter(df["recency"], df["history"], c=c_colors, alpha=0.22, s=10, edgecolors="none")
    ax.set_xlabel("前回購入からの月数（recency）")
    ax.set_ylabel("過去購入価値 history")
    ax.set_title(
        "recency と history の散布図（色=購入有無）\n"
        "青=未購入 / 赤=購入（点の重なりあり・透明度で表現）",
        fontsize=11,
    )
    h = [
        Patch(facecolor="#4c72b0", label="未購入 (0)"),
        Patch(facecolor="#c44e52", label="購入 (1)"),
    ]
    ax.legend(handles=h, title="conversion", loc="upper right")
    save_fig(FIG_IMP / "scatter_recency_history_by_conversion.png")
    return corr


def ols_history_regression(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """目的変数 history、説明変数は主要数値・二値。"""
    y = df["history"].astype(float)
    X = df[["recency", "used_discount", "used_bogo", "is_referral"]].astype(float)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type="HC1")
    summ = model.summary().as_text()
    (ART_DIR / "ols_history_summary.txt").write_text(summ, encoding="utf-8")

    ci = model.conf_int()
    coef = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "t": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_low": ci[0].values,
            "ci_high": ci[1].values,
        }
    )
    coef.to_csv(ART_DIR / "ols_history_coefficients.csv", index=False)

    init_plot_style()
    terms = coef[coef["term"] != "const"].copy()
    plt.figure(figsize=(7, 4))
    y_pos = np.arange(len(terms))
    err = [terms["coef"] - terms["ci_low"], terms["ci_high"] - terms["coef"]]
    plt.barh(y_pos, terms["coef"], xerr=err, color="#4c72b0", capsize=3)
    plt.yticks(y_pos, terms["term"])
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("係数（ロバストSEに基づく95%CI）")
    plt.title("重回帰（OLS）: 目的変数 = history\n説明: recency, used_discount, used_bogo, is_referral")
    save_fig(FIG_IMP / "ols_history_coefficients.png")

    diag = {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_stat": float(model.fvalue) if model.fvalue is not None else None,
        "nobs": int(model.nobs),
    }
    (ART_DIR / "ols_history_diag.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return model


def run_stats_extra(df: pd.DataFrame) -> dict:
    """相関図表と OLS を一括実行。"""
    correlation_numeric_analysis(df)
    ols_history_regression(df)
    return {"correlation_csv": str(ART_DIR / "correlation_numeric.csv")}
