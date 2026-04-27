"""BASE 特徴に TF-IDF+SVD（LSI 類似）潜在ベクトルを足した分類の比較（同一ホールドアウト）。"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from analytics.config import ART_DIR, FIG_IMP, HOLDOUT_RATIO, RANDOM_STATE
from analytics.figures_jp import init_plot_style, save_fig
from analytics.preprocess import build_preprocessor_dense
from analytics.row_text import rows_to_english_text
from analytics.utils_common import topk_metrics


def run_lsi_logreg_ablation(df: pd.DataFrame, n_svd: int = 24, max_tfidf_features: int = 400) -> pd.DataFrame:
    """訓練にのみフィットした TF-IDF+SVD を logreg に連結。"""
    X = df.drop(columns=["conversion"])
    y = df["conversion"].astype(int).to_numpy()
    texts = rows_to_english_text(df)

    idx = np.arange(len(df))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx,
        y,
        test_size=HOLDOUT_RATIO,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train = X.iloc[idx_train]
    X_test = X.iloc[idx_test]
    text_train = texts.iloc[idx_train]
    text_test = texts.iloc[idx_test]

    tfidf = TfidfVectorizer(max_features=max_tfidf_features, min_df=3, ngram_range=(1, 2))
    T_tr = tfidf.fit_transform(text_train)
    T_te = tfidf.transform(text_test)

    n_comp = min(n_svd, max(2, T_tr.shape[1] - 1), max(2, T_tr.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    Z_tr = svd.fit_transform(T_tr)
    Z_te = svd.transform(T_te)

    pre = build_preprocessor_dense(df)
    pre.fit(X_train)
    A_tr = pre.transform(X_train)
    A_te = pre.transform(X_test)
    if hasattr(A_tr, "toarray"):
        A_tr = A_tr.toarray()
        A_te = A_te.toarray()

    X_aug_tr = np.hstack([A_tr, Z_tr])
    X_aug_te = np.hstack([A_te, Z_te])

    clf_base = LogisticRegression(max_iter=4000, solver="lbfgs")
    clf_base.fit(A_tr, y_train)
    proba_base = clf_base.predict_proba(A_te)[:, 1]

    clf_aug = LogisticRegression(max_iter=4000, solver="lbfgs", C=0.5)
    clf_aug.fit(X_aug_tr, y_train)
    proba_aug = clf_aug.predict_proba(X_aug_te)[:, 1]

    rows = []
    for name, proba in [("base_only", proba_base), ("base_plus_lsi_tfidf_svd", proba_aug)]:
        tk = topk_metrics(y_test, proba, ks=[0.05, 0.10, 0.20])
        rows.append(
            {
                "variant": name,
                "auc": float(roc_auc_score(y_test, proba)),
                "brier": float(brier_score_loss(y_test, proba)),
                "average_precision": float(average_precision_score(y_test, proba)),
                **tk,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ART_DIR / "model_eval_ablation_lsi.csv", index=False)

    ev = {
        "svd_n_components": int(n_comp),
        "tfidf_max_features": max_tfidf_features,
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }
    (ART_DIR / "lsi_tfidf_diag.json").write_text(
        json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    init_plot_style()
    plot_df = out.set_index("variant")[
        ["auc", "brier", "average_precision", "top_5pct_cvr", "top_10pct_cvr"]
    ].T
    plt.figure(figsize=(8, 5))
    x = np.arange(len(plot_df))
    w = 0.35
    plt.bar(x - w / 2, plot_df.loc[:, "base_only"], width=w, label="BASE のみ（数値+カテゴリ）", color="#4c72b0")
    plt.bar(
        x + w / 2,
        plot_df.loc[:, "base_plus_lsi_tfidf_svd"],
        width=w,
        label=f"BASE + TF-IDF+SVD（次元={n_comp}）",
        color="#55a868",
    )
    for i in range(len(plot_df)):
        v0 = float(plot_df.iloc[i, 0])
        v1 = float(plot_df.iloc[i, 1])
        plt.text(i - w / 2, v0 + 0.015, f"{v0:.3f}", ha="center", fontsize=7)
        plt.text(i + w / 2, v1 + 0.015, f"{v1:.3f}", ha="center", fontsize=7)
    plt.xticks(x, ["AUC↑", "Brier↓", "AP↑", "Top5%CVR", "Top10%CVR"], rotation=15, ha="right")
    plt.ylabel("値（Brier は低いほど良）")
    plt.title(
        "同一ホールドアウトでの比較：ロジスティック回帰\n"
        "潜在特徴 = 行英語テンプレの TF-IDF → TruncatedSVD（学習は訓練のみ）",
        fontsize=11,
    )
    plt.legend(loc="lower right", fontsize=9)
    save_fig(FIG_IMP / "ablation_base_vs_lsi_logreg.png")

    return out
