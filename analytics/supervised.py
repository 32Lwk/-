from __future__ import annotations

from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from analytics.config import ART_DIR, FIG_DIR, FIG_IMP, HOLDOUT_RATIO, RANDOM_STATE
from analytics.figures_jp import save_fig
from analytics.preprocess import build_preprocessor, build_preprocessor_dense
from analytics.utils_common import topk_metrics, topk_precision_recall


def train_predictive_models(df: pd.DataFrame) -> Dict[str, object]:
    X = df.drop(columns=["conversion"])
    y = df["conversion"].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=HOLDOUT_RATIO, random_state=RANDOM_STATE, stratify=y
    )

    pre, _, _ = build_preprocessor(df)
    pre_dense = build_preprocessor_dense(df)

    models: Dict[str, Pipeline] = {
        "logreg": Pipeline(
            steps=[
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ]
        ),
        "hgb": Pipeline(
            steps=[
                ("pre", pre_dense),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        random_state=RANDOM_STATE,
                        max_depth=6,
                        learning_rate=0.08,
                        max_iter=300,
                    ),
                ),
            ]
        ),
    }

    results: Dict[str, object] = {"split": {"n_train": int(len(X_train)), "n_test": int(len(X_test))}}
    eval_rows: Dict[str, Dict[str, float]] = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        ap = average_precision_score(y_test, proba)
        topk = topk_metrics(y_test, proba, ks=[0.05, 0.10, 0.20, 0.30])
        eval_rows[name] = {
            "auc": float(auc_score),
            "brier": float(brier),
            "average_precision": float(ap),
            **topk,
        }

        RocCurveDisplay.from_predictions(y_test, proba)
        plt.title(f"ROC曲線（{name} / AUC={auc_score:.3f}）")
        plt.xlabel("偽陽性率（FPR）")
        plt.ylabel("真陽性率（TPR）")
        save_fig(FIG_DIR / f"roc_{name}.png")

        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        plt.figure(figsize=(5, 5))
        plt.plot(mean_pred, frac_pos, "o-", label=name)
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("予測購入確率（平均）")
        plt.ylabel("実測購入率")
        plt.title(f"キャリブレーション（{name} / Brier={brier:.3f}）")
        plt.legend()
        save_fig(FIG_DIR / f"calibration_{name}.png")

        disp = PrecisionRecallDisplay.from_predictions(y_test, proba, name=name)
        disp.ax_.set_title(f"PR曲線（{name} / AP={ap:.3f}）")
        disp.ax_.set_xlabel("再現率（Recall）")
        disp.ax_.set_ylabel("適合率（Precision）")
        save_fig(FIG_IMP / f"pr_curve_{name}.png")

    pr_table = topk_precision_recall(y_test, models["logreg"].predict_proba(X_test)[:, 1], ks=[0.05, 0.1, 0.2, 0.3])
    pr_table.to_csv(ART_DIR / "topk_precision_recall_logreg.csv", index=False)

    best = max(eval_rows.items(), key=lambda kv: kv[1]["auc"])[0]
    joblib.dump(models[best], ART_DIR / "model.joblib")

    results["model_eval"] = eval_rows
    results["best_model"] = best
    results["X_test"] = X_test
    results["y_test"] = y_test
    results["models"] = models
    return results
