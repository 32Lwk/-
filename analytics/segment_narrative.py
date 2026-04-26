from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from scipy import sparse as sp_sparse

from analytics.config import RANDOM_STATE


def _to_dense(a: Any) -> np.ndarray:
    if sp_sparse.issparse(a):
        return a.toarray()
    return np.asarray(a)


def segment_feature_zprofiles(
    df: pd.DataFrame,
    segment_labels: np.ndarray,
    feature_cols: List[str],
) -> Dict[int, str]:
    """各セグメントについて、全体平均からの z スコアが大きい特徴を1行で要約。"""
    tmp = df[feature_cols].copy()
    tmp["segment"] = segment_labels
    overall_mean = tmp[feature_cols].mean()
    overall_std = tmp[feature_cols].std().replace(0, np.nan)

    narratives: Dict[int, str] = {}
    for seg in sorted(np.unique(segment_labels)):
        sub = tmp.loc[tmp["segment"] == seg, feature_cols]
        m = sub.mean()
        z_signed = ((m - overall_mean) / overall_std).dropna()
        top_idx = z_signed.abs().sort_values(ascending=False).head(3).index
        parts = [f"{idx}: z={float(z_signed.loc[idx]):+.2f}" for idx in top_idx]
        narratives[int(seg)] = "主な特徴（全体比z）: " + ", ".join(parts)
    return narratives


def _segment_permutation_narratives(
    df: pd.DataFrame,
    segment_labels: np.ndarray,
    model: Any,
    max_per_seg: int = 1500,
    n_repeats: int = 4,
) -> Dict[int, str]:
    X = df.drop(columns=["conversion"])
    y = df["conversion"].astype(int)
    rng = np.random.RandomState(RANDOM_STATE)
    out: Dict[int, str] = {}
    names = X.columns.tolist()
    for seg in np.unique(segment_labels):
        m = segment_labels == seg
        Xs = X.loc[m]
        ys = y.loc[m]
        if len(Xs) < 30:
            out[int(seg)] = "置換重要度: サンプル不足"
            continue
        if len(Xs) > max_per_seg:
            idx = rng.choice(len(Xs), max_per_seg, replace=False)
            Xs = Xs.iloc[idx]
            ys = ys.iloc[idx]
        imp = permutation_importance(
            model,
            Xs,
            ys,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        order = np.argsort(imp.importances_mean)[::-1][:3]
        parts = [f"{names[i]}: {imp.importances_mean[i]:.4f}" for i in order]
        out[int(seg)] = "置換重要度（上位）: " + ", ".join(parts)
    return out


def _segment_shap_narratives(
    df: pd.DataFrame,
    segment_labels: np.ndarray,
    model: Any,
    max_background: int = 200,
    max_per_seg: int = 400,
) -> Dict[int, str]:
    """LogisticRegression パイプラインは前処理後空間で LinearExplainer を優先。"""
    try:
        import shap
    except ImportError:
        return {int(s): "" for s in np.unique(segment_labels)}

    X = df.drop(columns=["conversion"])
    rng = np.random.RandomState(RANDOM_STATE + 7)
    out: Dict[int, str] = {}
    steps = getattr(model, "named_steps", None)
    if steps is not None and "pre" in steps and "clf" in steps:
        pre, clf = steps["pre"], steps["clf"]
        cls_name = clf.__class__.__name__
        if cls_name == "LogisticRegression":
            bg_n = min(max_background, len(X))
            bg_idx = rng.choice(len(X), bg_n, replace=False)
            X_bg = X.iloc[bg_idx]
            try:
                X_bg_t = _to_dense(pre.transform(X_bg))
                feat_names = list(pre.get_feature_names_out())
                explainer = shap.LinearExplainer(clf, X_bg_t)
            except Exception:
                return {int(s): "" for s in np.unique(segment_labels)}

            for seg in np.unique(segment_labels):
                m = segment_labels == seg
                Xs = X.loc[m]
                if len(Xs) < 40:
                    out[int(seg)] = ""
                    continue
                take = min(max_per_seg, len(Xs))
                if len(Xs) > take:
                    idx = rng.choice(len(Xs), take, replace=False)
                    Xs = Xs.iloc[idx]
                try:
                    Xs_t = _to_dense(pre.transform(Xs))
                    sv = explainer(Xs_t)
                    arr = np.asarray(sv.values)
                    if arr.ndim == 3:
                        vals = np.abs(arr[:, :, 1]).mean(axis=0)
                    elif arr.ndim == 2:
                        vals = np.abs(arr).mean(axis=0)
                    else:
                        out[int(seg)] = ""
                        continue
                    if len(feat_names) != len(vals):
                        out[int(seg)] = ""
                        continue
                    order = np.argsort(vals)[::-1][:3]
                    parts = [f"{feat_names[i]}: {float(vals[i]):.4f}" for i in order]
                    out[int(seg)] = "SHAP線形（|φ|平均・購入クラス）: " + ", ".join(parts)
                except Exception:
                    out[int(seg)] = ""
            return out

    # フォールバック: 汎用 Explainer（失敗しやすい）
    bg_n = min(max_background, len(X))
    bg_idx = rng.choice(len(X), bg_n, replace=False)
    X_bg = X.iloc[bg_idx]
    feat_names = X.columns.tolist()
    for seg in np.unique(segment_labels):
        m = segment_labels == seg
        Xs = X.loc[m]
        if len(Xs) < 40:
            out[int(seg)] = ""
            continue
        take = min(80, len(Xs))
        if len(Xs) > take:
            idx = rng.choice(len(Xs), take, replace=False)
            Xs = Xs.iloc[idx]
        try:
            explainer = shap.Explainer(
                model.predict_proba,
                X_bg,
                max_evals=min(25 * X_bg.shape[1], 500),
            )
            sv = explainer(Xs)
            arr = np.asarray(sv.values)
            if arr.ndim == 3:
                vals = np.abs(arr[:, :, 1]).mean(axis=0)
            else:
                vals = np.abs(arr).mean(axis=0)
            if len(feat_names) != len(vals):
                out[int(seg)] = ""
                continue
            order = np.argsort(vals)[::-1][:3]
            parts = [f"{feat_names[i]}: {float(vals[i]):.4f}" for i in order]
            out[int(seg)] = "SHAP（|寄与|平均）: " + ", ".join(parts)
        except Exception:
            out[int(seg)] = ""
    return out


def build_segment_narratives(
    df: pd.DataFrame,
    segment_labels: np.ndarray,
    model: Optional[Any] = None,
    include_shap: bool = True,
) -> pd.DataFrame:
    feats = ["recency", "history", "used_discount", "used_bogo", "is_referral"]
    ztext = segment_feature_zprofiles(df, segment_labels, feats)
    perm: Dict[int, str] = {}
    shap_t: Dict[int, str] = {}
    if model is not None:
        perm = _segment_permutation_narratives(df, segment_labels, model)
        if include_shap:
            shap_t = _segment_shap_narratives(df, segment_labels, model)

    rows = []
    for seg in sorted(np.unique(segment_labels)):
        rows.append(
            {
                "segment": int(seg),
                "narrative_z": ztext.get(int(seg), ""),
                "narrative_perm": perm.get(int(seg), ""),
                "narrative_shap": shap_t.get(int(seg), ""),
            }
        )
    return pd.DataFrame(rows)
