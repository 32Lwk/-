from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import umap
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "exercise.csv"
FIG_DIR = ROOT / "figures"
FIG_HTML_DIR = ROOT / "figures_html"
ART_DIR = ROOT / "artifacts"

def set_japanese_font() -> None:
    # Pick the first available Japanese-capable font to avoid tofu glyphs.
    candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "MS UI Gothic", "Noto Sans CJK JP", "IPAexGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            return
    # Fallback (may not support Japanese fully, but avoids findfont spam)
    plt.rcParams["font.family"] = "DejaVu Sans"


# Global style: readability first (may override font)
sns.set_theme(style="whitegrid")
# Re-apply Japanese font after seaborn theme sets defaults.
set_japanese_font()

COLOR_PALETTE = {
    "購入": "#0072B2",  # Okabe-Ito blue
    "未購入": "#999999",
}


OFFER_JP = {
    "Discount": "割引",
    "Buy One Get One": "1つ買うと1つ無料",
    "No Offer": "オファーなし",
}
CHANNEL_JP = {
    "Web": "Web",
    "Phone": "電話",
    "Multichannel": "マルチチャネル",
}
ZIP_JP = {
    "Urban": "都市",
    "Surburban": "郊外",
    "Rural": "農村",
}


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["購入"] = out["conversion"].map({0: "未購入", 1: "購入"}).astype(str)
    out["オファー"] = out["offer"].map(OFFER_JP).fillna(out["offer"]).astype(str)
    out["チャネル"] = out["channel"].map(CHANNEL_JP).fillna(out["channel"]).astype(str)
    out["居住エリア"] = out["zip_code"].map(ZIP_JP).fillna(out["zip_code"]).astype(str)
    out["紹介流入"] = out["is_referral"].map({0: "なし", 1: "あり"}).astype(str)
    out["過去_割引利用"] = out["used_discount"].map({0: "なし", 1: "あり"}).astype(str)
    out["過去_BOGO利用"] = out["used_bogo"].map({0: "なし", 1: "あり"}).astype(str)
    return out


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_HTML_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_plotly(fig, stem: str) -> Tuple[Path, Path]:
    html_path = FIG_HTML_DIR / f"{stem}.html"
    png_path = FIG_DIR / f"{stem}.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    # Kaleido is required for this
    fig.write_image(str(png_path), scale=2)
    return png_path, html_path


def save_plotly_png(fig, stem: str) -> Path:
    png_path = FIG_DIR / f"{stem}.png"
    fig.write_image(str(png_path), scale=2)
    return png_path


def topk_metrics(y_true: np.ndarray, scores: np.ndarray, ks: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(y_true)
    order = np.argsort(scores)[::-1]
    for k in ks:
        m = max(1, int(round(n * k)))
        idx = order[:m]
        out[f"top_{int(k*100)}pct_cvr"] = float(y_true[idx].mean())
    return out


@dataclass(frozen=True)
class CostScenario:
    name: str
    r_bogo: float
    r_disc: float
    channel_rank_cost: Dict[str, float]  # relative multipliers to value proxy


DEFAULT_SCENARIOS: List[CostScenario] = [
    CostScenario(
        name="low_cost",
        r_bogo=0.10,
        r_disc=0.20,
        channel_rank_cost={"Web": 0.00, "Multichannel": 0.01, "Phone": 0.02},
    ),
    CostScenario(
        name="mid_cost",
        r_bogo=0.20,
        r_disc=0.40,
        channel_rank_cost={"Web": 0.00, "Multichannel": 0.02, "Phone": 0.04},
    ),
    CostScenario(
        name="high_cost",
        r_bogo=0.30,
        r_disc=0.60,
        channel_rank_cost={"Web": 0.00, "Multichannel": 0.03, "Phone": 0.06},
    ),
]


def offer_cost_multiplier(offer: str, scenario: CostScenario) -> float:
    if offer == "No Offer":
        return 0.0
    if offer == "Buy One Get One":
        return scenario.r_bogo
    if offer == "Discount":
        return scenario.r_disc
    raise ValueError(f"Unknown offer: {offer}")


def read_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Add stable row id for downstream artifacts
    df.insert(0, "customer_id", np.arange(len(df), dtype=np.int64))
    return df


def validate_schema(df: pd.DataFrame) -> None:
    expected = [
        "customer_id",
        "recency",
        "history",
        "used_discount",
        "used_bogo",
        "zip_code",
        "is_referral",
        "channel",
        "offer",
        "conversion",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def basic_qc_tables(df: pd.DataFrame) -> Dict[str, object]:
    qc: Dict[str, object] = {}
    qc["n_rows"] = int(len(df))
    qc["missing_by_col"] = df.isna().sum().to_dict()
    qc["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    qc["unique_zip_code"] = sorted(df["zip_code"].astype(str).unique().tolist())
    qc["unique_channel"] = sorted(df["channel"].astype(str).unique().tolist())
    qc["unique_offer"] = sorted(df["offer"].astype(str).unique().tolist())
    qc["conversion_rate"] = float(df["conversion"].mean())
    return qc


def make_eda_figures(df: pd.DataFrame) -> None:
    d = add_display_columns(df)
    # Distributions
    for col, jp in [("recency", "前回購入からの月数"), ("history", "過去購入価値（history）")]:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], bins=50, kde=True)
        plt.xlabel(jp)
        plt.ylabel("顧客数（n）")
        plt.title(f"{jp}の分布")
        save_fig(FIG_DIR / f"dist_{col}.png")

        plt.figure(figsize=(7, 4))
        sns.boxplot(x="購入", y=col, data=d, order=["未購入", "購入"])
        plt.xlabel("購入")
        plt.ylabel(jp)
        plt.title(f"{jp}（購入別）")
        save_fig(FIG_DIR / f"box_{col}_by_conversion.png")

    # CVR by categorical
    for col, col_jp in [
        ("居住エリア", "居住エリア"),
        ("チャネル", "チャネル"),
        ("オファー", "オファー"),
        ("過去_割引利用", "過去の割引利用"),
        ("過去_BOGO利用", "過去のBOGO利用"),
        ("紹介流入", "紹介流入（リファラル）"),
    ]:
        g = d.groupby(col)["conversion"].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=g.index.astype(str), y=g.values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("コンバージョン率（CVR）")
        plt.xlabel(col_jp)
        plt.title(f"{col_jp}別CVR（棒の高さ=CVR、nは表で併記）")
        safe = {
            "居住エリア": "zip_code",
            "チャネル": "channel",
            "オファー": "offer",
            "過去_割引利用": "used_discount",
            "過去_BOGO利用": "used_bogo",
            "紹介流入": "is_referral",
        }[col]
        save_fig(FIG_DIR / f"cvr_by_{safe}.png")

    # Crosstabs
    piv = (
        d.pivot_table(index="オファー", columns="チャネル", values="conversion", aggfunc="mean")
        .reindex(index=["オファーなし", "割引", "1つ買うと1つ無料"])
    )
    plt.figure(figsize=(7, 4))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis")
    plt.title("CVRヒートマップ（オファー×チャネル）")
    plt.xlabel("チャネル")
    plt.ylabel("オファー")
    save_fig(FIG_DIR / "heat_cvr_offer_x_channel.png")

    # Composition (pie): overall vs converted
    for subset_name, dd in [("overall", d), ("converted", d[d["conversion"] == 1])]:
        n_total = int(len(dd))
        for col, slug in [("オファー", "offer"), ("チャネル", "channel"), ("居住エリア", "zip_code")]:
            vc = dd[col].value_counts().reset_index()
            vc.columns = [col, "count"]
            fig = px.pie(
                vc,
                names=col,
                values="count",
                title=(
                    f"{subset_name}の構成比（{col}）".replace("overall", "全体").replace("converted", "購入者")
                    + f"（n={n_total}）"
                ),
                hole=0.35,
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            save_plotly_png(fig, stem=f"pie_{subset_name}_{slug}")


def chi2_tests(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    y = df["conversion"].astype(int)
    for col in cols:
        tab = pd.crosstab(df[col], y)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p, dof, expected = chi2_contingency(tab.values)
        n = tab.values.sum()
        # Cramer's V for effect size
        r, k = tab.shape
        denom = n * (min(r - 1, k - 1))
        cramer_v = float(math.sqrt(chi2 / denom)) if denom > 0 else float("nan")
        rows.append(
            {
                "feature": col,
                "chi2": float(chi2),
                "dof": int(dof),
                "p_value": float(p),
                "cramers_v": cramer_v,
                "n": int(n),
            }
        )
    out = pd.DataFrame(rows).sort_values("p_value")
    return out


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    y_col = "conversion"
    ignore = {"customer_id", y_col}
    features = [c for c in df.columns if c not in ignore]
    numeric = ["recency", "history", "used_discount", "used_bogo", "is_referral"]
    categorical = [c for c in features if c not in numeric]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ]
    )
    return pre, numeric, categorical


def build_preprocessor_dense(df: pd.DataFrame) -> ColumnTransformer:
    # For estimators that require dense input (e.g., HistGradientBoostingClassifier)
    y_col = "conversion"
    ignore = {"customer_id", y_col}
    features = [c for c in df.columns if c not in ignore]
    numeric = ["recency", "history", "used_discount", "used_bogo", "is_referral"]
    categorical = [c for c in features if c not in numeric]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        sparse_threshold=0.0,
    )


def train_predictive_models(df: pd.DataFrame) -> Dict[str, object]:
    X = df.drop(columns=["conversion"])
    y = df["conversion"].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pre, _, _ = build_preprocessor(df)
    pre_dense = build_preprocessor_dense(df)

    models: Dict[str, Pipeline] = {
        "logreg": Pipeline(
            steps=[
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        n_jobs=None,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "hgb": Pipeline(
            steps=[
                ("pre", pre_dense),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        random_state=42,
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
    saved_model_name = "logreg"

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        topk = topk_metrics(y_test, proba, ks=[0.05, 0.10, 0.20, 0.30])
        eval_rows[name] = {"auc": float(auc_score), "brier": float(brier), **topk}

        # Figures: ROC + Calibration
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

    # Save best model by AUC
    best = max(eval_rows.items(), key=lambda kv: kv[1]["auc"])[0]
    saved_model_name = best
    joblib.dump(models[best], ART_DIR / "model.joblib")

    results["model_eval"] = eval_rows
    results["best_model"] = saved_model_name
    results["X_test"] = X_test
    results["y_test"] = y_test
    return results


def make_latent_spaces(df: pd.DataFrame, preprocessor: ColumnTransformer) -> Dict[str, np.ndarray]:
    X = df.drop(columns=["conversion"])
    X_mat = preprocessor.fit_transform(X)

    pca = PCA(n_components=3, random_state=42)
    Z_pca = pca.fit_transform(X_mat)
    explained = pca.explained_variance_ratio_.tolist()
    with open(ART_DIR / "pca_explained_variance.json", "w", encoding="utf-8") as f:
        json.dump({"explained_variance_ratio": explained}, f, ensure_ascii=False, indent=2)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    Z_umap = reducer.fit_transform(X_mat)

    return {"pca3": Z_pca, "umap3": Z_umap}


def make_latent_spaces_2d(df: pd.DataFrame, preprocessor: ColumnTransformer) -> Dict[str, np.ndarray]:
    X = df.drop(columns=["conversion"])
    X_mat = preprocessor.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(X_mat)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    Z_umap = reducer.fit_transform(X_mat)
    return {"pca2": Z_pca, "umap2": Z_umap}


def plot_latent_2d(df: pd.DataFrame, Z: np.ndarray, name: str, color: str = "conversion") -> None:
    d = add_display_columns(df)
    tmp = d[["customer_id", "購入", "オファー", "チャネル"]].copy()
    tmp[f"{name}_1"] = Z[:, 0]
    tmp[f"{name}_2"] = Z[:, 1]
    color_col = "購入" if color == "conversion" else color
    fig = px.scatter(
        tmp.sample(n=min(len(tmp), 12000), random_state=42),
        x=f"{name}_1",
        y=f"{name}_2",
        color=color_col,
        symbol="オファー",
        title=f"潜在空間（2D）: {name}（色={color_col}）",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(
        legend_title_text="凡例",
        xaxis_title=f"{name}_1",
        yaxis_title=f"{name}_2",
    )
    # 2DはPNGのみで十分（Markdown埋め込み優先）
    save_plotly_png(fig, stem=f"latent2d_{name}_color_{color}")


def plot_latent_3d(
    df: pd.DataFrame, Z: np.ndarray, name: str, color: str = "conversion"
) -> None:
    d = add_display_columns(df)
    tmp = d[["customer_id", "conversion", "購入", "オファー", "チャネル"]].copy()
    tmp[f"{name}_1"] = Z[:, 0]
    tmp[f"{name}_2"] = Z[:, 1]
    tmp[f"{name}_3"] = Z[:, 2]
    color_col = "購入" if color == "conversion" else color
    fig = px.scatter_3d(
        tmp.sample(n=min(len(tmp), 8000), random_state=42),
        x=f"{name}_1",
        y=f"{name}_2",
        z=f"{name}_3",
        color=color_col,
        symbol="オファー",
        title=f"潜在空間（3D）: {name}（色={color_col}）",
        opacity=0.7,
    )
    fig.update_layout(
        legend_title_text="凡例",
        scene=dict(
            xaxis_title=f"{name}_1",
            yaxis_title=f"{name}_2",
            zaxis_title=f"{name}_3",
        ),
    )
    save_plotly(fig, stem=f"latent3d_{name}_color_{color}")


def choose_k_silhouette(Z: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    # Use a subset to keep silhouette fast
    n = len(Z)
    if n > 12000:
        idx = np.random.RandomState(42).choice(n, size=12000, replace=False)
        Z_use = Z[idx]
    else:
        Z_use = Z
    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(Z_use)
        s = silhouette_score(Z_use, labels)
        if s > best_score:
            best_score = float(s)
            best_k = k
    return best_k


def cluster_latent(Z: np.ndarray, name: str) -> np.ndarray:
    k = choose_k_silhouette(Z)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(Z)
    joblib.dump({"k": k, "model": km}, ART_DIR / f"kmeans_{name}.joblib")
    return labels


def cluster_latent_k(Z: np.ndarray, name: str, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(Z)
    joblib.dump({"k": k, "model": km}, ART_DIR / f"kmeans_{name}.joblib")
    return labels


def summarize_segments(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    best_targets_mid: pd.DataFrame,
) -> pd.DataFrame:
    tmp = df[["customer_id", "conversion", "history", "recency", "zip_code", "is_referral", "used_discount", "used_bogo"]].copy()
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
    # Segment-level recommended treatment by most frequent choice
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
    """
    For each segment, choose treatment that maximizes mean predicted P(conv) within the segment.
    (CVR-optimal policy proxy)
    """
    p_by_treat = policy_recommendations(df, out_model, base_features, treatments)
    p_by_treat["segment"] = segment_labels.astype(int)
    treat_cols = [c for c in p_by_treat.columns if c not in ["customer_id", "segment"]]
    seg_mean = p_by_treat.groupby("segment")[treat_cols].mean()
    best = seg_mean.idxmax(axis=1).rename("best_treatment_cvr")
    best_p = seg_mean.max(axis=1).rename("mean_p_best_cvr")
    out = pd.concat([best, best_p], axis=1).reset_index()
    # map to jp label
    def to_jp(t: str) -> str:
        offer, channel = t.split(" | ")
        return f"{OFFER_JP.get(offer, offer)} / {CHANNEL_JP.get(channel, channel)}"

    out["best_treatment_cvr_jp"] = out["best_treatment_cvr"].map(to_jp)
    return out


def build_joint_treatment(df: pd.DataFrame) -> pd.Series:
    return (df["offer"].astype(str) + " | " + df["channel"].astype(str)).astype(str)


def fit_propensity_and_outcome_models(
    df: pd.DataFrame, base_features: List[str], treatment: pd.Series
) -> Tuple[Pipeline, Pipeline, List[str]]:
    # Use base features X (exclude offer/channel themselves) for propensity.
    X_base = df[base_features].copy()
    T = treatment.astype(str)
    y = df["conversion"].astype(int).to_numpy()

    num_cols = ["recency", "history", "used_discount", "used_bogo", "is_referral"]
    cat_cols = [c for c in base_features if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    # Multinomial propensity
    prop = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    multi_class="multinomial",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    prop.fit(X_base, T)

    # Outcome model: include treatment as a categorical feature
    X_out = X_base.copy()
    X_out["treatment"] = T
    pre_out = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols + ["treatment"]),
        ]
    )
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
) -> pd.DataFrame:
    """
    Doubly-robust estimates of E[Y(t)] for each treatment t.
    Returns dataframe with columns: treatment, mu_dr
    """
    X_base = df[base_features].copy()
    T = build_joint_treatment(df).astype(str).to_numpy()
    y = df["conversion"].astype(int).to_numpy()

    # Propensity scores (n, K)
    e = prop_model.predict_proba(X_base)
    class_order = list(prop_model.named_steps["clf"].classes_)
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    rows = []
    for t in treatments:
        # Outcome regression mu_hat(t, x)
        X_out = X_base.copy()
        X_out["treatment"] = t
        mu = out_model.predict_proba(X_out)[:, 1]

        # IPW correction
        idx = class_to_idx[t]
        # Propensity trimming to stabilize DR when some treatments are rare / near-deterministic
        e_t = np.clip(e[:, idx], 0.01, 1.0)
        w = (T == t).astype(float) / e_t
        mu_dr = float(np.mean(mu + w * (y - mu)))
        rows.append({"treatment": t, "mu_dr": mu_dr, "mu_model": float(np.mean(mu)), "support": float((T == t).mean())})
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
    long = pd.concat(all_rows, axis=0, ignore_index=True)
    return long


def export_targets(
    profit_long: pd.DataFrame,
    p_by_treatment: pd.DataFrame,
    topks: List[float] = [0.05, 0.10, 0.20, 0.30],
) -> pd.DataFrame:
    # Choose best treatment per customer per scenario
    best = (
        profit_long.sort_values(["scenario", "customer_id", "expected_profit"], ascending=[True, True, False])
        .groupby(["scenario", "customer_id"], as_index=False)
        .first()
    )
    # Add offer/channel columns
    best[["recommend_offer", "recommend_channel"]] = best["treatment"].str.split(" \\| ", expand=True)

    # For convenience: add baseline max probability treatment as well
    treat_cols = [c for c in p_by_treatment.columns if c != "customer_id"]
    tmp = p_by_treatment.copy()
    tmp["best_p_treatment"] = tmp[treat_cols].idxmax(axis=1)
    tmp["best_p"] = tmp[treat_cols].max(axis=1)
    best = best.merge(tmp[["customer_id", "best_p_treatment", "best_p"]], on="customer_id", how="left")

    # Mark target rules
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
            g[f"target_top_{int(k*100)}pct"] = flag
        out_frames.append(g)

    out = pd.concat(out_frames, axis=0, ignore_index=True)
    out.to_csv(ART_DIR / "promo_targets.csv", index=False)
    return out


def summarize_profit(best_targets: pd.DataFrame) -> pd.DataFrame:
    """
    best_targets: output of export_targets (best per customer per scenario, with target flags)
    """
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


def write_report(
    qc: Dict[str, object],
    model_eval: Dict[str, Dict[str, float]],
    best_model: str,
    dr_table: pd.DataFrame,
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
) -> None:
    # Map labels in tables to Japanese for readability
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

    md: List[str] = []
    md.append("# 期待利益最大化のための分析レポート（分析担当向け）\n")

    md.append("## 0. 要約（結論）\n")
    top_treatment = str(policy_mid_summary.get("top_treatment", ""))
    top_share = float(policy_mid_summary.get("top_share", 0.0))
    md.append(f"- **顧客別最適ポリシー（mid_cost）で最頻の推奨施策**: `{top_treatment.replace(' | ', ' / ')}`（{top_share:.1%}）\n")
    md.append("- **示唆**: CVRが高い施策（例: 割引）は存在するが、コスト仮定が大きいと期待利益では負ける可能性がある。\n")
    md.append("- **推奨アクション**: 低コストなインセンティブへ置換し、上位スコア層（TopK%）でA/BテストしてROIが合う範囲を探索。\n")

    md.append("\n## 1. 目的と前提\n")
    md.append("- **目的**: `conversion`（購入）を増やしつつ、オファー/チャネルを含む施策設計で**期待利益**を最大化する。\n")
    md.append("- **価値のproxy**: `history` を購入価値のproxyとし、期待価値を \(P(購入|x)\\times history\) と置く。\n")
    md.append("- **コスト仮定（感度分析）**: Discount/BOGOのコストを `history` 相対でレンジ設定（詳細は `analysis.py`）。\n")
    md.append("- **注意**: 施策効果は観察データであるため、推定は因果を保証しない（後述）。\n")

    md.append("\n## 2. データ概要と品質\n")
    md.append(f"- **行数**: {qc['n_rows']}\n")
    md.append(f"- **全体CVR**: {qc['conversion_rate']:.4f}\n")
    md.append(f"- **居住エリア（zip_code）**: {[ZIP_JP.get(x, x) for x in qc['unique_zip_code']]}\n")
    md.append(f"- **チャネル**: {[CHANNEL_JP.get(x, x) for x in qc['unique_channel']]}\n")
    md.append(f"- **オファー**: {[OFFER_JP.get(x, x) for x in qc['unique_offer']]}\n")

    md.append("\n### 図（分布と購入差）\n")
    md.append("- **図中の n は顧客数（=行数）**を表す。\n\n")
    md.append("![前回購入からの月数の分布](figures/dist_recency.png)\n\n")
    md.append("![過去購入価値（history）の分布](figures/dist_history.png)\n\n")
    md.append("![前回購入からの月数（購入別）](figures/box_recency_by_conversion.png)\n\n")
    md.append("![過去購入価値（history）（購入別）](figures/box_history_by_conversion.png)\n\n")

    md.append("## 3. 基礎集計（CVR）と解釈\n")
    md.append("この章での **n は各カテゴリに属する顧客数（該当行数）**を表す。\n\n")
    md.append("### 3.1 オファー別/チャネル別\n")
    md.append("\n**オファー別CVR**\n\n")
    md.append("| オファー | n | CVR |\n| --- | ---: | ---: |\n")
    for k, v in cvr_offer_jp.items():
        md.append(f"| {k} | {int(n_offer_jp.get(k, 0))} | {v:.4f} |\n")
    md.append("\n**チャネル別CVR**\n\n")
    md.append("| チャネル | n | CVR |\n| --- | ---: | ---: |\n")
    for k, v in cvr_channel_jp.items():
        md.append(f"| {k} | {int(n_channel_jp.get(k, 0))} | {v:.4f} |\n")

    md.append("\n### 3.2 オファー×チャネル\n")
    md.append("施策の効きはチャネルと相互作用し得るため、平均CVRのヒートマップで俯瞰する。\n\n")
    md.append("![CVRヒートマップ（オファー×チャネル）](figures/heat_cvr_offer_x_channel.png)\n\n")
    md.append("- 表の `n/CVR` は **セル内の顧客数 n と、そのセルの平均CVR** を表す。\n\n")

    md.append("### 3.2.1 構成比（円グラフ）\n")
    md.append("円グラフは「構成比」を直感的に把握する用途に限定して利用する（比較は棒/積み上げが基本）。\n\n")
    md.append("- 円グラフのタイトル末尾の `n=...` は **円グラフに含めた母集団の顧客数**（全体 or 購入者）を表す。\n\n")
    md.append("![全体の構成比（オファー）](figures/pie_overall_offer.png)\n\n")
    md.append("![購入者の構成比（オファー）](figures/pie_converted_offer.png)\n\n")
    md.append("![全体の構成比（チャネル）](figures/pie_overall_channel.png)\n\n")
    md.append("![購入者の構成比（チャネル）](figures/pie_converted_channel.png)\n\n")
    md.append("![全体の構成比（居住エリア）](figures/pie_overall_zip_code.png)\n\n")
    md.append("![購入者の構成比（居住エリア）](figures/pie_converted_zip_code.png)\n\n")
    md.append("| オファー \\ チャネル | マルチチャネル（n/CVR） | 電話（n/CVR） | Web（n/CVR） |\n")
    md.append("| --- | --- | --- | --- |\n")
    for offer in cvr_offer_channel_jp.index:
        row = cvr_offer_channel_jp.loc[offer]
        nrow = n_offer_channel_jp.loc[offer]
        def fmt(cell_n, cell_cvr):
            if pd.isna(cell_cvr):
                return "-"
            return f"{int(cell_n)}/{float(cell_cvr):.4f}"
        md.append(
            f"| {offer} | {fmt(nrow.get('マルチチャネル', 0), row.get('マルチチャネル', float('nan')))} | {fmt(nrow.get('電話', 0), row.get('電話', float('nan')))} | {fmt(nrow.get('Web', 0), row.get('Web', float('nan')))} |\n"
        )

    md.append("\n### 3.3 比率差検定（カイ二乗）\n")
    md.append("関連の強さ（効果量）も併記し、強い特徴を優先的に追加分析・施策仮説に使う。\n\n")
    md.append("- 結果CSV: `artifacts/chi2_tests.csv`\n\n")
    md.append("| 特徴量 | n | p値 | Cramer's V |\n")
    md.append("| --- | ---: | ---: | ---: |\n")
    rename_feat = {
        "offer": "オファー",
        "channel": "チャネル",
        "zip_code": "居住エリア",
        "used_discount": "過去の割引利用",
        "used_bogo": "過去のBOGO利用",
        "is_referral": "紹介流入",
    }
    for _, r in chi2_table.head(10).iterrows():
        md.append(f"| {rename_feat.get(r['feature'], r['feature'])} | {int(r['n'])} | {r['p_value']:.3e} | {r['cramers_v']:.4f} |\n")

    md.append("\n## 4. 予測モデル（ターゲティング）\n")
    md.append("施策対象を絞るため、購入確率モデルを構築し、TopK%でのCVR改善（リフト）を重視して評価する。\n\n")
    md.append(f"- **採用モデル（AUC最大）**: {best_model}\n\n")
    md.append(f"- **学習/評価データ数（holdout）**: n_train={qc.get('n_train','?')}, n_test={qc.get('n_test','?')}（層化）\n\n")
    md.append("### 4.1 評価（holdout・層化）\n")
    md.append("| model | AUC | Brier | Top5% CVR | Top10% CVR | Top20% CVR | Top30% CVR |\n")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for name, row in model_eval.items():
        md.append(
            f"| {name} | {row['auc']:.4f} | {row['brier']:.4f} | {row['top_5pct_cvr']:.4f} | {row['top_10pct_cvr']:.4f} | {row['top_20pct_cvr']:.4f} | {row['top_30pct_cvr']:.4f} |\n"
        )

    md.append("\n### 4.2 図（ROC / キャリブレーション）\n")
    md.append("![ROC曲線（ロジスティック回帰）](figures/roc_logreg.png)\n\n")
    md.append("![キャリブレーション（ロジスティック回帰）](figures/calibration_logreg.png)\n\n")
    md.append("![ROC曲線（勾配ブースティング）](figures/roc_hgb.png)\n\n")
    md.append("![キャリブレーション（勾配ブースティング）](figures/calibration_hgb.png)\n\n")

    md.append("## 5. 潜在空間（2D）によるセグメント発見（主）\n")
    md.append("潜在空間は3Dより2Dの方が静的レポートで読みやすいため、**2Dを主**として提示する。\n\n")

    md.append("### 5.1 PCA(2D)\n")
    md.append("![潜在空間（PCA 2D、色=購入）](figures/latent2d_pca2_color_conversion.png)\n\n")

    md.append("### 5.2 UMAP(2D)\n")
    md.append("![潜在空間（UMAP 2D、色=購入）](figures/latent2d_umap2_color_conversion.png)\n\n")

    md.append("### 5.3 付録：3D可視化（任意）\n")
    md.append("- 3D（HTML）は操作性は高いが、静的文書では見にくいため付録扱いにする。\n")
    md.append("- [PCA 3Dを開く](figures_html/latent3d_pca3_color_conversion.html)\n")
    md.append("- [UMAP 3Dを開く](figures_html/latent3d_umap3_color_conversion.html)\n\n")

    md.append("### 5.4 セグメント（UMAP 3D上のクラスタ）\n")
    md.append("![潜在空間（UMAP 3D、色=セグメント）](figures/latent3d_umap3_color_segment.png)\n\n")
    md.append(f"- 3D（HTML）: [セグメント3Dを開く](figures_html/latent3d_umap3_color_segment.html)\n\n")
    md.append("- セグメント要約: `artifacts/segment_summary.csv`（利益最適の要約）\n")
    md.append("- セグメント別CVR最適: `artifacts/segment_best_cvr.csv`\n\n")
    md.append("| セグメント | 件数 | CVR | mean(history) | mean(recency) | mean(期待利益) | 推奨オファー（利益最適） | 推奨チャネル（利益最適） |\n")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
    for _, r in segment_summary.iterrows():
        md.append(
            f"| {int(r['segment'])} | {int(r['n'])} | {r['cvr']:.4f} | {r['mean_history']:.2f} | {r['mean_recency']:.2f} | {r['mean_expected_profit']:.2f} | {r['segment_offer']} | {r['segment_channel']} |\n"
        )

    md.append("\n**セグメント別の最適施策（CVR最適 vs 利益最適）**\n")
    md.append("同じセグメントでも、目的がCVR最大化か期待利益最大化かで推奨施策は変わり得る。\n\n")
    md.append("- **CVR最適（コスト無視の参考）**: `artifacts/segment_best_cvr.csv`\n")
    md.append("- **利益最適（コスト込み）**: `artifacts/segment_summary.csv`（mid_cost）\n\n")
    md.append("**セグメント別：CVR最適（予測）**\n\n")
    md.append("| セグメント | n | 推奨施策（CVR最適） | 平均予測購入確率 |\n")
    md.append("| --- | ---: | --- | ---: |\n")
    # Load precomputed CVR-optimal by segment if present (keeps report generation simple)
    seg_best_path = ART_DIR / "segment_best_cvr.csv"
    if seg_best_path.exists():
        seg_best = pd.read_csv(seg_best_path)
        for _, r in seg_best.sort_values("segment").iterrows():
            seg = int(r["segment"])
            nseg = int(segment_summary.set_index("segment").loc[seg, "n"]) if seg in set(segment_summary["segment"]) else 0
            md.append(f"| {seg} | {nseg} | {r['best_treatment_cvr_jp']} | {float(r['mean_p_best_cvr']):.4f} |\n")
    md.append("\n**解釈**:\n")
    md.append("- CVR最適では多くの場合「割引」が選ばれやすい（購入確率を押し上げるため）。\n")
    md.append("- 一方、利益最適ではコスト仮定によって「オファーなし」が選ばれることがあり、ここが“施策設計（コスト）”のボトルネックを示す。\n\n")

    md.append("### 5.5 セグメント別の深掘り（なぜそうなるか・何をすべきか）\n")
    md.append(
        "各セグメントの特徴量（`history`/`recency`/`is_referral`/過去利用）から、\n"
        "推奨施策がそうなる理由と、より具体的な施策案（訴求・オファー設計・チャネル）を整理する。\n\n"
    )
    seg_best_path = ART_DIR / "segment_best_cvr.csv"
    seg_best_df = pd.read_csv(seg_best_path) if seg_best_path.exists() else pd.DataFrame()
    seg_best_df = seg_best_df.set_index("segment") if not seg_best_df.empty else seg_best_df

    for _, r in segment_summary.sort_values("mean_expected_profit", ascending=False).iterrows():
        seg = int(r["segment"])
        n = int(r["n"])
        md.append(f"#### セグメント{seg}（n={n}）\n")
        md.append(
            f"- **CVR**: {r['cvr']:.4f}\n"
            f"- **平均history（価値proxy）**: {r['mean_history']:.2f}\n"
            f"- **平均recency（月）**: {r['mean_recency']:.2f}\n"
            f"- **紹介流入率**: {r['share_referral']:.3f}\n"
            f"- **過去の割引利用率**: {r['share_used_discount']:.3f}\n"
            f"- **過去のBOGO利用率**: {r['share_used_bogo']:.3f}\n"
        )
        md.append(
            f"- **利益最適（mid_cost）**: {r['segment_offer']} / {r['segment_channel']}（平均期待利益={r['mean_expected_profit']:.2f}）\n"
        )
        if not seg_best_df.empty and seg in seg_best_df.index:
            b = seg_best_df.loc[seg]
            md.append(
                f"- **CVR最適（参考）**: {b['best_treatment_cvr_jp']}（平均予測購入確率={float(b['mean_p_best_cvr']):.4f}）\n"
            )

        md.append("\n**考察（なぜそうなるか）**\n")
        md.append(
            "- CVR最適は購入確率のみを見るため、反応が出やすい『割引』が選ばれやすい。\n"
            "- 利益最適は『購入確率×history−コスト』のため、historyが大きいほど割引コスト（history×割引率）も大きくなり、コスト仮定次第で『オファーなし』が有利になり得る。\n"
            "- 特に高historyセグメントでは、値引き率が同じでもコスト総額が膨らむため、利益最大化はより保守的になりやすい。\n"
        )

        md.append("\n**具体施策（訴求・オファー設計・チャネル）**\n")
        md.append(
            "- **訴求**: 価格訴求だけでなく、価値訴求（限定性/利便性/新商品/まとめ買い等）でCVRを改善し、値引き依存を下げる。\n"
            "- **オファー設計**: 割引率ではなく『小額固定』『ポイント』『送料無料』等に置換し、`history`に比例してコストが膨らみにくい設計にする。\n"
            "- **チャネル**: 本分析の前提ではWebが低コスト。電話は高コスト想定のため、電話は高反応が見込める層に限定してテストする。\n"
        )
        md.append("\n")

    md.append("\n## 6. 施策効果の推定（傾向スコア + DR）\n")
    md.append(
        "観察データでは、オファー/チャネルが顧客特性に依存して配布されている可能性がある。"
        "そこで (オファー,チャネル) を多値介入として扱い、\n"
        "- 傾向スコア \(e(t|x)\)（多項ロジット）\n"
        "- アウトカムモデル \(\mu(t,x)\)\n"
        "を組み合わせたDR推定により、平均購入確率 \(E[Y(t)]\) を推定する。\n"
        "推定の安定化のため、傾向スコアは下限0.01でトリミングした。\n"
    )
    md.append("\n### 6.1 DR推定（上位のみ抜粋）\n")
    md.append("| 施策（オファー/チャネル） | n | 推定購入確率（DR） | アウトカムモデル平均 | 出現割合 |\n")
    md.append("| --- | ---: | ---: | ---: | ---: |\n")
    for _, r in dr_table.head(10).iterrows():
        offer, channel = str(r["treatment"]).split(" | ")
        t_jp = f"{OFFER_JP.get(offer, offer)} / {CHANNEL_JP.get(channel, channel)}"
        n_t = int(round(float(r["support"]) * qc["n_rows"]))
        md.append(f"| `{t_jp}` | {n_t} | {r['mu_dr']:.4f} | {r['mu_model']:.4f} | {r['support']:.4f} |\n")

    md.append("\n## 7. 期待利益最大化ポリシー\n")
    md.append(
        "購入価値のproxyとして `history` を用い、期待利益を\n"
        "\\[ E[profit]=P(購入|x,施策)\\times history - cost(offer) - cost(channel) \\]\n"
        "で定義。コストはレンジで感度分析し、結論の頑健性を確認する。\n"
    )
    md.append("- 配信対象リスト: `artifacts/promo_targets.csv`（上位K%フラグ + 期待利益>0フラグ + 推奨施策）\n\n")
    md.append("### 7.1 コストシナリオ別サマリ（顧客別最適）\n\n")
    md.append("| シナリオ | n | 利益>0割合 | 平均期待利益 | 中央値期待利益 | Top5%平均 | Top10%平均 | Top20%平均 | Top30%平均 |\n")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for _, r in profit_summary.iterrows():
        md.append(
            f"| {r['scenario']} | {int(r['n_customers'])} | {r['share_profit_positive']:.4f} | {r['mean_expected_profit']:.2f} | {r['median_expected_profit']:.2f} | {r['mean_expected_profit_top5pct']:.2f} | {r['mean_expected_profit_top10pct']:.2f} | {r['mean_expected_profit_top20pct']:.2f} | {r['mean_expected_profit_top30pct']:.2f} |\n"
        )

    md.append("\n## 8. 施策提案（考察）\n")
    md.append(
        "- CVR観点では割引が高いが、コスト仮定が大きいと期待利益で不利になり得る。\n"
        "- まずは低コストなインセンティブ設計へ置換し、TopK%でA/BテストしてROIが正となる範囲を探索する。\n"
        "- セグメント運用では、値引き額を増やさずに訴求/クリエイティブ最適化でCVR改善を狙う（`segment_summary.csv`）。\n"
    )

    md.append("\n## 9. 限界と次の検証\n")
    md.append("- 本分析は観察データに基づくため未観測交絡の可能性がある。\n")
    md.append("- DR推定は頑健性を高めるが、完全な因果推定を保証しない。\n")
    md.append("- 推奨ポリシーはA/Bテストで検証し、効果（CVR/利益）とコストを実測で更新する。\n")

    md.append("\n## 10. 追加で改善すると良い点（今後の拡張）\n")
    md.append(
        "- **割引コストの現実値**（割引率・粗利率・BOGO原価）を入れ、期待利益の絶対値を実務に合わせて再推定する。\n"
        "- **特徴量重要度/説明**: ロジスティック回帰の係数（標準化）やPermutation Importanceで“なぜ高確度か”を補強する。\n"
        "- **セグメント安定性**: UMAP/KMeansの乱数やkを変えたときの頑健性（再現性）を確認し、運用に耐えるセグメントを採用する。\n"
        "- **オファーの因果推定強化**: 多値介入のDRに加え、重みの分布・有効サンプルサイズ（ESS）を出して推定の信頼性を可視化する。\n"
        "- **ターゲティング運用**: 上位K%だけでなく“期待利益>0”の閾値運用（接触上限/予算制約）も含めた最適化に拡張する。\n"
    )

    md.append("\n## 付録：生成物\n")
    md.append("- `analysis.py` / `requirements.txt`\n")
    md.append("- `figures/`（PNG）・`figures_html/`（3D HTML）\n")
    md.append("- `artifacts/processed.csv`, `artifacts/model.joblib`, `artifacts/promo_targets.csv`\n")
    md.append("- `artifacts/chi2_tests.csv`, `artifacts/dr_treatment_effects.csv`, `artifacts/segment_summary.csv`\n")

    (ROOT / "final_report.md").write_text("".join(md), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = read_data()
    validate_schema(df)

    qc = basic_qc_tables(df)
    (ART_DIR / "qc_summary.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

    make_eda_figures(df)

    cvr_offer = df.groupby("offer")["conversion"].mean().sort_values(ascending=False)
    n_offer = df.groupby("offer")["conversion"].size().sort_values(ascending=False)
    cvr_channel = df.groupby("channel")["conversion"].mean().sort_values(ascending=False)
    n_channel = df.groupby("channel")["conversion"].size().sort_values(ascending=False)
    cvr_offer_channel = (
        df.pivot_table(index="offer", columns="channel", values="conversion", aggfunc="mean")
        .reindex(index=["No Offer", "Discount", "Buy One Get One"])
    )
    n_offer_channel = (
        df.pivot_table(index="offer", columns="channel", values="conversion", aggfunc="size")
        .reindex(index=["No Offer", "Discount", "Buy One Get One"])
    )

    chi2_table = chi2_tests(
        df,
        cols=["offer", "channel", "zip_code", "used_discount", "used_bogo", "is_referral"],
    )
    chi2_table.to_csv(ART_DIR / "chi2_tests.csv", index=False)

    # Predictive baseline
    res = train_predictive_models(df)
    best_model = res["best_model"]
    model_eval = res["model_eval"]
    qc["n_train"] = res.get("split", {}).get("n_train")
    qc["n_test"] = res.get("split", {}).get("n_test")

    # Latent spaces
    pre, _, _ = build_preprocessor(df)
    Zs = make_latent_spaces(df, preprocessor=pre)
    plot_latent_3d(df, Zs["pca3"], "pca3", color="conversion")
    plot_latent_3d(df, Zs["umap3"], "umap3", color="conversion")

    # 2D latent spaces (primary for readability)
    Z2 = make_latent_spaces_2d(df, preprocessor=pre)
    plot_latent_2d(df, Z2["pca2"], "pca2", color="conversion")
    plot_latent_2d(df, Z2["umap2"], "umap2", color="conversion")

    # Cluster in latent space (UMAP 3D) for segment discovery
    umap_seg = cluster_latent_k(Zs["umap3"], name="umap3", k=4)
    labels_df = pd.DataFrame({"customer_id": df["customer_id"].to_numpy(), "umap3_segment": umap_seg})
    labels_df.to_csv(ART_DIR / "cluster_labels.csv", index=False)
    # Plot latent colored by segment
    d_disp = add_display_columns(df)
    tmp_seg = d_disp[["customer_id", "購入", "オファー", "チャネル"]].copy()
    tmp_seg["segment"] = umap_seg.astype(int)
    tmp_seg["umap3_1"] = Zs["umap3"][:, 0]
    tmp_seg["umap3_2"] = Zs["umap3"][:, 1]
    tmp_seg["umap3_3"] = Zs["umap3"][:, 2]
    fig_seg = px.scatter_3d(
        tmp_seg.sample(n=min(len(tmp_seg), 8000), random_state=42),
        x="umap3_1",
        y="umap3_2",
        z="umap3_3",
        color="segment",
        title="潜在空間（3D）: UMAP（色=セグメント）",
        opacity=0.7,
    )
    fig_seg.update_layout(
        legend_title_text="セグメント",
        scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3"),
    )
    save_plotly(fig_seg, stem="latent3d_umap3_color_segment")

    # Causal-ish: joint (offer,channel) DR
    # Base features exclude offer/channel (treatment)
    base_features = ["recency", "history", "used_discount", "used_bogo", "zip_code", "is_referral"]
    T = build_joint_treatment(df)
    prop_model, out_model, treatments = fit_propensity_and_outcome_models(df, base_features, T)
    dr_table = dr_estimate_mu(df, prop_model, out_model, base_features, treatments)
    dr_table.to_csv(ART_DIR / "dr_treatment_effects.csv", index=False)

    # Policy: predict P(conv) for each treatment and compute profit under scenarios
    p_by_treat = policy_recommendations(df, out_model, base_features, treatments)
    profit_long = build_profit_tables(df, p_by_treat, DEFAULT_SCENARIOS)
    profit_long.to_csv(ART_DIR / "profit_long.csv", index=False)
    best_targets = export_targets(profit_long, p_by_treat)
    profit_summary = summarize_profit(best_targets)
    profit_summary.to_csv(ART_DIR / "profit_summary.csv", index=False)

    # Segment summary using mid_cost scenario (representative)
    best_mid = best_targets.loc[best_targets["scenario"] == "mid_cost"].copy()
    segment_summary = summarize_segments(
        df=df,
        cluster_labels=pd.Series(umap_seg, name="segment"),
        best_targets_mid=best_mid,
    )
    segment_summary.to_csv(ART_DIR / "segment_summary.csv", index=False)

    segment_best_cvr = segment_best_treatment_by_mean(
        df=df,
        out_model=out_model,
        base_features=base_features,
        treatments=treatments,
        segment_labels=umap_seg,
    )
    segment_best_cvr.to_csv(ART_DIR / "segment_best_cvr.csv", index=False)

    top_treat = best_mid["treatment"].value_counts().index[0]
    top_share = float((best_mid["treatment"] == top_treat).mean())
    policy_mid_summary = {"top_treatment": top_treat, "top_share": top_share}

    # Export processed dataset (simple: original + ids)
    df.to_csv(ART_DIR / "processed.csv", index=False)

    write_report(
        qc=qc,
        model_eval=model_eval,
        best_model=best_model,
        dr_table=dr_table,
        profit_summary=profit_summary,
        cvr_offer=cvr_offer,
        n_offer=n_offer,
        cvr_channel=cvr_channel,
        n_channel=n_channel,
        cvr_offer_channel=cvr_offer_channel,
        n_offer_channel=n_offer_channel,
        segment_summary=segment_summary,
        chi2_table=chi2_table,
        policy_mid_summary=policy_mid_summary,
    )

    print("Done. Outputs:")
    print(f"- Report: {ROOT / 'final_report.md'}")
    print(f"- Figures: {FIG_DIR}")
    print(f"- HTML figures: {FIG_HTML_DIR}")
    print(f"- Artifacts: {ART_DIR}")


if __name__ == "__main__":
    # Plotly image export defaults
    pio.defaults.default_scale = 2
    main()

