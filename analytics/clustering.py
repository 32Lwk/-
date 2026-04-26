from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from analytics.config import ART_DIR, FIG_DIR, FIG_HTML_DIR, FIG_IMP, RANDOM_STATE
from analytics.config import UMAP_MIN_DIST, UMAP_N_NEIGHBORS
from analytics.eda import save_plotly
from analytics.utils_common import add_display_columns


def make_latent_spaces(df, preprocessor) -> Dict[str, np.ndarray]:
    import json

    X = df.drop(columns=["conversion"])
    X_mat = preprocessor.fit_transform(X)

    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    Z_pca = pca.fit_transform(X_mat)
    explained = pca.explained_variance_ratio_.tolist()
    with open(ART_DIR / "pca_explained_variance.json", "w", encoding="utf-8") as f:
        json.dump({"explained_variance_ratio": explained}, f, ensure_ascii=False, indent=2)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    Z_umap = reducer.fit_transform(X_mat)

    return {"pca3": Z_pca, "umap3": Z_umap}


def make_latent_spaces_2d(df, preprocessor) -> Dict[str, np.ndarray]:
    X = df.drop(columns=["conversion"])
    X_mat = preprocessor.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z_pca = pca.fit_transform(X_mat)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    Z_umap = reducer.fit_transform(X_mat)
    return {"pca2": Z_pca, "umap2": Z_umap}


def plot_latent_2d(df, Z: np.ndarray, name: str, color: str = "conversion", out_dir=None) -> None:
    out_dir = out_dir or FIG_DIR
    d = add_display_columns(df)
    tmp = d[["customer_id", "購入", "オファー", "チャネル"]].copy()
    tmp[f"{name}_1"] = Z[:, 0]
    tmp[f"{name}_2"] = Z[:, 1]
    color_col = "購入" if color == "conversion" else color
    fig = px.scatter(
        tmp.sample(n=min(len(tmp), 12000), random_state=RANDOM_STATE),
        x=f"{name}_1",
        y=f"{name}_2",
        color=color_col,
        symbol="オファー",
        title=f"潜在空間（2D）: {name}（色={color_col}）",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(legend_title_text="凡例", xaxis_title=f"{name}_1", yaxis_title=f"{name}_2")
    png_path = out_dir / f"latent2d_{name}_color_{color}.png"
    fig.write_image(str(png_path), scale=2)


def plot_latent_3d(df, Z: np.ndarray, name: str, color: str = "conversion", out_dir_html=None) -> None:
    out_dir_html = out_dir_html or FIG_HTML_DIR
    d = add_display_columns(df)
    tmp = d[["customer_id", "conversion", "購入", "オファー", "チャネル"]].copy()
    tmp[f"{name}_1"] = Z[:, 0]
    tmp[f"{name}_2"] = Z[:, 1]
    tmp[f"{name}_3"] = Z[:, 2]
    color_col = "購入" if color == "conversion" else color
    fig = px.scatter_3d(
        tmp.sample(n=min(len(tmp), 8000), random_state=RANDOM_STATE),
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
        scene=dict(xaxis_title=f"{name}_1", yaxis_title=f"{name}_2", zaxis_title=f"{name}_3"),
    )
    save_plotly(fig, stem=f"latent3d_{name}_color_{color}")


def silhouette_curve(Z: np.ndarray, k_min: int, k_max: int) -> pd.DataFrame:
    n = len(Z)
    if n > 12000:
        idx = np.random.RandomState(RANDOM_STATE).choice(n, size=12000, replace=False)
        Z_use = Z[idx]
    else:
        Z_use = Z
    rows = []
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto").fit_predict(Z_use)
        s = float(silhouette_score(Z_use, labels))
        rows.append({"k": k, "silhouette": s})
    return pd.DataFrame(rows)


def choose_k_from_silhouette(curve: pd.DataFrame) -> int:
    row = curve.loc[curve["silhouette"].idxmax()]
    return int(row["k"])


def cluster_latent_k(Z: np.ndarray, name: str, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(Z)
    joblib.dump({"k": k, "model": km}, ART_DIR / f"kmeans_{name}_k{k}.joblib")
    return labels


def plot_silhouette_curve(curve: pd.DataFrame, path=None) -> None:
    path = path or (FIG_IMP / "silhouette_vs_k.png")
    plt.figure(figsize=(7, 4))
    plt.plot(curve["k"], curve["silhouette"], "o-")
    plt.xlabel("クラスタ数 k")
    plt.ylabel("シルエット係数")
    plt.title("シルエット係数 vs クラスタ数（UMAP 3D）")
    from analytics.figures_jp import save_fig

    save_fig(path)


def multiseed_ari_matrix(Z_base: np.ndarray, seeds: Tuple[int, ...], k: int) -> pd.DataFrame:
    labels_per_seed = []
    for sd in seeds:
        km = KMeans(n_clusters=k, random_state=sd, n_init="auto")
        labels_per_seed.append(km.fit_predict(Z_base))
    mat = np.zeros((len(seeds), len(seeds)))
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            mat[i, j] = adjusted_rand_score(labels_per_seed[i], labels_per_seed[j])
    return pd.DataFrame(mat, index=[f"seed{s}" for s in seeds], columns=[f"seed{s}" for s in seeds])


def plot_ari_heatmap(ari_df: pd.DataFrame, path=None) -> None:
    path = path or (FIG_IMP / "segment_stability_ari.png")
    import seaborn as sns

    plt.figure(figsize=(5, 4))
    sns.heatmap(ari_df, annot=True, fmt=".3f", cmap="Blues")
    plt.title("セグメント安定性（調整ランド指数 ARI）")
    from analytics.figures_jp import save_fig

    save_fig(path)


def plot_segment_3d(df, Z_umap3: np.ndarray, labels: np.ndarray, stem: str = "latent3d_umap3_color_segment") -> None:
    d_disp = add_display_columns(df)
    tmp_seg = d_disp[["customer_id", "購入", "オファー", "チャネル"]].copy()
    tmp_seg["segment"] = labels.astype(int)
    tmp_seg["umap3_1"] = Z_umap3[:, 0]
    tmp_seg["umap3_2"] = Z_umap3[:, 1]
    tmp_seg["umap3_3"] = Z_umap3[:, 2]
    fig_seg = px.scatter_3d(
        tmp_seg.sample(n=min(len(tmp_seg), 8000), random_state=RANDOM_STATE),
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
    save_plotly(fig_seg, stem=stem)
