from __future__ import annotations

import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt

from analytics.config import FIG_DIR, FIG_HTML_DIR
from analytics.figures_jp import save_fig
from analytics.utils_common import add_display_columns


def save_plotly(fig, stem: str):
    from pathlib import Path

    html_path = FIG_HTML_DIR / f"{stem}.html"
    png_path = FIG_DIR / f"{stem}.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    fig.write_image(str(png_path), scale=2)
    return png_path, html_path


def save_plotly_png(fig, stem: str):
    from pathlib import Path

    png_path = FIG_DIR / f"{stem}.png"
    fig.write_image(str(png_path), scale=2)
    return png_path


def make_eda_figures(df) -> None:
    d = add_display_columns(df)
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
