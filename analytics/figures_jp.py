from __future__ import annotations

import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt


def set_japanese_font() -> None:
    """日本語表示用フォント。ユーザー指定で BIZ UD ゴシック系を最優先（表記: BIZ UDP / BIGUDP 等のゆれに対応）。"""
    fonts = fm.fontManager.ttflist
    available = {f.name for f in fonts}

    preferred_exact = [
        "BIZ UDGothic",
        "BIZ UDゴシック",
        "BIZ UDPGothic",
        "BIZ UDPゴシック",
        "BIZ UD PGothic",
        "BIZ UD ゴシック",
        "BIZ UDMincho",
        "BIZ UD明朝",
    ]
    for name in preferred_exact:
        if name in available:
            plt.rcParams["font.family"] = name
            return

    for f in fonts:
        n = f.name
        if "BIZ UD" in n or "BIZUDP" in n.replace(" ", "").upper():
            plt.rcParams["font.family"] = n
            return

    candidates = [
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "MS UI Gothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
    ]
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


def init_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    set_japanese_font()


def save_fig(path, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
