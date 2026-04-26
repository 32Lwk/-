from __future__ import annotations

import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt


def set_japanese_font() -> None:
    candidates = [
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "MS UI Gothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
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
