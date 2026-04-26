from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# プロジェクトルート（analytics の親）
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "exercise.csv"
FIG_DIR = ROOT / "figures"
FIG_HTML_DIR = ROOT / "figures_html"
FIG_IMP = ROOT / "figures" / "improvements"
ART_DIR = ROOT / "artifacts"
TABLES_DIR = ART_DIR / "tables"
LATEX_DIR = ROOT / "latex"

RANDOM_STATE = 42
K_FOLD = 5
HOLDOUT_RATIO = 0.25
BOOTSTRAP_B = 200  # 計算負荷が高い場合は 80 程度に下げる
BOOTSTRAP_N = 15000  # 各ブートストラップ反復の層化部分標本（計算コスト抑制）
PROPENSITY_CLIP = 0.01
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
SILHOUETTE_K_MIN = 2
SILHOUETTE_K_MAX = 10
UMAP_STABILITY_SEEDS = (42, 43, 44)

# 制約付き配信（history 単位での相対スケール例）
CONSTRAINT_MAX_CONTACTS: int | None = 20000  # None なら無制限
CONSTRAINT_BUDGET_HISTORY_UNITS: float | None = 5_000_000.0  # sum(history) 上限の目安
CONSTRAINT_CHANNEL_CAP: Dict[str, int] | None = None  # 例: {"Phone": 5000}


@dataclass(frozen=True)
class CostScenario:
    name: str
    r_bogo: float
    r_disc: float
    channel_rank_cost: Dict[str, float]


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


def ensure_all_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_HTML_DIR.mkdir(parents=True, exist_ok=True)
    FIG_IMP.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
