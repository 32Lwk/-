from __future__ import annotations

import math

import pandas as pd

from analytics.config import ART_DIR, TABLES_DIR


def two_proportion_sample_size(
    p_control: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> float:
    """Balanced two-sample test for proportions (normal approx). Returns n per arm (approx)."""
    from scipy.stats import norm

    p_treat = p_control + mde_absolute
    p_pool = (p_control + p_treat) / 2
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(power)
    num = (za * math.sqrt(2 * p_pool * (1 - p_pool)) + zb * math.sqrt(p_control * (1 - p_control) + p_treat * (1 - p_treat))) ** 2
    den = mde_absolute**2
    n = num / den * (1 + 1 / ratio) / 2
    return float(n)


def write_ab_design_table(base_cvr: float) -> None:
    mdes = [0.005, 0.01, 0.015, 0.02]
    rows = []
    for mde in mdes:
        for power in (0.8, 0.9):
            n = two_proportion_sample_size(base_cvr, mde, alpha=0.05, power=power)
            rows.append(
                {
                    "baseline_cvr": base_cvr,
                    "mde_absolute": mde,
                    "alpha": 0.05,
                    "power": power,
                    "n_per_arm_approx": int(math.ceil(n)),
                    "total_n_approx": int(math.ceil(n * 2)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(ART_DIR / "ab_design.csv", index=False)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tex = [
        "\\begin{tabular}{rrrrr}\n\\toprule\n",
        "MDE & 検定力 & 片群n（近似） & 合計n（近似） \\\\\n\\midrule\n",
    ]
    for mde in mdes:
        for power in (0.8, 0.9):
            r = df[(df["mde_absolute"] == mde) & (df["power"] == power)].iloc[0]
            tex.append(f"{mde:.3f} & {power:.1f} & {int(r['n_per_arm_approx'])} & {int(r['total_n_approx'])} \\\\\n")
    tex.append("\\bottomrule\n\\end{tabular}\n")
    (TABLES_DIR / "ab_design.tex").write_text("".join(tex), encoding="utf-8")
