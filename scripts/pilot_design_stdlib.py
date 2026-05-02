"""
標準ライブラリのみでパイロット補助分析（環境で sklearn が使えない場合用）。
ロジスティック回帰を勾配降下で近似し、BASE 特徴に基づくスコアで上位％コホートを定義。
出力は analytics/pilot_design_analysis.py と同じパス先。
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "exercise.csv"
ART_DIR = ROOT / "artifacts"
FIG_IMP = ROOT / "figures" / "improvements"
LATEX_DIR = ROOT / "latex"

ZIP_MAP = {"Urban": 0.0, "Surburban": 1.0, "Rural": 2.0, "Suburban": 1.0}


def sigmoid(z: float) -> float:
    if z >= 30:
        return 1.0
    if z <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def main() -> None:
    random.seed(42)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    FIG_IMP.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "recency": float(r["recency"]),
                    "history": float(r["history"]),
                    "used_discount": float(r["used_discount"]),
                    "used_bogo": float(r["used_bogo"]),
                    "is_referral": float(r["is_referral"]),
                    "zip_num": ZIP_MAP.get(str(r["zip_code"]).strip(), 1.0),
                    "offer": r["offer"],
                    "conversion": int(float(r["conversion"])),
                }
            )

    n = len(rows)
    # 標準化
    def col(name: str) -> list[float]:
        return [r[name] for r in rows]

    def standardize(xs: list[float]) -> tuple[list[float], float, float]:
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
        s = math.sqrt(v) if v > 1e-12 else 1.0
        return [(x - m) / s for x in xs], m, s

    z_rec, _, _ = standardize(col("recency"))
    z_hist, _, _ = standardize(col("history"))
    z_ud, _, _ = standardize(col("used_discount"))
    z_ub, _, _ = standardize(col("used_bogo"))
    z_ir, _, _ = standardize(col("is_referral"))
    z_zip, _, _ = standardize(col("zip_num"))

    X = list(zip(z_rec, z_hist, z_ud, z_ub, z_ir, z_zip))
    y = [r["conversion"] for r in rows]

    # 勾配降下（L2）
    d = 6
    w = [0.0] * d
    b = 0.0
    lr = 0.15
    l2 = 0.01
    # 64000 件フルだと大イテレーションは遅い。収束に十分な回数（必要なら増やす）
    for _ in range(400):
        gw = [0.0] * d
        gb = 0.0
        for i in range(n):
            xi = X[i]
            p = sigmoid(sum(w[j] * xi[j] for j in range(d)) + b)
            err = p - y[i]
            gb += err
            for j in range(d):
                gw[j] += err * xi[j]
        gb /= n
        for j in range(d):
            gw[j] = gw[j] / n + l2 * w[j]
        b -= lr * gb
        for j in range(d):
            w[j] -= lr * gw[j]

    scores = [sigmoid(sum(w[j] * X[i][j] for j in range(d)) + b) for i in range(n)]
    for i, r in enumerate(rows):
        r["score_base"] = scores[i]

    sorted_idx = sorted(range(n), key=lambda i: scores[i], reverse=True)
    q90_i = int(max(0, math.ceil(0.10 * n) - 1))
    q95_i = int(max(0, math.ceil(0.05 * n) - 1))
    thresh90 = scores[sorted_idx[q90_i]]
    thresh95 = scores[sorted_idx[q95_i]]

    def summarize(mask: list[bool]) -> dict:
        sub = [rows[i] for i in range(n) if mask[i]]
        if not sub:
            return {"n": 0, "cvr": 0.0, "mean_history": 0.0, "mean_recency": 0.0}
        return {
            "n": len(sub),
            "cvr": sum(r["conversion"] for r in sub) / len(sub),
            "mean_history": sum(r["history"] for r in sub) / len(sub),
            "mean_recency": sum(r["recency"] for r in sub) / len(sub),
        }

    mask_all = [True] * n
    mask10 = [scores[i] >= thresh90 for i in range(n)]
    mask5 = [scores[i] >= thresh95 for i in range(n)]

    seg = [
        ("全体", mask_all),
        ("上位10%", mask10),
        ("上位5%", mask5),
    ]
    seg_rows = []
    for name, m in seg:
        s = summarize(m)
        seg_rows.append({"segment": name, **s})

    with (ART_DIR / "pilot_cohort_summary.csv").open("w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["segment", "n", "cvr", "mean_history", "mean_recency"])
        wtr.writeheader()
        for row in seg_rows:
            wtr.writerow(row)

    sub10 = [rows[i] for i in range(n) if mask10[i]]
    by_offer: dict[str, list[int]] = defaultdict(list)
    for r in sub10:
        by_offer[r["offer"]].append(r["conversion"])
    obs_rows = []
    for off, convs in by_offer.items():
        obs_rows.append({"offer": off, "n": len(convs), "cvr": sum(convs) / len(convs)})
    obs_rows.sort(key=lambda x: -x["cvr"])
    with (ART_DIR / "pilot_cohort_cvr_by_observed_offer.csv").open("w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["offer", "n", "cvr"])
        wtr.writeheader()
        for row in obs_rows:
            wtr.writerow(row)

    def prof_stats(m: list[bool]) -> dict:
        sub = [rows[i] for i in range(n) if m[i]]
        nn = len(sub)
        if nn == 0:
            return {"n": 0, "p_ud": 0.0, "p_ub": 0.0, "p_ir": 0.0}
        return {
            "n": nn,
            "p_ud": sum(r["used_discount"] for r in sub) / nn,
            "p_ub": sum(r["used_bogo"] for r in sub) / nn,
            "p_ir": sum(r["is_referral"] for r in sub) / nn,
        }

    prof_rows = [
        {"segment": "全体", **prof_stats(mask_all)},
        {"segment": "上位10%", **prof_stats(mask10)},
        {"segment": "上位5%", **prof_stats(mask5)},
    ]
    with (ART_DIR / "pilot_cohort_profile.csv").open("w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["segment", "n", "p_ud", "p_ub", "p_ir"])
        wtr.writeheader()
        for row in prof_rows:
            wtr.writerow(row)

    # スコア十分位ごとの観察差（Discount - No Offer）。交絡あり。
    order_score = sorted(range(n), key=lambda i: scores[i])
    dec_assign: dict[int, int] = {}
    for rank, i in enumerate(order_score):
        dec_assign[i] = min(rank // (n // 10), 9) + 1
    gap_rows: list[tuple[int, float, int, int]] = []
    for dec in range(1, 11):
        idx = [i for i in range(n) if dec_assign.get(i) == dec]
        d_arm = [i for i in idx if rows[i]["offer"] == "Discount"]
        n_arm = [i for i in idx if rows[i]["offer"] == "No Offer"]
        if len(d_arm) < 40 or len(n_arm) < 40:
            continue
        g = sum(rows[i]["conversion"] for i in d_arm) / len(d_arm) - sum(rows[i]["conversion"] for i in n_arm) / len(n_arm)
        gap_rows.append((dec, g, len(d_arm), len(n_arm)))
    with (ART_DIR / "pilot_uplift_crude_gap_by_decile.csv").open("w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["score_decile", "gap_crude", "n_discount", "n_no_offer"])
        wtr.writeheader()
        for dec, g, nd, nn in gap_rows:
            wtr.writerow({"score_decile": dec, "gap_crude": g, "n_discount": nd, "n_no_offer": nn})

    uplift_png = FIG_IMP / "uplift_crude_gap_by_score_decile.png"
    have_uplift_fig = False
    try:
        import sys

        import matplotlib.pyplot as plt  # type: ignore

        sys.path.insert(0, str(ROOT))
        from analytics.figures_jp import init_plot_style, save_fig

        if gap_rows:
            init_plot_style()
            _, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
            ax.bar([d for d, _, _, _ in gap_rows], [g * 100 for _, g, _, _ in gap_rows], color="#8172B3")
            ax.set_xlabel("購入スコア十分位（1=低・10=高）")
            ax.set_ylabel("観察CVR差（pt）\nDiscount − No Offer")
            ax.set_title("層別の観察差（交絡あり・真のupliftではない）")
            save_fig(uplift_png, dpi=150)
            have_uplift_fig = uplift_png.is_file()
    except Exception:
        have_uplift_fig = False

    p0 = summarize(mask10)["cvr"]

    def two_prop_n(p_control: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> float:
        # 正規近似（experiment_design.two_proportion_sample_size と同式、scipy なし）
        za = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else 1.96
        zb = 0.8416212335729143 if power <= 0.805 else 1.2815515655446004
        p_treat = min(max(p_control + mde, 1e-6), 1 - 1e-6)
        p_pool = (p_control + p_treat) / 2
        num = (za * math.sqrt(2 * p_pool * (1 - p_pool)) + zb * math.sqrt(p_control * (1 - p_control) + p_treat * (1 - p_treat))) ** 2
        den = mde**2
        return float(num / den)

    power_rows = []
    for mde in [0.005, 0.01, 0.015, 0.02, 0.03]:
        for pw in [0.8, 0.9]:
            try:
                nn = two_prop_n(p0, mde, power=pw)
            except ImportError:
                nn = two_prop_n(p0, mde, power=pw)
            power_rows.append(
                {
                    "baseline_cvr_pilot_pool": round(p0, 4),
                    "mde_absolute": mde,
                    "power": pw,
                    "n_per_arm_approx": int(math.ceil(nn)),
                    "total_n_3arm_approx": int(math.ceil(nn * 3)),
                }
            )
    with (ART_DIR / "pilot_power_assumptions.csv").open("w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(power_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(power_rows)

    scen_lines = []
    for label, lift in [("holdout_vs_lowcost", 0.01), ("holdout_vs_discount", 0.02)]:
        n80 = int(math.ceil(two_prop_n(p0, lift, power=0.8)))
        n90 = int(math.ceil(two_prop_n(p0, lift, power=0.9)))
        scen_lines.append({"comparison": label, "assumed_lift": lift, "n_per_arm_power80": n80, "n_per_arm_power90": n90})
    with (ART_DIR / "pilot_scenario_lift_n.csv").open("w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(scen_lines[0].keys()))
        wtr.writeheader()
        wtr.writerows(scen_lines)

    (ART_DIR / "pilot_analysis_meta.json").write_text(
        json.dumps(
            {
                "method": "stdlib_logistic_gd",
                "score_threshold_q90": thresh90,
                "score_threshold_q95": thresh95,
                "weights": w + [b],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 簡易 SVG の代わりに ASCII / ユーザーは matplotlib 版を pipeline で再生成
    # ここでは TeX 用数表のみ自動生成（図は後述でプレースホルダ or 数値から記述）

    r0 = next(x for x in seg_rows if x["segment"] == "全体")
    r10 = next(x for x in seg_rows if x["segment"] == "上位10%")
    r5 = next(x for x in seg_rows if x["segment"] == "上位5%")
    p0_pct = 100 * p0

    pr0 = next(x for x in prof_rows if x["segment"] == "全体")
    pr10 = next(x for x in prof_rows if x["segment"] == "上位10%")
    pr5 = next(x for x in prof_rows if x["segment"] == "上位5%")
    profile_tex = "\n".join(
        [
            r"% auto-generated by scripts/pilot_design_stdlib.py",
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \small",
            r"  \caption{スコア層のプロファイル（過去割引・BOGO・紹介の観察比率。記述）}",
            r"  \label{tab:pilot_profile_binary}",
            r"  \begin{tabular}{@{}lrrrr@{}}",
            r"    \toprule",
            r"    セグメント & $n$ & used\_discount & used\_bogo & is\_referral \\",
            r"    \midrule",
            rf"    全体 & {int(pr0['n'])} & {100*pr0['p_ud']:.1f}\% & {100*pr0['p_ub']:.1f}\% & {100*pr0['p_ir']:.1f}\% \\",
            rf"    上位10\% & {int(pr10['n'])} & {100*pr10['p_ud']:.1f}\% & {100*pr10['p_ub']:.1f}\% & {100*pr10['p_ir']:.1f}\% \\",
            rf"    上位5\% & {int(pr5['n'])} & {100*pr5['p_ud']:.1f}\% & {100*pr5['p_ub']:.1f}\% & {100*pr5['p_ir']:.1f}\% \\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    (LATEX_DIR / "generated_pilot_cohort_profile.tex").write_text(profile_tex, encoding="utf-8")

    def n_arm(mde: float, pw: float) -> int:
        return int(math.ceil(two_prop_n(p0, mde, power=pw)))

    tex = [
        r"% auto-generated by scripts/pilot_design_stdlib.py（標準ライブラリ・勾配ロジスティック）",
        r"% sklearn 環境では \texttt{python -m analytics.pilot\_design\_analysis} で図付きに置き換え可",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \caption{パイロット候補の記述（\texttt{recency}, \texttt{history}, \texttt{used\_discount}, \texttt{used\_bogo}, \texttt{is\_referral}, \texttt{zip\_code} のみから勾配ロジスティックでスコア化し上位\%で定義。演習用）}",
        r"  \label{tab:pilot_cohort}",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        r"    セグメント & $n$ & CVR & mean(history) & mean(recency) \\",
        r"    \midrule",
        rf"    全体 & {r0['n']} & {r0['cvr']:.4f} & {r0['mean_history']:.2f} & {r0['mean_recency']:.2f} \\",
        rf"    上位10\% & {r10['n']} & {r10['cvr']:.4f} & {r10['mean_history']:.2f} & {r10['mean_recency']:.2f} \\",
        rf"    上位5\% & {r5['n']} & {r5['cvr']:.4f} & {r5['mean_history']:.2f} & {r5['mean_recency']:.2f} \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        r"",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \caption{上位10\%コホート内の観察オファー別CVR（割当交絡あり・記述）}",
        r"  \label{tab:pilot_obs_offer}",
        r"  \begin{tabular}{@{}lrrr@{}}",
        r"    \toprule",
        r"    offer & $n$ & CVR \\",
        r"    \midrule",
    ]
    for o in obs_rows:
        tex.append(f"    {o['offer']} & {o['n']} & {o['cvr']:.4f} \\\\\n")
    tex.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            r"",
        ]
    )
    if have_uplift_fig:
        tex.extend(
            [
                r"\begin{figure}[htbp]",
                r"  \centering",
                r"  \includegraphics[width=0.88\linewidth]{../figures/improvements/uplift_crude_gap_by_score_decile.png}",
                r"  \caption{購入スコア（本スクリプトの6次元ロジスティック）十分位ごとの\textbf{観察}CVR差（Discount $-$ No Offer）。\textbf{割当は無作為ではない}ため真の増分（uplift）ではない。高スコア側で差が相対的に小さい層がありうる――\textbf{無駄割引}仮説の動機付けとして参照し、RCTで検証する。}",
                r"  \label{fig:uplift_crude_gap}",
                r"\end{figure}",
                r"",
            ]
        )
    tex.extend(
        [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \small",
            rf"  \caption{{仮定MDE（絶対差）に対する片群必要人数の目安（ベースラインCVR$=${p0_pct:.2f}\%、上位10\%コホート。二比例・$\alpha$=0.05・正規近似）}}",
            r"  \label{tab:pilot_power}",
            r"  \begin{tabular}{@{}rrrr@{}}",
            r"    \toprule",
            r"    MDE（絶対差） & 検定力 & 片群$n$（近似） & 3臂合計$n$目安 \\",
            r"    \midrule",
        ]
    )
    for mde in (0.01, 0.02):
        for pw in (0.8, 0.9):
            nn = n_arm(mde, pw)
            tex.append(f"    {mde:.3f} & {pw:.1f} & {nn} & {3 * nn} \\\\\n")
    tex.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    (LATEX_DIR / "pilot_design_analysis.tex").write_text("\n".join(tex), encoding="utf-8")
    print("Wrote pilot CSVs, latex/generated_pilot_cohort_profile.tex, latex/pilot_design_analysis.tex")


if __name__ == "__main__":
    main()
