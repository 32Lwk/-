"""
パイロット RCT 設計（上位スコア層・3 臂イメージ）の補助分析。
- BASE_FEATURES のみで購入確率を推定（現行 offer/channel は母集団定義に含めない）。
- 上位 5% / 10% コホートの記述統計・観察 offer 別 CVR（交絡あり・参考）。
- 仮定した効果量に対する片群必要 n（二比例・正規近似）を算出。

出力: artifacts/pilot_*.csv, figures/improvements/pilot_*.png, latex/pilot_design_analysis.tex
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from analytics.config import ART_DIR, FIG_IMP, LATEX_DIR
from analytics.constants import BASE_FEATURES
from analytics.experiment_design import two_proportion_sample_size
from analytics.figures_jp import init_plot_style, save_fig
from analytics.io_data import read_data
from analytics.preprocess import build_causal_preprocessor


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def main() -> None:
    init_plot_style()
    ART_DIR.mkdir(parents=True, exist_ok=True)
    FIG_IMP.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    df = read_data()
    X = df[BASE_FEATURES].copy()
    y = df["conversion"].astype(int).to_numpy()

    pre = build_causal_preprocessor(BASE_FEATURES)
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))])
    pipe.fit(X, y)
    score = pipe.predict_proba(X)[:, 1]

    df = df.copy()
    df["score_base"] = score

    q90 = np.quantile(score, 0.90)
    q95 = np.quantile(score, 0.95)
    mask10 = df["score_base"] >= q90
    mask5 = df["score_base"] >= q95

    rows = []
    for name, mask in [("全体", pd.Series(True, index=df.index)), ("上位10%", mask10), ("上位5%", mask5)]:
        sub = df.loc[mask]
        rows.append(
            {
                "segment": name,
                "n": int(len(sub)),
                "cvr": float(sub["conversion"].mean()),
                "mean_history": float(sub["history"].mean()),
                "mean_recency": float(sub["recency"].mean()),
            }
        )
    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(ART_DIR / "pilot_cohort_summary.csv", index=False)

    # 上位10% 内の観察 offer 別 CVR（記述・交絡）
    sub10 = df.loc[mask10].copy()
    obs = (
        sub10.groupby("offer", observed=True)
        .agg(n=("conversion", "size"), cvr=("conversion", "mean"))
        .reset_index()
        .sort_values("cvr", ascending=False)
    )
    obs.to_csv(ART_DIR / "pilot_cohort_cvr_by_observed_offer.csv", index=False)

    # コホート別プロファイル（バイナリ特徴の平均）
    prof_rows = []
    for name, mask in [("全体", pd.Series(True, index=df.index)), ("上位10%", mask10), ("上位5%", mask5)]:
        sub = df.loc[mask]
        prof_rows.append(
            {
                "segment": name,
                "n": int(len(sub)),
                "p_used_discount": float(sub["used_discount"].mean()),
                "p_used_bogo": float(sub["used_bogo"].mean()),
                "p_is_referral": float(sub["is_referral"].mean()),
            }
        )
    prof_df = pd.DataFrame(prof_rows)
    prof_df.to_csv(ART_DIR / "pilot_cohort_profile.csv", index=False)

    # 購入スコア十分位ごとの観察差：CVR(Discount) - CVR(No Offer)（割当交絡あり・参考）
    df_u = df.copy()
    try:
        df_u["decile"] = pd.qcut(df_u["score_base"], q=10, labels=False, duplicates="drop")
    except ValueError:
        df_u["decile"] = pd.qcut(df_u["score_base"].rank(method="first"), q=10, labels=False, duplicates="drop")
    gap_rows = []
    for d in sorted(int(x) for x in df_u["decile"].dropna().unique()):
        subd = df_u.loc[df_u["decile"] == d]
        d_arm = subd[subd["offer"] == "Discount"]
        n_arm = subd[subd["offer"] == "No Offer"]
        if len(d_arm) < 40 or len(n_arm) < 40:
            continue
        gap_rows.append(
            {
                "score_decile": d + 1,
                "n_discount": int(len(d_arm)),
                "n_no_offer": int(len(n_arm)),
                "cvr_discount": float(d_arm["conversion"].mean()),
                "cvr_no_offer": float(n_arm["conversion"].mean()),
                "gap_crude": float(d_arm["conversion"].mean() - n_arm["conversion"].mean()),
            }
        )
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(ART_DIR / "pilot_uplift_crude_gap_by_decile.csv", index=False)

    if len(gap_df) > 0:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        ax.bar(
            gap_df["score_decile"].astype(int).tolist(),
            (gap_df["gap_crude"] * 100).tolist(),
            color="#8172B3",
        )
        ax.set_xlabel("購入スコア十分位（1=低・10=高）")
        ax.set_ylabel("観察CVR差（pt）\nDiscount − No Offer")
        ax.set_title("層別の観察差（交絡あり・真のupliftではない）")
        fig.tight_layout()
        save_fig(FIG_IMP / "uplift_crude_gap_by_score_decile.png")

    # 図1: セグメント別 n と CVR
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(seg_df))
    ax.bar(x - 0.2, seg_df["n"] / 1000, width=0.4, label="件数（千件）", color="#4C72B0")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, seg_df["cvr"] * 100, width=0.4, label="CVR（%）", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(seg_df["segment"].tolist())
    ax.set_ylabel("件数（千件）")
    ax2.set_ylabel("CVR（%）")
    ax.set_title("パイロット候補母集団（BASE特徴のみのスコアで上位％）")
    fig.tight_layout()
    p1 = FIG_IMP / "pilot_cohort_n_cvr.png"
    save_fig(p1)

    # 図2: 上位10% で観察された offer 別 CVR
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = [str(r["offer"]) for _, r in obs.iterrows()]
    ax.barh(labels[::-1], (obs["cvr"] * 100).tolist()[::-1], color="#55A868")
    ax.set_xlabel("CVR（%）※観察・交絡あり")
    ax.set_title("上位10%コホート内：観察オファー別CVR（因果ではない）")
    fig.tight_layout()
    p2 = FIG_IMP / "pilot_cohort_observed_offer_cvr.png"
    save_fig(p2)

    # 仮定効果量での必要片群 n（コホート内ベースレート = 上位10% の観察CVR）
    p0 = float(sub10["conversion"].mean())
    mdes = [0.005, 0.01, 0.015, 0.02, 0.03]
    powers = [0.8, 0.9]
    power_rows = []
    for mde in mdes:
        for pw in powers:
            n = two_proportion_sample_size(p0, mde, alpha=0.05, power=pw)
            power_rows.append(
                {
                    "baseline_cvr_pilot_pool": round(p0, 4),
                    "mde_absolute": mde,
                    "power": pw,
                    "n_per_arm_approx": int(math.ceil(n)),
                    "total_n_3arm_approx": int(math.ceil(n * 3)),
                    "note": "2標本比例・片群n。3臂は目安で同一nを3倍",
                }
            )
    power_df = pd.DataFrame(power_rows)
    power_df.to_csv(ART_DIR / "pilot_power_assumptions.csv", index=False)

    # シナリオ: 低コスト臂が +1pt、割引臂が +2pt のとき（仮定）
    scenarios = [
        ("ホールドアウト対 低コスト訴求", p0, 0.01),
        ("ホールドアウト対 割引", p0, 0.02),
    ]
    scen_lines = []
    for label, p, lift in scenarios:
        n80 = int(math.ceil(two_proportion_sample_size(p, lift, alpha=0.05, power=0.8)))
        n90 = int(math.ceil(two_proportion_sample_size(p, lift, alpha=0.05, power=0.9)))
        scen_lines.append({"comparison": label, "assumed_lift": lift, "n_per_arm_power80": n80, "n_per_arm_power90": n90})
    scen_df = pd.DataFrame(scen_lines)
    scen_df.to_csv(ART_DIR / "pilot_scenario_lift_n.csv", index=False)

    (ART_DIR / "pilot_analysis_meta.json").write_text(
        json.dumps(
            {
                "score_quantile_q90": float(q90),
                "score_quantile_q95": float(q95),
                "base_features": BASE_FEATURES,
                "disclaimer": "スコアは全件学習のため演習用。本番は時系列・OOFで定義推奨。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # LaTeX 断片
    r0 = seg_df[seg_df["segment"] == "全体"].iloc[0]
    r10 = seg_df[seg_df["segment"] == "上位10%"].iloc[0]
    r5 = seg_df[seg_df["segment"] == "上位5%"].iloc[0]
    p0_pct = 100 * p0

    pr0 = prof_df[prof_df["segment"] == "全体"].iloc[0]
    pr10 = prof_df[prof_df["segment"] == "上位10%"].iloc[0]
    pr5 = prof_df[prof_df["segment"] == "上位5%"].iloc[0]
    profile_tex = "\n".join(
        [
            r"% auto-generated by analytics/pilot_design_analysis.py",
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \small",
            r"  \caption{スコア層のプロファイル（過去割引・BOGO・紹介の観察比率。記述）}",
            r"  \label{tab:pilot_profile_binary}",
            r"  \begin{tabular}{@{}lrrrr@{}}",
            r"    \toprule",
            r"    セグメント & $n$ & used\_discount & used\_bogo & is\_referral \\",
            r"    \midrule",
            rf"    全体 & {int(pr0['n'])} & {100*pr0['p_used_discount']:.1f}\\% & {100*pr0['p_used_bogo']:.1f}\\% & {100*pr0['p_is_referral']:.1f}\\% \\",
            rf"    上位10\\% & {int(pr10['n'])} & {100*pr10['p_used_discount']:.1f}\\% & {100*pr10['p_used_bogo']:.1f}\\% & {100*pr10['p_is_referral']:.1f}\\% \\",
            rf"    上位5\\% & {int(pr5['n'])} & {100*pr5['p_used_discount']:.1f}\\% & {100*pr5['p_used_bogo']:.1f}\\% & {100*pr5['p_is_referral']:.1f}\\% \\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    (LATEX_DIR / "generated_pilot_cohort_profile.tex").write_text(profile_tex, encoding="utf-8")

    tex = [
        r"% auto-generated by analytics/pilot_design_analysis.py",
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \includegraphics[width=0.88\linewidth]{../figures/improvements/pilot_cohort_n_cvr.png}",
        r"  \caption{パイロット候補母集団の規模と観察CVR（スコアは \texttt{recency}, \texttt{history} 等ベース特徴のみのロジスティック。演習用・参考）}",
        r"  \label{fig:pilot_cohort}",
        r"\end{figure}",
        r"",
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \includegraphics[width=0.88\linewidth]{../figures/improvements/pilot_cohort_observed_offer_cvr.png}",
        r"  \caption{上位10\%コホート内の\textbf{観察}オファー別CVR（割当交絡あり。RCT では無作為化により解消）}",
        r"  \label{fig:pilot_obs_offer}",
        r"\end{figure}",
        r"",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \caption{パイロット候補の記述（スコア上位％で定義）}",
        r"  \label{tab:pilot_cohort}",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        r"    セグメント & $n$ & CVR & mean(history) & mean(recency) \\",
        r"    \midrule",
        rf"    全体 & {int(r0['n'])} & {r0['cvr']:.4f} & {r0['mean_history']:.2f} & {r0['mean_recency']:.2f} \\",
        rf"    上位10\% & {int(r10['n'])} & {r10['cvr']:.4f} & {r10['mean_history']:.2f} & {r10['mean_recency']:.2f} \\",
        rf"    上位5\% & {int(r5['n'])} & {r5['cvr']:.4f} & {r5['mean_history']:.2f} & {r5['mean_recency']:.2f} \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        r"",
    ]
    # 上位10% 内の観察オファー別CVR（表：本文参照用）
    tex.extend(
        [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \small",
            r"  \caption{上位10\%コホート内の観察オファー別CVR（割当交絡あり・記述）}",
            r"  \label{tab:pilot_obs_offer}",
            r"  \begin{tabular}{@{}lrr@{}}",
            r"    \toprule",
            r"    offer & $n$ & CVR \\",
            r"    \midrule",
        ]
    )
    for _, rr in obs.iterrows():
        of = _tex_escape(str(rr["offer"]))
        tex.append(f"    {of} & {int(rr['n'])} & {float(rr['cvr']):.4f} \\\\\n")
    tex.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            r"",
        ]
    )
    if len(gap_df) > 0:
        tex.extend(
            [
                r"\begin{figure}[htbp]",
                r"  \centering",
                r"  \includegraphics[width=0.88\linewidth]{../figures/improvements/uplift_crude_gap_by_score_decile.png}",
                r"  \caption{購入スコア（ベース特徴のみのロジスティック）十分位ごとの\textbf{観察}CVR差（Discount $-$ No Offer）。\textbf{割当は無作為ではない}ため真の増分（uplift）ではないが、高スコア層で差が相対的に小さくなるパターンは「購入しやすい層への重いインセンティブは辺際的に効きにくい」という\textbf{仮説の動機付け}として参照できる。RCTで検証する。}",
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
            rf"  \caption{{仮定した絶対差（MDE）に対する片群必要人数の目安（ベースラインCVR$=${p0_pct:.2f}\%、上位10\%コホートの観察値。二比例・$\alpha$=0.05）}}",
            r"  \label{tab:pilot_power}",
            r"  \begin{tabular}{@{}rrrr@{}}",
            r"    \toprule",
            r"    MDE（絶対差） & 検定力 & 片群$n$（近似） & 3臂合計$n$目安 \\",
            r"    \midrule",
        ]
    )
    for mde in (0.01, 0.02):
        for pw in (0.8, 0.9):
            n = int(math.ceil(two_proportion_sample_size(p0, mde, alpha=0.05, power=pw)))
            tex.append(f"    {mde:.3f} & {pw:.1f} & {n} & {3 * n} \\\\\n")
    tex.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            r"",
            r"\noindent\footnotesize",
            r"\textbf{解釈（仮定）.} "
            r"低コスト訴求で購入率がベースから \textbf{+1pt}、割引で \textbf{+2pt} と仮定した場合の片群$n$（近似）は、"
            r"\texttt{artifacts/pilot\_scenario\_lift\_n.csv} を参照。"
            r"実際の設計では複数比較補正・クラスタ乱数化・無効検査除外を織り込む。",
        ]
    )
    (LATEX_DIR / "pilot_design_analysis.tex").write_text("\n".join(tex), encoding="utf-8")
    print("Wrote pilot artifacts and latex/pilot_design_analysis.tex")


if __name__ == "__main__":
    main()
