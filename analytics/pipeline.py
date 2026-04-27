from __future__ import annotations

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
from sklearn.model_selection import train_test_split

from analytics.causal import (
    bootstrap_dr_table,
    dr_estimate_mu,
    fit_propensity_and_outcome_models,
    plot_dr_bootstrap_forest,
    propensity_diagnostics,
)
from analytics.clustering import (
    choose_k_from_silhouette,
    cluster_latent_k,
    make_latent_spaces,
    make_latent_spaces_2d,
    multiseed_ari_matrix,
    plot_ari_heatmap,
    plot_latent_2d,
    plot_latent_3d,
    plot_segment_3d,
    plot_silhouette_curve,
    silhouette_curve,
)
from analytics.config import (
    ART_DIR,
    BOOTSTRAP_B,
    DEFAULT_SCENARIOS,
    FIG_DIR,
    FIG_IMP,
    HOLDOUT_RATIO,
    RANDOM_STATE,
    ensure_all_dirs,
)
from analytics.stats_extra import run_stats_extra
from analytics.supervised_ablation import run_lsi_logreg_ablation
from analytics.constants import BASE_FEATURES
from analytics.eda import make_eda_figures
from analytics.exploratory_strata import write_segment_treatment_exploratory
from analytics.experiment_design import write_ab_design_table
from analytics.figures_jp import init_plot_style
from analytics.governance import write_data_dictionary, write_governance_ethics_tex
from analytics.io_data import basic_qc_tables, export_processed, read_data, validate_schema, write_qc_summary
from analytics.monitoring import write_monitoring_spec
from analytics.policy import (
    benchmark_policies,
    build_profit_tables,
    export_targets,
    greedy_constrained_targets,
    oof_predictions_outcome,
    plot_policy_figures,
    policy_eval_compare_table,
    scenario_pairwise_sensitivity,
    segment_best_treatment_by_mean,
    summarize_profit,
    summarize_segments,
    policy_recommendations,
)
from analytics.preprocess import build_preprocessor
from analytics.report_md import write_final_report, write_run_config
from analytics.segment_narrative import build_segment_narratives
from analytics.stats_tests import chi2_tests, plot_cramers_v_bar
from analytics.strategy import plot_decision_flow, write_decision_gate_readme, write_kpi_bridge_sensitivity, write_risk_register
from analytics.supervised import train_predictive_models


def run_pipeline() -> None:
    init_plot_style()
    pio.defaults.default_scale = 2
    ensure_all_dirs()

    df = read_data()
    validate_schema(df)
    qc = basic_qc_tables(df)
    write_qc_summary(qc)

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
    chi2_table.to_csv(ART_DIR / "chi2_tests_with_fdr.csv", index=False)
    chi2_table.to_csv(ART_DIR / "chi2_tests.csv", index=False)
    plot_cramers_v_bar(chi2_table)

    res = train_predictive_models(df)
    best_model = res["best_model"]
    model_eval = res["model_eval"]
    qc["n_train"] = res.get("split", {}).get("n_train")
    qc["n_test"] = res.get("split", {}).get("n_test")

    run_stats_extra(df)
    run_lsi_logreg_ablation(df)

    pre, _, _ = build_preprocessor(df)
    Zs = make_latent_spaces(df, preprocessor=pre)
    plot_latent_3d(df, Zs["pca3"], "pca3", color="conversion")
    plot_latent_3d(df, Zs["umap3"], "umap3", color="conversion")

    Z2 = make_latent_spaces_2d(df, preprocessor=pre)
    plot_latent_2d(df, Z2["pca2"], "pca2", color="conversion")
    plot_latent_2d(df, Z2["umap2"], "umap2", color="conversion")
    for stem in ("latent2d_pca2_color_conversion.png", "latent2d_umap2_color_conversion.png"):
        src = FIG_DIR / stem
        if src.is_file():
            shutil.copy2(src, FIG_IMP / stem)

    sil_df = silhouette_curve(Zs["umap3"], k_min=2, k_max=10)
    sil_df.to_csv(ART_DIR / "k_silhouette.csv", index=False)
    plot_silhouette_curve(sil_df)
    k_auto = choose_k_from_silhouette(sil_df)
    umap_seg_auto = cluster_latent_k(Zs["umap3"], name="umap3_auto", k=k_auto)
    umap_seg_k4 = cluster_latent_k(Zs["umap3"], name="umap3_k4", k=4)

    pd.DataFrame({"customer_id": df["customer_id"], "segment": umap_seg_auto}).to_csv(
        ART_DIR / "cluster_labels_auto.csv", index=False
    )
    pd.DataFrame({"customer_id": df["customer_id"], "segment": umap_seg_k4}).to_csv(
        ART_DIR / "cluster_labels_k4.csv", index=False
    )
    pd.DataFrame({"customer_id": df["customer_id"], "umap3_segment": umap_seg_k4}).to_csv(
        ART_DIR / "cluster_labels.csv",
        index=False,
    )

    ari = multiseed_ari_matrix(Zs["umap3"], seeds=(42, 43, 44), k=k_auto)
    ari.to_csv(ART_DIR / "segment_ari_matrix.csv")
    plot_ari_heatmap(ari)

    plot_segment_3d(df, Zs["umap3"], umap_seg_auto, stem="latent3d_umap3_color_segment")
    seg_png = FIG_DIR / "latent3d_umap3_color_segment.png"
    if seg_png.is_file():
        shutil.copy2(seg_png, FIG_IMP / seg_png.name)

    T = df["offer"].astype(str) + " | " + df["channel"].astype(str)
    prop_full, out_full, treatments = fit_propensity_and_outcome_models(df, BASE_FEATURES, T)

    dr_table = dr_estimate_mu(df, prop_full, out_full, BASE_FEATURES, treatments)
    dr_table.to_csv(ART_DIR / "dr_treatment_effects.csv", index=False)

    df_tr, df_te = train_test_split(df, test_size=HOLDOUT_RATIO, random_state=RANDOM_STATE, stratify=df["conversion"])
    T_tr = df_tr["offer"].astype(str) + " | " + df_tr["channel"].astype(str)
    prop_h, out_h, _ = fit_propensity_and_outcome_models(df_tr, BASE_FEATURES, T_tr)
    dr_holdout = dr_estimate_mu(df_te, prop_h, out_h, BASE_FEATURES, treatments)
    dr_holdout.to_csv(ART_DIR / "dr_treatment_effects_holdout.csv", index=False)

    diag_df, _ = propensity_diagnostics(df, prop_full, BASE_FEATURES, treatments)
    diag_df.to_csv(ART_DIR / "propensity_diagnostics.csv", index=False)

    dr_boot = bootstrap_dr_table(df, BASE_FEATURES, treatments, n_boot=BOOTSTRAP_B, n_jobs=4)
    dr_boot.to_csv(ART_DIR / "dr_bootstrap_quantiles.csv", index=False)
    plot_dr_bootstrap_forest(dr_holdout, dr_boot)

    p_full = policy_recommendations(df, out_full, BASE_FEATURES, treatments)
    profit_long = build_profit_tables(df, p_full, DEFAULT_SCENARIOS)
    profit_long.to_csv(ART_DIR / "profit_long.csv", index=False)
    best_targets = export_targets(profit_long, p_full, csv_path=ART_DIR / "promo_targets.csv")
    profit_summary = summarize_profit(best_targets)
    profit_summary.to_csv(ART_DIR / "profit_summary.csv", index=False)

    p_oof = oof_predictions_outcome(df, BASE_FEATURES, treatments)
    profit_oof = build_profit_tables(df, p_oof, DEFAULT_SCENARIOS)
    best_oof = export_targets(profit_oof, p_oof, csv_path=ART_DIR / "promo_targets_oof.csv")

    p_holdout = policy_recommendations(df_te, out_h, BASE_FEATURES, treatments)
    profit_holdout = build_profit_tables(df_te, p_holdout, DEFAULT_SCENARIOS)
    best_holdout = export_targets(
        profit_holdout, p_holdout, csv_path=ART_DIR / "promo_targets_holdout.csv"
    )

    mid = next(s for s in DEFAULT_SCENARIOS if s.name == "mid_cost")
    bench = benchmark_policies(df, p_oof, mid)
    bench.to_csv(ART_DIR / "policy_benchmarks.csv", index=False)
    bench.to_csv(ART_DIR / "policy_benchmarks_mid_cost.csv", index=False)

    pec = policy_eval_compare_table(best_targets, best_oof, best_holdout)
    pec.to_csv(ART_DIR / "policy_eval_compare.csv", index=False)
    scenario_pairwise_sensitivity(best_targets).to_csv(
        ART_DIR / "policy_scenario_sensitivity.csv", index=False
    )

    plot_policy_figures(best_targets, best_oof, bench, scenario="mid_cost")

    best_mid = best_targets.loc[best_targets["scenario"] == "mid_cost"].copy()
    constrained = greedy_constrained_targets(best_mid, df, scenario="mid_cost")
    constrained.to_csv(ART_DIR / "promo_targets_constrained_mid_cost.csv", index=False)
    sel_rate = constrained["selected_constrained"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(["制約前（候補全体）", "制約後（貪欲選択）"], [1.0, sel_rate], color=["#4c72b0", "#55a868"])
    plt.ylabel("選択された顧客の割合（相対）")
    plt.title("制約付き配信（例：接触上限・予算）")
    from analytics.figures_jp import save_fig

    save_fig(FIG_IMP / "constrained_selection_summary.png")

    segment_summary = summarize_segments(df, pd.Series(umap_seg_auto), best_mid)
    segment_summary.to_csv(ART_DIR / "segment_summary.csv", index=False)

    write_segment_treatment_exploratory(df, umap_seg_auto)

    seg_best = segment_best_treatment_by_mean(df, out_full, BASE_FEATURES, treatments, umap_seg_auto)
    seg_best.to_csv(ART_DIR / "segment_best_cvr.csv", index=False)

    seg_narr = build_segment_narratives(df, umap_seg_auto, model=res["models"]["logreg"])
    seg_narr.to_csv(ART_DIR / "segment_narratives.csv", index=False)

    from analytics.report_visuals import write_report_summary_figures

    write_report_summary_figures(dr_table, dr_holdout, segment_summary, pec, diag_df, profit_summary)

    top_treat = best_mid["treatment"].value_counts().index[0]
    top_share = float((best_mid["treatment"] == top_treat).mean())
    policy_mid_summary = {"top_treatment": top_treat, "top_share": top_share}

    export_processed(df)

    write_kpi_bridge_sensitivity(int(qc["n_rows"]), float(df["history"].mean()), float(qc["conversion_rate"]))
    write_risk_register()
    write_decision_gate_readme()
    plot_decision_flow()

    write_ab_design_table(float(qc["conversion_rate"]))
    write_data_dictionary()
    write_governance_ethics_tex()
    write_monitoring_spec()

    write_final_report(
        qc=qc,
        model_eval=model_eval,
        best_model=best_model,
        dr_table=dr_table,
        dr_holdout=dr_holdout,
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
        segment_narrative=seg_narr,
    )

    write_run_config(
        {
            "random_state": RANDOM_STATE,
            "holdout_ratio": HOLDOUT_RATIO,
            "bootstrap_b": BOOTSTRAP_B,
            "evaluation_protocol": {
                "primary_kpi": "expected_profit_mid_cost",
                "secondary_kpi": "topk_lift",
                "primary_classifier": "logistic_regression",
                "secondary_classifier": "hist_gradient_boosting",
            },
        }
    )

    from analytics.report_tex import write_latex_bundle

    write_latex_bundle()

    print("Done. Outputs:")
    print(f"- Report: {ART_DIR.parent / 'final_report.md'}")
    print(f"- LaTeX: {ART_DIR.parent / 'latex' / 'final_report.tex'}")


if __name__ == "__main__":
    run_pipeline()
