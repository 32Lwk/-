from __future__ import annotations

import json

from analytics.config import ART_DIR, TABLES_DIR


def write_monitoring_spec() -> None:
    spec = {
        "model_calibration": {
            "metric": "Brier, ECE",
            "threshold_example": "Brier がベースライン+0.02 超で再学習検討",
        },
        "ranking": {
            "metric": "PR-AUC",
            "threshold_example": "週次で前週比 -5pt 以上なら調査",
        },
        "data_drift": {
            "metric": "PSI / KS（主要特徴）",
            "threshold_example": "PSI > 0.2 でアラート",
        },
        "propensity_ess": {
            "metric": "ESS（IPW重み）",
            "threshold_example": "ESS < 0.1 * n でトリミング強化または縮小運用",
        },
        "incident_response": {
            "actions": ["介入配布停止", "ルールベース縮退", "再学習"],
        },
    }
    (ART_DIR / "monitoring_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    yaml_text = """# 本番モニタリング仕様（テンプレ）
model_calibration:
  metric: \"Brier, ECE\"
  threshold_example: \"Brier がベースライン+0.02 超で再学習検討\"
ranking:
  metric: \"PR-AUC\"
  threshold_example: \"週次で前週比 -5pt 以上なら調査\"
data_drift:
  metric: \"PSI / KS（主要特徴）\"
  threshold_example: \"PSI > 0.2 でアラート\"
propensity_ess:
  metric: \"ESS（IPW重み）\"
  threshold_example: \"ESS < 0.1 * n でトリミング強化または縮小運用\"
incident_response:
  actions:
    - 介入配布停止
    - ルールベース縮退
    - 再学習
"""
    (ART_DIR / "monitoring_spec.yaml").write_text(yaml_text, encoding="utf-8")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tex = """\\begin{tabular}{ll}
\\toprule
領域 & 監視例 \\\\\n\\midrule
校正 & Brier / ECE 週次 \\\\\n
ドリフト & PSI / KS \\\\\n
傾向スコア & ESS・極小e割合 \\\\\n
\\bottomrule
\\end{tabular}
"""
    (TABLES_DIR / "monitoring_summary.tex").write_text(tex, encoding="utf-8")
