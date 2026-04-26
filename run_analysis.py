"""
キカガク分析パイプライン実行エントリ。

再現手順:
  cd プロジェクトルート
  python run_analysis.py

LuaLaTeX（任意）:
  cd latex
  latexmk -lualatex final_report.tex
"""

from __future__ import annotations

import plotly.io as pio

from analytics.pipeline import run_pipeline


def main() -> None:
    pio.defaults.default_scale = 2
    run_pipeline()


if __name__ == "__main__":
    main()
