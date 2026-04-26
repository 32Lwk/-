"""
後方互換エントリ。新パイプラインは `run_analysis.py` または
`python -m analytics.pipeline` を推奨。
"""

from __future__ import annotations

import plotly.io as pio


def main() -> None:
    pio.defaults.default_scale = 2
    from analytics.pipeline import run_pipeline

    run_pipeline()


if __name__ == "__main__":
    main()
