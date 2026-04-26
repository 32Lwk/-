from __future__ import annotations

from analytics.config import LATEX_DIR
from analytics.governance import write_governance_ethics_tex


def write_latex_bundle() -> None:
    """ガバナンス用の .tex を生成する。

    本文の `latex/final_report.tex` はリポジトリで保守する（理系レポートの構成・考察の編集は当該ファイルを直接編集）。
    本関数では **上書きしない**（過去の自動生成テンプレで手編集を潰すのを防ぐ）。
    """
    write_governance_ethics_tex()
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
