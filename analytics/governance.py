from __future__ import annotations

from analytics.config import ART_DIR, TABLES_DIR


def write_data_dictionary() -> None:
    lines = [
        "# データ辞書（目的限定・最小化のたたき台）\n\n",
        "本分析で用いる列と目的。**法的助言ではなく**社内テンプレとして利用すること。\n\n",
        "| 列 | 目的 | 保存期間の考え方（プレースホルダ） |\n",
        "|---|---|---|\n",
        "| customer_id | 再現性のための疑似ID | 分析プロジェクト期間 |\n",
        "| recency | 購買活動の説明 | 同上 |\n",
        "| history | 価値proxy | 同上 |\n",
        "| used_discount / used_bogo | 行動履歴 | 同上 |\n",
        "| zip_code | 地域カテゴリ（プロキシ） | 差別リスクレビュー |\n",
        "| is_referral | 流入経路 | 同上 |\n",
        "| channel / offer | 処置（観察） | 同上 |\n",
        "| conversion | アウトカム | 同上 |\n\n",
        "## 個人情報保護法（日本）に関する留意\n\n",
        "- 個人データに該当する場合は**利用目的の特定・公表**、**不要な個人データの削除**等を検討すること。\n",
        "- 自動化された決定に関する**説明・人の関与**の要否は、実装形態に依存する。\n",
    ]
    (ART_DIR / "data_dictionary.md").write_text("".join(lines), encoding="utf-8")


def write_governance_ethics_tex() -> None:
    """LaTeX \\input 用：倫理・人の関与・免責のたたき台（gov__02）。"""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tex = r"""\paragraph{目的限定と最小化}
分析目的に必要な列のみを用い、保存期間・アクセス権は社内規程に従う（詳細は \texttt{data\_dictionary.md}）。

\paragraph{説明可能性と人の関与}
セグメント別施策はモデル出力に基づくが、最終配信判断には担当者レビューを置く。自動化された決定のみで不利益を生じさせない運用を推奨。

\paragraph{差別・不利益リスク}
地域（\texttt{zip\_code} 等）などのプロキシ特性に依存する政策は、属性に基づく不当な差別に繋がらないよう設計・監査する。

\paragraph{オプトアウト・問い合わせ}
マーケティング配信の拒否・個人情報の開示等の請求手続きは、社内ポリシーおよび個人情報保護法（日本）の枠組みに従う。

\paragraph{法的助言ではない}
本稿・生成テンプレは\textbf{法的助言を構成しない}。実装・運用前に法務・DPO と整合を取ること。
"""
    (TABLES_DIR / "governance_ethics.tex").write_text(tex, encoding="utf-8")
