"""表形式の1行を英語テンプレの短文に変換（TF-IDF + SVD 用の疑似文書）。"""

from __future__ import annotations

import pandas as pd


def rows_to_english_text(df: pd.DataFrame) -> pd.Series:
    """customer_id / conversion を除き、各列を読みやすい英語フレーズに並べる。"""
    work = df.drop(columns=["conversion"], errors="ignore")
    out: list[str] = []
    for _, r in work.iterrows():
        parts = [
            f"recency_months={float(r['recency']):.2f}",
            f"history_value={float(r['history']):.2f}",
            f"used_discount={int(r['used_discount'])}",
            f"used_bogo={int(r['used_bogo'])}",
            f"zip_area={r['zip_code']}",
            f"referral={int(r['is_referral'])}",
            f"channel={r['channel']}",
            f"offer={r['offer']}",
        ]
        out.append(" ".join(parts))
    return pd.Series(out, index=df.index, name="row_text")
