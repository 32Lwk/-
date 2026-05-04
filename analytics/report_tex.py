from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics.config import ART_DIR, LATEX_DIR, TABLES_DIR
from analytics.governance import write_governance_ethics_tex


def _latex_escape_cell(val: object) -> str:
    t = str(val)
    repl = (
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
    )
    for a, b in repl:
        t = t.replace(a, b)
    return t


def write_data_sample_tex(processed_csv: Path | None = None) -> None:
    """レポート用：代表行を LaTeX の tabular 断片として出力する。"""
    path = processed_csv or (ART_DIR / "processed.csv")
    if not path.is_file():
        return
    df = pd.read_csv(path)
    picks: list[pd.Series] = []
    seen: set[int] = set()
    for sel in (
        df["conversion"] == 1,
        df["conversion"] == 0,
    ):
        sub = df.loc[sel]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        i = int(r.name)
        if i not in seen:
            picks.append(r)
            seen.add(i)
    if len(df) > 0:
        r = df.loc[df["history"].idxmax()]
        i = int(r.name)
        if i not in seen:
            picks.append(r)
            seen.add(i)
    if len(df) > 0:
        r = df.loc[df["recency"].idxmin()]
        i = int(r.name)
        if i not in seen:
            picks.append(r)
            seen.add(i)
    need = max(0, 5 - len(picks))
    pool = df.drop(index=list(seen), errors="ignore")
    if need > 0 and len(pool) > 0:
        extra = pool.sample(n=min(need, len(pool)), random_state=7, replace=False)
        for _, r in extra.iterrows():
            picks.append(r)
    picks = picks[:5]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{分析に用いた顧客レコードの例（抜粋。\texttt{customer\_id} は分析パイプラインが付与した行番号）。}",
        r"\label{tab:data_examples}",
        r"\footnotesize",
        r"\begin{tabular}{@{}rrrrllllr@{}}",
        r"\toprule",
        r"\texttt{id} & \texttt{recency} & \texttt{history} & \texttt{u\_disc} & \texttt{u\_bogo} & \texttt{zip} & \texttt{ref} & \texttt{channel} & \texttt{offer} & \texttt{conv} \\",
        r"\midrule",
    ]
    for r in picks:
        cells = [
            int(r["customer_id"]),
            int(r["recency"]),
            f"{float(r['history']):.2f}",
            int(r["used_discount"]),
            int(r["used_bogo"]),
            _latex_escape_cell(r["zip_code"]),
            int(r["is_referral"]),
            _latex_escape_cell(r["channel"]),
            _latex_escape_cell(r["offer"]),
            int(r["conversion"]),
        ]
        lines.append(" & ".join(str(c) for c in cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "data_sample_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# `final_report.md` の §3.4 / §3.5 / §4.5 と同じ図パス（LaTeX はメイン .tex からの相対パス）
_STATS_LSI_TEX = r"""\section{相関と補助的重回帰（history）}

\noindent\textbf{データと分析の見方（この章）.}\\
\textbf{データ}：\texttt{exercise.csv} から取り出した数値列（\texttt{recency}, \texttt{history}）と二値フラグ（割引・BOGO利用・紹介など）および \texttt{conversion}。\\
\textbf{やっていること}：変数同士が\textbf{同時に大きく動くか}を相関で眺め、続けて \texttt{history} を\textbf{目的変数}とした直線の当てはめ（OLS）で、購入予測とは別角度から\textbf{顧客価値の説明}を付ける。\\
\textbf{読み手が得るもの}：特徴量が\textbf{重複している（冗長）}かの手がかりと、価値proxyがどの軸で動くかの\textbf{補助的な説明}（因果の証明ではない）。

\subsection{Pearson 相関（数値・二値）}
図 \ref{fig:corr} は主要な数値・二値列の Pearson 相関を\textbf{下三角のみ}表示したものである（重複セルを避ける）。
強い相関は特徴設計上の\textbf{冗長性}の手がかりとなるが、因果ではない。

\begin{figure}[!htbp]
  \centering
  \begin{subfigure}[t]{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{../figures/improvements/corr_numeric_detailed.png}
    \caption{相関行列（Pearson、下三角）}
    \label{fig:corr}
  \end{subfigure}\hfill
  \begin{subfigure}[t]{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{../figures/improvements/scatter_recency_history_by_conversion.png}
    \caption{recency と history の散布図（色は購入有無）}
    \label{fig:scatter_rh}
  \end{subfigure}
  \caption{数値・二値列の Pearson 相関（下三角）と recency--history 散布図}
\end{figure}

\subsubsection*{解釈（相関・散布図）}
相関は「2変数が一緒に増減する度合い」の\textbf{要約}であり、どちらが原因とは言えない。
散布図は購入有無で色分けすることで、\textbf{同じ $(\texttt{recency},\texttt{history})$ 付近でも購入のばらつき}が残ることを直感的に示す——だから後段でロジスティック等の\textbf{多変量モデル}が必要になる。

\subsection{OLS（目的変数を history）}
演習で例示される重回帰に対応するため、連続の \texttt{history} を目的変数とした OLS（説明変数は \texttt{recency}, \texttt{used\_discount}, \texttt{used\_bogo}, \texttt{is\_referral}、ロバスト分散 HC1）を付す。
購入予測そのものは次節以降のロジスティック回帰が主である。係数表は \texttt{artifacts/ols\_history\_coefficients.csv}。

\begin{figure}[!htbp]
  \centering
  \includegraphics[width=0.78\linewidth]{../figures/improvements/ols_history_coefficients.png}
  \caption{OLS 係数と 95\% 信頼区間（history を目的変数）}
  \label{fig:ols_hist}
\end{figure}

\subsubsection*{解釈（OLS）}
ここでのOLSは「\texttt{history}（過去の購入規模proxy）を、時間軸とプロモ利用履歴・紹介でどれだけ説明できるか」を見る\textbf{補助分析}である。
係数の符号は\textbf{制御したあとの関連}の方向を示すが、未観測要因や設計バイアスが残れば誤解釈しうる。本稿の主目的である\textbf{購入の予測}そのものは次章以降の分類モデルが担う。

\subsection*{章末まとめ（相関・補助OLS）}
\begin{itemize}\setlength{\itemsep}{0.2em}
  \item 相関と散布図で、特徴量の\textbf{重なり}と分布の\textbf{異質性}を確認した。
  \item OLSは \texttt{history} の\textbf{説明}用で、購入予測の代替ではない。
\end{itemize}

\section{LSI類似表現と分類アブレーション}

\noindent\textbf{データと分析の見方（この章）.}\\
\textbf{データ}：各行を短い英語テンプレ文に変換したテキスト（外部APIなしで再現可能）と、従来の数値・カテゴリ特徴。\\
\textbf{やっていること}：テキストから「単語の出現パターン」を数ベクトルにし（TF-IDF）、次元を圧縮（SVD）してロジスティック回帰に足し、\textbf{ベース特徴だけ}のときと\textbf{足したとき}でホールドアウト性能を比較する（アブレーション）。\\
\textbf{読み手が得るもの}：テキスト由来の情報が、購入予測の\textbf{順位付け}を実務上どれだけ改善するかの\textbf{上限の手がかり}（改善が小さければ、まずベース特徴と運用データの整備が先、という判断材料になる）。

各行を英語テンプレの短文に変換し、\textbf{訓練標本のみ}に fit した TF-IDF と TruncatedSVD（次元は \texttt{artifacts/lsi\_tfidf\_diag.json}）を特徴に連結し、
ロジスティック回帰のホールドアウト性能を \textbf{BASE のみ} と比較する（外部 API 不要）。
図 \ref{fig:lsi_ab} は同一分割における AUC / Brier / AP / TopK\% CVR の対比である。

\begin{figure}[!htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{../figures/improvements/ablation_base_vs_lsi_logreg.png}
  \caption{BASE vs BASE+TF-IDF+SVD（同一ホールドアウト・値ラベル付き棒）}
  \label{fig:lsi_ab}
\end{figure}

\subsubsection*{解釈（LSI アブレーション）}
棒が大きく伸びなければ、「短文テンプレ由来の追加信号は、この分割では\textbf{限定的}」と読める。
逆にAUCやTopK\%が伸びれば、\textbf{テキスト化できる顧客記述}（コメント・カテゴリの組合せ等）を特徴に載せる価値がある。
いずれにせよ、本稿の主眼は\textbf{政策接続}のため、テキスト特徴は補助線として位置づける。

\subsection*{章末まとめ（LSI アブレーション）}
\begin{itemize}\setlength{\itemsep}{0.2em}
  \item TF-IDF+SVD を足したときの\textbf{ホールドアウト性能}を、同一条件でベースラインと比較した。
  \item 改善幅はデータとテンプレ設計に依存する——\textbf{再現手順}（\texttt{artifacts/lsi\_tfidf\_diag.json}）とセットで解釈する。
\end{itemize}

\FloatBarrier
"""


def write_generated_stats_lsi_sections_tex() -> None:
    """相関・OLS・LSI アブレーション節を `latex/generated_stats_lsi_sections.tex` に書き出す。

    `latex/final_report.tex` から \\input され、Markdown レポート（report_md.py）と図表パスを揃える。
    """
    path = LATEX_DIR / "generated_stats_lsi_sections.tex"
    path.write_text(_STATS_LSI_TEX.strip() + "\n", encoding="utf-8")


def write_latex_bundle() -> None:
    """ガバナンス用 .tex と、MD と対応する統計・LSI 断片を生成する。

    メインの `latex/final_report.tex` 本体はリポジトリで保守するが、
    相関・OLS・LSI 節は本モジュールが `generated_stats_lsi_sections.tex` として出力し、
    メイン文書は \\input で取り込む（report_md / pipeline と単一ソース化）。
    """
    write_governance_ethics_tex()
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    write_generated_stats_lsi_sections_tex()
    write_data_sample_tex()
