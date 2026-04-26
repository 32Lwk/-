from __future__ import annotations

from pathlib import Path

from analytics.config import LATEX_DIR


def write_latex_bundle() -> None:
    """理系最終レポート骨格（LuaLaTeX + ltjsarticle 想定）。"""
    tex = r"""\documentclass[11pt]{ltjsarticle}
\usepackage{luatexja-fontspec}
\setmainjfont{YuGothic-Medium}[BoldFont=YuGothic-Bold]
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}

\title{期待利益最大化に向けた分析レポート\\（観察データ・オフポリシー評価・実験計画）}
\author{分析パイプライン自動生成}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
本稿は、店舗顧客データに基づき、購入確率モデル、多値介入の傾向スコアと二重頑健（DR）推定、
期待利益最大化政策、およびセグメント頑健性を整理する。\textbf{観察データの限界}として未観測交絡があり、
識別仮定の下での参考推定である。増分効果の確定にはランダム化比較試験（RCT）を要する。
会計上の粗利・LTVは本データに含まれないため、\texttt{history} を売上proxyとし、
感度分析のみを付す（絶対額の断定はしない）。
\end{abstract}

\section{問題設定と記号}
処置 $T \in \mathcal{T}$ を（オファー, チャネル）の組とする。共変量 $X$、アウトカム $Y \in \{0,1\}$（購入）。
観察データ下で、各 $t$ について $\mu(t) = \mathbb{E}[Y(t)]$ のプラグイン推定として DR 推定量
\[
\hat{\mu}_{\mathrm{DR}}(t) = \frac{1}{n}\sum_{i=1}^n \left( \hat{\mu}(t, X_i) + \frac{\mathbf{1}\{T_i=t\}}{\hat{e}(t\mid X_i)} (Y_i - \hat{\mu}(t, X_i)) \right)
\]
を用いる。ここで $\hat{e}$ は多値ロジット傾向、$\hat{\mu}$ は結果回帰。$\hat{e}$ は閾値 $0.01$ でクリップ。

\section{期待利益（proxy）}
\[
\widehat{\Pi}_i(t) = \hat{P}(Y=1\mid X_i, t)\cdot \mathrm{history}_i - c_{\mathrm{offer}}(t)\cdot \mathrm{history}_i - c_{\mathrm{ch}}(t)\cdot \mathrm{history}_i.
\]
コスト係数はシナリオで与える（\texttt{analytics/config.py}）。

\section{オフポリシー評価とOOF}
全件学習による $\hat{P}(y\mid x,t)$ は楽観バイアスを生む。層化 $K$ 分割でアウトカムモデルを学習し、
ホールドアウト上で予測をスタックした期待利益を併記する。

\section{セグメントと頑健性}
UMAP 上で $k$ をシルエットで選び、別シードの KMeans 間で調整ランド指数（ARI）を算出する。

\section{実験計画（A/B）}
主アウトカムを購入率と置き、MDE・検定力・必要サンプルは表 \ref{tab:ab} を参照。

\begin{table}[ht]
\centering
\caption{A/B 必要サンプル（二標本比例・正規近似）}
\label{tab:ab}
\small
\input{../artifacts/tables/ab_design.tex}
\end{table}

\section{ガバナンス（テンプレ）}
個人情報保護法（日本）上の目的限定・最小化、説明可能性、不利益が生じうる属性プロキシ（地域等）については
\texttt{artifacts/data\_dictionary.md} を参照。本文書は法的助言を構成しない。

\section{モニタリング}
校正、ドリフト、傾向スコアの ESS を監視する（\texttt{artifacts/monitoring\_spec.json}）。

\begin{table}[ht]
\centering
\caption{監視指標の例}
\small
\input{../artifacts/tables/monitoring_summary.tex}
\end{table}

\section{図表（主要）}
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.75\linewidth]{../figures/improvements/dr_bootstrap_ci.png}
  \caption{DR（ホールドアウト点推定）とブートストラップ区間の例}
\end{figure}

\section{限界}
\begin{itemize}
  \item 未観測交絡・SUTVA・クーポン慣習化など（\texttt{risk\_register.json}）。
  \item ブートストラップは部分標本再学習であり、計算コストとのトレードオフがある。
\end{itemize}

\end{document}
"""
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    (LATEX_DIR / "final_report.tex").write_text(tex, encoding="utf-8")
