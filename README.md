# キカガク分析パイプライン

## 実行

```bash
pip install -r requirements.txt
python run_analysis.py
```

図中の日本語は **BIZ UDGothic**（BIZ UD ゴシック）を最優先します。未インストール時は Meiryo 等へ自動フォールバックします。推奨: [BIZ UD フォント（Google Fonts）](https://fonts.google.com/) を OS にインストール。

従来どおり `python analysis.py` でも実行可能（内部で同一パイプライン）。

## 主な成果物

| 種別 | パス |
|------|------|
| Markdown レポート | `final_report.md` |
| LaTeX 理系レポート | `latex/final_report.tex`（LuaLaTeX + `luatexja`、既定で Harano Aji 等） |
| 図（本編） | `figures/` |
| 図（改善・診断） | `figures/improvements/` |
| 表・CSV | `artifacts/`、`artifacts/tables/*.tex` |

## LaTeX（任意）

MiKTeX / TeX Live などで [LuaLaTeX](https://www.luatex.org/) と `luatexja` が入っている環境で（Harano Aji は多くの配布に同梱。`luatexja-preset` は古い `fontspec` と衝突することがあるため本テンプレでは未使用）:

```bash
cd latex
latexmk -lualatex final_report.tex
```

## パラメータ

`analytics/config.py` の `BOOTSTRAP_B`、`BOOTSTRAP_N`、`CONSTRAINT_*` などを調整可能。

## モジュール構成

- `analytics/pipeline.py` … 一括実行
- `analytics/causal.py` … DR・ブートストラップ
- `analytics/policy.py` … 期待利益・OOF・ベンチマーク・制約
- その他 `analytics/*.py`
