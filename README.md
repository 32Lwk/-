# キカガク分析パイプライン

## 実行

```bash
pip install -r requirements.txt
python run_analysis.py
```

従来どおり `python analysis.py` でも実行可能（内部で同一パイプライン）。

## 主な成果物

| 種別 | パス |
|------|------|
| Markdown レポート | `final_report.md` |
| LaTeX 理系レポート | `latex/final_report.tex`（LuaLaTeX + `ltjsarticle` 想定） |
| 図（本編） | `figures/` |
| 図（改善・診断） | `figures/improvements/` |
| 表・CSV | `artifacts/`、`artifacts/tables/*.tex` |

## LaTeX（任意）

Windows で [LuaLaTeX](https://www.luatex.org/) と日本語フォント（例: Yu Gothic）が利用可能な環境で:

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
