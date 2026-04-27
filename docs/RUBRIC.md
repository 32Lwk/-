# 採点ルーブリック（演習・再現性）

## 再現コマンド

```bash
pip install -r requirements.txt
python run_analysis.py
```

（従来どおり `python analysis.py` でも可。）

## 必須成果物（最低限）

| 項目 | 確認方法 |
|------|----------|
| `final_report.md` が生成される | ルートに存在 |
| `artifacts/run_config.json` に `evaluation_protocol` がある | JSON を開く |
| 相関・OLS | `figures/improvements/corr_numeric_detailed.png`, `ols_history_coefficients.png` |
| LSI アブレーション | `artifacts/model_eval_ablation_lsi.csv`, `figures/improvements/ablation_base_vs_lsi_logreg.png` |
| セグメント×処置（探索） | `artifacts/segment_treatment_cvr_exploratory.csv`, ヒートマップ PNG |
| 期待利益を第1指標として読める | `final_report.md` §0・§7 |
| 観察データの限界 | §12 または LaTeX 付録 |

## 許容差

- 乱数シード 42 固定のため、同一 `exercise.csv` では AUC 等は小数第3位程度まで一致することが多い。
- フォント未導入環境では図の日本語がフォールバックするが、ファイル生成自体は必須。
