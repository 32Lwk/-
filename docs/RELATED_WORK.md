# Related Work（業界テクニカルレポート用・短文）

- **表形式データの表現学習**: Table Representation Learning（TRL ワークショップ系）、行の言語化＋埋め込み、TabPFN / TabICL 等の表向け基盤モデル。
- **LSI / 低ランク**: 文書–語行列の SVD（LSI）に対応する実装として、本パイプラインでは **TF-IDF + TruncatedSVD** を「疑似文書」（行テンプレ）に適用。
- **因果 × 表現**: 観察データでの処置効果推定に対し、表現空間での分布バランス（例: representation balancing）や ITE 推定の研究がある。本稿は **DR + ホールドアウト** を主とし、層別は探索的記述に留める。
- **限界**: 本リポジトリは演習データ想定。未観測交絡・SUTVA・コスト proxy の妥当性は実務データで再検証が必要。
