# Flask から FastAPI への移行実装メモ

## 現状サマリ

このリポジトリには Flask アプリケーションのエントリポイントやルート定義が存在しなかったため、既存の Flask ルートを 1:1 で移植するのではなく、分析パイプラインを操作する ASGI アプリケーションを FastAPI で新規実装しました。

- 確認日: 2026-05-09
- 確認コマンド:
  - `rg -n "Flask|flask|@app\.route|Blueprint|request|jsonify|render_template|FastAPI|fastapi|uvicorn|app.py|wsgi" -g '*.py' .`
- 移行方針: Flask 実装が未検出のため、既存の `analytics.pipeline.run_pipeline()` を FastAPI から非同期バックグラウンド実行できる API として公開

## 1. FastAPI エントリポイント

| パス | 役割 |
|---|---|
| `web/main.py` | FastAPI アプリケーションファクトリ、lifespan hook、exception handler |
| `web/routers/health.py` | ヘルスチェックルート |
| `web/routers/analysis.py` | 分析実行・状態確認・成果物一覧・成果物取得ルート |
| `web/schemas.py` | Pydantic request / response model |
| `web/dependencies.py` | dependency injection 用の依存取得関数 |
| `web/services.py` | 分析パイプライン実行管理、成果物一覧・解決ロジック |
| `web/config.py` | API 用設定オブジェクト |
| `web/exceptions.py` | API ドメイン例外 |

ASGI アプリケーションは `web.main:app` です。

## 2. Flask 固有実装の棚卸し結果

現時点で検出された Flask 固有実装はありません。新規 FastAPI 実装で置き換えたため、Flask 依存は追加していません。

| 種別 | 検索対象 | 検出結果 | FastAPI 側の実装 |
|---|---|---:|---|
| ルート定義 | `@app.route` | 0 件 | `APIRouter` |
| Blueprint | `Blueprint` | 0 件 | `web/routers/*.py` |
| リクエスト参照 | `request` | 0 件 | Pydantic model / dependency |
| JSON レスポンス | `jsonify` | 0 件 | response model / `JSONResponse` |
| テンプレート描画 | `render_template` | 0 件 | なし。成果物は `FileResponse` で提供 |
| エラーハンドラ | `@app.errorhandler` | 0 件 | `app.add_exception_handler(...)` |
| 初期化フック | `before_request` など | 0 件 | lifespan hook |

## 3. 実装済み FastAPI ルート

| HTTP メソッド | URL パス | 入力 | レスポンス | ステータスコード | 副作用 | 実装 |
|---|---|---|---|---|---|---|
| `GET` | `/health` | なし | API 名、バージョン、データファイル有無 | `200` | なし | `web/routers/health.py` |
| `POST` | `/api/analysis/run` | JSON body: `force: bool = false` | 実行受付状態、開始時刻 | `202` / 実行中は `409` | `analytics.pipeline.run_pipeline()` をバックグラウンド実行し、`artifacts/`, `figures/`, `latex/`, `final_report.md` 等を更新 | `web/routers/analysis.py` |
| `GET` | `/api/analysis/status` | なし | 実行状態、開始・終了時刻、エラー | `200` | なし | `web/routers/analysis.py` |
| `GET` | `/api/artifacts` | なし | 成果物パス、サイズ、更新時刻の一覧 | `200` | なし | `web/routers/analysis.py` |
| `GET` | `/api/artifacts/{artifact_path}` | path parameter: 成果物の相対パス | ファイル本体 | `200` / 未検出は `404` | なし | `web/routers/analysis.py` |

## 4. Flask から FastAPI への対応実装

| Flask の要素 | FastAPI の移植先 | このリポジトリでの実装 |
|---|---|---|
| `@app.route` | `APIRouter` の `@router.get/post/...` | `web/routers/health.py`, `web/routers/analysis.py` |
| `Blueprint` | 機能単位の `APIRouter` | health / analysis で router 分割 |
| `request.get_json()` | Pydantic request model | `AnalysisRunRequest` |
| `jsonify(...)` | dict / Pydantic response model | `HealthResponse`, `AnalysisRunResponse`, `AnalysisStatusResponse`, `ArtifactListResponse` |
| `send_file(...)` | `FileResponse` | 成果物ダウンロード API |
| `abort(...)` | `HTTPException` またはドメイン例外 + handler | `ArtifactNotFoundError`, `PipelineAlreadyRunningError` |
| `@app.errorhandler` | `@app.exception_handler` / `add_exception_handler` | `web/main.py` |
| `before_request` / `after_request` | dependency / middleware / lifespan hook | `get_settings`, `get_analysis_runner`, `lifespan` |
| アプリ初期化 | application factory / lifespan hook | `create_app()`, `initialize_runtime()` |
| DB セッション | dependency injection | 現状 DB なし。将来追加する場合は `web/dependencies.py` に集約 |

## 5. 依存関係と起動手順

ASGI 実行依存と request / response model 定義のため、`requirements.txt` に以下を追加しました。

```text
fastapi>=0.115.0
pydantic>=2.7.0
uvicorn[standard]>=0.30.0
```

起動コマンド:

```bash
uvicorn web.main:app --reload
```

OpenAPI UI:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## 今後 Flask 実装が追加された場合の移植手順

1. 追加された Flask ルートを、上記「実装済み FastAPI ルート」と同じ形式で棚卸しする。
2. JSON 入力は Pydantic model、共通処理は dependency、初期化処理は lifespan hook に寄せる。
3. Flask セッションやグローバル状態に依存する処理は、明示的な dependency と設定オブジェクトに分離する。
4. 既存 API の response model とステータスコードを維持しながら、`web/routers/` に router を追加する。
