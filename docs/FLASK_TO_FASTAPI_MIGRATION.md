# Flask から FastAPI への移行準備メモ

## 現状サマリ

このリポジトリは、現時点では分析パイプラインとレポート生成物を中心に構成されており、Flask アプリケーションとして移行できるエントリポイントは確認できていません。

- 確認日: 2026-05-09
- 確認コマンド:
  - `rg -n "Flask|flask|@app\.route|Blueprint|request|jsonify|render_template|FastAPI|fastapi|uvicorn|app.py|wsgi" -g '*.py' .`
- 結果: 対象実装なし

そのため、FastAPI への「完全移行」は、まず移行対象となる Flask アプリケーションコードをリポジトリに追加するか、外部にある対象パスを明示してから実施してください。

## 1. 移行対象エントリポイント

| 候補 | 確認結果 | 備考 |
|---|---:|---|
| `app.py` | 未検出 | リポジトリ直下に Flask エントリポイントなし |
| `wsgi.py` | 未検出 | WSGI 起動用ファイルなし |
| `src/app.py` | 未検出 | `src/` 配下の Flask アプリなし |
| `web/app.py` | 未検出 | `web/` 配下の Flask アプリなし |
| その他 Python モジュール | 未検出 | `analytics/` と `scripts/` は分析・レポート生成用途 |

移行作業を進めるには、次のいずれかを実施してください。

1. Flask アプリをこのリポジトリに追加する。
   - 例: `web/app.py`, `web/routes.py`, `web/templates/`, `web/static/`
2. 既存アプリが別リポジトリや別ディレクトリにある場合は、対象パスを明示する。
   - 例: `/path/to/flask_app/app.py`

## 2. Flask 固有実装の一覧化

現時点で検出された Flask 固有実装はありません。

| 種別 | 検索対象 | 検出結果 |
|---|---|---:|
| ルート定義 | `@app.route` | 0 件 |
| Blueprint | `Blueprint` | 0 件 |
| リクエスト参照 | `request` | 0 件 |
| JSON レスポンス | `jsonify` | 0 件 |
| テンプレート描画 | `render_template` | 0 件 |
| Flask 本体 | `Flask`, `flask` | 0 件 |
| FastAPI/ASGI | `FastAPI`, `fastapi`, `uvicorn` | 0 件 |

Flask アプリが追加されたら、少なくとも次の観点で再棚卸ししてください。

- 認証・認可処理
- DB 接続、セッション、マイグレーション、初期化処理
- エラーハンドラ
- before/after request 相当のフック
- CORS、圧縮、ログ、メトリクスなどのミドルウェア相当処理
- テンプレート、静的ファイル、ファイルアップロード

## 3. ルート棚卸しテンプレート

Flask ルートが追加されたら、以下の表をルートごとに埋めてから FastAPI に移植してください。

| HTTP メソッド | URL パス | 入力 | レスポンス | ステータスコード | 副作用 | FastAPI 移植先 |
|---|---|---|---|---|---|---|
| 未定 | 未定 | 未定 | 未定 | 未定 | 未定 | `APIRouter` / Pydantic モデル / dependency |

入力は、少なくとも次の分類で整理します。

- path parameter
- query parameter
- JSON body
- form data
- file upload
- cookie / header
- session / current user

副作用は、少なくとも次の分類で整理します。

- DB 読み書き
- 外部 API 呼び出し
- ファイル I/O
- メール・通知送信
- キャッシュ更新
- ジョブキュー投入

## 4. FastAPI 移植方針

Flask 実装が確認できた後、次の対応表で移植します。

| Flask の要素 | FastAPI の移植先 |
|---|---|
| `@app.route` | `APIRouter` の `@router.get/post/...` |
| `Blueprint` | 機能単位の `APIRouter` |
| `request.args` | 型付き query parameter |
| `request.get_json()` | Pydantic request model |
| `jsonify(...)` | dict / Pydantic response model / `JSONResponse` |
| `render_template(...)` | `Jinja2Templates` または API 化 |
| `abort(...)` | `HTTPException` |
| `@app.errorhandler` | `@app.exception_handler` |
| `before_request` / `after_request` | dependency / middleware / lifespan hook |
| アプリ初期化 | application factory または lifespan hook |
| DB セッション | dependency injection |

## 5. 依存関係と起動手順について

`requirements.txt` にはまだ `fastapi` や `uvicorn` は追加していません。理由は、移行対象の Flask アプリが未確認であり、ASGI アプリケーションのエントリポイントも未定のためです。

移行対象が確定した時点で、次のような依存を追加してください。

```text
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
```

FastAPI エントリポイント作成後は、`README.md` に次のような起動手順を追加します。

```bash
uvicorn web.main:app --reload
```

## 次のアクション

1. Flask アプリのエントリポイントを追加、または対象パスを指定する。
2. 本ドキュメントの「Flask 固有実装の一覧化」と「ルート棚卸しテンプレート」を実コードに合わせて更新する。
3. 棚卸し結果をもとに FastAPI のルータ、Pydantic モデル、dependency、exception handler、lifespan hook を実装する。
4. ASGI 依存関係を `requirements.txt` に追加し、`README.md` に起動手順を追記する。
