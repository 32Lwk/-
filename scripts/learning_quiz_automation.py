"""
学習サイトの「解答を送信」までをブラウザ操作で自動化する補助スクリプト。

前提:
  - Playwright for Python
  - pip install playwright && playwright install chromium

利用上の注意:
  - サービス利用規約・試験ポリシーを遵守すること。
  - ログイン情報をコードに直書きしないこと（storage state ファイルを使用）。

例:
  # 1) ログイン状態を保存（ID は引数、パスワードは対話または環境変数 LEARNING_LOGIN_PASSWORD）
  python scripts/learning_quiz_automation.py save-login --start-url "https://example.com/login" --login-id "you@example.com"

  # 2) 問題ページを開き、選択肢テキストに一致するチェックをオンにして送信
  python scripts/learning_quiz_automation.py submit \\
    --quiz-url "https://example.com/quiz/..." \\
    --storage learning_auth.json \\
    --check-text "選択肢Aの一部" "選択肢Bの一部"

  # 3) タブパネル内の各設問ブロック（既定: div.css-pr2tx6）のチェックボックスについて全組み合わせを試す
  python scripts/learning_quiz_automation.py exhaustive \\
    --quiz-url "https://example.com/quiz/..." \\
    --storage learning_auth.json \\
    --out-csv artifacts/quiz_exhaustive_log.csv
"""

from __future__ import annotations

import argparse
import atexit
import csv
import getpass
import io
import itertools
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    print("playwright が未インストールです: pip install playwright && playwright install chromium", file=sys.stderr)
    raise


def _ts() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_stdio_line_buffered() -> None:
    """ターミナルでログが溜まらず即時表示されるよう stdout/stderr を行バッファにする。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
                pass


def _stderr_write_now(line: str) -> None:
    """
    sys.stderr の TextIO 経由で出力する（Windows コンソールのコードページと一致させ、日本語の文字化けを防ぐ）。
    行末ですぐ flush するので表示はほぼリアルタイム。
    """
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except (OSError, ValueError, UnicodeError):
        try:
            print(line, file=sys.stderr, flush=True)
        except OSError:
            pass


class _RunLogger:
    """標準エラーと任意のログファイルへ同一行を出す。"""

    def __init__(self, log_file: Path | None):
        self._log_fp = None
        if log_file:
            path = log_file.resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            # buffering=1: 行バッファ（テキストモード）
            self._log_fp = open(path, "a", encoding="utf-8", buffering=1)

    def log(self, msg: str) -> None:
        line = f"{_ts()} {msg}"
        _stderr_write_now(line)
        if self._log_fp:
            self._log_fp.write(line + "\n")
            self._log_fp.flush()

    def close(self) -> None:
        if self._log_fp:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except OSError:
                pass
            self._log_fp = None


def _wait_checkbox_attached(tab_panel, timeout_ms: int) -> None:
    try:
        tab_panel.locator('input[type="checkbox"]').first.wait_for(
            state="attached",
            timeout=timeout_ms,
        )
    except Exception as e:
        name = type(e).__name__
        if name == "TargetClosedError":
            raise SystemExit(
                "ブラウザまたはタブが閉じられました（TargetClosedError）。"
                "自動実行中は Playwright のウィンドウを閉じないでください。"
            ) from e
        raise


def _navigate_quiz(
    page,
    url: str,
    timeout_ms: int,
    wait_until: str,
    rlog: _RunLogger | None = None,
) -> None:
    """SPA 向け: goto →（任意）networkidle 追加待機。networkidle で goto が落ちたら load で再試行。"""
    try:
        page.goto(url, wait_until=wait_until, timeout=timeout_ms)
    except PlaywrightTimeout:
        if wait_until == "networkidle":
            if rlog:
                rlog.log(
                    "Page.goto が networkidle でタイムアウトしました。"
                    "wait_until=load で再試行します（--goto-wait load 推奨のこともあります）。"
                )
            page.goto(url, wait_until="load", timeout=timeout_ms)
        else:
            raise
    try:
        page.wait_for_load_state("networkidle", timeout=min(30_000, timeout_ms))
    except PlaywrightTimeout:
        pass


def _frame_checkbox_input_count(frame) -> int:
    try:
        return frame.locator('input[type="checkbox"]').count()
    except Exception:
        return 0


def _page_has_checkbox_input_anywhere(page) -> bool:
    """メインフレーム・子 iframe いずれかにネイティブ checkbox input があるか。"""
    return any(_frame_checkbox_input_count(fr) > 0 for fr in page.frames)


def _selector_targets_tabpanel(root_selector: str | None) -> bool:
    if not root_selector:
        return False
    return "tabpanel" in root_selector.lower()


def _first_visible_tabpanel_with_checkbox(fr, timeout_ms: int):
    """Chakra Tabs: 非表示パネル内の input は操作できないため is_visible なパネルを優先。"""
    panels = fr.locator('[role="tabpanel"]')
    n = panels.count()
    fallback = None
    for i in range(n):
        pan = panels.nth(i)
        try:
            pan.wait_for(state="attached", timeout=min(5_000, timeout_ms))
        except PlaywrightTimeout:
            continue
        if pan.locator('input[type="checkbox"]').count() == 0:
            continue
        if fallback is None:
            fallback = pan
        try:
            if pan.is_visible():
                return pan
        except Exception:
            continue
    return fallback


def _wait_until_page_has_checkbox(
    page,
    total_timeout_ms: int,
    rlog: _RunLogger | None,
) -> None:
    """全フレームをポーリングし input[type=checkbox] が付くまで待つ。"""
    deadline = time.perf_counter() + total_timeout_ms / 1000
    last_log = 0.0
    while time.perf_counter() < deadline:
        if _page_has_checkbox_input_anywhere(page):
            return
        now = time.perf_counter()
        if rlog and now - last_log >= 10.0:
            left = max(0, int(deadline - now))
            try:
                cur_url = page.url
            except Exception:
                cur_url = "(取得不可)"
            rlog.log(
                f"チェックボックス待機中… 残り約 {left}s "
                f"現在URL={cur_url!r} frames={len(page.frames)}"
            )
            last_log = now
        time.sleep(0.45)
    raise PlaywrightTimeout(
        f"{total_timeout_ms}ms 以内にチェックボックスが現れませんでした"
    )


def _quiz_root_for_frame(
    fr,
    root_selector: str | None,
    timeout_ms: int,
):
    """単一 Frame 内でチェックを含む Locator を返す。見つからなければ None。"""
    if _frame_checkbox_input_count(fr) == 0:
        return None
    if root_selector and _selector_targets_tabpanel(root_selector):
        hit = _first_visible_tabpanel_with_checkbox(fr, timeout_ms)
        if hit is not None:
            return hit
    if root_selector:
        loc = fr.locator(root_selector).first
        try:
            loc.wait_for(state="attached", timeout=min(10_000, timeout_ms))
            if loc.locator('input[type="checkbox"]').count() > 0:
                return loc
        except PlaywrightTimeout:
            pass
    hit = _first_visible_tabpanel_with_checkbox(fr, timeout_ms)
    if hit is not None:
        return hit
    panels = fr.locator('[role="tabpanel"]')
    for i in range(panels.count()):
        pan = panels.nth(i)
        if pan.locator('input[type="checkbox"]').count() > 0:
            return pan
    main = fr.locator("main").first
    try:
        main.wait_for(state="attached", timeout=min(10_000, timeout_ms))
    except PlaywrightTimeout:
        pass
    if main.locator('input[type="checkbox"]').count() > 0:
        return main
    body = fr.locator("body")
    if body.locator('input[type="checkbox"]').count() > 0:
        return body
    return None


def _quiz_root_with_inputs(
    page,
    root_selector: str | None,
    timeout_ms: int,
) -> object:
    """
    チェックボックスを含むルートを返す。
    iframe 内の設問も対象。--root-selector は各フレームで試す。
    """
    for fr in page.frames:
        hit = _quiz_root_for_frame(fr, root_selector, timeout_ms)
        if hit is not None:
            return hit
    if root_selector:
        loc = page.locator(root_selector).first
        loc.wait_for(state="attached", timeout=timeout_ms)
        if loc.locator('input[type="checkbox"]').count() > 0:
            return loc
    pan_hit = _first_visible_tabpanel_with_checkbox(page.main_frame, timeout_ms)
    if pan_hit is not None:
        return pan_hit
    main = page.locator("main").first
    try:
        main.wait_for(state="attached", timeout=min(10_000, timeout_ms))
    except PlaywrightTimeout:
        pass
    if main.locator('input[type="checkbox"]').count() > 0:
        return main
    body = page.locator("body")
    if body.locator('input[type="checkbox"]').count() > 0:
        return body
    return _resolve_quiz_root(page, root_selector, timeout_ms=timeout_ms)


def _open_quiz_ready(
    page,
    quiz_url: str,
    root_selector: str | None,
    navigation_timeout_ms: int,
    goto_wait: str,
    checkbox_wait_ms: int,
    rlog: _RunLogger | None,
) -> object:
    """遷移 → チェック出現ポーリング → 入力付きルート解決。"""
    _navigate_quiz(page, quiz_url, navigation_timeout_ms, goto_wait, rlog)
    if rlog:
        try:
            rlog.log(f"遷移直後 URL={page.url!r} frames={len(page.frames)}")
        except Exception:
            pass
    _wait_until_page_has_checkbox(page, checkbox_wait_ms, rlog)
    return _quiz_root_with_inputs(page, root_selector, navigation_timeout_ms)


def _first_visible_tabpanel(page):
    panels = page.locator('[role="tabpanel"]')
    n = panels.count()
    for i in range(n):
        loc = panels.nth(i)
        if loc.is_visible():
            return loc
    return panels.first


def _resolve_quiz_root(page, root_selector: str | None, timeout_ms: int):
    """
    設問の包含要素。--root-selector 指定時はそれを使用。
    未指定時は可視の tabpanel、無ければ main（ログイン直後など tab が無い場合のフォールバック）。
    """
    if root_selector and _selector_targets_tabpanel(root_selector):
        for fr in page.frames:
            hit = _first_visible_tabpanel_with_checkbox(fr, timeout_ms)
            if hit is not None:
                return hit
        loc = page.locator(root_selector).first
        loc.wait_for(state="attached", timeout=timeout_ms)
        return loc
    if root_selector:
        loc = page.locator(root_selector).first
        loc.wait_for(state="visible", timeout=timeout_ms)
        return loc
    for fr in page.frames:
        hit = _first_visible_tabpanel_with_checkbox(fr, timeout_ms)
        if hit is not None:
            return hit
    panels = page.locator('[role="tabpanel"]')
    for i in range(panels.count()):
        loc = panels.nth(i)
        if loc.is_visible():
            loc.wait_for(state="visible", timeout=timeout_ms)
            return loc
    main = page.locator("main").first
    main.wait_for(state="visible", timeout=timeout_ms)
    return main


def check_choices_in_panel(tab_panel, texts: list[str], timeout_ms: int) -> None:
    """tab_panel 内で、表示テキストが texts のいずれかに部分一致するチェックボックスを選択。"""
    for fragment in texts:
        fragment = fragment.strip()
        if not fragment:
            continue
        by_role = tab_panel.get_by_role("checkbox", name=re.compile(re.escape(fragment), re.I))
        if by_role.count() > 0:
            by_role.first.check(timeout=timeout_ms, force=True)
            continue
        label = tab_panel.locator("label").filter(has_text=fragment).first
        label.wait_for(state="visible", timeout=timeout_ms)
        label.click()


def click_submit(page, timeout_ms: int) -> None:
    btn = page.get_by_role("button", name="解答を送信")
    btn.wait_for(state="visible", timeout=timeout_ms)
    btn.click()


DEFAULT_QUESTION_CARD_SELECTOR = "div.css-pr2tx6"

# 既定セレクタで 0 件／チェック0 のときに順に試す（ビルドで class が変わる場合の救済）
QUESTION_CARD_FALLBACK_SELECTORS = (
    'div[class*="pr2tx6"]',
    'div.chakra-stack:has(input[type="checkbox"])',
    'section:has(input[type="checkbox"])',
    'div:has(> label input[type="checkbox"])',
)


def _checkboxes_in_card(card):
    return card.locator('input[type="checkbox"]')


def _resolve_card_strategy(
    tab_panel,
    question_card_selector: str,
) -> tuple[str, object]:
    """
    戻り値: (説明文字列, cards_locator | None)
    cards_locator は .count() / .nth(i) 可能な Locator。None の場合は Qラベル分割を試す。
    """
    candidates = (question_card_selector,) + QUESTION_CARD_FALLBACK_SELECTORS
    for sel in candidates:
        loc = tab_panel.locator(sel)
        n = loc.count()
        if n == 0:
            continue
        ok = True
        for i in range(n):
            if _checkboxes_in_card(loc.nth(i)).count() == 0:
                ok = False
                break
        if ok:
            return (f"selector:{sel}", loc)
    # カードはあるがチェックが付いていないセレクタはスキップ済み → 最後に件数だけ返す
    for sel in candidates:
        loc = tab_panel.locator(sel)
        if loc.count() > 0:
            return (f"selector_partial:{sel}", loc)
    return ("none", None)


def _group_checkbox_indices_by_q_labels(tab_panel) -> list[list[int]] | None:
    """
    タブパネル内を文書順に走査し、直近に現れた Qn. ラベルにチェックボックスを紐づける。
    キカガクのように Q1/Q2 が同一パネルに並ぶ構成向け。
    """
    data = tab_panel.evaluate(
        """(root) => {
        const cbs = Array.from(root.querySelectorAll('input[type="checkbox"]'));
        if (!cbs.length) return null;
        let lastQ = 1;
        const idxByQ = {};
        const visit = (parent) => {
            for (const child of parent.childNodes) {
                if (child.nodeType === 3) {
                    const re = /Q(\\d+)\\./g;
                    let m;
                    while ((m = re.exec(child.textContent)) !== null) {
                        lastQ = parseInt(m[1], 10);
                    }
                } else if (child.nodeType === 1) {
                    if (child.matches && child.matches('input[type="checkbox"]')) {
                        const idx = cbs.indexOf(child);
                        if (idx < 0) continue;
                        if (!idxByQ[lastQ]) idxByQ[lastQ] = [];
                        idxByQ[lastQ].push(idx);
                    } else {
                        visit(child);
                    }
                }
            }
        };
        visit(root);
        const qs = Object.keys(idxByQ).map(Number).sort((a, b) => a - b);
        if (!qs.length) return null;
        return qs.map((q) => idxByQ[q].sort((a, b) => a - b));
    }"""
    )
    if not data:
        return None
    out = [sorted(set(g)) for g in data]
    out = [g for g in out if g]
    return out if out else None


def _set_checkbox_state(box, want: bool, timeout_ms: int) -> None:
    """ネイティブ input が非表示（Chakra 等）のときは祖先 label / ラッパーをクリックしてトグルする。"""

    def state_ok() -> bool:
        try:
            return box.is_checked() == want
        except Exception:
            return False

    box.wait_for(state="attached", timeout=timeout_ms)
    if state_ok():
        return
    try:
        if want:
            box.check(timeout=timeout_ms, force=True)
        else:
            box.uncheck(timeout=timeout_ms, force=True)
    except Exception:
        pass
    if state_ok():
        return
    label = box.locator("xpath=ancestor::label[1]")
    wrap = box.locator("xpath=ancestor::*[contains(@class,'chakra-checkbox')][1]")
    for _ in range(2):
        try:
            if label.count() > 0:
                label.first.click(timeout=timeout_ms, force=True)
                if state_ok():
                    return
        except Exception:
            pass
        try:
            if wrap.count() > 0:
                wrap.first.click(timeout=timeout_ms, force=True)
                if state_ok():
                    return
        except Exception:
            pass
        try:
            box.click(timeout=timeout_ms, force=True)
        except Exception:
            pass
        if state_ok():
            return


def _apply_mask_to_boxes(boxes: list, mask: int, timeout_ms: int) -> None:
    for i, box in enumerate(boxes):
        want = bool((mask >> i) & 1)
        _set_checkbox_state(box, want, timeout_ms)


def _mask_range(bit_width: int, allow_empty: bool):
    if bit_width <= 0:
        yield 0
        return
    start = 0 if allow_empty else 1
    for m in range(start, 1 << bit_width):
        yield m


def _apply_mask_to_card(card, mask: int, timeout_ms: int) -> None:
    _apply_mask_to_boxes(_checkboxes_in_card(card).all(), mask, timeout_ms)


def _checkbox_caption(box) -> str:
    return (
        box.evaluate(
            """el => {
              const l = el.labels && el.labels[0];
              if (l && l.innerText) return l.innerText.trim().slice(0, 200);
              const a = el.getAttribute('aria-label');
              if (a) return a.trim().slice(0, 200);
              return '';
            }"""
        )
        or ""
    )


def _snapshot_result_text(tab_panel) -> str:
    try:
        t = tab_panel.inner_text(timeout=5000)
    except PlaywrightTimeout:
        return ""
    return re.sub(r"\s+", " ", t).strip()[:4000]


def _infer_verdict(snippet: str) -> str:
    if not snippet:
        return "empty"
    # 複数設問が同一パネルにあると「正解」「不正解」が混在しうる
    if "不正解" in snippet:
        return "不正解を含む"
    if "正解" in snippet:
        return "正解を含む"
    return "不明"


def _likely_all_correct_after_submit(snippet: str) -> bool:
    """
    採点後パネル文言から「全設問とも不正解ではない（＝全問正解の可能性が高い）」か推定する。
    キカガク系で誤答時に「不正解」が出る前提。表示仕様依存のため誤判定があり得る。
    """
    if not snippet or len(snippet) < 40:
        return False
    if "不正解" in snippet:
        return False
    if "正解" not in snippet:
        return False
    return True


def _wrong_question_ids_near_incorrect(snippet: str) -> set[int]:
    """「Qn. … 不正解」に近い箇所から、誤答と思われる設問番号を拾う（参考用）。"""
    out: set[int] = set()
    for m in re.finditer(r"Q\s*(\d+)\s*\.[\s\S]{0,160}?不正解", snippet):
        out.add(int(m.group(1)))
    return out


def _selections_from_mask(mask: int, labels: list[str]) -> list[str]:
    sel: list[str] = []
    for i, lab in enumerate(labels):
        if (mask >> i) & 1:
            sel.append(lab)
    return sel


def _build_correct_answer_report(
    hits: list[dict],
    label_rows: list[list[str]],
    strat: str,
    quiz_url: str,
) -> dict:
    matches = []
    seen_masks: set[tuple] = set()
    for h in hits:
        combo = h["masks"]
        key = tuple(combo)
        if key in seen_masks:
            continue
        seen_masks.add(key)
        if label_rows and len(label_rows) == len(combo):
            selected = [_selections_from_mask(combo[i], label_rows[i]) for i in range(len(combo))]
        else:
            selected = []
        matches.append(
            {
                "trial_idx": h["idx"],
                "masks_per_question": list(combo),
                "selected_labels_per_question": selected,
            }
        )
    unique_combo = len(matches) == 1
    return {
        "quiz_url": quiz_url,
        "detect_strategy": strat,
        "note": "採点エリアに「不正解」が含まれず「正解」が含まれる試行を全問正解候補とした。表示が変わると誤る可能性あり。",
        "candidate_count": len(matches),
        "unique_mask_pattern": unique_combo,
        "matches": matches,
    }


def run_report_correct(from_csv: Path) -> None:
    """exhaustive 出力 CSV から、全問正解候補を JSON 風に標準出力へ。"""
    from_csv = from_csv.resolve()
    if not from_csv.is_file():
        raise SystemExit(f"CSV が見つかりません: {from_csv}")

    hits: list[dict] = []
    with from_csv.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            snippet = row.get("panel_text_snippet") or ""
            flag = (row.get("likely_all_correct") or "").strip().lower() in ("1", "true", "yes")
            if not flag:
                flag = _likely_all_correct_after_submit(snippet)
            if not flag:
                continue
            try:
                masks = json.loads(row["masks_decimal"])
            except (json.JSONDecodeError, KeyError):
                continue
            idx = int(row.get("idx", -1))
            hits.append({"idx": idx, "masks": masks})

    if not hits:
        print("全問正解候補となる行は見つかりませんでした。", file=sys.stderr)
        return

    label_rows: list[list[str]] = []
    with from_csv.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            cj = row.get("captions_json")
            if not cj:
                continue
            try:
                label_rows = json.loads(cj)
                break
            except json.JSONDecodeError:
                continue

    strat = ""
    with from_csv.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            strat = row.get("detect_strategy") or ""
            break

    report = _build_correct_answer_report(hits, label_rows, strat, "")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _build_question_box_groups(
    tab_panel,
    question_card_selector: str,
) -> tuple[str, list[list]]:
    """Q1./Q2. テキスト走査または設問カードセレクタで、設問ごとのチェック Locator 列を得る。"""
    flat = tab_panel.locator('input[type="checkbox"]')
    n = flat.count()
    if n == 0:
        raise SystemExit(
            "チェックボックスが 0 件です。章末テストのタブを開いた状態か、"
            "--root-selector '[role=\"tabpanel\"]' を指定して inspect-quiz で確認してください。"
        )

    gi = _group_checkbox_indices_by_q_labels(tab_panel)
    if gi:
        covered = sorted({i for g in gi for i in g})
        disjoint = len(covered) == sum(len(g) for g in gi)
        if disjoint and len(covered) == n:
            return ("q-label-dom-walk", [[flat.nth(i) for i in g] for g in gi])
        if disjoint and len(covered) < n:
            print(
                f"警告: Qラベル分割がチェックを取りこぼし（全{n}件のうち{len(covered)}件のみ）。"
                "セレクタ方式にフォールバックします。",
                file=sys.stderr,
            )

    desc, cards = _resolve_card_strategy(tab_panel, question_card_selector)
    if cards is not None and cards.count() > 0:
        groups = [_checkboxes_in_card(cards.nth(i)).all() for i in range(cards.count())]
        if all(len(g) > 0 for g in groups):
            return (desc, groups)

    if gi and all(len(g) > 0 for g in gi):
        return ("q-label-dom-walk-partial", [[flat.nth(i) for i in g] for g in gi])

    raise SystemExit(
        "Q1/Q2 の回答欄を特定できませんでした。\n"
        "  py -3.13 scripts/learning_quiz_automation.py inspect-quiz "
        "--quiz-url （URL） --storage learning_auth.json\n"
        "でチェック件数・Q分割・カード候補を確認してください。"
    )


def run_inspect_quiz(
    quiz_url: str,
    storage_path: Path | None,
    root_selector: str | None,
    question_card_selector: str,
    headless: bool,
    timeout_ms: int,
) -> None:
    storage_path = storage_path.resolve() if storage_path else None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx_kw: dict = {}
        if storage_path and storage_path.is_file():
            ctx_kw["storage_state"] = str(storage_path)
        ctx = browser.new_context(**ctx_kw)
        page = ctx.new_page()
        try:
            tab_panel = _open_quiz_ready(
                page,
                quiz_url,
                root_selector,
                timeout_ms,
                "load",
                max(timeout_ms, 90_000),
                None,
            )
        except PlaywrightTimeout:
            print(
                "タイムアウト: チェックボックスが規定時間内に現れませんでした。"
                " save-login を章末テスト表示状態でやり直すか --timeout-ms を延ばしてください。",
                file=sys.stderr,
                flush=True,
            )
            tab_panel = _resolve_quiz_root(page, root_selector, timeout_ms=timeout_ms)

        n_chk = tab_panel.locator('input[type="checkbox"]').count()
        print(f"ルート内チェックボックス数: {n_chk}")
        gi = _group_checkbox_indices_by_q_labels(tab_panel)
        print(f"Qラベル走査によるインデックス群: {json.dumps(gi, ensure_ascii=False) if gi else None}")

        for sel in (question_card_selector,) + QUESTION_CARD_FALLBACK_SELECTORS:
            loc = tab_panel.locator(sel)
            cnt = loc.count()
            if cnt == 0:
                print(f"  [{sel!r}] → 0 ブロック")
                continue
            chk_counts = [_checkboxes_in_card(loc.nth(i)).count() for i in range(cnt)]
            print(f"  [{sel!r}] → {cnt} ブロック, 各チェック数 {chk_counts}")

        try:
            strat, groups = _build_question_box_groups(tab_panel, question_card_selector)
            print(f"採用ストラテジ: {strat}")
            for qi, grp in enumerate(groups):
                caps = [_checkbox_caption(b) for b in grp]
                print(f"  グループ{qi + 1}: {len(grp)} 件 — {caps[:2]!r} ...")
        except SystemExit as e:
            print(f"_build_question_box_groups: {e.args[0] if e.args else e}", file=sys.stderr)

        browser.close()


def _parse_freeze_masks_1based_json(raw: str | None) -> dict[int, int] | None:
    """'{\"1\": 54}' のような 1 始まり設問番号→十進マスク。"""
    if not raw or not raw.strip():
        return None
    d = json.loads(raw)
    return {int(k): int(v) for k, v in d.items()}


def _resolve_freeze_masks_from_args(args: argparse.Namespace) -> dict[int, int] | None:
    """--freeze-masks-1based-json-file があれば優先、なければ --freeze-masks-1based-json。"""
    fp = getattr(args, "freeze_masks_1based_json_file", None)
    if fp:
        fp = Path(fp).resolve()
        if not fp.is_file():
            raise SystemExit(f"--freeze-masks-1based-json-file が見つかりません: {fp}")
        return _parse_freeze_masks_1based_json(fp.read_text(encoding="utf-8"))
    return _parse_freeze_masks_1based_json(args.freeze_masks_1based_json)


def run_exhaustive(
    quiz_url: str,
    storage_path: Path | None,
    root_selector: str | None,
    question_card_selector: str,
    allow_empty: bool,
    max_total: int,
    reload_each_attempt: bool,
    pause_ms: int,
    headless: bool,
    timeout_ms: int,
    out_csv: Path,
    correct_out: Path | None,
    stop_on_full_correct: bool,
    vary_questions_1based: list[int] | None,
    cap_options_per_question: int | None,
    max_trials: int | None,
    log_file: Path | None,
    goto_wait: str,
    checkbox_wait_ms: int,
    freeze_masks_1based: dict[int, int] | None = None,
) -> None:
    storage_path = storage_path.resolve() if storage_path else None
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _ensure_stdio_line_buffered()
    rlog = _RunLogger(log_file)
    atexit.register(rlog.close)
    if "..." in quiz_url or "\u2026" in quiz_url:
        raise SystemExit(
            "--quiz-url に ... や … が含まれています（例の省略をそのまま貼っていませんか）。\n"
            "ブラウザのアドレスバーから、次のようなパスまで含めた完全な URL を指定してください。\n"
            "例:\n"
            "https://www.kikagaku.ai/learning/learn/benesse-i-career3/chapter-test1/end-of-chapter1/"
        )
    if storage_path and not storage_path.is_file():
        rlog.log(
            f"警告: storage state が見つかりません: {storage_path} "
            "（先に save-login で保存してください）"
        )

    strat = ""
    label_rows: list[list[str]] = []
    correct_hits: list[dict] = []
    planned_product = 0
    exec_limit = 0
    stopped_early = False
    trials_done = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context_kwargs: dict = {}
        if storage_path and storage_path.is_file():
            context_kwargs["storage_state"] = str(storage_path)
        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        try:
            tab_panel = _open_quiz_ready(
                page,
                quiz_url,
                root_selector,
                timeout_ms,
                goto_wait,
                checkbox_wait_ms,
                rlog,
            )
        except PlaywrightTimeout:
            browser.close()
            raise SystemExit(
                "チェックボックスが規定時間内に現れませんでした。\n"
                "  • --quiz-url はアドレスバーからコピーした完全なものか（... 省略や誤ったパスで sign-in/?redirect=%2F404 になりやすい）\n"
                "  • learning_auth.json を、章末テストが表示された画面で save-login し直す\n"
                "  • --checkbox-wait-ms を延ばす（例: 180000）\n"
                "  • --goto-wait load（networkidle はタイムアウトしやすい）\n"
                "  • --root-selector と inspect-quiz で構造を確認\n"
                "  • セッション切れなら再ログイン後に storage を更新"
            ) from None
        except SystemExit:
            browser.close()
            raise

        strat, question_boxes = _build_question_box_groups(tab_panel, question_card_selector)
        n_groups = len(question_boxes)
        widths_full: list[int] = [len(g) for g in question_boxes]
        label_rows: list[list[str]] = [[_checkbox_caption(b) for b in g] for g in question_boxes]

        if vary_questions_1based is None:
            active = list(range(n_groups))
        else:
            active = sorted({q - 1 for q in vary_questions_1based})
            for a in active:
                if a < 0 or a >= n_groups:
                    browser.close()
                    raise SystemExit(
                        f"--vary-questions の番号が範囲外です（1〜{n_groups}）: {vary_questions_1based}"
                    )
        active_set = set(active)
        cap = cap_options_per_question

        widths_active: list[int] = []
        for qi in active:
            full_w = widths_full[qi]
            w = min(cap, full_w) if cap is not None else full_w
            if w < 1:
                browser.close()
                raise SystemExit(f"設問 Q{qi + 1} の対象チェックが 0 件です（cap 大きすぎ？）")
            widths_active.append(w)

        mask_lists_active = [list(_mask_range(w, allow_empty)) for w in widths_active]
        planned_product = 1
        for ml in mask_lists_active:
            planned_product *= max(len(ml), 1)

        exec_limit = planned_product
        if max_trials is not None:
            exec_limit = min(planned_product, max_trials)

        if max_trials is None and planned_product > max_total:
            browser.close()
            raise SystemExit(
                f"組み合わせ数が {planned_product}（--max-total {max_total} を超過）。"
                "--vary-questions / --cap-options / --max-trials で削るか max_total を上げてください。"
            )
        if max_trials is not None and planned_product > max_total:
            rlog.log(
                f"注意: 理論組み合わせ {planned_product} > --max-total {max_total} ですが、"
                f"--max-trials {max_trials} により実際は {exec_limit} 試行のみ実行します。"
            )

        rlog.log(
            f"開始 url={quiz_url!r} strat={strat!r} groups={n_groups} "
            f"vary_1based={[i + 1 for i in active]!r} cap={cap} "
            f"freeze_1based={freeze_masks_1based!r} "
            f"planned_product={planned_product} exec_limit={exec_limit} reload={reload_each_attempt}"
        )

        correct_hits.clear()
        trials_done = 0
        stopped_early = False

        product_iter = itertools.product(*mask_lists_active)
        if max_trials is not None:
            product_iter = islice(product_iter, max_trials)

        with out_csv.open("w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(
                [
                    "idx",
                    "masks_decimal",
                    "vary_questions_1based",
                    "cap_options",
                    "detect_strategy",
                    "captions_json",
                    "verdict_guess",
                    "likely_all_correct",
                    "wrong_q_guess",
                    "panel_text_snippet",
                ]
            )

            for idx, combo_active in enumerate(product_iter):
                t0 = time.perf_counter()
                try:
                    rlog.log(f"試行 {idx + 1}/{exec_limit} 開始 combo_active={list(combo_active)!r}")
                    if reload_each_attempt:
                        tab_panel = _open_quiz_ready(
                            page,
                            quiz_url,
                            root_selector,
                            timeout_ms,
                            goto_wait,
                            checkbox_wait_ms,
                            rlog,
                        )
                    else:
                        tab_panel = _resolve_quiz_root(page, root_selector, timeout_ms=timeout_ms)
                        _wait_checkbox_attached(tab_panel, timeout_ms)
                    strat2, qb = _build_question_box_groups(tab_panel, question_card_selector)
                    if strat2 != strat or [len(x) for x in qb] != widths_full:
                        browser.close()
                        raise SystemExit(
                            f"試行 {idx}: 設問構造が変化しました "
                            f"({strat!r}/{widths_full!r} → {strat2!r}/{[len(x) for x in qb]!r})"
                        )

                    combo_full = [0] * n_groups
                    if freeze_masks_1based:
                        for qi in range(n_groups):
                            key = qi + 1
                            if key in freeze_masks_1based:
                                combo_full[qi] = freeze_masks_1based[key]
                    for j, qi in enumerate(active):
                        combo_full[qi] = combo_active[j]

                    for ci in range(n_groups):
                        boxes = qb[ci]
                        if ci in active_set:
                            cap_w = min(cap, len(boxes)) if cap is not None else len(boxes)
                            sub = boxes[:cap_w]
                            _apply_mask_to_boxes(sub, combo_full[ci], timeout_ms=timeout_ms)
                            for j in range(cap_w, len(boxes)):
                                _set_checkbox_state(boxes[j], False, timeout_ms)
                        else:
                            _apply_mask_to_boxes(boxes, combo_full[ci], timeout_ms=timeout_ms)

                    click_submit(page, timeout_ms=timeout_ms)
                    if pause_ms > 0:
                        time.sleep(pause_ms / 1000.0)

                    tab_panel = _resolve_quiz_root(page, root_selector, timeout_ms=timeout_ms)
                    snippet = _snapshot_result_text(tab_panel)
                    verdict = _infer_verdict(snippet)
                    likely_ok = _likely_all_correct_after_submit(snippet)
                    wrong_q = sorted(_wrong_question_ids_near_incorrect(snippet))
                    wrong_q_cell = json.dumps(wrong_q, ensure_ascii=False) if wrong_q else ""
                    trials_done = idx + 1
                    dt = time.perf_counter() - t0
                    snippet_1l = re.sub(r"\s+", " ", snippet).strip()
                    if len(snippet_1l) > 200:
                        snippet_1l = snippet_1l[:200] + "…"
                    # 採点直後・CSV 書き込み前に結果をターミナルへ（パターンごとのリアルタイム確認用）
                    rlog.log(
                        f"結果 試行 {idx + 1}/{exec_limit} dt={dt:.2f}s "
                        f"masks={combo_full!r} verdict={verdict!r} "
                        f"likely_all_correct={likely_ok!r} wrong_q={wrong_q!r} "
                        f"panel_preview={snippet_1l!r}"
                    )
                    w.writerow(
                        [
                            idx,
                            json.dumps(combo_full, ensure_ascii=False),
                            json.dumps([i + 1 for i in active], ensure_ascii=False),
                            cap if cap is not None else "",
                            strat,
                            json.dumps(label_rows, ensure_ascii=False),
                            verdict,
                            "yes" if likely_ok else "no",
                            wrong_q_cell,
                            snippet,
                        ]
                    )
                    fp.flush()

                    if likely_ok:
                        correct_hits.append({"idx": idx, "masks": list(combo_full)})
                    if stop_on_full_correct and likely_ok:
                        stopped_early = True
                        rlog.log("早期終了: 全問正解候補を検出しました。")
                        break
                except SystemExit:
                    browser.close()
                    raise
                except Exception as e:
                    browser.close()
                    name = type(e).__name__
                    if name == "TargetClosedError":
                        raise SystemExit(
                            "ブラウザまたはタブが閉じられました（TargetClosedError）。"
                            "実行中はウィンドウを閉じないでください。"
                        ) from e
                    raise

        browser.close()

    if correct_out:
        correct_out = correct_out.resolve()
        if correct_hits:
            correct_out.parent.mkdir(parents=True, exist_ok=True)
            report = _build_correct_answer_report(correct_hits, label_rows, strat, quiz_url)
            correct_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"正解候補を {correct_out} に書き出しました（{len(correct_hits)} 件）。",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "全問正解候補は検出されず、correct-out は作成しませんでした。",
                file=sys.stderr,
                flush=True,
            )

    if stopped_early:
        rlog.log(f"早期終了: 全問正解候補検出のため {trials_done}/{exec_limit} 試行で停止。")
    rlog.log(f"完了: {out_csv}（実行 {trials_done} / 上限 {exec_limit}、理論積 {planned_product}）")


def _first_input_locator(page, selectors: tuple[str, ...]):
    for sel in selectors:
        loc = page.locator(sel)
        if loc.count() > 0:
            return loc.first
    return None


def _perform_auto_login(page, login_id: str, password: str, timeout_ms: int) -> None:
    """一般的なメール／ID・パスワードフォームを想定（キカガク等の Next/Chakra 系も含む）。"""
    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    user_el = _first_input_locator(
        page,
        (
            'input[type="email"]',
            'input[name="email"]',
            'input[autocomplete="username"]',
            'input[autocomplete="email"]',
            'input[id*="mail" i]',
            'input[id*="login" i]',
            'input[name="login"]',
            'input[type="text"]',
        ),
    )
    if user_el is None:
        raise ValueError(
            "ログイン用の ID/メール入力欄が見つかりません。"
            "ログインページを --start-url で開くか、手動で入力してください。"
        )
    if page.locator('input[type="password"]').count() == 0:
        raise ValueError("パスワード入力欄が見つかりません。")
    pwd_el = page.locator('input[type="password"]').first

    user_el.click(timeout=timeout_ms)
    user_el.fill(login_id, timeout=timeout_ms)
    pwd_el.click(timeout=timeout_ms)
    pwd_el.fill(password, timeout=timeout_ms)

    login_btn = page.get_by_role(
        "button",
        name=re.compile(r"ログイン|Login|サインイン|Sign\s*in", re.I),
    )
    if login_btn.count() > 0:
        login_btn.first.click(timeout=timeout_ms)
    else:
        sub = page.locator('button[type="submit"], input[type="submit"]')
        if sub.count() > 0:
            sub.first.click(timeout=timeout_ms)
        else:
            raise ValueError(
                "ログインボタンを特定できませんでした。手動でログインしてください。"
            )

    try:
        page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 30_000))
    except PlaywrightTimeout:
        pass


def _resolve_login_secrets(
    login_id_arg: str | None,
    password_arg: str | None,
    credentials_json: Path | None,
) -> tuple[str | None, str | None]:
    login_id = (login_id_arg or "").strip() or None
    password = password_arg or None

    if credentials_json and credentials_json.is_file():
        data = json.loads(credentials_json.read_text(encoding="utf-8"))
        login_id = login_id or (data.get("login_id") or data.get("email") or "").strip() or None
        password = password or data.get("password")

    if not login_id:
        login_id = (os.environ.get("LEARNING_LOGIN_ID") or os.environ.get("KIKAGAKU_LOGIN_ID") or "").strip() or None
    if not password:
        password = os.environ.get("LEARNING_LOGIN_PASSWORD") or os.environ.get("KIKAGAKU_PASSWORD")

    if login_id and not password:
        password = getpass.getpass("ログインパスワード（画面に表示されません）: ")

    return login_id, password


def save_login_state(
    start_url: str,
    out_path: Path,
    timeout_ms: int,
    login_id: str | None = None,
    password: str | None = None,
) -> None:
    out_path = out_path.resolve()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(start_url, wait_until="domcontentloaded", timeout=timeout_ms)

        if login_id and password:
            try:
                _perform_auto_login(page, login_id, password, timeout_ms=timeout_ms)
            except ValueError as e:
                print(f"自動ログインをスキップ: {e}", file=sys.stderr)

        print(
            "ブラウザで 2FA や遷移を完了し、storage を保存してよい状態になったら Enter を押してください。"
            "（手動ログインのみの場合もここで Enter）"
        )
        input()

        context.storage_state(path=str(out_path))
        browser.close()
    print(f"保存しました: {out_path}")


def run_quiz(
    quiz_url: str,
    storage_path: Path | None,
    root_selector: str | None,
    check_texts: list[str],
    headless: bool,
    timeout_ms: int,
    dry_run: bool,
) -> None:
    storage_path = storage_path.resolve() if storage_path else None
    if storage_path and not storage_path.is_file():
        print(
            f"警告: storage state が見つかりません: {storage_path}",
            file=sys.stderr,
        )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context_kwargs = {}
        if storage_path and storage_path.is_file():
            context_kwargs["storage_state"] = str(storage_path)
        context = browser.new_context(**context_kwargs)
        page = context.new_page()
        page.goto(quiz_url, wait_until="domcontentloaded", timeout=timeout_ms)

        tab_panel = _resolve_quiz_root(page, root_selector, timeout_ms=timeout_ms)

        if check_texts:
            check_choices_in_panel(tab_panel, check_texts, timeout_ms=timeout_ms)

        if dry_run:
            print("dry-run: 送信はスキップしました。")
            browser.close()
            return

        click_submit(page, timeout_ms=timeout_ms)
        browser.close()


def _parse_vary_questions(s: str | None) -> list[int] | None:
    if s is None or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="学習サイトの解答送信操作（Playwright）")
    sub = parser.add_subparsers(dest="command")

    p_login = sub.add_parser("save-login", help="ログイン後の storage state を保存")
    p_login.add_argument("--start-url", required=True, help="ログインページなどの開始 URL")
    p_login.add_argument(
        "--out",
        type=Path,
        default=Path("learning_auth.json"),
        help="保存先（既定: learning_auth.json）",
    )
    p_login.add_argument("--timeout-ms", type=int, default=60_000)
    p_login.add_argument(
        "--login-id",
        default=None,
        help="ログイン ID またはメール（未指定時は環境変数 LEARNING_LOGIN_ID / KIKAGAKU_LOGIN_ID）",
    )
    p_login.add_argument(
        "--password",
        default=None,
        help="パスワード（履歴に残るため非推奨。環境変数 LEARNING_LOGIN_PASSWORD または対話入力を推奨）",
    )
    p_login.add_argument(
        "--credentials-json",
        type=Path,
        default=None,
        help='{"login_id":"...","password":"..."} 形式（.gitignore 推奨）',
    )

    p_run = sub.add_parser("submit", help="問題ページで選択して送信")
    p_run.add_argument("--quiz-url", required=True)
    p_run.add_argument("--storage", type=Path, default=Path("learning_auth.json"))
    p_run.add_argument(
        "--check-text",
        nargs="*",
        default=[],
        help="選択するチェックボックスに紐づく表示テキスト（部分一致・複数可）",
    )
    p_run.add_argument("--config-json", type=Path, help='{"check_texts": ["...", "..."]} 形式')
    p_run.add_argument(
        "--root-selector",
        default=None,
        help='設問のルート（例: [role="tabpanel"]）。未指定時は tabpanel → main の順で解決',
    )
    p_run.add_argument("--headless", action="store_true")
    p_run.add_argument("--timeout-ms", type=int, default=60_000)
    p_run.add_argument("--dry-run", action="store_true", help="送信ボタンは押さない")

    p_ex = sub.add_parser(
        "exhaustive",
        help="各設問ブロック（div.css-pr2tx6 相当）内チェックの全組み合わせを試行し CSV に記録",
    )
    p_ex.add_argument("--quiz-url", required=True)
    p_ex.add_argument("--storage", type=Path, default=Path("learning_auth.json"))
    p_ex.add_argument(
        "--root-selector",
        default=None,
        help='設問のルート。未指定時は可視 tabpanel、無ければ main（キカガク等）',
    )
    p_ex.add_argument(
        "--question-card-selector",
        default=DEFAULT_QUESTION_CARD_SELECTOR,
        help=f"設問ラッパー（既定: {DEFAULT_QUESTION_CARD_SELECTOR!r}）",
    )
    p_ex.add_argument(
        "--allow-empty",
        action="store_true",
        help="各設問で「すべてオフ」も含める（既定は除く）",
    )
    p_ex.add_argument(
        "--max-total",
        type=int,
        default=8192,
        help="試行上限（組み合わせ総数）。超えると起動前に終了",
    )
    p_ex.add_argument(
        "--no-reload",
        action="store_true",
        help="各試行でページ遷移しない（状態が残るサイト向け。不具合時は外す）",
    )
    p_ex.add_argument(
        "--pause-ms",
        type=int,
        default=400,
        help="送信直後に待つ毫秒（描画・採点表示用）",
    )
    p_ex.add_argument("--headless", action="store_true")
    p_ex.add_argument("--timeout-ms", type=int, default=60_000)
    p_ex.add_argument(
        "--out-csv",
        type=Path,
        default=Path("artifacts/quiz_exhaustive_log.csv"),
    )
    p_ex.add_argument(
        "--correct-out",
        type=Path,
        default=Path("artifacts/quiz_inferred_correct.json"),
        help="全問正解候補のマスク・選択肢ラベルを JSON で保存（候補が 1 件以上のとき）",
    )
    p_ex.add_argument(
        "--no-correct-out",
        action="store_true",
        help="--correct-out の既定パスへの書き出しを無効化",
    )
    p_ex.add_argument(
        "--stop-on-full-correct",
        action="store_true",
        help="全問正解候補を初めて検出した試行で打ち切る（試行数削減）",
    )
    p_ex.add_argument(
        "--vary-questions",
        default=None,
        metavar="N,N,...",
        help="変化させる設問番号（1始まり・カンマ区切り）。例: 1,2 で Q1・Q2 のみ。省略時は全設問",
    )
    p_ex.add_argument(
        "--cap-options",
        type=int,
        default=None,
        metavar="K",
        help="各設問で先頭 K 個のチェックボックスだけを組み合わせ対象にする（例: 5）",
    )
    p_ex.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="試行回数の上限（理論積より小さいとき先頭から打ち切り。例: 240）",
    )
    p_ex.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="実行ログを追記するファイル（時刻・試行・判定を記録）",
    )
    p_ex.add_argument(
        "--goto-wait",
        default="load",
        choices=("domcontentloaded", "load", "networkidle"),
        help='page.goto の wait_until（既定 load。networkidle は常時通信のページで 60s タイムアウトしやすい）',
    )
    p_ex.add_argument(
        "--checkbox-wait-ms",
        type=int,
        default=120_000,
        help="遷移後、ページ内にチェックが現るまでポーリングする最大時間（ms）",
    )
    p_ex.add_argument(
        "--freeze-masks-1based-json",
        default=None,
        metavar="JSON",
        help=(
            '1 始まり設問番号→十進マスクの JSON（例: {"1": 54}）。'
            "PowerShell では --freeze-masks-1based-json-file 推奨。"
            "文字列で渡す場合は全体を単引用符で囲む例: "
            "--freeze-masks-1based-json '{\"1\": 54}'"
        ),
    )
    p_ex.add_argument(
        "--freeze-masks-1based-json-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "上記と同じ内容の UTF-8 JSON ファイル（例: {\"1\": 54}）。"
            "指定時は --freeze-masks-1based-json より優先。"
        ),
    )

    p_rep = sub.add_parser(
        "report-correct",
        help="exhaustive の CSV から全問正解候補を抽出し JSON を標準出力に出す",
    )
    p_rep.add_argument("--from-csv", type=Path, required=True)

    p_insp = sub.add_parser(
        "inspect-quiz",
        help="設問ページのチェックボックス数・Q1/Q2分割・カードセレクタ候補を表示（デバッグ用）",
    )
    p_insp.add_argument("--quiz-url", required=True)
    p_insp.add_argument("--storage", type=Path, default=Path("learning_auth.json"))
    p_insp.add_argument("--root-selector", default=None)
    p_insp.add_argument(
        "--question-card-selector",
        default=DEFAULT_QUESTION_CARD_SELECTOR,
    )
    p_insp.add_argument("--headless", action="store_true")
    p_insp.add_argument("--timeout-ms", type=int, default=60_000)

    args = parser.parse_args()

    if args.command == "save-login":
        lid, pw = _resolve_login_secrets(
            args.login_id,
            args.password,
            args.credentials_json,
        )
        save_login_state(
            args.start_url,
            args.out,
            timeout_ms=args.timeout_ms,
            login_id=lid,
            password=pw,
        )
        return

    if args.command == "submit":
        texts = list(args.check_text)
        if args.config_json:
            data = json.loads(args.config_json.read_text(encoding="utf-8"))
            texts.extend(data.get("check_texts", []))
        run_quiz(
            quiz_url=args.quiz_url,
            storage_path=args.storage,
            root_selector=args.root_selector,
            check_texts=texts,
            headless=args.headless,
            timeout_ms=args.timeout_ms,
            dry_run=args.dry_run,
        )
        return

    if args.command == "exhaustive":
        co = None if args.no_correct_out else args.correct_out
        run_exhaustive(
            quiz_url=args.quiz_url,
            storage_path=args.storage,
            root_selector=args.root_selector,
            question_card_selector=args.question_card_selector,
            allow_empty=args.allow_empty,
            max_total=args.max_total,
            reload_each_attempt=not args.no_reload,
            pause_ms=args.pause_ms,
            headless=args.headless,
            timeout_ms=args.timeout_ms,
            out_csv=args.out_csv,
            correct_out=co,
            stop_on_full_correct=args.stop_on_full_correct,
            vary_questions_1based=_parse_vary_questions(args.vary_questions),
            cap_options_per_question=args.cap_options,
            max_trials=args.max_trials,
            log_file=args.log_file,
            goto_wait=args.goto_wait,
            checkbox_wait_ms=args.checkbox_wait_ms,
            freeze_masks_1based=_resolve_freeze_masks_from_args(args),
        )
        return

    if args.command == "report-correct":
        run_report_correct(args.from_csv)
        return

    if args.command == "inspect-quiz":
        run_inspect_quiz(
            quiz_url=args.quiz_url,
            storage_path=args.storage,
            root_selector=args.root_selector,
            question_card_selector=args.question_card_selector,
            headless=args.headless,
            timeout_ms=args.timeout_ms,
        )
        return

    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    _ensure_stdio_line_buffered()
    # argparse のサブコマンド無しでも動くよう、フラグだけの互換
    if len(sys.argv) > 1 and sys.argv[1] not in (
        "save-login",
        "submit",
        "exhaustive",
        "inspect-quiz",
        "report-correct",
        "-h",
        "--help",
    ):
        sys.argv.insert(1, "submit")
    main()
