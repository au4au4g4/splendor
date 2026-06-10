#!/usr/bin/env python3
"""Browser UI for playing Splendor with AlphaZero move suggestions.

This is intentionally a small local web app: it reuses the same upstream
``cestpasphoto/alpha-zero-general`` game, checkpoint and MCTS code as the
terminal launcher, but renders the current board, suggested moves and legal move
buttons in a browser.
"""

from __future__ import annotations

import argparse
import html
import http.server
import io
import json
import re
import socketserver
import traceback
import urllib.parse
import webbrowser
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Optional

from azg_human_hints import (
    HistoryEntry,
    UndoMode,
    add_upstream_to_path,
    build_advisor,
    checkpoint_file,
    choose_best_action,
    collect_hint_rows,
    import_move_formatter,
    import_upstream_modules,
    rewind_history,
    row_sort_key,
)

DEFAULT_CHECKPOINT = "../alpha-zero-general/splendor/pretrained_2players.pt"
OFFICIAL_GUI_URL = "https://cestpasphoto.github.io/splendor.html?players=2"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass
class GameState:
    board: object
    current_player: int
    turn: int = 0
    history: list[HistoryEntry] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AppContext:
    game: object
    ai_advisor: object
    hint_advisor: object
    mcts_cls: object
    formatter: object
    top: int
    sort_by: str
    hint_temperature: float
    human_player: int
    state: GameState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a browser Splendor UI with AlphaZero hints.")
    parser.add_argument(
        "--azg-path",
        default="../alpha-zero-general",
        help="Path to a cestpasphoto/alpha-zero-general checkout.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="AlphaZero checkpoint used by the AI and hint advisor.",
    )
    parser.add_argument("--port", type=int, default=8766, help="Local web server port.")
    parser.add_argument("--top", type=int, default=8, help="Number of suggested moves to show.")
    parser.add_argument(
        "--sort-by",
        choices=("q", "visits", "probability", "prior"),
        default="q",
        help="Metric used to rank suggested moves.",
    )
    parser.add_argument("--numMCTSSims", "-m", type=int, default=200, help="MCTS simulations per move.")
    parser.add_argument("--cpuct", "-c", type=float, default=None, help="Override cpuct.")
    parser.add_argument("--fpu", "-f", type=float, default=None, help="Override first-play urgency.")
    parser.add_argument(
        "--modern-onnx-export",
        action="store_true",
        help=(
            "Use PyTorch's default ONNX exporter. By default this GUI forces "
            "the legacy exporter for better compatibility with the Splendor model."
        ),
    )
    parser.add_argument(
        "--hint-temperature",
        type=float,
        default=0.01,
        help="Temperature used for displayed visit distribution.",
    )
    parser.add_argument("--ai-first", action="store_true", help="Let the checkpoint AI move first.")
    parser.add_argument(
        "--integrated-card-ui",
        action="store_true",
        help=(
            "Open the local data-integrated card UI. The board, hints, moves, AI, "
            "and undo all use the same backend state."
        ),
    )
    parser.add_argument(
        "--official-card-hints",
        action="store_true",
        help=(
            "Open an integrated page with the official card UI and a floating "
            "local AlphaZero hint panel. The panes are not automatically synchronized."
        ),
    )
    parser.add_argument(
        "--official-companion",
        action="store_true",
        help=(
            "Open a side-by-side page with the official card UI and the local "
            "hint panel. The two panes are not automatically synchronized."
        ),
    )
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser automatically.")
    return parser.parse_args()


def html_page(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #0f172a; color: #e2e8f0; }}
    header {{ background: #111827; border-bottom: 1px solid #334155; padding: 1rem 1.25rem; }}
    main {{ display: grid; grid-template-columns: minmax(360px, 1fr) minmax(420px, 1fr); gap: 1rem; padding: 1rem; }}
    h1 {{ margin: 0 0 .4rem; font-size: 1.35rem; }}
    h2 {{ margin-top: 0; }}
    .panel {{ background: #111827; border: 1px solid #334155; border-radius: .8rem; padding: 1rem; }}
    .board {{ white-space: pre-wrap; overflow: auto; max-height: 72vh; color: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #334155; padding: .4rem .35rem; text-align: left; vertical-align: top; }}
    th {{ color: #bfdbfe; }}
    .moves {{ display: flex; flex-wrap: wrap; gap: .45rem; }}
    button, .button {{ background: #2563eb; border: 0; color: white; border-radius: .5rem; cursor: pointer; display: inline-block; font-weight: 700; padding: .45rem .65rem; text-decoration: none; }}
    button:hover, .button:hover {{ background: #1d4ed8; }}
    .danger {{ background: #b91c1c; }}
    .muted {{ color: #cbd5e1; }}
    .log {{ max-height: 14rem; overflow: auto; }}
    .pill {{ background: #334155; border-radius: 999px; display: inline-block; margin-right: .35rem; padding: .15rem .55rem; }}
    @media (max-width: 980px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>{body}</body>
</html>""".encode()



def dependency_fix_for(exc: BaseException) -> str:
    message = str(exc)
    if "adaptive_max_pool2d" in message and "ONNX" in message:
        return (
            "<p><strong>ONNX exporter 不相容。</strong> 你的 PyTorch 正在用新版 "
            "dynamo ONNX exporter 匯出 Splendor model，但它無法轉換 "
            "<code>adaptive_max_pool2d</code>。</p>"
            "<p>本專案最新版預設會強制使用 legacy ONNX exporter。請確認你已更新 "
            "<code>tools/azg_human_hints.py</code> 與 <code>tools/splendor_hint_gui.py</code>，"
            "然後重新啟動同一個指令。</p>"
            "<p>如果你有加 <code>--modern-onnx-export</code>，請先拿掉它。</p>"
        )
    if isinstance(exc, ModuleNotFoundError) and exc.name == "onnxscript":
        return (
            "<p><strong>缺少 onnxscript。</strong> 新版 PyTorch 的 ONNX export 會用到 "
            "<code>onnxscript</code>；請在你的 upstream/目前 Python 環境安裝：</p>"
            "<pre>pip install onnxscript</pre>"
            "<p>如果仍有 ONNX 相關錯誤，建議一併更新：</p>"
            "<pre>pip install --upgrade onnx onnxscript onnxruntime</pre>"
        )
    return (
        "<p>請先確認已依 README 安裝 upstream dependencies，且 "
        "<code>--azg-path</code> / <code>--checkpoint</code> 指向正確位置。</p>"
    )


def render_runtime_error(exc: BaseException) -> bytes:
    trace = html.escape("".join(traceback.format_exception(exc)))
    body = (
        "<header><h1>Splendor hint GUI 啟動或推論失敗</h1>"
        "<a class='button' href='/'>重新整理</a></header>"
        "<main><section class='panel'>"
        f"<h2>{html.escape(type(exc).__name__)}</h2>"
        f"<p>{html.escape(str(exc))}</p>"
        f"{dependency_fix_for(exc)}"
        "<h2>詳細 traceback</h2>"
        f"<pre class='board'>{trace}</pre>"
        "</section></main>"
    )
    return html_page("Splendor hint GUI error", body)

def percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def board_text(game, board) -> str:
    stream = io.StringIO()
    with redirect_stdout(stream):
        game.printBoard(board)
    return stream.getvalue()


def board_html(game, board) -> str:
    return html.escape(strip_ansi(board_text(game, board)))


def move_label(game, formatter, action: int, player: int) -> str:
    if formatter:
        return formatter(action)
    return game.moveToString(action, player)


def legal_move_buttons(ctx: AppContext, canonical_board) -> str:
    valid_moves = ctx.game.getValidMoves(canonical_board, 0)
    buttons = []
    for action, is_valid in enumerate(valid_moves):
        if not is_valid:
            continue
        label = html.escape(move_label(ctx.game, ctx.formatter, action, ctx.human_player))
        buttons.append(
            f'<form method="post" action="/move" style="display:inline">'
            f'<input type="hidden" name="action" value="{action}">'
            f'<button title="{label}">{action}</button></form>'
        )
    return "\n".join(buttons)



def hint_rows_for_api(ctx: AppContext):
    if game_ended(ctx) or ctx.state.current_player != ctx.human_player:
        return []
    canonical_board = ctx.game.getCanonicalForm(ctx.state.board, ctx.state.current_player)
    probabilities, _, _ = ctx.hint_advisor.getActionProb(
        canonical_board,
        temp=ctx.hint_temperature,
        force_full_search=True,
    )
    rows = collect_hint_rows(ctx.game, ctx.hint_advisor, canonical_board, probabilities, ctx.formatter)
    ranked = sorted(rows, key=lambda row: row_sort_key(ctx.sort_by, row), reverse=True)[: ctx.top]
    return [
        {
            "rank": index,
            "action": row.action,
            "win_pct": None if row.estimated_win_rate is None else round(100.0 * row.estimated_win_rate, 1),
            "visits": row.visits,
            "prob_pct": round(100.0 * row.probability, 1),
            "prior_pct": round(100.0 * row.prior, 1),
            "label": row.label,
        }
        for index, row in enumerate(ranked, start=1)
    ]


def legal_moves_for_api(ctx: AppContext):
    if game_ended(ctx) or ctx.state.current_player != ctx.human_player:
        return []
    canonical_board = ctx.game.getCanonicalForm(ctx.state.board, ctx.state.current_player)
    valid_moves = ctx.game.getValidMoves(canonical_board, 0)
    return [
        {
            "action": action,
            "label": move_label(ctx.game, ctx.formatter, action, ctx.human_player),
        }
        for action, is_valid in enumerate(valid_moves)
        if is_valid
    ]


def state_payload(ctx: AppContext) -> dict:
    ended = game_ended(ctx)
    board = ctx.state.board
    status = [f"Turn {ctx.state.turn + 1}", f"目前玩家 P{ctx.state.current_player}"]
    if ended:
        status.append(f"Game over: {ctx.game.getGameEnded(board, ctx.state.current_player)}")
    elif ctx.state.current_player == ctx.human_player:
        status.append("輪到你")
    else:
        status.append("AI 思考中")
    return {
        "turn": ctx.state.turn,
        "current_player": ctx.state.current_player,
        "human_player": ctx.human_player,
        "ended": ended,
        "status": status,
        "board_text": strip_ansi(board_text(ctx.game, board)),
        "hints": hint_rows_for_api(ctx),
        "legal_moves": legal_moves_for_api(ctx),
        "log": list(reversed(ctx.state.log[-80:])),
    }


def json_response(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")

def hints_table(ctx: AppContext, canonical_board) -> str:
    probabilities, _, _ = ctx.hint_advisor.getActionProb(
        canonical_board,
        temp=ctx.hint_temperature,
        force_full_search=True,
    )
    rows = collect_hint_rows(ctx.game, ctx.hint_advisor, canonical_board, probabilities, ctx.formatter)
    ranked = sorted(rows, key=lambda row: row_sort_key(ctx.sort_by, row), reverse=True)[: ctx.top]
    body = []
    for index, row in enumerate(ranked, start=1):
        label = html.escape(row.label)
        body.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{row.action}</td>"
            f"<td>{percent(row.estimated_win_rate)}</td>"
            f"<td>{row.visits}</td>"
            f"<td>{percent(row.probability)}</td>"
            f"<td>{percent(row.prior)}</td>"
            f"<td>{label}</td>"
            f"<td><form method='post' action='/move'>"
            f"<input type='hidden' name='action' value='{row.action}'>"
            "<button>下這手</button></form></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>#</th><th>action</th><th>win%</th><th>visits</th><th>prob%</th><th>prior%</th><th>move</th><th></th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def append_log(ctx: AppContext, text: str) -> None:
    ctx.state.log.append(text)
    del ctx.state.log[:-80]


def snapshot(ctx: AppContext) -> HistoryEntry:
    return HistoryEntry(
        board=ctx.state.board.copy(),
        current_player=ctx.state.current_player,
        turn=ctx.state.turn,
    )


def apply_action(ctx: AppContext, action: int, actor: str) -> None:
    canonical_board = ctx.game.getCanonicalForm(ctx.state.board, ctx.state.current_player)
    valid_moves = ctx.game.getValidMoves(canonical_board, 0)
    if action < 0 or action >= len(valid_moves) or not valid_moves[action]:
        append_log(ctx, f"忽略非法動作：{action}")
        return

    ctx.state.history.append(snapshot(ctx))
    move_text = ctx.game.moveToString(action, ctx.state.current_player)
    append_log(ctx, f"Turn {ctx.state.turn + 1}: {actor} P{ctx.state.current_player} → {move_text}")
    ctx.state.board, ctx.state.current_player = ctx.game.getNextState(
        ctx.state.board,
        ctx.state.current_player,
        action,
        random_seed=0,
    )
    ctx.state.current_player = int(ctx.state.current_player)
    ctx.state.turn += 1


def game_ended(ctx: AppContext) -> bool:
    return bool(ctx.game.getGameEnded(ctx.state.board, ctx.state.current_player).any())


def advance_ai_until_human(ctx: AppContext) -> None:
    while not game_ended(ctx) and ctx.state.current_player != ctx.human_player:
        canonical_board = ctx.game.getCanonicalForm(ctx.state.board, ctx.state.current_player)
        action = choose_best_action(ctx.ai_advisor, canonical_board)
        apply_action(ctx, action, "AI")



def status_pills(ctx: AppContext, compact: bool = False) -> str:
    board = ctx.state.board
    status = [f"Turn {ctx.state.turn + 1}", f"目前玩家 P{ctx.state.current_player}"]
    if game_ended(ctx):
        status.append(f"Game over: {ctx.game.getGameEnded(board, ctx.state.current_player)}")
    elif ctx.state.current_player == ctx.human_player:
        status.append("輪到你")
    else:
        status.append("AI 思考中；請重新整理")
    if compact:
        status.append("請手動同步官方畫面")
    return "".join(f'<span class="pill">{html.escape(item)}</span>' for item in status)


def render_hints_panel(ctx: AppContext) -> bytes:
    ended = game_ended(ctx)
    body = (
        "<header><h1>AlphaZero 建議</h1>"
        f"<p>{status_pills(ctx, compact=True)}</p>"
        "<form method='post' action='/undo' style='display:inline'><button class='danger'>Undo</button></form> "
        "<a class='button' href='/hints-panel'>重新整理</a>"
        "<p class='muted'>這個面板不會自動讀取左側官方網頁；請在官方 GUI 手動下同一手。</p>"
        "</header>"
    )
    if ended:
        body += "<main><section class='panel'><h2>遊戲結束</h2></section></main>"
        return html_page("AlphaZero 建議", body)

    if ctx.state.current_player == ctx.human_player:
        canonical_board = ctx.game.getCanonicalForm(ctx.state.board, ctx.state.current_player)
        body += (
            "<main style='display:block'>"
            "<section class='panel'><h2>建議走法</h2>"
            f"{hints_table(ctx, canonical_board)}"
            "<h2>所有合法動作</h2>"
            f"<div class='moves'>{legal_move_buttons(ctx, canonical_board)}</div>"
            "</section>"
        )
    else:
        body += "<main style='display:block'><section class='panel'><h2>AI 回合</h2></section>"
    log_items = "".join(f"<li>{html.escape(item)}</li>" for item in reversed(ctx.state.log))
    body += f"<section class='panel'><h2>紀錄</h2><ol class='log'>{log_items}</ol></section></main>"
    return html_page("AlphaZero 建議", body)

def render(ctx: AppContext) -> bytes:
    ended = game_ended(ctx)
    board = ctx.state.board
    status = [f"Turn {ctx.state.turn + 1}", f"目前玩家 P{ctx.state.current_player}"]
    if ended:
        status.append(f"Game over: {ctx.game.getGameEnded(board, ctx.state.current_player)}")
    elif ctx.state.current_player == ctx.human_player:
        status.append("輪到你")
    else:
        status.append("AI 思考中；請重新整理")

    status_html = "".join(f'<span class="pill">{html.escape(item)}</span>' for item in status)
    board_panel = (
        "<section class='panel'><h2>盤面</h2>"
        "<p class='muted'>提示版 GUI 目前使用 alpha-zero-general 的文字盤面；"
        "它不是官方 cestpasphoto.github.io 的卡片式前端。</p>"
        f"<pre class='board'>{board_html(ctx.game, board)}</pre></section>"
    )

    if ended:
        right_panel = "<section class='panel'><h2>遊戲結束</h2><p>請關閉 server 後重新啟動以開始新局。</p></section>"
    elif ctx.state.current_player == ctx.human_player:
        canonical_board = ctx.game.getCanonicalForm(board, ctx.state.current_player)
        right_panel = (
            "<section class='panel'><h2>建議走法</h2>"
            "<p class='muted'>依 MCTS root node 排序；按「下這手」或下方 action 按鈕即可落子。</p>"
            f"{hints_table(ctx, canonical_board)}"
            "<h2>所有合法動作</h2>"
            f"<div class='moves'>{legal_move_buttons(ctx, canonical_board)}</div>"
            "</section>"
        )
    else:
        right_panel = "<section class='panel'><h2>AI 回合</h2><p>AI 會自動走，請重新整理。</p></section>"

    log_items = "".join(f"<li>{html.escape(item)}</li>" for item in reversed(ctx.state.log))
    body = (
        "<header><h1>Splendor AlphaZero 建議走法 GUI</h1>"
        f"<p>{status_html}</p>"
        "<form method='post' action='/undo' style='display:inline'><button class='danger'>Undo</button></form> "
        "<a class='button' href='/'>重新整理</a>"
        "</header>"
        f"<main>{board_panel}{right_panel}"
        f"<section class='panel'><h2>紀錄</h2><ol class='log'>{log_items}</ol></section>"
        "</main>"
    )
    return html_page("Splendor AlphaZero 建議走法 GUI", body)





def render_integrated_card_ui() -> bytes:
    body = """
<header>
  <h1>Splendor AlphaZero 資料層整合版</h1>
  <p class="muted">此頁不用官方 hosted iframe；盤面、建議、下棋、AI 回合與 Undo 全部共用同一個 Python 後端狀態，因此不會不同步。</p>
  <button onclick="undoMove()" class="danger">Undo</button>
  <button onclick="loadState()">重新整理</button>
</header>
<main style="grid-template-columns:minmax(520px,1.2fr) minmax(420px,.8fr);">
  <section class="panel">
    <h2>卡片式盤面區</h2>
    <div id="status" class="moves"></div>
    <div id="boardCards" style="display:grid;grid-template-columns:repeat(5,minmax(90px,1fr));gap:.75rem;margin-top:1rem;"></div>
    <details style="margin-top:1rem"><summary>文字盤面 / debug</summary><pre id="boardText" class="board"></pre></details>
  </section>
  <section class="panel">
    <h2>建議走法</h2>
    <p class="muted">點「下這手」會直接送到同一個後端狀態，AI 也會自動回應；這裡是真正同步的資料層。</p>
    <div id="hints"></div>
    <h2>所有合法動作</h2>
    <div id="legalMoves" class="moves"></div>
    <h2>紀錄</h2>
    <ol id="log" class="log"></ol>
  </section>
</main>
<script>
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
}
function cardColor(label) {
  if (label.includes('blue') || label.includes('B')) return '#2563eb';
  if (label.includes('green') || label.includes('G')) return '#16a34a';
  if (label.includes('red') || label.includes('R')) return '#dc2626';
  if (label.includes('white') || label.includes('W')) return '#e5e7eb';
  if (label.includes('black') || label.includes('K')) return '#27272a';
  return '#475569';
}
function renderPseudoCards(moves) {
  const cards = moves.slice(0, 25).map(move => `
    <button onclick="playMove(${move.action})" style="min-height:112px;text-align:left;border-radius:12px;background:${cardColor(move.label)};border:1px solid rgba(255,255,255,.35);box-shadow:0 8px 20px rgba(0,0,0,.22);">
      <strong style="font-size:1.2rem">#${move.action}</strong><br>
      <span style="font-size:.82rem">${escapeHtml(move.label)}</span>
    </button>`).join('');
  document.getElementById('boardCards').innerHTML = cards || '<p class="muted">目前沒有可顯示的合法動作卡片。</p>';
}
function renderHints(hints) {
  if (!hints.length) {
    document.getElementById('hints').innerHTML = '<p class="muted">目前沒有建議，可能是 AI 回合或遊戲結束。</p>';
    return;
  }
  document.getElementById('hints').innerHTML = `<table><thead><tr><th>#</th><th>action</th><th>win%</th><th>visits</th><th>prob%</th><th>prior%</th><th>move</th><th></th></tr></thead><tbody>${hints.map(row => `
    <tr><td>${row.rank}</td><td>${row.action}</td><td>${row.win_pct ?? 'n/a'}%</td><td>${row.visits}</td><td>${row.prob_pct}%</td><td>${row.prior_pct}%</td><td>${escapeHtml(row.label)}</td><td><button onclick="playMove(${row.action})">下這手</button></td></tr>`).join('')}</tbody></table>`;
}
function renderLegalMoves(moves) {
  document.getElementById('legalMoves').innerHTML = moves.map(move => `<button title="${escapeHtml(move.label)}" onclick="playMove(${move.action})">${move.action}</button>`).join('');
}
async function loadState() {
  const response = await fetch('/api/state');
  const state = await response.json();
  document.getElementById('status').innerHTML = state.status.map(item => `<span class="pill">${escapeHtml(item)}</span>`).join('');
  document.getElementById('boardText').textContent = state.board_text;
  renderPseudoCards(state.legal_moves);
  renderHints(state.hints);
  renderLegalMoves(state.legal_moves);
  document.getElementById('log').innerHTML = state.log.map(item => `<li>${escapeHtml(item)}</li>`).join('');
}
async function playMove(action) {
  await fetch('/api/move', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action})});
  await loadState();
}
async function undoMove() {
  await fetch('/api/undo', {method:'POST'});
  await loadState();
}
loadState();
</script>
"""
    return html_page("Splendor AlphaZero 資料層整合版", body)

def render_official_card_hints() -> bytes:
    body = f"""
<div style="position:fixed; inset:0; background:white;">
  <iframe title="Official Splendor card UI" src="{OFFICIAL_GUI_URL}" style="position:absolute; inset:0; width:100%; height:100%; border:0; background:white;"></iframe>
</div>
<aside style="position:fixed; right:16px; top:16px; bottom:16px; width:min(560px, 42vw); min-width:420px; z-index:10; box-shadow:0 18px 55px rgba(0,0,0,.45); border:1px solid #475569; border-radius:16px; overflow:hidden; background:#0f172a;">
  <iframe title="AlphaZero hint panel" src="/hints-panel" style="width:100%; height:100%; border:0;"></iframe>
</aside>
<div style="position:fixed; left:16px; bottom:16px; z-index:11; max-width:720px; background:rgba(15,23,42,.92); color:#e2e8f0; padding:.75rem 1rem; border-radius:12px; border:1px solid #475569; font-family:system-ui,sans-serif;">
  <strong>官方卡片外觀 + 建議走法</strong><br>
  左側是官方卡片 GUI；右側是本機建議面板。兩者目前不能自動同步，請在官方 GUI 手動下右側推薦的同一手。
  <a href="/official-companion" style="color:#93c5fd; margin-left:.5rem;">改用左右分割</a>
</div>
"""
    return html_page("Splendor 官方卡片外觀 + 建議走法", body)

def render_official_companion() -> bytes:
    body = f"""
<header>
  <h1>Splendor 官方卡片外觀 + 建議走法</h1>
  <p class="muted">左邊是官方 cestpasphoto 卡片式 GUI；右邊是本機 AlphaZero / MCTS 建議面板。</p>
  <p class="muted"><strong>注意：</strong>兩邊目前無法自動同步。請把右側建議當參考，在左側官方 GUI 手動下同一手；如果你也在右側按「下這手」，請確保左側同步操作相同局面。</p>
  <a class="button" href="/" target="_blank">只開建議面板</a>
  <a class="button" href="{OFFICIAL_GUI_URL}" target="_blank" rel="noopener noreferrer">另開官方 GUI</a>
</header>
<main style="grid-template-columns: minmax(640px, 1.15fr) minmax(520px, .85fr); height: calc(100vh - 132px);">
  <section class="panel" style="padding:0; overflow:hidden;">
    <iframe title="Official Splendor GUI" src="{OFFICIAL_GUI_URL}" style="width:100%; height:100%; border:0; background:white;"></iframe>
  </section>
  <section class="panel" style="padding:0; overflow:hidden;">
    <iframe title="Local AlphaZero hints" src="/" style="width:100%; height:100%; border:0;"></iframe>
  </section>
</main>
"""
    return html_page("Splendor 官方卡片外觀 + 建議走法", body)

class HintGuiHandler(http.server.BaseHTTPRequestHandler):
    ctx: AppContext

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        return

    def send_html(self, content: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict, status: int = 200) -> None:
        content = json_response(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def redirect_home(self) -> None:
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def do_GET(self) -> None:
        try:
            if self.path.startswith("/api/state"):
                advance_ai_until_human(self.ctx)
                self.send_json(state_payload(self.ctx))
                return
            if self.path.startswith("/integrated-card-ui"):
                self.send_html(render_integrated_card_ui())
                return
            if self.path.startswith("/official-card-hints"):
                self.send_html(render_official_card_hints())
                return
            if self.path.startswith("/official-companion"):
                self.send_html(render_official_companion())
                return
            advance_ai_until_human(self.ctx)
            if self.path.startswith("/hints-panel"):
                self.send_html(render_hints_panel(self.ctx))
                return
            self.send_html(render(self.ctx))
        except Exception as exc:  # Keep browser UI helpful instead of dropping the socket.
            self.send_html(render_runtime_error(exc), status=500)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode()
            if self.headers.get("Content-Type", "").startswith("application/json"):
                data = json.loads(raw_body or "{}")
            else:
                data = urllib.parse.parse_qs(raw_body)
            if self.path in {"/move", "/api/move"}:
                action_value = data.get("action", -1)
                if isinstance(action_value, list):
                    action_value = action_value[0]
                action = int(action_value)
                if not game_ended(self.ctx) and self.ctx.state.current_player == self.ctx.human_player:
                    apply_action(self.ctx, action, "Human")
                    advance_ai_until_human(self.ctx)
                if self.path == "/api/move":
                    self.send_json(state_payload(self.ctx))
                    return
            elif self.path in {"/undo", "/api/undo"}:
                previous = rewind_history(
                    self.ctx.state.history,
                    UndoMode.SAME_PLAYER,
                    self.ctx.human_player,
                )
                if previous is None:
                    append_log(self.ctx, "沒有可 undo 的人類回合。")
                else:
                    self.ctx.state.board = previous.board.copy()
                    self.ctx.state.current_player = previous.current_player
                    self.ctx.state.turn = previous.turn
                    self.ctx.mcts_cls.reset_all_search_trees()
                    append_log(self.ctx, "Undo：回到你上一次決策前。")
                if self.path == "/api/undo":
                    self.send_json(state_payload(self.ctx))
                    return
            self.redirect_home()
        except Exception as exc:  # Keep browser UI helpful instead of dropping the socket.
            self.send_html(render_runtime_error(exc), status=500)


def build_context(args: argparse.Namespace) -> AppContext:
    add_upstream_to_path(args.azg_path)
    import_game, mcts_cls, dotdict = import_upstream_modules()
    game_cls, nnet_cls, _, _ = import_game("splendor")
    game = game_cls()
    checkpoint = checkpoint_file(args.checkpoint)
    ai_advisor = build_advisor(game, nnet_cls, mcts_cls, dotdict, checkpoint, args)
    hint_advisor = build_advisor(game, nnet_cls, mcts_cls, dotdict, checkpoint, args)
    human_player = 1 if args.ai_first else 0
    state = GameState(board=game.getInitBoard(), current_player=0)
    ctx = AppContext(
        game=game,
        ai_advisor=ai_advisor,
        hint_advisor=hint_advisor,
        mcts_cls=mcts_cls,
        formatter=import_move_formatter("splendor"),
        top=args.top,
        sort_by=args.sort_by,
        hint_temperature=args.hint_temperature,
        human_player=human_player,
        state=state,
    )
    return ctx


def main() -> None:
    args = parse_args()
    ctx = build_context(args)
    handler = type("ConfiguredHintGuiHandler", (HintGuiHandler,), {"ctx": ctx})
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as server:
        url = f"http://127.0.0.1:{args.port}/"
        if args.integrated_card_ui:
            url = f"http://127.0.0.1:{args.port}/integrated-card-ui"
        elif args.official_card_hints:
            url = f"http://127.0.0.1:{args.port}/official-card-hints"
        elif args.official_companion:
            url = f"http://127.0.0.1:{args.port}/official-companion"
        print(f"Splendor hint GUI: {url}")
        print("Press Ctrl+C to stop.")
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped Splendor hint GUI.")


if __name__ == "__main__":
    main()
