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


def board_html(game, board) -> str:
    stream = io.StringIO()
    with redirect_stdout(stream):
        game.printBoard(board)
    return html.escape(stream.getvalue())


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
    board_panel = f"<section class='panel'><h2>盤面</h2><pre class='board'>{board_html(ctx.game, board)}</pre></section>"

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

    def redirect_home(self) -> None:
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def do_GET(self) -> None:
        try:
            advance_ai_until_human(self.ctx)
            self.send_html(render(self.ctx))
        except Exception as exc:  # Keep browser UI helpful instead of dropping the socket.
            self.send_html(render_runtime_error(exc), status=500)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = urllib.parse.parse_qs(self.rfile.read(length).decode())
            if self.path == "/move":
                action = int(data.get("action", ["-1"])[0])
                if not game_ended(self.ctx) and self.ctx.state.current_player == self.ctx.human_player:
                    apply_action(self.ctx, action, "Human")
                    advance_ai_until_human(self.ctx)
            elif self.path == "/undo":
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
