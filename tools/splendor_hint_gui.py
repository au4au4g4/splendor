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
import mimetypes
import re
import socketserver
import traceback
import urllib.parse
import webbrowser
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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
OFFICIAL_LOCAL_UI_ROOT = Path(__file__).resolve().parent / "static" / "official_splendor"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


GEM_NAMES = ["white", "blue", "green", "red", "black", "gold"]
CARD_COLOR_NAMES = GEM_NAMES[:5]


def int_list(values, length: Optional[int] = None) -> list[int]:
    items = [int(value) for value in list(values)]
    return items if length is None else items[:length]


def gem_counts(values, include_gold: bool = True) -> dict[str, int]:
    count = 6 if include_gold else 5
    return {name: int(values[index]) for index, name in enumerate(GEM_NAMES[:count])}


def card_from_rows(cost_row, value_row, *, action: Optional[int] = None, reserve_action: Optional[int] = None) -> Optional[dict[str, Any]]:
    cost = int_list(cost_row, 7)
    value = int_list(value_row, 7)
    if sum(cost[:5]) == 0 and sum(value[:5]) == 0 and int(value[6]) == 0:
        return None
    color_index = next((index for index, amount in enumerate(value[:5]) if amount), None)
    return {
        "color": CARD_COLOR_NAMES[color_index] if color_index is not None else "unknown",
        "points": int(value[6]),
        "cost": gem_counts(cost, include_gold=False),
        "raw_cost": cost,
        "raw_value": value,
        "action": action,
        "reserve_action": reserve_action,
    }


def noble_from_row(row, *, index: int) -> Optional[dict[str, Any]]:
    values = int_list(row, 7)
    if int(values[6]) == 0:
        return None
    return {
        "index": index,
        "points": int(values[6]),
        "cost": gem_counts(values, include_gold=False),
        "raw": values,
    }


def legal_move_kind(action: int) -> str:
    if action < 12:
        return "buy_visible"
    if action < 27:
        return "reserve"
    if action < 30:
        return "buy_reserved"
    if action < 60:
        return "take_gems"
    if action < 80:
        return "give_gems"
    return "pass"


def player_gems_row_index(num_players: int, player: int) -> int:
    num_nobles = num_players + 1
    gems_start = 32 + num_nobles
    return gems_start + player


def gem_delta(before, after) -> dict[str, int]:
    return {name: int(after[index]) - int(before[index]) for index, name in enumerate(GEM_NAMES)}


def enrich_legal_moves(moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for move in moves:
        action = int(move["action"])
        item = dict(move)
        item["kind"] = legal_move_kind(action)
        if action < 12:
            item["tier"] = action // 4
            item["index"] = action % 4
        elif action < 24:
            reserve_index = action - 12
            item["tier"] = reserve_index // 4
            item["index"] = reserve_index % 4
        elif action < 27:
            item["tier"] = action - 24
        elif action < 30:
            item["index"] = action - 27
        enriched.append(item)
    return enriched


def serialize_splendor_board(ctx: AppContext, legal_moves: list[dict[str, Any]], hints: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert the upstream Splendor ndarray into a browser-friendly board model.

    The upstream engine stores the board as a compact 2-D ndarray.  Keeping that
    knowledge here avoids making the browser parse ``printBoard()`` text and
    gives every UI mode the same synchronized backend data.
    """
    board = ctx.state.board
    num_players = int(ctx.game.getNumberOfPlayers()) if hasattr(ctx.game, "getNumberOfPlayers") else 2
    num_nobles = num_players + 1
    legal_actions = {int(move["action"]) for move in legal_moves}

    visible_cards_by_tier = []
    for tier in range(3):
        cards = []
        for index in range(4):
            row = 1 + 8 * tier + 2 * index
            buy_action = tier * 4 + index
            reserve_action = 12 + tier * 4 + index
            card = card_from_rows(
                board[row],
                board[row + 1],
                action=buy_action if buy_action in legal_actions else None,
                reserve_action=reserve_action if reserve_action in legal_actions else None,
            )
            if card is None:
                card = {
                    "color": "empty",
                    "points": 0,
                    "cost": gem_counts([0, 0, 0, 0, 0], include_gold=False),
                    "action": None,
                    "reserve_action": None,
                }
            card["tier"] = tier
            card["index"] = index
            cards.append(card)
        deck_count = int(sum(int(value) for value in board[25 + 2 * tier][:5])) if len(board) > 25 + 2 * tier else 0
        visible_cards_by_tier.append({"tier": tier, "deck_count": deck_count, "cards": cards})

    nobles = [
        noble
        for noble in (noble_from_row(board[31 + index], index=index) for index in range(num_nobles))
        if noble is not None
    ]

    players = []
    gems_start = 32 + num_nobles
    player_nobles_start = 32 + 2 * num_players
    cards_start = 32 + 3 * num_players + num_players * num_players
    reserved_start = 32 + 4 * num_players + num_players * num_players
    for player in range(num_players):
        noble_rows = board[player_nobles_start + player * num_nobles : player_nobles_start + (player + 1) * num_nobles]
        player_nobles = [noble for noble in (noble_from_row(row, index=index) for index, row in enumerate(noble_rows)) if noble]
        reserved = []
        for index in range(3):
            row = reserved_start + 6 * player + 2 * index
            reserve_card = card_from_rows(
                board[row],
                board[row + 1],
                action=(27 + index if player == ctx.human_player and 27 + index in legal_actions else None),
            )
            if reserve_card is not None:
                reserve_card["index"] = index
                reserved.append(reserve_card)
        cards_row = int_list(board[cards_start + player], 7)
        try:
            score = int(ctx.game.getScore(board, player))
        except Exception:
            score = int(cards_row[6] + sum(noble["points"] for noble in player_nobles))
        players.append({
            "id": player,
            "is_human": player == ctx.human_player,
            "score": score,
            "gems": gem_counts(board[gems_start + player], include_gold=True),
            "cards": {name: int(cards_row[index]) for index, name in enumerate(CARD_COLOR_NAMES)},
            "card_points": int(cards_row[6]),
            "nobles": player_nobles,
            "reserved": reserved,
        })

    return {
        "visible_cards_by_tier": visible_cards_by_tier,
        "nobles": nobles,
        "bank_gems": gem_counts(board[0], include_gold=True),
        "players": players,
        "legal_moves": enrich_legal_moves(legal_moves),
        "hints": hints,
    }


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
        "--official-local-ui",
        action="store_true",
        help=(
            "Open the vendored same-origin official-style frontend. It is driven by "
            "/api/state, /api/move, and /api/undo instead of hosted iframe state."
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
    num_players = int(ctx.game.getNumberOfPlayers()) if hasattr(ctx.game, "getNumberOfPlayers") else 2
    gems_row = player_gems_row_index(num_players, ctx.human_player)
    moves = []
    for action, is_valid in enumerate(valid_moves):
        if not is_valid:
            continue
        item = {
            "action": action,
            "label": move_label(ctx.game, ctx.formatter, action, ctx.human_player),
        }
        kind = legal_move_kind(action)
        if kind in {"take_gems", "give_gems"}:
            try:
                next_board, _ = ctx.game.getNextState(
                    ctx.state.board.copy(),
                    ctx.state.current_player,
                    action,
                    random_seed=0,
                )
                item["gem_delta"] = gem_delta(ctx.state.board[gems_row], next_board[gems_row])
            except Exception:
                item["gem_delta"] = {}
        moves.append(item)
    return moves


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
    hints = hint_rows_for_api(ctx)
    legal_moves = legal_moves_for_api(ctx)
    return {
        "turn": ctx.state.turn,
        "current_player": ctx.state.current_player,
        "human_player": ctx.human_player,
        "ended": ended,
        "status": status,
        "board_text": strip_ansi(board_text(ctx.game, board)),
        "hints": hints,
        "legal_moves": legal_moves,
        "board": serialize_splendor_board(ctx, legal_moves, hints),
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
<header class="official-header">
  <div>
    <h1>Splendor AlphaZero 官方風格同步版</h1>
    <p class="muted">此頁不用官方 hosted iframe；卡片、token、玩家區、建議、下棋、AI 回合與 Undo 全部由本機 Python 後端狀態驅動。</p>
  </div>
  <div class="toolbar">
    <button onclick="undoMove()" class="danger">Undo</button>
    <button onclick="loadState()">重新整理</button>
  </div>
</header>
<style>
  body { background: radial-gradient(circle at top, #29405f 0, #162238 42%, #0b1020 100%); }
  main.official-layout { display:grid; grid-template-columns:minmax(720px,1.45fr) minmax(440px,.75fr); gap:1rem; padding:1rem; }
  .official-header { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }
  .toolbar { display:flex; gap:.5rem; flex-wrap:wrap; }
  .tableau { background:linear-gradient(145deg, rgba(15,23,42,.88), rgba(30,41,59,.78)); border:1px solid rgba(226,232,240,.18); border-radius:18px; box-shadow:0 20px 60px rgba(0,0,0,.35); padding:1rem; }
  .top-strip { display:grid; grid-template-columns:minmax(220px,.65fr) 1fr; gap:1rem; margin:.85rem 0 1rem; }
  .bank, .nobles, .player, .side-card { background:rgba(2,6,23,.42); border:1px solid rgba(148,163,184,.24); border-radius:16px; padding:.75rem; }
  .section-title { color:#fde68a; font-weight:900; letter-spacing:.04em; text-transform:uppercase; font-size:.78rem; margin-bottom:.55rem; }
  .tokens { display:flex; gap:.45rem; flex-wrap:wrap; align-items:center; }
  .gem { width:38px; height:38px; border-radius:50%; display:inline-grid; place-items:center; font-weight:900; border:3px solid rgba(255,255,255,.55); box-shadow: inset 0 2px 8px rgba(255,255,255,.28), 0 6px 14px rgba(0,0,0,.35); color:#0f172a; }
  .gem.small { width:25px; height:25px; border-width:2px; font-size:.72rem; }
  .gem.white { background:#f8fafc; } .gem.blue { background:#3b82f6; color:#eff6ff; } .gem.green { background:#22c55e; } .gem.red { background:#ef4444; color:#fff1f2; } .gem.black { background:#171717; color:#f8fafc; } .gem.gold { background:#facc15; }
  .tier-row { display:grid; grid-template-columns:88px repeat(4, minmax(122px, 1fr)); gap:.75rem; align-items:stretch; margin-bottom:.85rem; }
  .deck { border-radius:14px; background:linear-gradient(160deg,#6b4a2b,#27160b); border:2px solid rgba(253,230,138,.45); display:grid; place-items:center; text-align:center; font-weight:900; color:#fde68a; box-shadow:0 10px 22px rgba(0,0,0,.35); min-height:154px; }
  .spl-card { min-height:154px; border:0; border-radius:15px; padding:.65rem; color:#0f172a; cursor:default; text-align:left; position:relative; overflow:hidden; box-shadow:0 12px 28px rgba(0,0,0,.42); outline:1px solid rgba(255,255,255,.28); }
  .spl-card::after { content:""; position:absolute; inset:36px 10px 44px; border-radius:50%; background:rgba(255,255,255,.16); filter:blur(.5px); }
  .spl-card.playable { cursor:pointer; transform:translateY(0); transition:transform .12s ease, box-shadow .12s ease; }
  .spl-card.playable:hover { transform:translateY(-3px); box-shadow:0 18px 34px rgba(0,0,0,.55), 0 0 0 3px rgba(250,204,21,.55); }
  .card-white { background:linear-gradient(145deg,#fff,#cbd5e1); } .card-blue { background:linear-gradient(145deg,#60a5fa,#1d4ed8); color:#eff6ff; } .card-green { background:linear-gradient(145deg,#4ade80,#15803d); } .card-red { background:linear-gradient(145deg,#fb7185,#b91c1c); color:#fff1f2; } .card-black { background:linear-gradient(145deg,#525252,#020617); color:#f8fafc; } .card-empty { background:rgba(71,85,105,.45); color:#cbd5e1; }
  .points { position:absolute; top:.5rem; left:.65rem; font-size:1.65rem; font-weight:1000; text-shadow:0 2px 5px rgba(0,0,0,.3); z-index:1; }
  .card-action { position:absolute; top:.55rem; right:.55rem; z-index:2; display:flex; gap:.25rem; }
  .mini-btn { padding:.18rem .35rem; border-radius:999px; font-size:.72rem; background:rgba(15,23,42,.76); color:#fff; border:1px solid rgba(255,255,255,.32); }
  .costs { position:absolute; left:.55rem; bottom:.5rem; display:flex; gap:.24rem; flex-wrap:wrap; z-index:2; max-width:82%; }
  .noble-list, .player-grid { display:flex; gap:.6rem; flex-wrap:wrap; }
  .noble-tile { min-width:92px; border-radius:12px; padding:.5rem; background:linear-gradient(145deg,#fef3c7,#92400e); color:#1f2937; font-weight:800; box-shadow:0 8px 18px rgba(0,0,0,.3); }
  .players { display:grid; gap:.75rem; }
  .player.current { outline:2px solid #facc15; }
  .player-head { display:flex; justify-content:space-between; gap:.6rem; align-items:center; margin-bottom:.55rem; }
  .score { font-size:1.5rem; color:#fde68a; font-weight:1000; }
  .reserved { display:grid; grid-template-columns:repeat(3,minmax(88px,1fr)); gap:.45rem; margin-top:.55rem; }
  .reserve-card { min-height:86px; border-radius:10px; padding:.4rem; font-size:.78rem; }
  .owned { display:flex; gap:.35rem; flex-wrap:wrap; }
  .owned-pill { border-radius:999px; padding:.12rem .45rem; background:rgba(15,23,42,.58); border:1px solid rgba(255,255,255,.18); }
  @media (max-width: 1120px) { main.official-layout, .top-strip { grid-template-columns:1fr; } .tier-row { grid-template-columns:70px repeat(2,minmax(120px,1fr)); } }
</style>
<main class="official-layout">
  <section class="tableau">
    <div id="status" class="moves"></div>
    <div class="top-strip">
      <div class="bank"><div class="section-title">Bank tokens</div><div id="bank" class="tokens"></div></div>
      <div class="nobles"><div class="section-title">Nobles</div><div id="nobles" class="noble-list"></div></div>
    </div>
    <div id="tiers"></div>
    <details style="margin-top:1rem"><summary>文字盤面 / debug</summary><pre id="boardText" class="board"></pre></details>
  </section>
  <aside class="panel">
    <h2>玩家區</h2>
    <div id="players" class="players"></div>
    <h2>建議走法</h2>
    <p class="muted">點卡片、token 動作或「下這手」都會呼叫 <code>/api/move</code>，因此建議、落子與 Undo 都同步。</p>
    <div id="hints"></div>
    <h2>所有合法動作</h2>
    <div id="legalMoves" class="moves"></div>
    <h2>紀錄</h2>
    <ol id="log" class="log"></ol>
  </aside>
</main>
<script>
const gemOrder = ['white','blue','green','red','black','gold'];
const gemLabel = {white:'白', blue:'藍', green:'綠', red:'紅', black:'黑', gold:'金'};
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
}
function gemToken(color, value, small=false) {
  return `<span class="gem ${color} ${small ? 'small' : ''}" title="${color}">${value}</span>`;
}
function costTokens(cost) {
  return gemOrder.slice(0,5).filter(color => cost[color] > 0).map(color => gemToken(color, cost[color], true)).join('') || '<span class="muted">free</span>';
}
function renderCard(card) {
  const playable = card.action !== null || card.reserve_action !== null;
  const onclick = card.action !== null ? `onclick="playMove(${card.action})"` : '';
  const reserve = card.reserve_action !== null ? `<button class="mini-btn" onclick="event.stopPropagation(); playMove(${card.reserve_action})">保留</button>` : '';
  const buy = card.action !== null ? `<span class="mini-btn">買 #${card.action}</span>` : '';
  if (card.color === 'empty') return `<div class="spl-card card-empty"><span class="muted">empty</span></div>`;
  return `<div class="spl-card card-${card.color} ${playable ? 'playable' : ''}" ${onclick}>
    <div class="points">${card.points || ''}</div>
    <div class="card-action">${buy}${reserve}</div>
    <div class="costs">${costTokens(card.cost)}</div>
  </div>`;
}
function renderBoard(board) {
  document.getElementById('bank').innerHTML = gemOrder.map(color => gemToken(color, board.bank_gems[color])).join('');
  document.getElementById('nobles').innerHTML = board.nobles.map(noble => `<div class="noble-tile"><div>${noble.points} pts</div><div class="tokens">${costTokens(noble.cost)}</div></div>`).join('') || '<p class="muted">沒有可顯示的貴族。</p>';
  document.getElementById('tiers').innerHTML = board.visible_cards_by_tier.slice().reverse().map(tier => `
    <div class="tier-row">
      <div class="deck">Tier ${tier.tier + 1}<br><span class="muted">deck ${tier.deck_count}</span></div>
      ${tier.cards.map(renderCard).join('')}
    </div>`).join('');
  renderPlayers(board.players);
}
function renderPlayers(players) {
  document.getElementById('players').innerHTML = players.map(player => `<section class="player ${player.is_human ? 'current' : ''}">
    <div class="player-head"><strong>P${player.id}${player.is_human ? '（你）' : ''}</strong><span class="score">${player.score}</span></div>
    <div class="section-title">Gems</div><div class="tokens">${gemOrder.map(color => gemToken(color, player.gems[color], true)).join('')}</div>
    <div class="section-title" style="margin-top:.5rem">已購買卡</div><div class="owned">${gemOrder.slice(0,5).map(color => `<span class="owned-pill">${gemLabel[color]} ${player.cards[color]}</span>`).join('')}<span class="owned-pill">卡分 ${player.card_points}</span></div>
    <div class="section-title" style="margin-top:.5rem">貴族</div><div>${player.nobles.map(n => `<span class="owned-pill">${n.points} pts</span>`).join('') || '<span class="muted">無</span>'}</div>
    <div class="section-title" style="margin-top:.5rem">保留卡</div><div class="reserved">${player.reserved.map(card => `<div class="reserve-card card-${card.color} ${card.action !== null ? 'playable' : ''}" ${card.action !== null ? `onclick="playMove(${card.action})"` : ''}><strong>${card.points || ''}</strong><div class="costs">${costTokens(card.cost)}</div></div>`).join('') || '<span class="muted">無</span>'}</div>
  </section>`).join('');
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
  renderBoard(state.board);
  renderHints(state.board.hints || state.hints);
  renderLegalMoves(state.board.legal_moves || state.legal_moves);
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
    return html_page("Splendor AlphaZero 官方風格同步版", body)

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

    def send_static_file(self, root: Path, request_prefix: str, index_name: str = "index.html") -> None:
        parsed = urllib.parse.urlparse(self.path)
        relative = parsed.path.removeprefix(request_prefix).lstrip("/") or index_name
        candidate = (root / urllib.parse.unquote(relative)).resolve()
        root_resolved = root.resolve()
        if root_resolved != candidate and root_resolved not in candidate.parents:
            self.send_html(b"Not found", status=404)
            return
        if candidate.is_dir():
            candidate = candidate / index_name
        if not candidate.exists() or not candidate.is_file():
            self.send_html(b"Not found", status=404)
            return
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
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
            if self.path.startswith("/official-local-ui"):
                self.send_static_file(OFFICIAL_LOCAL_UI_ROOT, "/official-local-ui")
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
        elif args.official_local_ui:
            url = f"http://127.0.0.1:{args.port}/official-local-ui"
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
