#!/usr/bin/env python3
"""Run cestpasphoto/alpha-zero-general human games with AlphaZero move hints.

This launcher is intentionally kept outside the upstream repository.  Point
``--azg-path`` at a checkout of cestpasphoto/alpha-zero-general and use
``human-hints`` as one of the two players.  Before each human input prompt, the
script runs the same MCTS+network stack as an AlphaZero player and prints the
best-looking legal moves for the current canonical board.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import os
import sys
import zlib
from dataclasses import dataclass
from enum import Enum
from math import isnan
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


class UndoMode(Enum):
    SAME_PLAYER = "same-player"
    ONE_PLY = "one-ply"


class UndoRequest(Exception):
    def __init__(self, mode: UndoMode):
        self.mode = mode
        super().__init__(mode.value)


@dataclass(frozen=True)
class HistoryEntry:
    board: Any
    current_player: int
    turn: int


@dataclass(frozen=True)
class HintRow:
    action: int
    q_value: Optional[float]
    visits: int
    probability: float
    prior: float
    label: str

    @property
    def estimated_win_rate(self) -> Optional[float]:
        if self.q_value is None:
            return None
        return max(0.0, min(1.0, (self.q_value + 1.0) / 2.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play alpha-zero-general games and show MCTS hints before human moves."
    )
    parser.add_argument(
        "--azg-path",
        default=os.environ.get("ALPHA_ZERO_GENERAL", "../alpha-zero-general"),
        help="Path to a cestpasphoto/alpha-zero-general checkout."
    )
    parser.add_argument("--top", type=int, default=8, help="Number of hinted moves to print.")
    parser.add_argument(
        "--sort-by",
        choices=("q", "visits", "probability", "prior"),
        default="q",
        help="Metric used to rank hinted moves."
    )
    parser.add_argument("--num-games", "-n", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--display", action="store_true", help="Ask Arena to display every board.")
    parser.add_argument(
        "--disable-undo",
        action="store_true",
        help="Disable console undo commands for human players."
    )
    parser.add_argument("--state", "-s", default="", help="Initial state passed to upstream Arena.")
    parser.add_argument(
        "--numMCTSSims", "-m", type=int, default=None, help="MCTS simulations per move."
    )
    parser.add_argument("--cpuct", "-c", type=float, default=None, help="Override cpuct.")
    parser.add_argument("--fpu", "-f", type=float, default=None, help="Override first-play urgency.")
    parser.add_argument(
        "--modern-onnx-export",
        action="store_true",
        help=(
            "Use PyTorch's default ONNX exporter. By default this launcher forces "
            "the legacy exporter for better compatibility with the Splendor model."
        ),
    )
    parser.add_argument(
        "--hint-temperature",
        type=float,
        default=0.01,
        help="Temperature used for the displayed visit distribution."
    )
    parser.add_argument("game", help="Game name understood by alpha-zero-general, e.g. splendor.")
    parser.add_argument(
        "players",
        nargs=2,
        help="Two players: checkpoint path/directory, random, greedy, human, or human-hints."
    )
    return parser.parse_args()


def add_upstream_to_path(azg_path: str) -> Path:
    root = Path(azg_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"alpha-zero-general path does not exist: {root}")
    sys.path.insert(0, str(root))
    return root


def import_upstream_modules():
    from GameSwitcher import import_game
    from MCTS import MCTS
    from utils import dotdict

    return import_game, MCTS, dotdict


def import_move_formatter(game_name: str) -> Optional[Callable[[int], str]]:
    if game_name != "splendor":
        return None
    from splendor.SplendorLogic import move_to_str

    return lambda action: move_to_str(action, short=True)


def checkpoint_file(player_name: str) -> str:
    return os.path.join(player_name, "best.pt") if os.path.isdir(player_name) else player_name


def mcts_args_from_checkpoint(args: argparse.Namespace, additional_keys: dict, dotdict):
    cpuct = additional_keys.get("cpuct")
    cpuct = float(cpuct[0]) if isinstance(cpuct, list) else cpuct
    return dotdict({
        "numMCTSSims": (
            args.numMCTSSims if args.numMCTSSims else additional_keys.get("numMCTSSims", 100)
        ),
        "fpu": args.fpu if args.fpu else additional_keys.get("fpu", 0.0),
        "universes": additional_keys.get("universes", 1),
        "cpuct": args.cpuct if args.cpuct else cpuct,
        "prob_fullMCTS": 1.0,
        "forced_playouts": additional_keys.get("forced_playouts", False),
        "no_mem_optim": False,
    })



def install_legacy_onnx_export_patch() -> None:
    """Force PyTorch's legacy ONNX exporter when available.

    Newer PyTorch releases default to the dynamo ONNX exporter.  The upstream
    Splendor network uses ``adaptive_max_pool2d``, which currently fails in that
    exporter on some Windows/Python 3.13 installations.  The legacy exporter is
    sufficient for this model and matches older working environments.
    """
    try:
        import torch.onnx
    except ImportError:
        return

    if getattr(torch.onnx.export, "_splendor_legacy_patch", False):
        return

    original_export = torch.onnx.export
    export_parameters = inspect.signature(original_export).parameters

    def export_with_legacy_default(*args, **kwargs):
        if "dynamo" in export_parameters:
            kwargs.setdefault("dynamo", False)
        return original_export(*args, **kwargs)

    export_with_legacy_default._splendor_legacy_patch = True
    torch.onnx.export = export_with_legacy_default

def build_advisor(game, nnet_cls, mcts_cls, dotdict, checkpoint: str, args: argparse.Namespace):
    if not getattr(args, "modern_onnx_export", False):
        install_legacy_onnx_export_patch()

    nn_args = {
        "lr": None,
        "dropout": 0.0,
        "epochs": None,
        "batch_size": None,
        "nn_version": -1,
    }
    net = nnet_cls(game, nn_args)
    cpt_dir, cpt_file = os.path.split(checkpoint)
    additional_keys = net.load_checkpoint(cpt_dir or ".", cpt_file)
    return mcts_cls(game, net, mcts_args_from_checkpoint(args, additional_keys, dotdict))


def numeric_or_none(value: float) -> Optional[float]:
    value = float(value)
    return None if isnan(value) or value <= -40.0 else value


def action_label(formatter: Optional[Callable[[int], str]], action: int) -> str:
    return formatter(action) if formatter else str(action)


def collect_hint_rows(game, mcts, board, probabilities: Iterable[float], formatter) -> list[HintRow]:
    state_key = game.stringRepresentation(board)
    _, valid_moves, priors, _, q_values, visits, _, _ = mcts.nodes_data[state_key]
    rows: list[HintRow] = []
    for action, is_valid in enumerate(valid_moves):
        if not is_valid:
            continue
        rows.append(HintRow(
            action=action,
            q_value=numeric_or_none(q_values[action]),
            visits=int(visits[action]),
            probability=float(probabilities[action]),
            prior=float(priors[action]),
            label=action_label(formatter, action),
        ))
    return rows


def row_sort_key(sort_by: str, row: HintRow) -> float:
    if sort_by == "q":
        return -1.0 if row.q_value is None else row.q_value
    if sort_by == "visits":
        return float(row.visits)
    if sort_by == "probability":
        return row.probability
    return row.prior


def percent(value: Optional[float]) -> str:
    return "   n/a" if value is None else f"{100.0 * value:6.1f}%"


def print_hints(rows: list[HintRow], top: int, sort_by: str) -> None:
    ranked = sorted(rows, key=lambda row: row_sort_key(sort_by, row), reverse=True)[:top]
    print("\nAlphaZero hints for current position:")
    print(
        f"{'#':<3} {'action':<7} {'win%':>7} {'visits':>7} "
        f"{'prob%':>7} {'prior%':>7}  move"
    )
    for index, row in enumerate(ranked, start=1):
        print(
            f"{index:<3} {row.action:<7} {percent(row.estimated_win_rate):>7} "
            f"{row.visits:>7} {percent(row.probability):>7} {percent(row.prior):>7}  {row.label}"
        )
    print()


class ConsoleHumanPlayer:
    def __init__(self, game, upstream_human_player, undo_enabled: bool = True):
        self.game = game
        self.upstream_human_player = upstream_human_player
        self.undo_enabled = undo_enabled

    def _show_moves(self, valid_moves):
        if hasattr(self.upstream_human_player, "show_main_moves"):
            self.upstream_human_player.show_main_moves(valid_moves)
            return

        for action, is_valid in enumerate(valid_moves):
            if is_valid:
                print(f"{action} = {self.game.moveToString(action, 0)}", end=" ")
        print()

    def _parse_undo(self, value: str) -> Optional[UndoMode]:
        if not self.undo_enabled:
            return None
        normalized = value.strip().lower()
        if normalized in {"u", "undo"}:
            return UndoMode.SAME_PLAYER
        if normalized in {"u1", "undo1", "undo 1", "undo-one", "undo-ply"}:
            return UndoMode.ONE_PLY
        return None

    def play(self, board, nb_moves):
        valid_moves = self.game.getValidMoves(board, 0)
        self._show_moves(valid_moves)
        if self.undo_enabled:
            print("undo/u = 回到你上一次決策前；undo 1/u1 = 只退回上一手")

        while True:
            input_move = input()
            undo_mode = self._parse_undo(input_move)
            if undo_mode:
                raise UndoRequest(undo_mode)
            if input_move == "+" and hasattr(self.upstream_human_player, "show_all_moves"):
                self.upstream_human_player.show_all_moves(valid_moves)
                continue
            try:
                action = int(input_move)
                if action < 0 or action >= len(valid_moves) or not valid_moves[action]:
                    raise ValueError
                return action
            except ValueError:
                print("Invalid move:", input_move)


class HintedHumanPlayer:
    def __init__(self, game, human_player, advisor, args: argparse.Namespace, formatter):
        self.game = game
        self.human_player = human_player
        self.advisor = advisor
        self.args = args
        self.formatter = formatter

    def play(self, board, nb_moves):
        probabilities, _, _ = self.advisor.getActionProb(
            board,
            temp=self.args.hint_temperature,
            force_full_search=True,
        )
        rows = collect_hint_rows(self.game, self.advisor, board, probabilities, self.formatter)
        print_hints(rows, self.args.top, self.args.sort_by)
        return self.human_player.play(board, nb_moves)


def choose_best_action(advisor, board) -> int:
    probabilities = advisor.getActionProb(board, temp=0.01, force_full_search=True)[0]
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def create_player(name, args, game, nnet_cls, players_module, mcts_cls, dotdict, formatter):
    if name == "random":
        return players_module.RandomPlayer(game).play
    if name == "greedy":
        return players_module.GreedyPlayer(game).play
    if name == "human":
        human = ConsoleHumanPlayer(
            game, players_module.HumanPlayer(game), undo_enabled=not args.disable_undo
        )
        return human.play
    if name == "human-hints":
        opponent_checkpoints = [checkpoint_file(p) for p in args.players if p != "human-hints"]
        if not opponent_checkpoints or opponent_checkpoints[0] in {"human", "random", "greedy"}:
            raise ValueError("human-hints needs the other player to be an AlphaZero checkpoint.")
        advisor = build_advisor(game, nnet_cls, mcts_cls, dotdict, opponent_checkpoints[0], args)
        human = ConsoleHumanPlayer(
            game, players_module.HumanPlayer(game), undo_enabled=not args.disable_undo
        )
        return HintedHumanPlayer(game, human, advisor, args, formatter).play

    checkpoint = checkpoint_file(name)
    advisor = build_advisor(game, nnet_cls, mcts_cls, dotdict, checkpoint, args)
    return lambda board, nb_moves: choose_best_action(advisor, board)


def snapshot(board: Any, current_player: int, turn: int) -> HistoryEntry:
    return HistoryEntry(board=board.copy(), current_player=int(current_player), turn=int(turn))


def load_initial_state(game, initial_state: str):
    board = game.getInitBoard()
    current_player, turn = 0, 0
    if initial_state:
        import numpy as np

        data = zlib.decompress(base64.b64decode(initial_state), wbits=-15)
        board = np.frombuffer(data[:-3], dtype=np.int8).copy().reshape(board.shape)
        current_player = int(data[-3])
        turn = int.from_bytes(data[-2:])
    return board, current_player, turn


def rewind_history(
    history: list[HistoryEntry], mode: UndoMode, requesting_player: int
) -> Optional[HistoryEntry]:
    if not history:
        return None

    if mode == UndoMode.ONE_PLY:
        return history.pop()

    rewind_index = next(
        (
            index
            for index in range(len(history) - 1, -1, -1)
            if history[index].current_player == requesting_player
        ),
        None,
    )
    if rewind_index is None:
        return None

    entry = history[rewind_index]
    del history[rewind_index:]
    return entry


class UndoableArena:
    def __init__(self, player1, player2, game, mcts_cls, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.mcts_cls = mcts_cls
        self.display = display

    def player_order(self, other_way: bool):
        if not other_way:
            return [self.player1] + [self.player2] * (self.game.getNumberOfPlayers() - 1)
        return [self.player2] + [self.player1] * (self.game.getNumberOfPlayers() - 1)

    def play_game(self, initial_state: str = "", verbose: bool = False, other_way: bool = False):
        players = self.player_order(other_way)
        current_player = 0
        board, current_player, turn = load_initial_state(self.game, initial_state)
        history: list[HistoryEntry] = []

        while not self.game.getGameEnded(board, current_player).any():
            turn += 1
            if verbose:
                if self.display:
                    self.display(board)
                print()
                print(f"Turn {turn} Player {current_player}: ", end="")

            canonical_board = self.game.getCanonicalForm(board, current_player)
            try:
                action = players[current_player](canonical_board, turn)
            except UndoRequest as request:
                previous = rewind_history(history, request.mode, current_player)
                if previous is None:
                    print("No move to undo.")
                    turn -= 1
                    continue
                board = previous.board.copy()
                current_player = previous.current_player
                turn = previous.turn
                self.mcts_cls.reset_all_search_trees()
                print(f"Undone. Back to turn {turn + 1}, player {current_player}.")
                continue

            valid_moves = self.game.getValidMoves(canonical_board, 0)
            if verbose:
                print(f"P{current_player} decided to {self.game.moveToString(action, current_player)}")
            assert valid_moves[action] > 0

            history.append(snapshot(board, current_player, turn - 1))
            board, current_player = self.game.getNextState(
                board, current_player, action, random_seed=0
            )
            current_player = int(current_player)

        if verbose:
            if self.display:
                self.display(board)
            print("Game over: Turn ", str(turn), "Result ", self.game.getGameEnded(board, current_player))
        elif initial_state:
            print(f"Game over: {self.game.getScore(board, 0)} - {self.game.getScore(board, 1)}")

        self.mcts_cls.reset_all_search_trees()
        return self.game.getGameEnded(board, current_player)[0]

    def play_games(self, num: int, initial_state: str = "", verbose: bool = False):
        one_won, two_won, draws = 0, 0, 0
        for index in range(num):
            one_vs_two = (index % 4 == 0) or (index % 4 == 3) or bool(initial_state)
            game_result = self.play_game(
                verbose=verbose, initial_state=initial_state, other_way=not one_vs_two
            )
            if game_result == (1.0 if one_vs_two else -1.0):
                one_won += 1
            elif game_result == (-1.0 if one_vs_two else 1.0):
                two_won += 1
            else:
                draws += 1
        return one_won, two_won, draws


def main() -> None:
    args = parse_args()
    add_upstream_to_path(args.azg_path)
    import_game, MCTS, dotdict = import_upstream_modules()
    game_cls, nnet_cls, players_module, _ = import_game(args.game)
    game = game_cls()
    formatter = import_move_formatter(args.game)
    player1 = create_player(
        args.players[0], args, game, nnet_cls, players_module, MCTS, dotdict, formatter
    )
    player2 = create_player(
        args.players[1], args, game, nnet_cls, players_module, MCTS, dotdict, formatter
    )
    arena = UndoableArena(player1, player2, game, MCTS, display=game.printBoard)
    human_game = any(player in {"human", "human-hints"} for player in args.players)
    result = arena.play_games(args.num_games, initial_state=args.state, verbose=args.display or human_game)
    print("Result:", result)


if __name__ == "__main__":
    main()
