import importlib.util
import sys
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "azg_human_hints.py"
spec = importlib.util.spec_from_file_location("azg_human_hints", MODULE_PATH)
azg_human_hints = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = azg_human_hints
spec.loader.exec_module(azg_human_hints)


class DummyBoard:
    def __init__(self, value):
        self.value = value

    def copy(self):
        return DummyBoard(self.value)


class UndoHistoryTest(unittest.TestCase):
    def entry(self, value, player, turn):
        return azg_human_hints.HistoryEntry(DummyBoard(value), player, turn)

    def test_one_ply_rewinds_only_last_entry(self):
        history = [self.entry("before-p0", 0, 0), self.entry("before-p1", 1, 1)]

        previous = azg_human_hints.rewind_history(
            history, azg_human_hints.UndoMode.ONE_PLY, requesting_player=0
        )

        self.assertEqual(previous.board.value, "before-p1")
        self.assertEqual([entry.board.value for entry in history], ["before-p0"])

    def test_same_player_rewinds_to_requesting_players_previous_decision(self):
        history = [
            self.entry("before-p0-first", 0, 0),
            self.entry("before-p1", 1, 1),
            self.entry("before-p0-second", 0, 2),
            self.entry("before-p1-second", 1, 3),
        ]

        previous = azg_human_hints.rewind_history(
            history, azg_human_hints.UndoMode.SAME_PLAYER, requesting_player=0
        )

        self.assertEqual(previous.board.value, "before-p0-second")
        self.assertEqual(
            [entry.board.value for entry in history],
            ["before-p0-first", "before-p1"],
        )

    def test_same_player_without_prior_decision_keeps_history(self):
        history = [self.entry("before-ai-first", 0, 0)]

        previous = azg_human_hints.rewind_history(
            history, azg_human_hints.UndoMode.SAME_PLAYER, requesting_player=1
        )

        self.assertIsNone(previous)
        self.assertEqual([entry.board.value for entry in history], ["before-ai-first"])


if __name__ == "__main__":
    unittest.main()
