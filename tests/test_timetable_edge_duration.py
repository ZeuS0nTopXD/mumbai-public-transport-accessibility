import ast
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")


def _load_scheduled_edge_duration():
    tree = ast.parse(APP_SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_scheduled_edge_duration"
    )
    namespace = {"math": math}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace["_scheduled_edge_duration"]


def _load_service_group_columns():
    tree = ast.parse(APP_SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_service_group_columns"
    )
    namespace = {}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace["_service_group_columns"]


scheduled_edge_duration = _load_scheduled_edge_duration()
service_group_columns = _load_service_group_columns()


class TimetableEdgeDurationTests(unittest.TestCase):
    def test_rejects_small_negative_timetable_reversal(self):
        self.assertIsNone(scheduled_edge_duration(10 * 60 + 42, 10 * 60 + 41))

    def test_keeps_genuine_late_night_to_early_morning_crossing(self):
        self.assertAlmostEqual(
            5.0,
            scheduled_edge_duration(23 * 60 + 59, 4),
        )

    def test_keeps_normal_positive_consecutive_stop_duration(self):
        self.assertAlmostEqual(
            3.0,
            scheduled_edge_duration(10 * 60 + 42, 10 * 60 + 45),
        )

    def test_rejects_early_morning_reversal_as_invalid(self):
        self.assertIsNone(scheduled_edge_duration(6, 3))

    def test_separates_reused_train_ids_by_timetable_provenance(self):
        class Frame:
            columns = {
                "line",
                "train_id",
                "source_file",
                "source_page",
            }

        self.assertEqual(
            ["line", "train_id", "source_file", "source_page"],
            service_group_columns(Frame()),
        )


if __name__ == "__main__":
    unittest.main()
