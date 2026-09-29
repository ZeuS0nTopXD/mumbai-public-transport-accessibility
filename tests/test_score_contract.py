from pathlib import Path
import ast
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
TEMPLATE_SOURCE = (
    ROOT / "templates" / "index.html"
).read_text(encoding="utf-8")


def _load_evaluation_rank_key():
    tree = ast.parse(APP_SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_evaluation_rank_key"
    )
    namespace = {"np": np}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace["_evaluation_rank_key"]


def _load_tie_break_efficiency_score():
    tree = ast.parse(APP_SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_tie_break_efficiency_score"
    )
    namespace = {"np": np}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace["_tie_break_efficiency_score"]


evaluation_rank_key = _load_evaluation_rank_key()
tie_break_efficiency_score = _load_tie_break_efficiency_score()


class BenchmarkScoreContractTests(unittest.TestCase):
    def test_dashboard_uses_backend_score_description(self):
        expected_description = (
            "Ranking = lowest route-cost RMSE; ties are decided by a 50/50 "
            "efficiency score using measured compute time and average nodes "
            "checked."
        )

        self.assertIn(
            "BENCHMARK_RANKING_DESCRIPTION =",
            APP_SOURCE,
        )
        assignment = next(
            node
            for node in ast.walk(ast.parse(APP_SOURCE))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "BENCHMARK_RANKING_DESCRIPTION"
                for target in node.targets
            )
        )
        self.assertEqual(
            expected_description,
            ast.literal_eval(assignment.value),
        )
        self.assertIn(
            "{{ benchmark_ranking_description }}",
            TEMPLATE_SOURCE,
        )
        self.assertNotIn(
            "55% route optimality + 30% search efficiency + 15% runtime efficiency",
            TEMPLATE_SOURCE,
        )

    def test_home_context_passes_ranking_description_to_template(self):
        normalized_source = "".join(APP_SOURCE.split())
        self.assertIn(
            "benchmark_ranking_description=BENCHMARK_RANKING_DESCRIPTION",
            normalized_source,
        )

    def test_lower_rmse_ranks_before_secondary_score(self):
        slower_optimal = {
            "route_rmse": 0.0,
            "average_nodes_expanded": 82.51,
            "success_rate": 1.0,
            "algorithm": "A* Search",
        }
        faster_suboptimal = {
            "route_rmse": 13.329,
            "average_nodes_expanded": 8.99,
            "success_rate": 1.0,
            "algorithm": "Greedy Best First Search",
        }

        self.assertLess(
            evaluation_rank_key(slower_optimal),
            evaluation_rank_key(faster_suboptimal),
        )

    def test_equal_rmse_uses_efficiency_formula_as_tie_breaker(self):
        a_star = {
            "average_execution_time": 0.2,
            "average_nodes_expanded": 80.0,
        }
        ucs = {
            "average_execution_time": 0.1,
            "average_nodes_expanded": 100.0,
        }

        ucs_score = tie_break_efficiency_score(
            ucs,
            fastest_time=0.1,
            fewest_nodes=80.0,
        )
        a_star_score = tie_break_efficiency_score(
            a_star,
            fastest_time=0.1,
            fewest_nodes=80.0,
        )

        self.assertAlmostEqual(0.9, ucs_score)
        self.assertAlmostEqual(0.75, a_star_score)
        self.assertGreater(ucs_score, a_star_score)

    def test_equal_rmse_uses_higher_tie_break_score(self):
        a_star = {
            "route_rmse": 0.0,
            "tie_break_efficiency_score": 0.9,
            "success_rate": 1.0,
            "algorithm": "A* Search",
        }
        ucs = {
            "route_rmse": 0.0,
            "tie_break_efficiency_score": 0.8,
            "success_rate": 1.0,
            "algorithm": "Uniform Cost Search",
        }

        self.assertLess(
            evaluation_rank_key(a_star),
            evaluation_rank_key(ucs),
        )

    def test_dashboard_shows_evaluation_rank_not_aggregate_score(self):
        self.assertIn("Evaluation Rank", TEMPLATE_SOURCE)
        self.assertIn("Tie-Break Efficiency", TEMPLATE_SOURCE)
        self.assertNotIn("Overall Evaluation Score", TEMPLATE_SOURCE)

    def test_runtime_efficiency_is_exposed_to_the_dashboard(self):
        self.assertIn("'runtime_efficiency':", APP_SOURCE)
        self.assertIn("{{ result.runtime_efficiency }}%", TEMPLATE_SOURCE)

    def test_dashboard_labels_all_research_metrics(self):
        for label in (
            "Success Rate",
            "Route Optimality",
            "Search Efficiency",
            "Runtime Efficiency",
        ):
            self.assertIn(label, TEMPLATE_SOURCE)


if __name__ == "__main__":
    unittest.main()
