from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
TEMPLATE_SOURCE = (
    ROOT / "templates" / "index.html"
).read_text(encoding="utf-8")


class BenchmarkScoreContractTests(unittest.TestCase):
    def test_dashboard_uses_backend_score_description(self):
        expected_description = (
            "Score = success rate × (55% route optimality + 30% search "
            "efficiency + 15% runtime efficiency)"
        )

        self.assertIn(
            "BENCHMARK_SCORE_DESCRIPTION =",
            APP_SOURCE,
        )
        assignment = next(
            node
            for node in ast.walk(ast.parse(APP_SOURCE))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "BENCHMARK_SCORE_DESCRIPTION"
                for target in node.targets
            )
        )
        self.assertEqual(
            expected_description,
            ast.literal_eval(assignment.value),
        )
        self.assertIn(
            "{{ benchmark_score_description }}",
            TEMPLATE_SOURCE,
        )
        self.assertNotIn(
            "35% route quality + 30% search efficiency + 20% consistency + 15% success",
            TEMPLATE_SOURCE,
        )

    def test_home_context_passes_score_description_to_template(self):
        normalized_source = "".join(APP_SOURCE.split())
        self.assertIn(
            "benchmark_score_description=BENCHMARK_SCORE_DESCRIPTION",
            normalized_source,
        )

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
