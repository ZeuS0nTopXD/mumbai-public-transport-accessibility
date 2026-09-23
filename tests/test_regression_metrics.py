import ast
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")


def _load_route_cost_regression_metrics():
    tree = ast.parse(APP_SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_route_cost_regression_metrics"
    )
    namespace = {"np": np}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace["_route_cost_regression_metrics"]


route_cost_regression_metrics = _load_route_cost_regression_metrics()


class RouteCostRegressionMetricTests(unittest.TestCase):
    def test_perfect_route_cost_predictions_have_zero_error(self):
        metrics = route_cost_regression_metrics(
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
        )

        self.assertEqual(3, metrics["samples"])
        self.assertAlmostEqual(0.0, metrics["mae"])
        self.assertAlmostEqual(0.0, metrics["mse"])
        self.assertAlmostEqual(0.0, metrics["rmse"])
        self.assertAlmostEqual(0.0, metrics["mape"])
        self.assertAlmostEqual(1.0, metrics["r2"])

    def test_mse_is_mean_squared_route_cost_error(self):
        metrics = route_cost_regression_metrics(
            [10.0, 20.0],
            [12.0, 16.0],
        )

        self.assertAlmostEqual(3.0, metrics["mae"])
        self.assertAlmostEqual(10.0, metrics["mse"])
        self.assertAlmostEqual(np.sqrt(10.0), metrics["rmse"])

    def test_dashboard_exposes_route_cost_regression_metrics(self):
        template = (ROOT / "templates" / "index.html").read_text(
            encoding="utf-8"
        )
        for label in ("MAE", "MSE", "RMSE", "MAPE", "R²"):
            self.assertIn(label, template)

    def test_benchmark_uses_three_hundred_routes(self):
        self.assertIn("BENCHMARK_ROUTE_COUNT = 300", APP_SOURCE)
        self.assertIn("number_of_pairs=BENCHMARK_ROUTE_COUNT", APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
