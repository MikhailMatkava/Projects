import unittest

import numpy as np

from ridge_cv import run, search_ridge


class RidgeCVTests(unittest.TestCase):
    def test_selection_uses_smallest_mean_cv_mae(self):
        report = run()
        winner = min(report["candidates"], key=lambda candidate: candidate["cv_mae_mean"])
        self.assertEqual(report["selected_alpha"], winner["alpha"])
        self.assertEqual((report["train_size"], report["excluded_test_size"]), (353, 89))
        self.assertEqual(report, run())

    def test_grid_order_and_duplicates_do_not_change_selection(self):
        x = np.arange(60, dtype=float).reshape(30, 2)
        y = 3 * x[:, 0] + np.sin(x[:, 0])
        first = search_ridge(x, y, [10, 0.1, 1, 1])
        second = search_ridge(x, y, [0.1, 1, 10])
        self.assertEqual(first.best_params_, second.best_params_)
        np.testing.assert_allclose(first.predict(x[:3]), second.predict(x[:3]))
        np.testing.assert_array_equal(
            first.cv_results_["mean_test_score"], second.cv_results_["mean_test_score"]
        )

    def test_invalid_alpha_grid_rejected(self):
        for alphas in ([], [[1]], [0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(alphas=alphas), self.assertRaises(ValueError):
                search_ridge([[0]], [0], alphas)


if __name__ == "__main__":
    unittest.main()
