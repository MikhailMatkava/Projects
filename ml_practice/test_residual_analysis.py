import unittest

import numpy as np

from residual_analysis import run, summarize_errors


class ResidualTests(unittest.TestCase):
    def test_metrics_signs_and_original_sample_indices(self):
        report = summarize_errors([10, 20, 30], [12, 15, 30], [9, 4, 7])
        self.assertAlmostEqual(report["mae"], 7 / 3)
        self.assertAlmostEqual(report["rmse"], np.sqrt(29 / 3))
        self.assertAlmostEqual(report["mean_residual"], 1)
        self.assertEqual(report["median_absolute_error"], 2)
        self.assertEqual([row["dataset_index"] for row in report["largest_errors"]], [4, 9, 7])
        self.assertEqual(report["largest_errors"][1]["residual"], -2)

    def test_invalid_inputs_rejected(self):
        cases = [([], [], []), ([[1]], [[1]], [[0]]), ([1], [1, 2], [0]),
                 ([1], [1], []), ([float("nan")], [1], [0]), ([1], [float("inf")], [0])]
        for observed, predicted, ids in cases:
            with self.subTest(observed=observed, predicted=predicted), self.assertRaises(ValueError):
                summarize_errors(observed, predicted, ids)

    def test_out_of_fold_report_is_reproducible(self):
        report = run()
        self.assertEqual(report, run())
        self.assertEqual((report["train_size"], report["excluded_test_size"]), (353, 89))
        self.assertEqual(len(report["errors"]["largest_errors"]), 5)
        self.assertGreaterEqual(report["errors"]["rmse"], report["errors"]["mae"])


if __name__ == "__main__":
    unittest.main()
