import unittest

import numpy as np

from learning_curve_demo import run


class LearningCurveTests(unittest.TestCase):
    def test_curve_sizes_respect_cross_validation_training_limit(self):
        report = run()
        sizes = [point["fit_size_per_fold"] for point in report["points"]]
        self.assertEqual(len(sizes), 5)
        self.assertTrue(all(left < right for left, right in zip(sizes, sizes[1:])))
        self.assertEqual((report["train_pool_size"], report["excluded_test_size"]), (353, 89))
        self.assertLessEqual(sizes[-1], 353 - 71)
        for point in report["points"]:
            for key in ("train_mae_mean", "validation_mae_mean", "validation_mae_std"):
                self.assertTrue(np.isfinite(point[key]))
                self.assertGreaterEqual(point[key], 0)

    def test_repeated_seed_reproduces_curve(self):
        self.assertEqual(run(seed=7), run(seed=7))


if __name__ == "__main__":
    unittest.main()
