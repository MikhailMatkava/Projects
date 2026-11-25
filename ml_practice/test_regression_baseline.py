import unittest

import numpy as np

from regression_baseline import make_models, run


class RegressionTests(unittest.TestCase):
    def test_scaler_does_not_fit_on_prediction_data(self):
        model = make_models()["ridge"]
        x_train = np.array([[0.0], [2.0], [4.0]])
        model.fit(x_train, [1.0, 5.0, 9.0])
        scaler = model.named_steps["standardscaler"]
        model.predict([[1000.0]])
        np.testing.assert_allclose(scaler.mean_, [2.0])
        self.assertEqual(scaler.n_samples_seen_, 3)

    def test_report_is_reproducible_and_beats_mean_baseline(self):
        report = run(seed=42)
        self.assertEqual(report, run(seed=42))
        self.assertEqual(report["train_size"] + report["test_size"], 442)
        baseline = report["results"]["mean_baseline"]
        ridge = report["results"]["ridge"]
        self.assertLess(ridge["cv_mae_mean"], baseline["cv_mae_mean"])
        self.assertLess(ridge["test_mae"], baseline["test_mae"])
        for result in report["results"].values():
            self.assertTrue(all(np.isfinite(value) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
