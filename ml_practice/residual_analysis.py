import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from regression_baseline import make_models


def summarize_errors(observed, predicted, sample_ids):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    sample_ids = np.asarray(sample_ids)
    if observed.ndim != 1 or not len(observed):
        raise ValueError("observed must be a non-empty one-dimensional array")
    if predicted.shape != observed.shape or sample_ids.shape != observed.shape:
        raise ValueError("observed, predicted and sample_ids must have matching shapes")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("observed and predicted must be finite")
    residuals = observed - predicted
    absolute_errors = np.abs(residuals)
    worst = np.argsort(-absolute_errors, kind="stable")[:5]
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(root_mean_squared_error(observed, predicted)),
        "mean_residual": float(residuals.mean()),
        "median_absolute_error": float(np.median(absolute_errors)),
        "p90_absolute_error": float(np.quantile(absolute_errors, 0.9)),
        "largest_errors": [
            {
                "dataset_index": int(sample_ids[index]),
                "observed": float(observed[index]),
                "predicted": float(predicted[index]),
                "residual": float(residuals[index]),
                "absolute_error": float(absolute_errors[index]),
            }
            for index in worst
        ],
    }


def run(seed=42):
    data = load_diabetes(scaled=False)
    train_ids, excluded_ids = train_test_split(
        np.arange(len(data.target)), test_size=0.2, random_state=seed
    )
    x_train, y_train = data.data[train_ids], data.target[train_ids]
    predicted = cross_val_predict(
        make_models()["ridge"], x_train, y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
    )
    return {
        "dataset": "sklearn.datasets.load_diabetes(scaled=False)",
        "seed": seed,
        "ridge_alpha": 1.0,
        "prediction_source": "out-of-fold predictions within train, fixed alpha",
        "residual_definition": "observed minus predicted",
        "train_size": len(train_ids),
        "excluded_test_size": len(excluded_ids),
        "cv_folds": 5,
        "errors": summarize_errors(y_train, predicted, train_ids),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(args.seed), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
