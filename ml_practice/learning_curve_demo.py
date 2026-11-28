import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, learning_curve, train_test_split

from regression_baseline import make_models


def run(seed=42):
    data = load_diabetes(scaled=False)
    train_ids, excluded_ids = train_test_split(
        np.arange(len(data.target)), test_size=0.2, random_state=seed
    )
    sizes, train_scores, validation_scores = learning_curve(
        make_models()["ridge"],
        data.data[train_ids],
        data.target[train_ids],
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="neg_mean_absolute_error",
        shuffle=True,
        random_state=seed,
        error_score="raise",
    )
    train_mae, validation_mae = -train_scores, -validation_scores
    return {
        "dataset": "sklearn.datasets.load_diabetes(scaled=False)",
        "seed": seed,
        "ridge_alpha": 1.0,
        "train_pool_size": len(train_ids),
        "excluded_test_size": len(excluded_ids),
        "cv_folds": 5,
        "points": [
            {
                "fit_size_per_fold": int(size),
                "train_mae_mean": float(train_mae[index].mean()),
                "validation_mae_mean": float(validation_mae[index].mean()),
                "validation_mae_std": float(validation_mae[index].std()),
            }
            for index, size in enumerate(sizes)
        ],
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
