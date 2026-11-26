import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def search_ridge(x_train, y_train, alphas, seed=42):
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or not len(alphas):
        raise ValueError("alphas must be a non-empty one-dimensional sequence")
    if not np.isfinite(alphas).all() or (alphas <= 0).any():
        raise ValueError("alphas must be finite and positive")
    model = make_pipeline(StandardScaler(), Ridge())
    search = GridSearchCV(
        model,
        {"ridge__alpha": sorted(set(alphas.tolist()))},
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        error_score="raise",
    )
    return search.fit(x_train, y_train)


def run(seed=42, alphas=(0.01, 0.1, 1, 10, 100, 1000)):
    data = load_diabetes(scaled=False)
    train_ids, excluded_ids = train_test_split(
        np.arange(len(data.target)), test_size=0.2, random_state=seed
    )
    search = search_ridge(data.data[train_ids], data.target[train_ids], alphas, seed)
    candidates = []
    scores = search.cv_results_
    for index, params in enumerate(scores["params"]):
        candidates.append({
            "alpha": params["ridge__alpha"],
            "cv_mae_mean": float(-scores["mean_test_score"][index]),
            "cv_mae_std": float(scores["std_test_score"][index]),
        })
    return {
        "dataset": "sklearn.datasets.load_diabetes(scaled=False)",
        "seed": seed,
        "train_size": len(train_ids),
        "excluded_test_size": len(excluded_ids),
        "cv_folds": 5,
        "selected_alpha": search.best_params_["ridge__alpha"],
        "selection_metric": "minimum mean CV MAE within train",
        "candidates": candidates,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.1, 1, 10, 100, 1000])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = json.dumps(run(args.seed, args.alphas), indent=2) + "\n"
    except ValueError as error:
        parser.error(str(error))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
