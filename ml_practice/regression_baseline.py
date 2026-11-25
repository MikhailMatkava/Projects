import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_models():
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }


def run(seed=42):
    data = load_diabetes(scaled=False)
    train_ids, test_ids = train_test_split(
        np.arange(len(data.target)), test_size=0.2, random_state=seed
    )
    x_train, x_test = data.data[train_ids], data.data[test_ids]
    y_train, y_test = data.target[train_ids], data.target[test_ids]
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = {}
    for name, model in make_models().items():
        cv_mae = -cross_val_score(
            model, x_train, y_train, cv=folds, scoring="neg_mean_absolute_error"
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        results[name] = {
            "cv_mae_mean": float(cv_mae.mean()),
            "cv_mae_std": float(cv_mae.std()),
            "test_mae": float(mean_absolute_error(y_test, prediction)),
            "test_rmse": float(root_mean_squared_error(y_test, prediction)),
            "test_r2": float(r2_score(y_test, prediction)),
        }
    return {
        "dataset": "sklearn.datasets.load_diabetes(scaled=False)",
        "seed": seed,
        "ridge_alpha": 1.0,
        "train_size": len(train_ids),
        "test_size": len(test_ids),
        "features": data.feature_names,
        "cv_folds": 5,
        "target": "dataset disease progression score",
        "results": results,
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
    report = json.dumps(run(args.seed), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
