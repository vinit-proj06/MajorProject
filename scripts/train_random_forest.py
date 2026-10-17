"""Train and evaluate a patient-isolated Random Forest sepsis baseline."""
from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from src.model_training import load_model_config, run_training, script_arguments, training_weight


def fit_random_forest(train, validation, features, config, model_dir):
    """All training rows remain eligible; per-tree bootstrap size is bounded."""
    settings = config["random_forest"]
    positive_weight, _ = training_weight(train["y"], config)
    parameters = {
        "n_estimators": 0, "warm_start": True, "random_state": config["seed"], "n_jobs": config["n_jobs"],
        "max_depth": settings["max_depth"], "max_leaf_nodes": settings["max_leaf_nodes"],
        "min_samples_leaf": settings["min_samples_leaf"], "max_features": settings["max_features"],
        "bootstrap": True, "max_samples": min(settings["max_samples"], len(train["y"])),
        "class_weight": {0: 1.0, 1: positive_weight}, "oob_score": False,
    }
    forest = RandomForestClassifier(**parameters)
    started = time.perf_counter()
    with threadpool_limits(limits=config["n_jobs"]):
        for count in range(settings["trees_per_batch"], settings["n_estimators"] + settings["trees_per_batch"], settings["trees_per_batch"]):
            forest.set_params(n_estimators=min(count, settings["n_estimators"]))
            forest.fit(train["X"], train["y"])
            logging.info("Random Forest: fitted %s/%s trees; %.1fs", len(forest.estimators_), settings["n_estimators"], time.perf_counter() - started)
    forest.set_params(warm_start=False)
    return forest, {"parameters": forest.get_params(), "trees": len(forest.estimators_),
                    "bootstrap_note": "Each tree draws at most max_samples rows with replacement from the full training pool; no global patient/row subset is discarded.",
                    "memory_note": "Input is float32 memory-mapped data. sklearn is not incremental/out-of-core; tree fitting can page input and allocate working memory. Models run sequentially with bounded tree depth and worker count."}


def predict_random_forest(model, X):
    return model.predict_proba(X)[:, 1]


def save_random_forest(model, model_dir):
    path = model_dir / "random_forest.joblib"
    joblib.dump(model, path, compress=3)
    return path


def main():
    args = script_arguments(__doc__)
    run_training("random_forest", fit_random_forest, predict_random_forest, save_random_forest,
                 load_model_config(args.config), args.overwrite, args.rebuild_cache)


if __name__ == "__main__":
    main()
