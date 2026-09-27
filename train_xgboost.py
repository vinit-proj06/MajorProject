"""Train and evaluate XGBoost using disk-backed CPU histogram data."""
from __future__ import annotations

import gc
import logging

import numpy as np
import xgboost as xgb

from src.common import save_json
from src.model_training import load_model_config, run_training, script_arguments, training_weight


class MemmapIterator(xgb.DataIter):
    """Resettable bounded NumPy batches for CPU external-memory quantization."""
    def __init__(self, X, y, features, batch_rows, cache_prefix):
        self.X, self.y, self.features = X, y, features
        self.batch_rows, self.position = batch_rows, 0
        super().__init__(cache_prefix=str(cache_prefix), release_data=True)

    def reset(self):
        self.position = 0

    def next(self, input_data):
        if self.position >= len(self.y):
            return False
        end = min(self.position + self.batch_rows, len(self.y))
        input_data(data=np.asarray(self.X[self.position:end], dtype=np.float32),
                   label=np.asarray(self.y[self.position:end], dtype=np.float32), feature_names=self.features)
        self.position = end
        return True


def fit_xgboost(train, validation, features, config, model_dir):
    settings = config["xgboost"]
    weight, _ = training_weight(train["y"], config)
    cache_dir = model_dir / "external_memory"
    cache_dir.mkdir()
    train_iter = MemmapIterator(train["X"], train["y"], features, config["batch_rows"], cache_dir / "train")
    validation_iter = MemmapIterator(validation["X"], validation["y"], features, config["batch_rows"], cache_dir / "validation")
    logging.info("Building external-memory training quantiles (max_bin=%s)", settings["max_bin"])
    dtrain = xgb.ExtMemQuantileDMatrix(train_iter, max_bin=settings["max_bin"], nthread=config["n_jobs"])
    # Use TRAINING quantile cuts for validation; never learn a held-out transform.
    dvalidation = xgb.ExtMemQuantileDMatrix(validation_iter, ref=dtrain, max_bin=settings["max_bin"], nthread=config["n_jobs"])
    parameters = {
        "objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist", "device": "cpu",
        "grow_policy": "depthwise", "max_bin": settings["max_bin"], "max_depth": settings["max_depth"],
        "eta": settings["learning_rate"], "min_child_weight": settings["min_child_weight"],
        "subsample": settings["subsample"], "colsample_bytree": settings["colsample_bytree"],
        "lambda": settings["reg_lambda"], "scale_pos_weight": weight, "seed": config["seed"], "nthread": config["n_jobs"],
    }
    history = {}
    stop = xgb.callback.EarlyStopping(rounds=settings["early_stopping_rounds"], metric_name="aucpr", data_name="validation", maximize=True, save_best=True)
    booster = xgb.train(parameters, dtrain, num_boost_round=settings["num_boost_round"], evals=[(dvalidation, "validation")],
                        evals_result=history, callbacks=[stop], verbose_eval=settings["log_every"])
    details = {"parameters": parameters, "best_iteration": int(booster.best_iteration), "best_validation_xgboost_aucpr": float(booster.best_score),
               "trees_saved": booster.num_boosted_rounds(), "early_stopping_rounds": settings["early_stopping_rounds"],
               "early_stopping_note": "Validation only. XGBoost aucpr is its own PR-area metric; final AUPRC(AP) is separately computed with sklearn average_precision_score.",
               "memory_note": "CPU ExtMemQuantileDMatrix with disk pages; validation quantiles reference training cuts."}
    save_json(model_dir / "learning_history.json", history)
    del dvalidation, dtrain, validation_iter, train_iter
    gc.collect()
    return booster, details


def predict_xgboost(model, X):
    return model.inplace_predict(X, iteration_range=(0, model.best_iteration + 1))


def save_xgboost(model, model_dir):
    path = model_dir / "xgboost.ubj"
    model.save_model(path)
    return path


def main():
    args = script_arguments(__doc__)
    run_training("xgboost", fit_xgboost, predict_xgboost, save_xgboost,
                 load_model_config(args.config), args.overwrite, args.rebuild_cache)


if __name__ == "__main__":
    main()
