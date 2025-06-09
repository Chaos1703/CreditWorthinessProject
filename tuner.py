import optuna
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import lightgbm as lgb


def objective_factory(model_type, X, y , n_splits = 5 , cat_features = None , cat_idxs=None, cat_dims=None):
    def objective(trial):
        if model_type == "lightgbm":
            params = {
                "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "num_leaves":        trial.suggest_int("num_leaves", 16, 128),
                "max_depth":         trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 1.0),
                "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
                "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 0.5, 2.0),
                "n_estimators":      trial.suggest_int("n_estimators", 100, 2000),
                "random_state":      42,
                "n_jobs":            -1,
                "class_weight":      "balanced"
            }

            # 2) 5-fold stratified CV
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            macro_f1_scores = []

            for train_idx, valid_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

                model = LGBMClassifier(**params)
                # no early_stopping here to avoid compatibility issues
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)

                # optimize for balanced performance across both classes
                macro_f1 = f1_score(y_val, preds, average="macro")
                macro_f1_scores.append(macro_f1)

            return np.mean(macro_f1_scores)

        # CatBoost
        elif model_type == "catboost":
            params = {
                "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3),
                "depth":           trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg":     trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "iterations":      trial.suggest_int("iterations", 100, 1000),
                "verbose":         0,
                "random_seed":     42,
                "auto_class_weights": "Balanced"
            }

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            macro_f1_scores = []

            for train_idx, valid_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

                model = CatBoostClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features = cat_features , use_best_model=False)
                y_pred = model.predict(X_val)

                macro_f1 = f1_score(y_val, y_pred, average="macro")
                macro_f1_scores.append(macro_f1)

            return np.mean(macro_f1_scores)

        # Random Forest
        elif model_type == "random_forest":
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
                "max_depth":         trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
                "class_weight":      "balanced",
                "random_state":      42,
                "n_jobs":            -1,
            }

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            macro_f1_scores = []

            for train_idx, valid_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

                model = RandomForestClassifier(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)

                macro_f1 = f1_score(y_val, preds, average="macro")
                macro_f1_scores.append(macro_f1)

            return np.mean(macro_f1_scores)


        # TabNet
        elif model_type == "tabnet":
            params = {
                "n_d": trial.suggest_int("n_d", 8, 32),
                "n_a": trial.suggest_int("n_a", 8, 32),
                "n_steps": trial.suggest_int("n_steps", 3, 7),
                "gamma": trial.suggest_float("gamma", 1.0, 2.0),
                "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
                "optimizer_params": dict(lr=trial.suggest_float("lr", 2e-2, 0.1, log=True)),
                "mask_type": "entmax",
                "verbose": 0,
                ## FIX: Added crucial parameters for categorical features.
                "cat_idxs": cat_idxs,
                "cat_dims": cat_dims
            }
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            macro_f1_scores = []
        
            if not isinstance(X, np.ndarray):
                raise TypeError("For TabNet, please provide X and y as NumPy arrays.")

            for train_idx, valid_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[valid_idx]
                y_tr, y_val = y[train_idx], y[valid_idx]
                model = TabNetClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    patience=20,
                    max_epochs=200,
                    batch_size=1024,
                    drop_last=False,
                    weights=1
                )
                y_pred = model.predict(X_val)
                macro_f1_scores.append(f1_score(y_val, y_pred, average="macro"))
            return np.mean(macro_f1_scores)
        else:
            raise ValueError("Unsupported model type")

    return objective

def tune_model(model_type, X, y, n_trials=30, n_splits=5, cat_features=None, cat_idxs=None, cat_dims=None):
    study = optuna.create_study(direction="maximize")
    objective_func = objective_factory(model_type, X, y, n_splits=n_splits, cat_features=cat_features, cat_idxs=cat_idxs, cat_dims=cat_dims)
    study.optimize(objective_func, n_trials=n_trials)
    return study.best_params
