"""Win probability model — stacked ensemble of XGBoost + LightGBM + logistic meta-learner."""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from bbbot.models.base import BaseModel

log = structlog.get_logger()

# Keep top N features by combined importance from initial training pass
TOP_N_FEATURES = 40


class WinProbabilityModel(BaseModel):
    """Stacked ensemble: XGBoost + LightGBM base learners, logistic meta-learner."""

    name = "win_probability"

    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.meta_learner = None
        self.feature_names = None
        self.selected_features = None
        self.feature_medians = None
        self._trained = False

    def _clean(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Impute missing values and select features.

        When fit=True (training), computes medians and selects features.
        When fit=False (prediction), applies stored medians and feature subset.
        """
        X_clean = X.copy()
        for col in X_clean.columns:
            X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

        if fit:
            self.feature_medians = X_clean.median()

        # Fill NaN with training medians (league-average values)
        if self.feature_medians is not None:
            X_clean = X_clean.fillna(self.feature_medians)
        X_clean = X_clean.fillna(0)  # fallback for any remaining NaN

        # Apply feature selection
        if not fit and self.selected_features is not None:
            missing = set(self.selected_features) - set(X_clean.columns)
            for col in missing:
                X_clean[col] = 0.0
            X_clean = X_clean[self.selected_features]

        return X_clean

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        """Train a quick XGBoost to rank features, return top N names."""
        from xgboost import XGBClassifier

        selector = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        selector.fit(X, y)
        importance = pd.Series(selector.feature_importances_, index=X.columns)
        top = importance.nlargest(TOP_N_FEATURES).index.tolist()
        log.info("feature_selection", total=len(X.columns), selected=len(top))
        return top

    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                              n_trials: int = 50) -> dict:
        """Use Optuna to find optimal hyperparameters via time-series CV."""
        import optuna
        from sklearn.metrics import log_loss

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

            from xgboost import XGBClassifier

            tscv = TimeSeriesSplit(n_splits=5)
            fold_losses = []

            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = XGBClassifier(
                    **params, random_state=42, verbosity=0,
                    early_stopping_rounds=20,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                probs = model.predict_proba(X_val)[:, 1]
                fold_losses.append(log_loss(y_val, probs))

            return np.mean(fold_losses)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        log.info("tuning_complete",
                 best_log_loss=f"{study.best_value:.4f}",
                 best_params=study.best_params)
        return study.best_params

    def train(self, X: pd.DataFrame, y: pd.Series, tune: bool = False,
              n_trials: int = 50, **kwargs) -> dict:
        """Train the stacked ensemble with time-series cross-validation.

        Args:
            tune: If True, run Optuna hyperparameter search first.
            n_trials: Number of Optuna trials (default 50).

        Returns a dict of evaluation metrics.
        """
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        self.feature_names = list(X.columns)

        # Impute with medians instead of filling zeros
        X_clean = self._clean(X, fit=True)

        # Feature selection: quick initial pass to find the top features
        self.selected_features = self._select_features(X_clean, y)
        X_clean = X_clean[self.selected_features]
        log.info("training_on_selected_features", n_features=len(self.selected_features))

        # Hyperparameter tuning
        if tune:
            log.info("starting_hyperparameter_tuning", n_trials=n_trials)
            best_params = self._tune_hyperparameters(X_clean, y, n_trials=n_trials)
        else:
            best_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            }

        # XGBoost base learner with tuned (or default) params
        self.xgb_model = XGBClassifier(
            **best_params,
            random_state=42,
            verbosity=0,
        )

        # LightGBM base learner — translate XGBoost params
        self.lgb_model = LGBMClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            min_child_samples=max(best_params["min_child_weight"], 5),
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            random_state=42,
            verbose=-1,
        )

        # Generate out-of-fold predictions for stacking
        tscv = TimeSeriesSplit(n_splits=5)
        oof_xgb = np.full(len(X_clean), np.nan)
        oof_lgb = np.full(len(X_clean), np.nan)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            self.xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                               verbose=False)
            oof_xgb[val_idx] = self.xgb_model.predict_proba(X_val)[:, 1]

            self.lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            oof_lgb[val_idx] = self.lgb_model.predict_proba(X_val)[:, 1]

            log.info("fold_complete", fold=fold + 1,
                     val_size=len(val_idx))

        # Train final base models on all data
        self.xgb_model.fit(X_clean, y, verbose=False)
        self.lgb_model.fit(X_clean, y)

        # Train meta-learner on out-of-fold predictions
        # Use NaN mask to identify indices that were in at least one validation fold
        mask = ~(np.isnan(oof_xgb) | np.isnan(oof_lgb))
        meta_X = np.column_stack([oof_xgb[mask], oof_lgb[mask]])
        meta_y = y.values[mask]

        self.meta_learner = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000),
            cv=3,
            method="isotonic",
        )
        self.meta_learner.fit(meta_X, meta_y)

        self._trained = True

        # Evaluation metrics — use OOF predictions (honest estimate, not training accuracy)
        from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
        oof_probs = (oof_xgb[mask] + oof_lgb[mask]) / 2  # average base learner OOF preds
        metrics = {
            "accuracy": accuracy_score(meta_y, (oof_probs >= 0.5).astype(int)),
            "log_loss": log_loss(meta_y, oof_probs),
            "brier_score": brier_score_loss(meta_y, oof_probs),
        }
        log.info("training_complete", **metrics)
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(home_win) for each game."""
        if not self._trained:
            from bbbot.models.baseline import BaselineWinModel
            return BaselineWinModel().predict_proba(X)

        X_clean = self._clean(X, fit=False)

        xgb_probs = self.xgb_model.predict_proba(X_clean)[:, 1]
        lgb_probs = self.lgb_model.predict_proba(X_clean)[:, 1]

        meta_X = np.column_stack([xgb_probs, lgb_probs])
        return self.meta_learner.predict_proba(meta_X)[:, 1]

    def get_feature_importance(self) -> pd.Series | None:
        """Return feature importance from XGBoost (gain-based)."""
        if not self._trained or not self.selected_features:
            return None
        importance = self.xgb_model.feature_importances_
        return pd.Series(importance, index=self.selected_features).sort_values(ascending=False)
