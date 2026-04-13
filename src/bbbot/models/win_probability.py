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


class WinProbabilityModel(BaseModel):
    """Stacked ensemble: XGBoost + LightGBM base learners, logistic meta-learner."""

    name = "win_probability"

    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.meta_learner = None
        self.feature_names = None
        self._trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict:
        """Train the stacked ensemble with time-series cross-validation.

        Returns a dict of evaluation metrics.
        """
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        self.feature_names = list(X.columns)

        # Replace NaN/inf with 0 for tree models
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)

        # XGBoost base learner
        self.xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )

        # LightGBM base learner
        self.lgb_model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )

        # Generate out-of-fold predictions for stacking
        tscv = TimeSeriesSplit(n_splits=5)
        oof_xgb = np.zeros(len(X_clean))
        oof_lgb = np.zeros(len(X_clean))

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
        # Only use indices that have valid OOF predictions (skip first fold's training data)
        mask = (oof_xgb != 0) | (oof_lgb != 0)
        meta_X = np.column_stack([oof_xgb[mask], oof_lgb[mask]])
        meta_y = y.values[mask]

        self.meta_learner = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000),
            cv=3,
            method="isotonic",
        )
        self.meta_learner.fit(meta_X, meta_y)

        self._trained = True

        # Evaluation metrics
        from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
        final_probs = self.predict_proba(X)
        metrics = {
            "accuracy": accuracy_score(y, (final_probs >= 0.5).astype(int)),
            "log_loss": log_loss(y, final_probs),
            "brier_score": brier_score_loss(y, final_probs),
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

        X_clean = X.copy()
        for col in X_clean.columns:
            X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")
        X_clean = X_clean.fillna(0).replace([np.inf, -np.inf], 0)

        # Ensure feature alignment
        if self.feature_names:
            missing = set(self.feature_names) - set(X_clean.columns)
            for col in missing:
                X_clean[col] = 0.0
            X_clean = X_clean[self.feature_names]

        xgb_probs = self.xgb_model.predict_proba(X_clean)[:, 1]
        lgb_probs = self.lgb_model.predict_proba(X_clean)[:, 1]

        meta_X = np.column_stack([xgb_probs, lgb_probs])
        return self.meta_learner.predict_proba(meta_X)[:, 1]

    def get_feature_importance(self) -> pd.Series | None:
        """Return feature importance from XGBoost (gain-based)."""
        if not self._trained or not self.feature_names:
            return None
        importance = self.xgb_model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
