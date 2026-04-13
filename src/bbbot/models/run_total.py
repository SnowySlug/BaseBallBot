"""Run total model — Poisson-based regression for predicting runs scored."""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from scipy.stats import poisson
from sklearn.model_selection import TimeSeriesSplit

from bbbot.models.base import BaseModel

log = structlog.get_logger()


class RunTotalModel(BaseModel):
    """Predicts expected runs for home and away teams using gradient boosting.

    Uses the predicted means to fit Poisson distributions for deriving
    over/under and run-line probabilities.
    """

    name = "run_total"

    def __init__(self):
        self.home_model = None
        self.away_model = None
        self.feature_names = None
        self._trained = False

    def train(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series,
              **kwargs) -> dict:
        """Train separate models for home and away run prediction."""
        from lightgbm import LGBMRegressor

        self.feature_names = list(X.columns)
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)

        self.home_model = LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            random_state=42,
            verbose=-1,
        )

        self.away_model = LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            random_state=42,
            verbose=-1,
        )

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        home_maes = []
        away_maes = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            yh_tr, yh_val = y_home.iloc[train_idx], y_home.iloc[val_idx]
            ya_tr, ya_val = y_away.iloc[train_idx], y_away.iloc[val_idx]

            self.home_model.fit(X_tr, yh_tr)
            self.away_model.fit(X_tr, ya_tr)

            h_pred = self.home_model.predict(X_val)
            a_pred = self.away_model.predict(X_val)

            home_maes.append(np.mean(np.abs(h_pred - yh_val)))
            away_maes.append(np.mean(np.abs(a_pred - ya_val)))

        # Final fit on all data
        self.home_model.fit(X_clean, y_home)
        self.away_model.fit(X_clean, y_away)
        self._trained = True

        metrics = {
            "home_mae": np.mean(home_maes),
            "away_mae": np.mean(away_maes),
            "total_mae": (np.mean(home_maes) + np.mean(away_maes)),
        }
        log.info("run_total_training_complete", **metrics)
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted [home_runs, away_runs] for each game."""
        if not self._trained:
            from bbbot.models.baseline import BaselineRunsModel
            return BaselineRunsModel().predict(X)

        X_clean = X.copy()
        for col in X_clean.columns:
            X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")
        X_clean = X_clean.fillna(0).replace([np.inf, -np.inf], 0)

        if self.feature_names:
            missing = set(self.feature_names) - set(X_clean.columns)
            for col in missing:
                X_clean[col] = 0.0
            X_clean = X_clean[self.feature_names]

        home_pred = np.clip(self.home_model.predict(X_clean), 0.5, 15.0)
        away_pred = np.clip(self.away_model.predict(X_clean), 0.5, 15.0)
        return np.column_stack([home_pred, away_pred])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)

    def predict_distributions(self, X: pd.DataFrame, max_runs: int = 20
                              ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return Poisson PMFs for each game.

        Returns list of (home_pmf, away_pmf) tuples.
        """
        preds = self.predict(X)
        distributions = []
        for home_mu, away_mu in preds:
            home_pmf = poisson.pmf(range(max_runs + 1), home_mu)
            away_pmf = poisson.pmf(range(max_runs + 1), away_mu)
            distributions.append((home_pmf, away_pmf))
        return distributions

    def predict_over_under(self, X: pd.DataFrame, line: float = 8.5
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Predict P(over) and P(under) for a given total line."""
        distributions = self.predict_distributions(X)
        over_probs = np.zeros(len(X))
        under_probs = np.zeros(len(X))

        for i, (home_pmf, away_pmf) in enumerate(distributions):
            total_pmf = np.convolve(home_pmf, away_pmf)
            threshold = int(line)
            over_probs[i] = sum(total_pmf[threshold + 1:])
            under_probs[i] = sum(total_pmf[:threshold + 1])

        return over_probs, under_probs

    def simulate_run_line(self, X: pd.DataFrame, n_sims: int = 10000,
                          spread: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
        """Monte Carlo simulation for run-line probabilities.

        Returns (P(home covers -spread), P(away covers +spread))
        """
        preds = self.predict(X)
        home_cover = np.zeros(len(X))
        away_cover = np.zeros(len(X))

        rng = np.random.default_rng(42)
        for i, (home_mu, away_mu) in enumerate(preds):
            home_sims = rng.poisson(home_mu, n_sims)
            away_sims = rng.poisson(away_mu, n_sims)
            margins = home_sims - away_sims
            home_cover[i] = np.mean(margins > spread)
            away_cover[i] = np.mean(margins < -spread)

        return home_cover, away_cover
