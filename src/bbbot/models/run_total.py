"""Run total model — Poisson-based regression for predicting runs scored."""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from scipy.stats import poisson
from sklearn.model_selection import TimeSeriesSplit

from bbbot.models.base import BaseModel

log = structlog.get_logger()

# Keep top N features by importance from initial training pass
TOP_N_FEATURES = 40


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

    def _select_features(self, X: pd.DataFrame, y_home: pd.Series,
                         y_away: pd.Series) -> list[str]:
        """Train quick LGBMs to rank features, return union of top N for both targets."""
        from lightgbm import LGBMRegressor

        home_sel = LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbose=-1,
        )
        away_sel = LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbose=-1,
        )
        home_sel.fit(X, y_home)
        away_sel.fit(X, y_away)

        home_imp = pd.Series(home_sel.feature_importances_, index=X.columns)
        away_imp = pd.Series(away_sel.feature_importances_, index=X.columns)

        # Combined importance: average of normalized importances
        combined = (home_imp / home_imp.sum() + away_imp / away_imp.sum()) / 2
        top = combined.nlargest(TOP_N_FEATURES).index.tolist()
        log.info("feature_selection", total=len(X.columns), selected=len(top))
        return top

    def _tune_hyperparameters(self, X: pd.DataFrame, y_home: pd.Series,
                              y_away: pd.Series, n_trials: int = 50) -> dict:
        """Use Optuna to find optimal hyperparameters via time-series CV."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

            from lightgbm import LGBMRegressor

            tscv = TimeSeriesSplit(n_splits=5)
            fold_maes = []

            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                yh_tr, yh_val = y_home.iloc[train_idx], y_home.iloc[val_idx]
                ya_tr, ya_val = y_away.iloc[train_idx], y_away.iloc[val_idx]

                home_m = LGBMRegressor(**params, random_state=42, verbose=-1)
                away_m = LGBMRegressor(**params, random_state=42, verbose=-1)

                home_m.fit(X_tr, yh_tr)
                away_m.fit(X_tr, ya_tr)

                h_pred = home_m.predict(X_val)
                a_pred = away_m.predict(X_val)

                total_mae = np.mean(np.abs(
                    (h_pred + a_pred) - (yh_val.values + ya_val.values)
                ))
                fold_maes.append(total_mae)

            return np.mean(fold_maes)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        log.info("tuning_complete",
                 best_total_mae=f"{study.best_value:.4f}",
                 best_params=study.best_params)
        return study.best_params

    def train(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series,
              tune: bool = False, n_trials: int = 50, **kwargs) -> dict:
        """Train separate models for home and away run prediction.

        Args:
            tune: If True, run Optuna hyperparameter search first.
            n_trials: Number of Optuna trials (default 50).
        """
        from lightgbm import LGBMRegressor

        self.feature_names = list(X.columns)

        # Impute with medians instead of filling zeros
        X_clean = self._clean(X, fit=True)

        # Feature selection: quick initial pass to find the top features
        self.selected_features = self._select_features(X_clean, y_home, y_away)
        X_clean = X_clean[self.selected_features]
        log.info("training_on_selected_features", n_features=len(self.selected_features))

        # Hyperparameter tuning
        if tune:
            log.info("starting_hyperparameter_tuning", n_trials=n_trials)
            best_params = self._tune_hyperparameters(
                X_clean, y_home, y_away, n_trials=n_trials,
            )
        else:
            best_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 10,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            }

        self.home_model = LGBMRegressor(
            **best_params, random_state=42, verbose=-1,
        )
        self.away_model = LGBMRegressor(
            **best_params, random_state=42, verbose=-1,
        )

        # Time-series cross-validation with OOF predictions
        tscv = TimeSeriesSplit(n_splits=5)
        oof_home = np.full(len(X_clean), np.nan)
        oof_away = np.full(len(X_clean), np.nan)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            yh_tr, yh_val = y_home.iloc[train_idx], y_home.iloc[val_idx]
            ya_tr, ya_val = y_away.iloc[train_idx], y_away.iloc[val_idx]

            self.home_model.fit(X_tr, yh_tr)
            self.away_model.fit(X_tr, ya_tr)

            oof_home[val_idx] = self.home_model.predict(X_val)
            oof_away[val_idx] = self.away_model.predict(X_val)

        # Final fit on all data
        self.home_model.fit(X_clean, y_home)
        self.away_model.fit(X_clean, y_away)
        self._trained = True

        # Metrics from OOF predictions (honest estimate)
        mask = ~(np.isnan(oof_home) | np.isnan(oof_away))
        metrics = {
            "home_mae": np.mean(np.abs(oof_home[mask] - y_home.values[mask])),
            "away_mae": np.mean(np.abs(oof_away[mask] - y_away.values[mask])),
            "total_mae": np.mean(np.abs(
                (oof_home[mask] + oof_away[mask])
                - (y_home.values[mask] + y_away.values[mask])
            )),
        }
        log.info("run_total_training_complete", **metrics)
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted [home_runs, away_runs] for each game."""
        if not self._trained:
            from bbbot.models.baseline import BaselineRunsModel
            return BaselineRunsModel().predict(X)

        X_clean = self._clean(X, fit=False)

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
