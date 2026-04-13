"""Baseline prediction model using historical base rates and simple heuristics.

This model doesn't require training — it uses MLB historical averages
and adjusts based on available features (park factors, SP quality, etc.).
It serves as a starting point until enough data is collected to train
the full XGBoost/LightGBM ensemble.
"""

import numpy as np
import pandas as pd

from bbbot.models.base import BaseModel


# MLB historical averages (2019-2025)
MLB_AVG_RUNS_PER_TEAM = 4.5
MLB_HOME_WIN_PCT = 0.538
MLB_AVG_ERA = 4.08
MLB_AVG_K_PER_9 = 8.6


class BaselineWinModel(BaseModel):
    """Heuristic win probability model using available features."""

    name = "baseline_win"

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        pass  # No training needed

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (1 = home win)."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate home win probability using available features."""
        probs = np.full(len(X), MLB_HOME_WIN_PCT)

        for i, (_, row) in enumerate(X.iterrows()):
            adj = 0.0

            # Adjust for starting pitcher quality (ERA-based)
            home_era = row.get("home_sp_era")
            away_era = row.get("away_sp_era")
            if home_era is not None and away_era is not None and home_era > 0 and away_era > 0:
                # Better ERA = lower = advantage. Scale: 1 run of ERA ~ 3% win prob
                era_diff = away_era - home_era  # positive = home SP is better
                adj += np.clip(era_diff * 0.03, -0.15, 0.15)

            # Adjust for xFIP if available (more predictive than ERA)
            home_xfip = row.get("home_sp_xfip")
            away_xfip = row.get("away_sp_xfip")
            if home_xfip is not None and away_xfip is not None and home_xfip > 0 and away_xfip > 0:
                xfip_diff = away_xfip - home_xfip
                adj += np.clip(xfip_diff * 0.02, -0.10, 0.10)

            # Adjust for recent team batting (runs per game)
            home_rpg = row.get("home_bat_runs_per_game_14d")
            away_rpg = row.get("away_bat_runs_per_game_14d")
            if home_rpg is not None and away_rpg is not None:
                rpg_diff = home_rpg - away_rpg
                adj += np.clip(rpg_diff * 0.02, -0.10, 0.10)

            # Park factor adjustment
            pf = row.get("park_factor_r")
            if pf is not None and pf != 1.0:
                # High park factor slightly favors the home team (more variance)
                adj += (pf - 1.0) * 0.02

            # Elevation bonus (Coors effect)
            elevation = row.get("elevation_ft")
            if elevation is not None and elevation > 4000:
                adj += 0.02  # Slight home advantage at elevation

            probs[i] = np.clip(MLB_HOME_WIN_PCT + adj, 0.25, 0.75)

        return probs


class BaselineRunsModel(BaseModel):
    """Heuristic run total prediction model."""

    name = "baseline_runs"

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted [home_runs, away_runs] for each game."""
        predictions = np.full((len(X), 2), MLB_AVG_RUNS_PER_TEAM)

        for i, (_, row) in enumerate(X.iterrows()):
            home_runs = MLB_AVG_RUNS_PER_TEAM
            away_runs = MLB_AVG_RUNS_PER_TEAM

            # Adjust for opposing SP quality
            away_sp_era = row.get("away_sp_era")
            if away_sp_era is not None and away_sp_era > 0:
                # Home team faces away SP
                home_runs += (away_sp_era - MLB_AVG_ERA) * 0.25

            home_sp_era = row.get("home_sp_era")
            if home_sp_era is not None and home_sp_era > 0:
                # Away team faces home SP
                away_runs += (home_sp_era - MLB_AVG_ERA) * 0.25

            # Adjust for team batting (recent form)
            home_rpg = row.get("home_bat_runs_per_game_14d")
            if home_rpg is not None:
                home_runs = 0.6 * home_runs + 0.4 * home_rpg

            away_rpg = row.get("away_bat_runs_per_game_14d")
            if away_rpg is not None:
                away_runs = 0.6 * away_runs + 0.4 * away_rpg

            # Park factor
            pf = row.get("park_factor_r")
            if pf is not None:
                home_runs *= pf
                away_runs *= pf

            # Elevation (Coors)
            elevation = row.get("elevation_ft")
            if elevation is not None and elevation > 4000:
                home_runs *= 1.15
                away_runs *= 1.10

            predictions[i] = [
                np.clip(home_runs, 1.0, 15.0),
                np.clip(away_runs, 1.0, 15.0),
            ]

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)

    def predict_total_probs(self, X: pd.DataFrame, line: float = 8.5
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Predict over/under probabilities for a given total line.

        Uses a Poisson approximation from predicted means.
        """
        from scipy.stats import poisson

        preds = self.predict(X)
        over_probs = np.zeros(len(X))
        under_probs = np.zeros(len(X))

        for i, (home_r, away_r) in enumerate(preds):
            # Simulate Poisson draws
            total_mean = home_r + away_r
            # P(total > line) = 1 - P(total <= floor(line))
            # Using convolution of two Poissons
            max_runs = 25
            home_pmf = poisson.pmf(range(max_runs + 1), home_r)
            away_pmf = poisson.pmf(range(max_runs + 1), away_r)

            # Convolve to get total distribution
            total_pmf = np.convolve(home_pmf, away_pmf)

            # P(over) = P(total > line)
            threshold = int(line)  # e.g., 8 for 8.5 line
            over_probs[i] = sum(total_pmf[threshold + 1:])
            under_probs[i] = sum(total_pmf[:threshold + 1])

        return over_probs, under_probs
