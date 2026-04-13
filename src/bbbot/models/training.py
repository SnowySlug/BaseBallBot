"""Training pipeline — builds feature matrix and trains all models."""

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
from sqlalchemy.orm import Session

from bbbot.db.engine import get_session, init_db
from bbbot.db.models import Game
from bbbot.features.builder import build_game_features, create_default_registry

log = structlog.get_logger()

MODEL_DIR = Path("data/models")


def prepare_training_data(session: Session, season: int) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build feature matrix and targets for a season.

    Returns (X, y_win, y_home_runs, y_away_runs)
    """
    registry = create_default_registry()

    # Get all completed games for the season
    games = list(session.query(Game).filter(
        Game.season == season,
        Game.status == "final",
        Game.home_score.isnot(None),
    ).order_by(Game.game_date).all())

    log.info("preparing_training_data", season=season, games=len(games))

    rows = []
    targets_win = []
    targets_home = []
    targets_away = []

    for game in games:
        try:
            features = build_game_features(session, game, registry)
            rows.append(features)
            targets_win.append(1 if game.home_score > game.away_score else 0)
            targets_home.append(game.home_score)
            targets_away.append(game.away_score)
        except Exception as e:
            log.warning("feature_build_error", game_pk=game.mlb_game_pk, error=str(e))
            continue

    X = pd.DataFrame(rows)
    # Drop non-feature columns
    drop_cols = [c for c in X.columns if c in ("game_id",)]
    X = X.drop(columns=drop_cols, errors="ignore")

    # Ensure all columns are numeric (None values create 'object' dtype)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return (
        X,
        pd.Series(targets_win, name="home_win"),
        pd.Series(targets_home, name="home_runs"),
        pd.Series(targets_away, name="away_runs"),
    )


def train_win_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train the win probability ensemble model."""
    from bbbot.models.win_probability import WinProbabilityModel

    model = WinProbabilityModel()
    metrics = model.train(X, y)

    # Save
    model_path = MODEL_DIR / "win_probability" / "latest"
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path / "model.pkl")

    # Save metadata
    import json
    meta = {
        "model_name": "win_probability",
        "features": list(X.columns),
        "n_samples": len(X),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(model_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        importance.to_csv(model_path / "feature_importance.csv")

    log.info("win_model_saved", path=str(model_path))
    return metrics


def train_run_model(X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series) -> dict:
    """Train the run total model."""
    from bbbot.models.run_total import RunTotalModel

    model = RunTotalModel()
    metrics = model.train(X, y_home, y_away)

    model_path = MODEL_DIR / "run_total" / "latest"
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path / "model.pkl")

    import json
    meta = {
        "model_name": "run_total",
        "features": list(X.columns),
        "n_samples": len(X),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(model_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("run_model_saved", path=str(model_path))
    return metrics


def train_all(season: int) -> dict:
    """Train all models for a season. Returns combined metrics."""
    init_db()
    session = get_session()

    try:
        X, y_win, y_home, y_away = prepare_training_data(session, season)
        log.info("training_data_ready", features=X.shape[1], samples=X.shape[0])

        if len(X) < 50:
            log.warning("insufficient_data",
                        msg=f"Only {len(X)} samples. Need 50+ for meaningful training.")
            return {"error": "insufficient_data", "samples": len(X)}

        win_metrics = train_win_model(X, y_win)
        run_metrics = train_run_model(X, y_home, y_away)

        return {"win_model": win_metrics, "run_model": run_metrics}
    finally:
        session.close()


def load_trained_model(model_name: str):
    """Load a trained model from disk."""
    model_path = MODEL_DIR / model_name / "latest" / "model.pkl"
    if not model_path.exists():
        log.warning("model_not_found", model=model_name, path=str(model_path))
        return None
    return joblib.load(model_path)
