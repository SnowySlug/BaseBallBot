"""Feature builder — assembles all feature groups into a game-level feature matrix."""

from datetime import date

import pandas as pd
import structlog
from sqlalchemy.orm import Session

from bbbot.db.models import Game
from bbbot.features.bullpen import BullpenFeatures
from bbbot.features.registry import FeatureRegistry
from bbbot.features.situational import SituationalFeatures
from bbbot.features.starting_pitching import StartingPitchingFeatures
from bbbot.features.team_batting import TeamBattingFeatures

log = structlog.get_logger()


def create_default_registry() -> FeatureRegistry:
    """Create a registry with all standard feature groups."""
    registry = FeatureRegistry()
    registry.register(TeamBattingFeatures())
    registry.register(StartingPitchingFeatures())
    registry.register(BullpenFeatures())
    registry.register(SituationalFeatures())
    return registry


def build_game_features(session: Session, game: Game,
                        registry: FeatureRegistry | None = None) -> dict[str, float | None]:
    """Build the full feature vector for a single game.

    Returns a flat dict with features prefixed by 'home_' or 'away_' for team-specific
    features, unprefixed for game-level (situational) features, and 'diff_' for
    home-minus-away differentials.
    """
    if registry is None:
        registry = create_default_registry()

    as_of = game.game_date
    features: dict[str, float | None] = {"game_id": game.id}

    SITUATIONAL_PREFIXES = ("park_", "elevation", "is_dome", "temperature", "wind_",
                            "humidity", "precipitation", "is_day", "is_inter", "is_double")

    # Home team features
    home_feats = registry.compute_all(session, game.id, game.home_team_id, as_of)
    home_team_keys = []
    for key, val in home_feats.items():
        # Situational features are game-level, don't prefix
        if key.startswith(SITUATIONAL_PREFIXES):
            features[key] = val
        else:
            features[f"home_{key}"] = val
            home_team_keys.append(key)

    # Away team features
    away_feats = registry.compute_all(session, game.id, game.away_team_id, as_of)
    for key, val in away_feats.items():
        if not key.startswith(SITUATIONAL_PREFIXES):
            features[f"away_{key}"] = val

    # Differential features: home - away for every shared team-level feature
    for key in home_team_keys:
        home_val = home_feats.get(key)
        away_val = away_feats.get(key)
        if home_val is not None and away_val is not None:
            features[f"diff_{key}"] = home_val - away_val
        else:
            features[f"diff_{key}"] = None

    return features


def build_feature_matrix(session: Session, games: list[Game],
                         registry: FeatureRegistry | None = None) -> pd.DataFrame:
    """Build features for multiple games, returning a DataFrame."""
    if registry is None:
        registry = create_default_registry()

    rows = []
    for game in games:
        row = build_game_features(session, game, registry)
        rows.append(row)

    return pd.DataFrame(rows)
