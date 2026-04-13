"""Feature registry and builder."""

from datetime import date

import structlog
from sqlalchemy.orm import Session

from bbbot.features.base import FeatureGroup

log = structlog.get_logger()


class FeatureRegistry:
    """Manages and orchestrates feature group computation."""

    def __init__(self):
        self._groups: dict[str, FeatureGroup] = {}

    def register(self, group: FeatureGroup) -> None:
        self._groups[group.name] = group

    def compute_all(self, session: Session, game_id: int, team_id: int,
                    as_of_date: date) -> dict[str, float | None]:
        """Compute all registered features for a team in a game."""
        features: dict[str, float | None] = {}
        for name, group in self._groups.items():
            try:
                group_features = group.compute(session, game_id, team_id, as_of_date)
                features.update(group_features)
            except Exception as e:
                log.warning("feature_compute_error", group=name, error=str(e))
                for fname in group.feature_names:
                    features[fname] = None
        return features

    @property
    def all_feature_names(self) -> list[str]:
        names = []
        for group in self._groups.values():
            names.extend(group.feature_names)
        return names
