"""Abstract base class for feature groups."""

from abc import ABC, abstractmethod
from datetime import date

from sqlalchemy.orm import Session


class FeatureGroup(ABC):
    """Base class for all feature computation groups.

    Each subclass computes a related set of features for one team in one game.
    """

    name: str
    feature_names: list[str]

    @abstractmethod
    def compute(self, session: Session, game_id: int, team_id: int,
                as_of_date: date) -> dict[str, float | None]:
        """Compute features for a team in a game.

        Returns a dict mapping feature_name -> value. Missing values should be None.
        """
        ...

    def describe(self) -> dict[str, str]:
        """Return feature_name -> description mapping for documentation."""
        return {name: "" for name in self.feature_names}
