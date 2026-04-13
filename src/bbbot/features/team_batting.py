"""Team batting feature group — rolling offensive metrics."""

from datetime import date, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from bbbot.db.models import Game, TeamBattingDaily
from bbbot.features.base import FeatureGroup


class TeamBattingFeatures(FeatureGroup):
    name = "team_batting"

    WINDOWS = [7, 14, 30]
    METRICS = ["runs_per_game", "hits_per_game", "hr_per_game", "bb_per_game",
               "k_per_game", "sb_per_game"]

    @property
    def feature_names(self) -> list[str]:
        names = []
        for window in self.WINDOWS:
            for metric in self.METRICS:
                names.append(f"bat_{metric}_{window}d")
        # Season-long averages
        for metric in self.METRICS:
            names.append(f"bat_{metric}_season")
        # Recent trend (7d vs 30d ratio)
        names.append("bat_trend_runs")
        return names

    def compute(self, session: Session, game_id: int, team_id: int,
                as_of_date: date) -> dict[str, float | None]:
        features: dict[str, float | None] = {}

        for window in self.WINDOWS:
            start = as_of_date - timedelta(days=window)
            stats = self._get_rolling_stats(session, team_id, start, as_of_date)
            suffix = f"_{window}d"
            if stats["games"] > 0:
                g = stats["games"]
                features[f"bat_runs_per_game{suffix}"] = stats["runs"] / g
                features[f"bat_hits_per_game{suffix}"] = stats["hits"] / g
                features[f"bat_hr_per_game{suffix}"] = stats["home_runs"] / g
                features[f"bat_bb_per_game{suffix}"] = stats["walks"] / g
                features[f"bat_k_per_game{suffix}"] = stats["strikeouts"] / g
                features[f"bat_sb_per_game{suffix}"] = stats["stolen_bases"] / g
            else:
                for metric in self.METRICS:
                    features[f"bat_{metric}{suffix}"] = None

        # Season stats (use year start as window)
        season_start = date(as_of_date.year, 3, 1)
        season_stats = self._get_rolling_stats(session, team_id, season_start, as_of_date)
        if season_stats["games"] > 0:
            g = season_stats["games"]
            features["bat_runs_per_game_season"] = season_stats["runs"] / g
            features["bat_hits_per_game_season"] = season_stats["hits"] / g
            features["bat_hr_per_game_season"] = season_stats["home_runs"] / g
            features["bat_bb_per_game_season"] = season_stats["walks"] / g
            features["bat_k_per_game_season"] = season_stats["strikeouts"] / g
            features["bat_sb_per_game_season"] = season_stats["stolen_bases"] / g
        else:
            for metric in self.METRICS:
                features[f"bat_{metric}_season"] = None

        # Trend: 7d / 30d ratio (>1 means trending up)
        r7 = features.get("bat_runs_per_game_7d")
        r30 = features.get("bat_runs_per_game_30d")
        if r7 is not None and r30 is not None and r30 > 0:
            features["bat_trend_runs"] = r7 / r30
        else:
            features["bat_trend_runs"] = None

        return features

    def _get_rolling_stats(self, session: Session, team_id: int,
                           start: date, end: date) -> dict:
        """Aggregate team batting stats over a date range."""
        row = session.execute(
            select(
                func.count().label("games"),
                func.coalesce(func.sum(TeamBattingDaily.runs), 0).label("runs"),
                func.coalesce(func.sum(TeamBattingDaily.hits), 0).label("hits"),
                func.coalesce(func.sum(TeamBattingDaily.home_runs), 0).label("home_runs"),
                func.coalesce(func.sum(TeamBattingDaily.walks), 0).label("walks"),
                func.coalesce(func.sum(TeamBattingDaily.strikeouts), 0).label("strikeouts"),
                func.coalesce(func.sum(TeamBattingDaily.stolen_bases), 0).label("stolen_bases"),
            ).where(and_(
                TeamBattingDaily.team_id == team_id,
                TeamBattingDaily.game_date >= start,
                TeamBattingDaily.game_date < end,
            ))
        ).one()

        return {
            "games": row.games,
            "runs": row.runs,
            "hits": row.hits,
            "home_runs": row.home_runs,
            "walks": row.walks,
            "strikeouts": row.strikeouts,
            "stolen_bases": row.stolen_bases,
        }
