"""Situational and environmental features."""

from datetime import date

from sqlalchemy.orm import Session

from bbbot.db.models import Game, Park, Weather
from bbbot.features.base import FeatureGroup


class SituationalFeatures(FeatureGroup):
    name = "situational"

    @property
    def feature_names(self) -> list[str]:
        return [
            "park_factor_r", "park_factor_hr",
            "elevation_ft", "is_dome",
            "temperature_f", "wind_speed_mph",
            "wind_out", "wind_in",
            "humidity_pct", "precipitation_pct",
            "is_day_game", "is_interleague",
            "is_doubleheader",
        ]

    def compute(self, session: Session, game_id: int, team_id: int,
                as_of_date: date) -> dict[str, float | None]:
        features: dict[str, float | None] = {name: None for name in self.feature_names}

        game = session.get(Game, game_id)
        if not game:
            return features

        # Park factors
        if game.venue:
            park = game.venue
            features["park_factor_r"] = park.park_factor_r or 1.0
            features["park_factor_hr"] = park.park_factor_hr or 1.0
            features["elevation_ft"] = float(park.elevation_ft or 0)
            features["is_dome"] = 1.0 if park.roof_type == "dome" else 0.0

        # Weather
        weather = session.query(Weather).filter_by(game_id=game_id).first()
        if weather:
            features["temperature_f"] = weather.temperature_f
            features["wind_speed_mph"] = weather.wind_speed_mph
            features["humidity_pct"] = weather.humidity_pct
            features["precipitation_pct"] = weather.precipitation_pct

            wd = (weather.wind_direction or "").lower()
            features["wind_out"] = 1.0 if "out" in wd else 0.0
            features["wind_in"] = 1.0 if "in" in wd else 0.0
        elif features.get("is_dome") == 1.0:
            # Dome: neutral weather
            features["temperature_f"] = 72.0
            features["wind_speed_mph"] = 0.0
            features["humidity_pct"] = 50.0
            features["precipitation_pct"] = 0.0
            features["wind_out"] = 0.0
            features["wind_in"] = 0.0

        # Game context
        if game.game_time_utc:
            hour_utc = game.game_time_utc.hour
            features["is_day_game"] = 1.0 if hour_utc < 22 else 0.0  # rough proxy
        features["is_doubleheader"] = 1.0 if game.is_doubleheader else 0.0

        # Interleague
        if game.home_team and game.away_team:
            features["is_interleague"] = (
                1.0 if game.home_team.league != game.away_team.league else 0.0
            )

        return features
