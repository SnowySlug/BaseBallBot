"""MLB Stats API client (statsapi.mlb.com)."""

from datetime import date, datetime

import structlog

from bbbot.config import get_settings
from bbbot.ingest.base_client import BaseClient

log = structlog.get_logger()


class MLBStatsClient(BaseClient):
    """Client for the MLB Stats API v1."""

    def __init__(self):
        settings = get_settings()
        super().__init__(
            base_url=settings.mlb_api.base_url,
            timeout=settings.mlb_api.timeout,
            max_retries=settings.mlb_api.max_retries,
            rate_limit=5.0,  # 5 requests/sec is safe for MLB API
        )

    def get_schedule(self, game_date: date) -> list[dict]:
        """Fetch the schedule for a given date.

        Returns a list of game dicts with keys:
            gamePk, gameDate, status, teams (home/away with team info),
            venue, probablePitchers, doubleHeader, gameNumber
        """
        data = self._get("schedule", params={
            "date": game_date.isoformat(),
            "sportId": 1,  # MLB
            "hydrate": "probablePitcher,venue,team",
        })

        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                games.append(game)

        log.info("fetched_schedule", date=game_date.isoformat(), games=len(games))
        return games

    def get_boxscore(self, game_pk: int) -> dict:
        """Fetch the boxscore for a completed game."""
        data = self._get(f"game/{game_pk}/boxscore")
        log.debug("fetched_boxscore", game_pk=game_pk)
        return data

    def get_linescore(self, game_pk: int) -> dict:
        """Fetch the linescore (inning-by-inning) for a game."""
        data = self._get(f"game/{game_pk}/linescore")
        log.debug("fetched_linescore", game_pk=game_pk)
        return data

    def get_game_feed(self, game_pk: int) -> dict:
        """Fetch the full live game feed (includes linescore, boxscore, plays)."""
        # The feed endpoint is at the v1.1 path
        data = self._get(f"game/{game_pk}/feed/live")
        log.debug("fetched_game_feed", game_pk=game_pk)
        return data

    def get_roster(self, team_mlb_id: int, season: int | None = None) -> list[dict]:
        """Fetch the 40-man roster for a team."""
        params = {"rosterType": "40Man"}
        if season:
            params["season"] = season
        data = self._get(f"teams/{team_mlb_id}/roster", params=params)
        roster = data.get("roster", [])
        log.debug("fetched_roster", team_id=team_mlb_id, players=len(roster))
        return roster

    def get_player(self, player_mlb_id: int) -> dict:
        """Fetch player biographical info."""
        data = self._get(f"people/{player_mlb_id}")
        people = data.get("people", [])
        if not people:
            return {}
        return people[0]

    def get_standings(self, season: int, league_id: int | None = None) -> dict:
        """Fetch current standings."""
        params = {"season": season, "leagueId": league_id or "103,104"}
        data = self._get("standings", params=params)
        return data


def parse_game_datetime(game_data: dict) -> datetime | None:
    """Parse the gameDate field from the MLB API into a UTC datetime."""
    raw = game_data.get("gameDate")
    if not raw:
        return None
    # Format: "2026-04-13T18:10:00Z"
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def parse_game_status(game_data: dict) -> str:
    """Map MLB API status codes to our GameStatus values."""
    status_code = game_data.get("status", {}).get("statusCode", "")
    mapping = {
        "S": "scheduled",
        "P": "pregame",
        "PW": "pregame",
        "I": "live",
        "F": "final",
        "O": "final",  # "Game Over"
        "D": "postponed",
        "DR": "postponed",
        "U": "suspended",
    }
    return mapping.get(status_code, "scheduled")


def extract_team_id(team_data: dict) -> int | None:
    """Extract the MLB team ID from a team node in the schedule."""
    team = team_data.get("team", {})
    return team.get("id")


def extract_pitcher_id(pitchers_data: dict, side: str) -> int | None:
    """Extract probable pitcher MLB ID from probablePitchers node."""
    pitcher = pitchers_data.get(side, {})
    return pitcher.get("id")


def extract_pitcher_name(pitchers_data: dict, side: str) -> str | None:
    """Extract probable pitcher name."""
    pitcher = pitchers_data.get(side, {})
    return pitcher.get("fullName")
