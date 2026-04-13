"""The Odds API client for fetching sportsbook odds."""

from datetime import datetime

import structlog

from bbbot.config import get_settings
from bbbot.ingest.base_client import BaseClient

log = structlog.get_logger()


class OddsAPIClient(BaseClient):
    """Client for The Odds API (https://the-odds-api.com/)."""

    def __init__(self):
        settings = get_settings()
        super().__init__(
            base_url=settings.odds_api.base_url,
            timeout=30,
            rate_limit=1.0,  # conservative rate limit
        )
        self.api_key = settings.odds_api.api_key
        self.regions = settings.odds_api.regions
        self.markets = settings.odds_api.markets

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # Bookmakers to explicitly request (Kalshi isn't in default US region)
    BOOKMAKERS = "kalshi,draftkings,fanduel,betmgm,betrivers,bovada,betonlineag,lowvig,mybookieag,betus"

    def get_mlb_odds(self) -> list[dict]:
        """Fetch current MLB odds from all configured sportsbooks.

        Returns a list of game dicts, each containing:
            id, sport_key, commence_time, home_team, away_team, bookmakers[]
        """
        if not self.configured:
            log.warning("odds_api_not_configured",
                        msg="Set ODDS_API__API_KEY in .env to fetch real odds")
            return []

        data = self._get("sports/baseball_mlb/odds", params={
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": "american",
            "bookmakers": self.BOOKMAKERS,
        })

        if isinstance(data, list):
            log.info("fetched_odds", games=len(data))
            return data
        return []

    def parse_odds_for_game(self, odds_data: dict) -> list[dict]:
        """Parse a single game's odds into a flat list of snapshots.

        Returns list of dicts suitable for creating OddsSnapshot records.
        """
        snapshots = []
        now = datetime.utcnow()

        for bookmaker in odds_data.get("bookmakers", []):
            book_name = bookmaker.get("key", "unknown")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if market_key == "h2h" and len(outcomes) >= 2:
                    home_odds = None
                    away_odds = None
                    for outcome in outcomes:
                        if outcome.get("name") == odds_data.get("home_team"):
                            home_odds = outcome.get("price")
                        else:
                            away_odds = outcome.get("price")

                    snapshots.append({
                        "captured_at": now,
                        "sportsbook": book_name,
                        "market_type": "h2h",
                        "home_line": home_odds,
                        "away_line": away_odds,
                        "total_line": None,
                        "over_odds": None,
                        "under_odds": None,
                    })

                elif market_key == "totals" and len(outcomes) >= 2:
                    total_line = None
                    over_odds = None
                    under_odds = None
                    for outcome in outcomes:
                        if outcome.get("name") == "Over":
                            total_line = outcome.get("point")
                            over_odds = outcome.get("price")
                        elif outcome.get("name") == "Under":
                            under_odds = outcome.get("price")

                    snapshots.append({
                        "captured_at": now,
                        "sportsbook": book_name,
                        "market_type": "totals",
                        "home_line": None,
                        "away_line": None,
                        "total_line": total_line,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                    })

                elif market_key == "spreads" and len(outcomes) >= 2:
                    home_spread = None
                    home_spread_odds = None
                    away_spread_odds = None
                    for outcome in outcomes:
                        if outcome.get("name") == odds_data.get("home_team"):
                            home_spread = outcome.get("point")
                            home_spread_odds = outcome.get("price")
                        else:
                            away_spread_odds = outcome.get("price")

                    snapshots.append({
                        "captured_at": now,
                        "sportsbook": book_name,
                        "market_type": "spreads",
                        "home_line": home_spread_odds,
                        "away_line": away_spread_odds,
                        "total_line": home_spread,
                        "over_odds": None,
                        "under_odds": None,
                    })

        return snapshots
