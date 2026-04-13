"""Odds ingestion — fetches and stores live odds from The Odds API."""

from datetime import datetime

import structlog
from sqlalchemy.orm import Session

from bbbot.db.engine import get_session, init_db
from bbbot.db.models import Game, OddsSnapshot, Team
from bbbot.ingest.odds import OddsAPIClient

log = structlog.get_logger()

# Mapping from The Odds API team names to our abbreviations
ODDS_TEAM_MAP = {
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Athletics": "OAK",
    "Sacramento Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def ingest_odds() -> int:
    """Fetch current odds and store snapshots for matching games.

    Returns the number of odds snapshots stored.
    """
    init_db()
    client = OddsAPIClient()
    if not client.configured:
        log.warning("odds_api_not_configured")
        return 0

    odds_data = client.get_mlb_odds()
    if not odds_data:
        return 0

    session = get_session()
    count = 0

    try:
        # Build team lookup
        teams = {t.abbreviation: t for t in session.query(Team).all()}

        for game_odds in odds_data:
            home_name = game_odds.get("home_team", "")
            away_name = game_odds.get("away_team", "")

            home_abbr = ODDS_TEAM_MAP.get(home_name)
            away_abbr = ODDS_TEAM_MAP.get(away_name)

            if not home_abbr or not away_abbr:
                log.warning("unknown_odds_team", home=home_name, away=away_name)
                continue

            home_team = teams.get(home_abbr)
            away_team = teams.get(away_abbr)
            if not home_team or not away_team:
                continue

            # Find matching game in DB (today's games)
            game = session.query(Game).filter(
                Game.home_team_id == home_team.id,
                Game.away_team_id == away_team.id,
                Game.status.in_(["scheduled", "pregame", "live"]),
            ).order_by(Game.game_date.desc()).first()

            if not game:
                continue

            # Parse and store odds snapshots
            snapshots = client.parse_odds_for_game(game_odds)
            for snap in snapshots:
                odds_snap = OddsSnapshot(
                    game_id=game.id,
                    **snap,
                )
                session.add(odds_snap)
                count += 1

        session.commit()
        log.info("ingested_odds", snapshots=count)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return count


def get_best_odds_for_game(session: Session, game_id: int) -> dict:
    """Get the best available odds across all books for a game.

    Returns dict with keys: home_ml, away_ml, total_line, over_odds, under_odds,
    and the sportsbook offering each.
    """
    result = {
        "home_ml": None, "home_ml_book": None,
        "away_ml": None, "away_ml_book": None,
        "total_line": None,
        "over_odds": None, "over_book": None,
        "under_odds": None, "under_book": None,
    }

    # Get most recent h2h odds
    h2h_snaps = session.query(OddsSnapshot).filter(
        OddsSnapshot.game_id == game_id,
        OddsSnapshot.market_type == "h2h",
    ).order_by(OddsSnapshot.captured_at.desc()).all()

    best_home = None
    best_away = None
    for snap in h2h_snaps:
        if snap.home_line is not None:
            if best_home is None or snap.home_line > best_home:
                best_home = snap.home_line
                result["home_ml"] = snap.home_line
                result["home_ml_book"] = snap.sportsbook
        if snap.away_line is not None:
            if best_away is None or snap.away_line > best_away:
                best_away = snap.away_line
                result["away_ml"] = snap.away_line
                result["away_ml_book"] = snap.sportsbook

    # Get most recent totals
    total_snaps = session.query(OddsSnapshot).filter(
        OddsSnapshot.game_id == game_id,
        OddsSnapshot.market_type == "totals",
    ).order_by(OddsSnapshot.captured_at.desc()).all()

    for snap in total_snaps:
        if snap.total_line is not None and result["total_line"] is None:
            result["total_line"] = snap.total_line
        if snap.over_odds is not None:
            if result["over_odds"] is None or snap.over_odds > result["over_odds"]:
                result["over_odds"] = snap.over_odds
                result["over_book"] = snap.sportsbook
        if snap.under_odds is not None:
            if result["under_odds"] is None or snap.under_odds > result["under_odds"]:
                result["under_odds"] = snap.under_odds
                result["under_book"] = snap.sportsbook

    return result
