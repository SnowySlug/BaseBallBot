"""Box score ingestion — populates batting and pitching game logs."""

from datetime import date

import structlog
from sqlalchemy.orm import Session

from bbbot.db.engine import get_session, init_db
from bbbot.db.models import Game, PitcherGameLog, TeamBattingDaily
from bbbot.db.queries import get_games_by_date, upsert_player
from bbbot.ingest.mlb_stats import MLBStatsClient

log = structlog.get_logger()


class BoxScoreIngestor:
    """Ingests box score data for completed games."""

    def __init__(self):
        self.client = MLBStatsClient()

    def ingest_boxscores(self, game_date: date) -> int:
        """Fetch and store box scores for all final games on a date."""
        init_db()
        session = get_session()
        count = 0

        try:
            games = get_games_by_date(session, game_date)
            final_games = [g for g in games if g.status == "final"]

            for game in final_games:
                try:
                    self._ingest_game_boxscore(session, game)
                    count += 1
                except Exception as e:
                    log.warning("boxscore_error", game_pk=game.mlb_game_pk, error=str(e))

            session.commit()
            log.info("ingested_boxscores", date=game_date.isoformat(), count=count)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return count

    def _ingest_game_boxscore(self, session: Session, game: Game) -> None:
        """Ingest a single game's box score."""
        # Check if already ingested
        existing = session.query(TeamBattingDaily).filter_by(
            team_id=game.home_team_id, game_date=game.game_date
        ).first()
        if existing:
            return

        box = self.client.get_boxscore(game.mlb_game_pk)
        teams_data = box.get("teams", {})

        for side in ("home", "away"):
            team_id = game.home_team_id if side == "home" else game.away_team_id
            team_box = teams_data.get(side, {})

            # Team batting stats
            batting = team_box.get("teamStats", {}).get("batting", {})
            if batting:
                tbd = TeamBattingDaily(
                    team_id=team_id,
                    game_date=game.game_date,
                    runs=_int(batting.get("runs")),
                    hits=_int(batting.get("hits")),
                    doubles=_int(batting.get("doubles")),
                    triples=_int(batting.get("triples")),
                    home_runs=_int(batting.get("homeRuns")),
                    rbi=_int(batting.get("rbi")),
                    walks=_int(batting.get("baseOnBalls")),
                    strikeouts=_int(batting.get("strikeOuts")),
                    stolen_bases=_int(batting.get("stolenBases")),
                    at_bats=_int(batting.get("atBats")),
                    left_on_base=_int(batting.get("leftOnBase")),
                )
                session.add(tbd)

            # Pitcher game logs
            pitchers = team_box.get("pitchers", [])
            players = team_box.get("players", {})

            for idx, pitcher_id in enumerate(pitchers):
                player_key = f"ID{pitcher_id}"
                player_data = players.get(player_key, {})
                pitching_stats = player_data.get("stats", {}).get("pitching", {})

                if not pitching_stats:
                    continue

                person = player_data.get("person", {})
                name = person.get("fullName", "Unknown")

                # Upsert the player
                player = upsert_player(
                    session, mlb_id=pitcher_id, name=name,
                    team_id=team_id, position="P",
                )

                # Check for existing log
                existing_log = session.query(PitcherGameLog).filter_by(
                    player_id=player.id, game_id=game.id
                ).first()
                if existing_log:
                    continue

                ip_str = pitching_stats.get("inningsPitched", "0")
                ip = _parse_ip(ip_str)

                pgl = PitcherGameLog(
                    player_id=player.id,
                    game_id=game.id,
                    game_date=game.game_date,
                    team_id=team_id,
                    is_starter=(idx == 0),
                    innings_pitched=ip,
                    hits_allowed=_int(pitching_stats.get("hits")),
                    runs_allowed=_int(pitching_stats.get("runs")),
                    earned_runs=_int(pitching_stats.get("earnedRuns")),
                    walks=_int(pitching_stats.get("baseOnBalls")),
                    strikeouts=_int(pitching_stats.get("strikeOuts")),
                    home_runs_allowed=_int(pitching_stats.get("homeRuns")),
                    pitches_thrown=_int(pitching_stats.get("pitchesThrown")),
                    win=pitching_stats.get("wins", 0) > 0 if "wins" in pitching_stats else None,
                    loss=pitching_stats.get("losses", 0) > 0 if "losses" in pitching_stats else None,
                    save=pitching_stats.get("saves", 0) > 0 if "saves" in pitching_stats else None,
                )
                session.add(pgl)

        session.flush()


def _int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_ip(ip_str: str) -> float:
    """Parse innings pitched string like '6.1' -> 6.333."""
    try:
        parts = str(ip_str).split(".")
        whole = int(parts[0])
        if len(parts) > 1:
            thirds = int(parts[1])
            return whole + thirds / 3.0
        return float(whole)
    except (ValueError, TypeError):
        return 0.0
