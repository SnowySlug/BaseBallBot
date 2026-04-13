"""Daily ingestion orchestrator — fetches schedule, scores, and player data."""

from datetime import date

import structlog
from sqlalchemy.orm import Session

from bbbot.db.engine import get_session, init_db
from bbbot.db.models import Game, Player, Team
from bbbot.db.queries import (
    get_game_by_pk, get_park_by_venue_id, get_team_by_mlb_id,
    upsert_game, upsert_player,
)
from bbbot.db.seed import seed_all
from bbbot.ingest.mlb_stats import (
    MLBStatsClient, extract_pitcher_id, extract_pitcher_name,
    extract_team_id, parse_game_datetime, parse_game_status,
)

log = structlog.get_logger()


class DailyIngestor:
    """Orchestrates daily data ingestion from MLB Stats API."""

    def __init__(self):
        self.client = MLBStatsClient()

    def ensure_db(self):
        """Initialize DB and seed if needed."""
        init_db()
        session = get_session()
        try:
            teams = session.query(Team).count()
            if teams == 0:
                seed_all(session)
            session.commit()
        finally:
            session.close()

    def ingest_schedule(self, game_date: date) -> int:
        """Fetch schedule for a date and store games in DB.

        Returns the number of games ingested.
        """
        self.ensure_db()
        raw_games = self.client.get_schedule(game_date)

        session = get_session()
        count = 0
        try:
            for raw in raw_games:
                game_pk = raw.get("gamePk")
                if not game_pk:
                    continue

                teams_data = raw.get("teams", {})
                home_mlb_id = extract_team_id(teams_data.get("home", {}))
                away_mlb_id = extract_team_id(teams_data.get("away", {}))

                if not home_mlb_id or not away_mlb_id:
                    log.warning("missing_team_ids", game_pk=game_pk)
                    continue

                home_team = get_team_by_mlb_id(session, home_mlb_id)
                away_team = get_team_by_mlb_id(session, away_mlb_id)

                if not home_team or not away_team:
                    log.warning("unknown_team", game_pk=game_pk,
                                home=home_mlb_id, away=away_mlb_id)
                    continue

                # Resolve venue
                venue_data = raw.get("venue", {})
                venue_mlb_id = venue_data.get("id")
                park = get_park_by_venue_id(session, venue_mlb_id) if venue_mlb_id else None

                # Resolve probable pitchers
                pitchers = raw.get("probablePitcher",
                                   raw.get("teams", {}).get("home", {}).get("probablePitcher"))
                # The hydrated schedule puts probablePitchers at teams.home/away level
                home_sp_id = self._resolve_pitcher(
                    session, teams_data.get("home", {}), home_team
                )
                away_sp_id = self._resolve_pitcher(
                    session, teams_data.get("away", {}), away_team
                )

                # Scores (if game is final)
                status = parse_game_status(raw)
                home_score = teams_data.get("home", {}).get("score")
                away_score = teams_data.get("away", {}).get("score")

                # Doubleheader info
                dh = raw.get("doubleHeader", "N")
                is_dh = dh in ("Y", "S")
                dh_num = raw.get("gameNumber", 1)

                game_kwargs = dict(
                    mlb_game_pk=game_pk,
                    game_date=game_date,
                    game_time_utc=parse_game_datetime(raw),
                    status=status,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    venue_id=park.id if park else None,
                    home_sp_id=home_sp_id,
                    away_sp_id=away_sp_id,
                    season=game_date.year,
                    is_doubleheader=is_dh,
                    doubleheader_num=dh_num,
                )

                if status == "final" and home_score is not None and away_score is not None:
                    game_kwargs["home_score"] = home_score
                    game_kwargs["away_score"] = away_score
                    game_kwargs["total_runs"] = home_score + away_score
                    if home_score > away_score:
                        game_kwargs["winning_team_id"] = home_team.id
                    elif away_score > home_score:
                        game_kwargs["winning_team_id"] = away_team.id

                upsert_game(session, **game_kwargs)
                count += 1

            session.commit()
            log.info("ingested_schedule", date=game_date.isoformat(), games=count)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return count

    def ingest_scores(self, game_date: date) -> int:
        """Update scores for games on a given date. Returns count of scored games."""
        self.ensure_db()
        raw_games = self.client.get_schedule(game_date)

        session = get_session()
        scored = 0
        try:
            for raw in raw_games:
                game_pk = raw.get("gamePk")
                status = parse_game_status(raw)
                if status != "final":
                    continue

                game = get_game_by_pk(session, game_pk)
                if not game:
                    continue

                teams_data = raw.get("teams", {})
                home_score = teams_data.get("home", {}).get("score")
                away_score = teams_data.get("away", {}).get("score")

                if home_score is None or away_score is None:
                    continue

                game.status = "final"
                game.home_score = home_score
                game.away_score = away_score
                game.total_runs = home_score + away_score
                if home_score > away_score:
                    game.winning_team_id = game.home_team_id
                elif away_score > home_score:
                    game.winning_team_id = game.away_team_id
                else:
                    game.winning_team_id = None

                scored += 1

            session.commit()
            log.info("ingested_scores", date=game_date.isoformat(), scored=scored)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return scored

    def _resolve_pitcher(self, session: Session, team_side_data: dict,
                         team: Team) -> int | None:
        """Extract and upsert probable pitcher, returning the DB player ID."""
        pitcher_data = team_side_data.get("probablePitcher", {})
        if not pitcher_data:
            return None

        mlb_id = pitcher_data.get("id")
        name = pitcher_data.get("fullName", "Unknown")
        if not mlb_id:
            return None

        player = upsert_player(
            session,
            mlb_id=mlb_id,
            name=name,
            team_id=team.id,
            position="P",
        )
        return player.id
