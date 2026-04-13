"""Seed the database with MLB teams and parks."""

import structlog
from sqlalchemy.orm import Session

from bbbot.constants import MLB_PARKS, MLB_TEAMS
from bbbot.db.models import Park, Team

log = structlog.get_logger()


def seed_teams(session: Session) -> dict[str, Team]:
    """Insert or update all 30 MLB teams. Returns abbr -> Team mapping."""
    teams: dict[str, Team] = {}
    for abbr, (name, league, division, mlb_id) in MLB_TEAMS.items():
        team = session.query(Team).filter_by(mlb_id=mlb_id).first()
        if team is None:
            team = Team(
                mlb_id=mlb_id,
                abbreviation=abbr,
                name=name,
                league=league,
                division=division,
            )
            session.add(team)
            log.debug("seeded_team", team=abbr)
        else:
            team.abbreviation = abbr
            team.name = name
            team.league = league
            team.division = division
        teams[abbr] = team
    session.flush()
    return teams


def seed_parks(session: Session, teams: dict[str, Team]) -> None:
    """Insert or update all 30 MLB parks."""
    for abbr, (name, venue_id, roof, elevation) in MLB_PARKS.items():
        park = session.query(Park).filter_by(mlb_venue_id=venue_id).first()
        team = teams.get(abbr)
        if park is None:
            park = Park(
                mlb_venue_id=venue_id,
                name=name,
                team_id=team.id if team else None,
                roof_type=roof,
                elevation_ft=elevation,
            )
            session.add(park)
            log.debug("seeded_park", park=name)
        else:
            park.name = name
            park.roof_type = roof
            park.elevation_ft = elevation
            if team:
                park.team_id = team.id
    session.flush()


def seed_all(session: Session) -> None:
    """Run all seed operations."""
    teams = seed_teams(session)
    seed_parks(session, teams)
    session.commit()
    log.info("database_seeded", teams=len(teams), parks=len(MLB_PARKS))
