"""Reusable database query functions."""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from bbbot.db.models import Game, Park, Player, Team


def get_team_by_abbr(session: Session, abbr: str) -> Team | None:
    return session.execute(
        select(Team).where(Team.abbreviation == abbr)
    ).scalar_one_or_none()


def get_team_by_mlb_id(session: Session, mlb_id: int) -> Team | None:
    return session.execute(
        select(Team).where(Team.mlb_id == mlb_id)
    ).scalar_one_or_none()


def get_all_teams(session: Session) -> list[Team]:
    return list(session.execute(select(Team)).scalars().all())


def get_park_by_venue_id(session: Session, venue_id: int) -> Park | None:
    return session.execute(
        select(Park).where(Park.mlb_venue_id == venue_id)
    ).scalar_one_or_none()


def get_player_by_mlb_id(session: Session, mlb_id: int) -> Player | None:
    return session.execute(
        select(Player).where(Player.mlb_id == mlb_id)
    ).scalar_one_or_none()


def upsert_player(session: Session, mlb_id: int, name: str,
                   team_id: int | None = None, position: str | None = None,
                   bats: str | None = None, throws: str | None = None) -> Player:
    player = get_player_by_mlb_id(session, mlb_id)
    if player is None:
        player = Player(mlb_id=mlb_id, name=name, team_id=team_id,
                        position=position, bats=bats, throws=throws)
        session.add(player)
    else:
        player.name = name
        if team_id is not None:
            player.team_id = team_id
        if position is not None:
            player.position = position
        if bats is not None:
            player.bats = bats
        if throws is not None:
            player.throws = throws
    session.flush()
    return player


def get_game_by_pk(session: Session, game_pk: int) -> Game | None:
    return session.execute(
        select(Game).where(Game.mlb_game_pk == game_pk)
    ).scalar_one_or_none()


def get_games_by_date(session: Session, game_date: date) -> list[Game]:
    return list(session.execute(
        select(Game).where(Game.game_date == game_date).order_by(Game.game_time_utc)
    ).scalars().all())


def upsert_game(session: Session, **kwargs) -> Game:
    """Insert or update a game by mlb_game_pk."""
    game_pk = kwargs["mlb_game_pk"]
    game = get_game_by_pk(session, game_pk)
    if game is None:
        game = Game(**kwargs)
        session.add(game)
    else:
        for key, value in kwargs.items():
            if key != "mlb_game_pk" and value is not None:
                setattr(game, key, value)
    session.flush()
    return game
