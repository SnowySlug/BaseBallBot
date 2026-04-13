"""SQLAlchemy ORM models for all database tables."""

from datetime import date, datetime

from sqlalchemy import (
    Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    abbreviation: Mapped[str] = mapped_column(String(3), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    league: Mapped[str] = mapped_column(String(2), nullable=False)
    division: Mapped[str] = mapped_column(String(12), nullable=False)


class Park(Base):
    __tablename__ = "parks"

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_venue_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    roof_type: Mapped[str | None] = mapped_column(String(16))
    elevation_ft: Mapped[int | None] = mapped_column(Integer)
    park_factor_r: Mapped[float | None] = mapped_column(Float)
    park_factor_hr: Mapped[float | None] = mapped_column(Float)
    park_factor_lhb: Mapped[float | None] = mapped_column(Float)
    park_factor_rhb: Mapped[float | None] = mapped_column(Float)


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    position: Mapped[str | None] = mapped_column(String(8))
    bats: Mapped[str | None] = mapped_column(String(1))
    throws: Mapped[str | None] = mapped_column(String(1))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (
        Index("ix_games_date", "game_date"),
        Index("ix_games_season_home", "season", "home_team_id"),
        Index("ix_games_season_away", "season", "away_team_id"),
        Index("ix_games_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_game_pk: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    game_time_utc: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="scheduled")
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("parks.id"))
    home_sp_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    away_sp_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    total_runs: Mapped[int | None] = mapped_column(Integer)
    winning_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    innings: Mapped[int | None] = mapped_column(Integer)
    is_doubleheader: Mapped[bool] = mapped_column(Boolean, default=False)
    doubleheader_num: Mapped[int | None] = mapped_column(Integer)
    season: Mapped[int] = mapped_column(Integer, nullable=False)

    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])
    venue: Mapped["Park"] = relationship(foreign_keys=[venue_id])
    home_sp: Mapped["Player"] = relationship(foreign_keys=[home_sp_id])
    away_sp: Mapped["Player"] = relationship(foreign_keys=[away_sp_id])


class Lineup(Base):
    __tablename__ = "lineups"
    __table_args__ = (
        UniqueConstraint("game_id", "team_id", "batting_order"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    batting_order: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[str | None] = mapped_column(String(4))


class TeamBattingDaily(Base):
    __tablename__ = "team_batting_daily"
    __table_args__ = (
        UniqueConstraint("team_id", "game_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    runs: Mapped[int | None] = mapped_column(Integer)
    hits: Mapped[int | None] = mapped_column(Integer)
    doubles: Mapped[int | None] = mapped_column(Integer)
    triples: Mapped[int | None] = mapped_column(Integer)
    home_runs: Mapped[int | None] = mapped_column(Integer)
    rbi: Mapped[int | None] = mapped_column(Integer)
    walks: Mapped[int | None] = mapped_column(Integer)
    strikeouts: Mapped[int | None] = mapped_column(Integer)
    stolen_bases: Mapped[int | None] = mapped_column(Integer)
    at_bats: Mapped[int | None] = mapped_column(Integer)
    left_on_base: Mapped[int | None] = mapped_column(Integer)


class PitcherGameLog(Base):
    __tablename__ = "pitcher_game_log"
    __table_args__ = (
        UniqueConstraint("player_id", "game_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    is_starter: Mapped[bool] = mapped_column(Boolean, nullable=False)
    innings_pitched: Mapped[float | None] = mapped_column(Float)
    hits_allowed: Mapped[int | None] = mapped_column(Integer)
    runs_allowed: Mapped[int | None] = mapped_column(Integer)
    earned_runs: Mapped[int | None] = mapped_column(Integer)
    walks: Mapped[int | None] = mapped_column(Integer)
    strikeouts: Mapped[int | None] = mapped_column(Integer)
    home_runs_allowed: Mapped[int | None] = mapped_column(Integer)
    pitches_thrown: Mapped[int | None] = mapped_column(Integer)
    win: Mapped[bool | None] = mapped_column(Boolean)
    loss: Mapped[bool | None] = mapped_column(Boolean)
    save: Mapped[bool | None] = mapped_column(Boolean)


class StatcastPitcherMetrics(Base):
    __tablename__ = "statcast_pitcher_metrics"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "as_of_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    era: Mapped[float | None] = mapped_column(Float)
    fip: Mapped[float | None] = mapped_column(Float)
    xfip: Mapped[float | None] = mapped_column(Float)
    siera: Mapped[float | None] = mapped_column(Float)
    k_per_9: Mapped[float | None] = mapped_column(Float)
    bb_per_9: Mapped[float | None] = mapped_column(Float)
    hr_per_9: Mapped[float | None] = mapped_column(Float)
    k_pct: Mapped[float | None] = mapped_column(Float)
    bb_pct: Mapped[float | None] = mapped_column(Float)
    whip: Mapped[float | None] = mapped_column(Float)
    babip: Mapped[float | None] = mapped_column(Float)
    lob_pct: Mapped[float | None] = mapped_column(Float)
    gb_pct: Mapped[float | None] = mapped_column(Float)
    fb_pct: Mapped[float | None] = mapped_column(Float)
    hr_fb_pct: Mapped[float | None] = mapped_column(Float)
    avg_velocity: Mapped[float | None] = mapped_column(Float)
    xba: Mapped[float | None] = mapped_column(Float)
    xslg: Mapped[float | None] = mapped_column(Float)
    xwoba: Mapped[float | None] = mapped_column(Float)
    barrel_pct: Mapped[float | None] = mapped_column(Float)
    hard_hit_pct: Mapped[float | None] = mapped_column(Float)
    whiff_pct: Mapped[float | None] = mapped_column(Float)


class StatcastBatterMetrics(Base):
    __tablename__ = "statcast_batter_metrics"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "as_of_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    pa: Mapped[int | None] = mapped_column(Integer)
    avg: Mapped[float | None] = mapped_column(Float)
    obp: Mapped[float | None] = mapped_column(Float)
    slg: Mapped[float | None] = mapped_column(Float)
    woba: Mapped[float | None] = mapped_column(Float)
    xba: Mapped[float | None] = mapped_column(Float)
    xslg: Mapped[float | None] = mapped_column(Float)
    xwoba: Mapped[float | None] = mapped_column(Float)
    wrc_plus: Mapped[float | None] = mapped_column(Float)
    barrel_pct: Mapped[float | None] = mapped_column(Float)
    hard_hit_pct: Mapped[float | None] = mapped_column(Float)
    k_pct: Mapped[float | None] = mapped_column(Float)
    bb_pct: Mapped[float | None] = mapped_column(Float)
    sprint_speed: Mapped[float | None] = mapped_column(Float)
    iso: Mapped[float | None] = mapped_column(Float)
    babip: Mapped[float | None] = mapped_column(Float)


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"
    __table_args__ = (
        Index("ix_odds_game_market", "game_id", "market_type", "sportsbook", "captured_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    sportsbook: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(16), nullable=False)
    home_line: Mapped[float | None] = mapped_column(Float)
    away_line: Mapped[float | None] = mapped_column(Float)
    total_line: Mapped[float | None] = mapped_column(Float)
    over_odds: Mapped[float | None] = mapped_column(Float)
    under_odds: Mapped[float | None] = mapped_column(Float)


class Weather(Base):
    __tablename__ = "weather"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), unique=True, nullable=False)
    temperature_f: Mapped[float | None] = mapped_column(Float)
    humidity_pct: Mapped[float | None] = mapped_column(Float)
    wind_speed_mph: Mapped[float | None] = mapped_column(Float)
    wind_direction: Mapped[str | None] = mapped_column(String(16))
    precipitation_pct: Mapped[float | None] = mapped_column(Float)
    condition: Mapped[str | None] = mapped_column(String(32))
    is_dome: Mapped[bool] = mapped_column(Boolean, default=False)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("game_id", "model_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    home_win_prob: Mapped[float | None] = mapped_column(Float)
    away_win_prob: Mapped[float | None] = mapped_column(Float)
    home_runs_pred: Mapped[float | None] = mapped_column(Float)
    away_runs_pred: Mapped[float | None] = mapped_column(Float)
    total_runs_pred: Mapped[float | None] = mapped_column(Float)
    over_prob: Mapped[float | None] = mapped_column(Float)
    under_prob: Mapped[float | None] = mapped_column(Float)
    home_cover_prob: Mapped[float | None] = mapped_column(Float)
    away_cover_prob: Mapped[float | None] = mapped_column(Float)
    home_ml_ev: Mapped[float | None] = mapped_column(Float)
    away_ml_ev: Mapped[float | None] = mapped_column(Float)
    over_ev: Mapped[float | None] = mapped_column(Float)
    under_ev: Mapped[float | None] = mapped_column(Float)
    confidence_tier: Mapped[str | None] = mapped_column(String(1))
    kelly_fraction: Mapped[float | None] = mapped_column(Float)
    recommended_units: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)


class BankrollLedger(Base):
    __tablename__ = "bankroll_ledger"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    prediction_id: Mapped[int | None] = mapped_column(ForeignKey("predictions.id"))
    bet_type: Mapped[str | None] = mapped_column(String(16))
    sportsbook: Mapped[str | None] = mapped_column(String(32))
    odds_taken: Mapped[float | None] = mapped_column(Float)
    stake_units: Mapped[float | None] = mapped_column(Float)
    stake_dollars: Mapped[float | None] = mapped_column(Float)
    result: Mapped[str | None] = mapped_column(String(8))
    pnl_units: Mapped[float | None] = mapped_column(Float)
    pnl_dollars: Mapped[float | None] = mapped_column(Float)
    bankroll_after: Mapped[float | None] = mapped_column(Float)
