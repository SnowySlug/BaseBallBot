"""Statcast / Baseball Savant data ingestion via pybaseball."""

from datetime import date
from pathlib import Path

import pandas as pd
import structlog

from bbbot.db.engine import get_session, init_db
from bbbot.db.models import Player, StatcastBatterMetrics, StatcastPitcherMetrics

log = structlog.get_logger()

CACHE_DIR = Path("data/cache/pybaseball")


def _cache_path(name: str, season: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_{season}.csv"


# ---------------------------------------------------------------------------
# Fetchers — pull from Baseball Savant via pybaseball
# ---------------------------------------------------------------------------

def fetch_pitcher_expected(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch pitcher expected stats (xBA, xSLG, xwOBA, xERA) from Savant."""
    cache = _cache_path("pitcher_xstats", season)
    if cache.exists() and not force_refresh:
        log.info("loading_cached_pitcher_xstats", path=str(cache))
        return pd.read_csv(cache)

    from pybaseball import statcast_pitcher_expected_stats

    log.info("fetching_pitcher_xstats", season=season)
    df = statcast_pitcher_expected_stats(season, minPA=50)
    df.to_csv(cache, index=False)
    log.info("cached_pitcher_xstats", path=str(cache), rows=len(df))
    return df


def fetch_pitcher_barrels(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch pitcher exit velo / barrel data from Savant."""
    cache = _cache_path("pitcher_barrels", season)
    if cache.exists() and not force_refresh:
        log.info("loading_cached_pitcher_barrels", path=str(cache))
        return pd.read_csv(cache)

    from pybaseball import statcast_pitcher_exitvelo_barrels

    log.info("fetching_pitcher_barrels", season=season)
    df = statcast_pitcher_exitvelo_barrels(season, minBBE=50)
    df.to_csv(cache, index=False)
    log.info("cached_pitcher_barrels", path=str(cache), rows=len(df))
    return df


def fetch_batter_expected(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch batter expected stats (xBA, xSLG, xwOBA) from Savant."""
    cache = _cache_path("batter_xstats", season)
    if cache.exists() and not force_refresh:
        log.info("loading_cached_batter_xstats", path=str(cache))
        return pd.read_csv(cache)

    from pybaseball import statcast_batter_expected_stats

    log.info("fetching_batter_xstats", season=season)
    df = statcast_batter_expected_stats(season, minPA=50)
    df.to_csv(cache, index=False)
    log.info("cached_batter_xstats", path=str(cache), rows=len(df))
    return df


def fetch_batter_barrels(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch batter exit velo / barrel data from Savant."""
    cache = _cache_path("batter_barrels", season)
    if cache.exists() and not force_refresh:
        log.info("loading_cached_batter_barrels", path=str(cache))
        return pd.read_csv(cache)

    from pybaseball import statcast_batter_exitvelo_barrels

    log.info("fetching_batter_barrels", season=season)
    df = statcast_batter_exitvelo_barrels(season, minBBE=50)
    df.to_csv(cache, index=False)
    log.info("cached_batter_barrels", path=str(cache), rows=len(df))
    return df


# ---------------------------------------------------------------------------
# Ingestors — merge Savant data and store in DB
# ---------------------------------------------------------------------------

def _find_player_by_mlb_id(session, mlb_id: int) -> Player | None:
    """Look up player by MLB (Savant) player_id."""
    return session.query(Player).filter(Player.mlb_id == mlb_id).first()


def ingest_pitcher_statcast(season: int, force_refresh: bool = False) -> int:
    """Ingest pitcher Statcast metrics from Baseball Savant.

    Merges expected stats (xBA, xSLG, xwOBA, xERA) with barrel data.
    Returns number of records stored.
    """
    init_db()
    df_xstats = fetch_pitcher_expected(season, force_refresh)
    df_barrels = fetch_pitcher_barrels(season, force_refresh)

    if df_xstats.empty:
        log.warning("no_pitcher_xstats", season=season)
        return 0

    # Merge on player_id
    df = df_xstats.merge(df_barrels, on="player_id", how="left", suffixes=("", "_barrel"))

    session = get_session()
    count = 0
    as_of = date.today()

    try:
        for _, row in df.iterrows():
            mlb_id = _int(row.get("player_id"))
            if not mlb_id:
                continue

            player = _find_player_by_mlb_id(session, mlb_id)
            if not player:
                continue

            # Skip if already exists for today
            existing = session.query(StatcastPitcherMetrics).filter_by(
                player_id=player.id, season=season, as_of_date=as_of
            ).first()
            if existing:
                continue

            metrics = StatcastPitcherMetrics(
                player_id=player.id,
                season=season,
                as_of_date=as_of,
                # Expected stats from Savant
                xba=_float(row.get("est_ba")),
                xslg=_float(row.get("est_slg")),
                xwoba=_float(row.get("est_woba")),
                # Barrel / exit velo data
                barrel_pct=_float(row.get("brl_percent")),
                hard_hit_pct=_float(row.get("ev95percent")),
                avg_velocity=_float(row.get("avg_hit_speed")),
                # xERA if available
                era=_float(row.get("era")),
                # Remaining fields null — we don't have FanGraphs data
                fip=None,
                xfip=None,
                siera=None,
                k_per_9=None,
                bb_per_9=None,
                hr_per_9=None,
                k_pct=None,
                bb_pct=None,
                whip=None,
                babip=None,
                lob_pct=None,
                gb_pct=None,
                fb_pct=None,
                hr_fb_pct=None,
                whiff_pct=None,
            )
            session.add(metrics)
            count += 1

        session.commit()
        log.info("ingested_pitcher_statcast", season=season, count=count)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return count


def ingest_batter_statcast(season: int, force_refresh: bool = False) -> int:
    """Ingest batter Statcast metrics from Baseball Savant.

    Merges expected stats (xBA, xSLG, xwOBA) with barrel data.
    Returns number of records stored.
    """
    init_db()
    df_xstats = fetch_batter_expected(season, force_refresh)
    df_barrels = fetch_batter_barrels(season, force_refresh)

    if df_xstats.empty:
        log.warning("no_batter_xstats", season=season)
        return 0

    # Merge on player_id
    df = df_xstats.merge(df_barrels, on="player_id", how="left", suffixes=("", "_barrel"))

    session = get_session()
    count = 0
    as_of = date.today()

    try:
        for _, row in df.iterrows():
            mlb_id = _int(row.get("player_id"))
            if not mlb_id:
                continue

            player = _find_player_by_mlb_id(session, mlb_id)
            if not player:
                continue

            existing = session.query(StatcastBatterMetrics).filter_by(
                player_id=player.id, season=season, as_of_date=as_of
            ).first()
            if existing:
                continue

            metrics = StatcastBatterMetrics(
                player_id=player.id,
                season=season,
                as_of_date=as_of,
                pa=_int(row.get("pa")),
                avg=_float(row.get("ba")),
                obp=None,
                slg=_float(row.get("slg")),
                woba=_float(row.get("woba")),
                xba=_float(row.get("est_ba")),
                xslg=_float(row.get("est_slg")),
                xwoba=_float(row.get("est_woba")),
                wrc_plus=None,
                barrel_pct=_float(row.get("brl_percent")),
                hard_hit_pct=_float(row.get("ev95percent")),
                k_pct=None,
                bb_pct=None,
                sprint_speed=None,
                iso=None,
                babip=None,
            )
            session.add(metrics)
            count += 1

        session.commit()
        log.info("ingested_batter_statcast", season=season, count=count)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _int(val) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _pct(val) -> float | None:
    """Convert percentage strings like '25.3%' or 0.253 to decimal 0.253."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        val = val.strip().rstrip("%")
        try:
            v = float(val)
            return v / 100 if v > 1 else v
        except ValueError:
            return None
    try:
        v = float(val)
        return v / 100 if v > 1 else v
    except (ValueError, TypeError):
        return None
