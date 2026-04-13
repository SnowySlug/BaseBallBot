"""Starting pitcher feature group."""

from datetime import date, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from bbbot.db.models import PitcherGameLog, StatcastPitcherMetrics
from bbbot.features.base import FeatureGroup


class StartingPitchingFeatures(FeatureGroup):
    name = "starting_pitching"

    @property
    def feature_names(self) -> list[str]:
        return [
            # Season statcast metrics
            "sp_era", "sp_fip", "sp_xfip", "sp_siera",
            "sp_k_per_9", "sp_bb_per_9", "sp_hr_per_9",
            "sp_whip", "sp_babip", "sp_lob_pct",
            "sp_gb_pct", "sp_fb_pct",
            "sp_xba", "sp_xslg", "sp_xwoba",
            "sp_barrel_pct", "sp_hard_hit_pct", "sp_whiff_pct",
            "sp_avg_velocity",
            # Recent form (last 3 starts)
            "sp_recent_era", "sp_recent_k_per_9", "sp_recent_bb_per_9",
            "sp_recent_ip_avg",
            # Workload
            "sp_rest_days", "sp_last_pitch_count",
            "sp_season_ip",
        ]

    def compute(self, session: Session, game_id: int, team_id: int,
                as_of_date: date) -> dict[str, float | None]:
        from bbbot.db.models import Game

        features: dict[str, float | None] = {name: None for name in self.feature_names}

        # Find the starter for this team in this game
        game = session.get(Game, game_id)
        if not game:
            return features

        sp_id = game.home_sp_id if game.home_team_id == team_id else game.away_sp_id
        if not sp_id:
            return features

        # Get latest statcast metrics
        metrics = session.execute(
            select(StatcastPitcherMetrics)
            .where(and_(
                StatcastPitcherMetrics.player_id == sp_id,
                StatcastPitcherMetrics.season == as_of_date.year,
                StatcastPitcherMetrics.as_of_date <= as_of_date,
            ))
            .order_by(StatcastPitcherMetrics.as_of_date.desc())
            .limit(1)
        ).scalar_one_or_none()

        if metrics:
            features["sp_era"] = metrics.era
            features["sp_fip"] = metrics.fip
            features["sp_xfip"] = metrics.xfip
            features["sp_siera"] = metrics.siera
            features["sp_k_per_9"] = metrics.k_per_9
            features["sp_bb_per_9"] = metrics.bb_per_9
            features["sp_hr_per_9"] = metrics.hr_per_9
            features["sp_whip"] = metrics.whip
            features["sp_babip"] = metrics.babip
            features["sp_lob_pct"] = metrics.lob_pct
            features["sp_gb_pct"] = metrics.gb_pct
            features["sp_fb_pct"] = metrics.fb_pct
            features["sp_xba"] = metrics.xba
            features["sp_xslg"] = metrics.xslg
            features["sp_xwoba"] = metrics.xwoba
            features["sp_barrel_pct"] = metrics.barrel_pct
            features["sp_hard_hit_pct"] = metrics.hard_hit_pct
            features["sp_whiff_pct"] = metrics.whiff_pct
            features["sp_avg_velocity"] = metrics.avg_velocity

        # Recent game logs (last 3 starts)
        recent_logs = list(session.execute(
            select(PitcherGameLog)
            .where(and_(
                PitcherGameLog.player_id == sp_id,
                PitcherGameLog.is_starter == True,
                PitcherGameLog.game_date < as_of_date,
            ))
            .order_by(PitcherGameLog.game_date.desc())
            .limit(3)
        ).scalars().all())

        if recent_logs:
            total_ip = sum(l.innings_pitched or 0 for l in recent_logs)
            total_er = sum(l.earned_runs or 0 for l in recent_logs)
            total_k = sum(l.strikeouts or 0 for l in recent_logs)
            total_bb = sum(l.walks or 0 for l in recent_logs)
            n = len(recent_logs)

            if total_ip > 0:
                features["sp_recent_era"] = (total_er / total_ip) * 9
                features["sp_recent_k_per_9"] = (total_k / total_ip) * 9
                features["sp_recent_bb_per_9"] = (total_bb / total_ip) * 9
            features["sp_recent_ip_avg"] = total_ip / n

            # Rest days since last start
            last_start_date = recent_logs[0].game_date
            features["sp_rest_days"] = (as_of_date - last_start_date).days

            # Last start pitch count
            features["sp_last_pitch_count"] = float(recent_logs[0].pitches_thrown or 0)

        # Season IP total
        season_ip = session.execute(
            select(func.sum(PitcherGameLog.innings_pitched))
            .where(and_(
                PitcherGameLog.player_id == sp_id,
                PitcherGameLog.is_starter == True,
                PitcherGameLog.game_date >= date(as_of_date.year, 1, 1),
                PitcherGameLog.game_date < as_of_date,
            ))
        ).scalar()
        features["sp_season_ip"] = float(season_ip or 0)

        return features
