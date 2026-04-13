"""Bullpen feature group — relief corps quality and fatigue."""

from datetime import date, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from bbbot.db.models import PitcherGameLog
from bbbot.features.base import FeatureGroup


class BullpenFeatures(FeatureGroup):
    name = "bullpen"

    @property
    def feature_names(self) -> list[str]:
        return [
            "bp_era_14d", "bp_k_per_9_14d", "bp_bb_per_9_14d",
            "bp_whip_14d",
            "bp_ip_last_3d",  # fatigue: total bullpen IP last 3 days
            "bp_pitches_last_3d",  # fatigue: total bullpen pitches last 3 days
            "bp_appearances_7d",  # workload: total reliever appearances last 7 days
        ]

    def compute(self, session: Session, game_id: int, team_id: int,
                as_of_date: date) -> dict[str, float | None]:
        features: dict[str, float | None] = {name: None for name in self.feature_names}

        # 14-day bullpen aggregate stats
        start_14d = as_of_date - timedelta(days=14)
        bp_logs_14d = list(session.execute(
            select(PitcherGameLog).where(and_(
                PitcherGameLog.team_id == team_id,
                PitcherGameLog.is_starter == False,
                PitcherGameLog.game_date >= start_14d,
                PitcherGameLog.game_date < as_of_date,
            ))
        ).scalars().all())

        if bp_logs_14d:
            total_ip = sum(l.innings_pitched or 0 for l in bp_logs_14d)
            total_er = sum(l.earned_runs or 0 for l in bp_logs_14d)
            total_h = sum(l.hits_allowed or 0 for l in bp_logs_14d)
            total_bb = sum(l.walks or 0 for l in bp_logs_14d)
            total_k = sum(l.strikeouts or 0 for l in bp_logs_14d)

            if total_ip > 0:
                features["bp_era_14d"] = (total_er / total_ip) * 9
                features["bp_k_per_9_14d"] = (total_k / total_ip) * 9
                features["bp_bb_per_9_14d"] = (total_bb / total_ip) * 9
                features["bp_whip_14d"] = (total_h + total_bb) / total_ip

        # 3-day fatigue metrics
        start_3d = as_of_date - timedelta(days=3)
        bp_logs_3d = list(session.execute(
            select(PitcherGameLog).where(and_(
                PitcherGameLog.team_id == team_id,
                PitcherGameLog.is_starter == False,
                PitcherGameLog.game_date >= start_3d,
                PitcherGameLog.game_date < as_of_date,
            ))
        ).scalars().all())

        if bp_logs_3d:
            features["bp_ip_last_3d"] = sum(l.innings_pitched or 0 for l in bp_logs_3d)
            features["bp_pitches_last_3d"] = float(
                sum(l.pitches_thrown or 0 for l in bp_logs_3d)
            )

        # 7-day appearance count
        start_7d = as_of_date - timedelta(days=7)
        appearances = session.execute(
            select(func.count()).where(and_(
                PitcherGameLog.team_id == team_id,
                PitcherGameLog.is_starter == False,
                PitcherGameLog.game_date >= start_7d,
                PitcherGameLog.game_date < as_of_date,
            ))
        ).scalar() or 0
        features["bp_appearances_7d"] = float(appearances)

        return features
