"""Backtesting engine — walk-forward simulation on historical data."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy.orm import Session

from bbbot.betting.kelly import fractional_kelly, kelly_to_units
from bbbot.betting.odds_math import american_to_decimal, calculate_ev
from bbbot.db.engine import get_session, init_db
from bbbot.db.models import Game
from bbbot.db.queries import get_games_by_date
from bbbot.features.builder import build_game_features, create_default_registry
from bbbot.models.baseline import BaselineRunsModel, BaselineWinModel

log = structlog.get_logger()


class BacktestEngine:
    """Walk-forward backtesting engine."""

    def __init__(self, kelly_fraction: float = 0.25,
                 starting_bankroll: float = 1000.0,
                 unit_size: float = 100.0,
                 min_edge: float = 0.03):
        self.kelly_fraction = kelly_fraction
        self.starting_bankroll = starting_bankroll
        self.unit_size = unit_size
        self.min_edge = min_edge

    def run(self, start: date, end: date) -> pd.DataFrame:
        """Run backtest over a date range.

        Returns a DataFrame with one row per bet placed, including:
            date, game_id, bet_type, side, model_prob, odds, ev, units,
            result, pnl, bankroll
        """
        init_db()
        session = get_session()
        registry = create_default_registry()
        win_model = BaselineWinModel()
        runs_model = BaselineRunsModel()

        bankroll = self.starting_bankroll
        results = []
        current = start

        try:
            while current <= end:
                games = get_games_by_date(session, current)
                final_games = [g for g in games if g.status == "final"
                               and g.home_score is not None]

                for game in final_games:
                    features = build_game_features(session, game, registry)
                    feature_df = pd.DataFrame([features])

                    # Win probability
                    home_wp = float(win_model.predict_proba(feature_df)[0])
                    away_wp = 1.0 - home_wp

                    # Run totals
                    run_preds = runs_model.predict(feature_df)[0]
                    total_pred = float(run_preds[0] + run_preds[1])

                    # O/U probabilities
                    over_probs, under_probs = runs_model.predict_total_probs(
                        feature_df, line=8.5
                    )

                    # Check for ML bets
                    for prob, side, team_id in [
                        (home_wp, "home", game.home_team_id),
                        (away_wp, "away", game.away_team_id),
                    ]:
                        if prob < 0.55:
                            continue
                        # Simulate odds based on our probability
                        # Use -150 for moderate favorites as a rough benchmark
                        odds = american_to_decimal(-130 if prob < 0.6 else -160)
                        ev = calculate_ev(prob, odds)

                        if ev >= self.min_edge:
                            kf = fractional_kelly(prob, odds, self.kelly_fraction)
                            units = kelly_to_units(kf, bankroll, self.unit_size)
                            if units < 0.1:
                                continue

                            stake = units * self.unit_size
                            home_won = game.home_score > game.away_score
                            won = (side == "home") == home_won

                            if won:
                                pnl = stake * (odds - 1)
                            else:
                                pnl = -stake

                            bankroll += pnl
                            team = game.home_team if side == "home" else game.away_team

                            results.append({
                                "date": current,
                                "game_pk": game.mlb_game_pk,
                                "bet_type": "ML",
                                "side": f"{team.abbreviation} ({side})",
                                "model_prob": prob,
                                "odds": odds,
                                "ev": ev,
                                "units": units,
                                "stake": stake,
                                "won": won,
                                "pnl": pnl,
                                "bankroll": bankroll,
                            })

                    # Check for O/U bets
                    for prob, side in [
                        (float(over_probs[0]), "over"),
                        (float(under_probs[0]), "under"),
                    ]:
                        if prob < 0.55:
                            continue
                        odds = american_to_decimal(-110)
                        ev = calculate_ev(prob, odds)

                        if ev >= self.min_edge:
                            kf = fractional_kelly(prob, odds, self.kelly_fraction)
                            units = kelly_to_units(kf, bankroll, self.unit_size)
                            if units < 0.1:
                                continue

                            stake = units * self.unit_size
                            actual_total = game.total_runs or 0
                            if side == "over":
                                won = actual_total > 8.5
                            else:
                                won = actual_total < 8.5

                            if won:
                                pnl = stake * (odds - 1)
                            else:
                                pnl = -stake

                            bankroll += pnl
                            results.append({
                                "date": current,
                                "game_pk": game.mlb_game_pk,
                                "bet_type": "O/U",
                                "side": f"{side.title()} 8.5",
                                "model_prob": prob,
                                "odds": odds,
                                "ev": ev,
                                "units": units,
                                "stake": stake,
                                "won": won,
                                "pnl": pnl,
                                "bankroll": bankroll,
                            })

                current += timedelta(days=1)

        finally:
            session.close()

        return pd.DataFrame(results)

    def render_report(self, results: pd.DataFrame, console: Console | None = None):
        """Display backtest results."""
        if console is None:
            console = Console()

        if results.empty:
            console.print("[yellow]No bets placed during backtest period.[/yellow]")
            return

        total_bets = len(results)
        wins = results["won"].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets
        total_pnl = results["pnl"].sum()
        total_staked = results["stake"].sum()
        roi = total_pnl / total_staked if total_staked > 0 else 0
        final_bankroll = results["bankroll"].iloc[-1]
        max_bankroll = results["bankroll"].max()
        min_bankroll = results["bankroll"].min()
        max_drawdown = (max_bankroll - min_bankroll) / max_bankroll if max_bankroll > 0 else 0

        # By bet type
        ml_results = results[results["bet_type"] == "ML"]
        ou_results = results[results["bet_type"] == "O/U"]

        console.print()
        console.print(Panel(
            f"[bold]Backtest Results[/bold]\n"
            f"[dim]{results['date'].min()} to {results['date'].max()} | "
            f"{total_bets} bets placed[/dim]",
            border_style="blue",
        ))

        # Summary
        summary = Table(show_header=False, padding=(0, 2))
        summary.add_column("Metric", style="bold")
        summary.add_column("Value", justify="right")

        pnl_style = "green" if total_pnl >= 0 else "red"
        summary.add_row("Record", f"{wins}W - {losses}L ({win_rate:.1%})")
        summary.add_row("Total P&L", f"[{pnl_style}]${total_pnl:+,.2f}[/{pnl_style}]")
        summary.add_row("ROI", f"[{pnl_style}]{roi:+.1%}[/{pnl_style}]")
        summary.add_row("Starting Bankroll", f"${self.starting_bankroll:,.2f}")
        summary.add_row("Final Bankroll", f"${final_bankroll:,.2f}")
        summary.add_row("Max Drawdown", f"{max_drawdown:.1%}")
        summary.add_row("Avg EV per Bet", f"{results['ev'].mean():+.1%}")

        console.print(summary)

        # By type
        if not ml_results.empty:
            ml_wins = ml_results["won"].sum()
            ml_pnl = ml_results["pnl"].sum()
            console.print(
                f"\n[bold]ML Bets:[/bold] {ml_wins}W-{len(ml_results) - ml_wins}L | "
                f"P&L: ${ml_pnl:+,.2f}"
            )

        if not ou_results.empty:
            ou_wins = ou_results["won"].sum()
            ou_pnl = ou_results["pnl"].sum()
            console.print(
                f"[bold]O/U Bets:[/bold] {ou_wins}W-{len(ou_results) - ou_wins}L | "
                f"P&L: ${ou_pnl:+,.2f}"
            )

        # Recent bets table
        console.print()
        recent = Table(title="Last 10 Bets")
        recent.add_column("Date", width=10)
        recent.add_column("Type", width=5)
        recent.add_column("Side", width=15)
        recent.add_column("Prob", width=5, justify="right")
        recent.add_column("EV", width=6, justify="right")
        recent.add_column("Units", width=5, justify="right")
        recent.add_column("Result", width=6, justify="center")
        recent.add_column("P&L", width=10, justify="right")

        for _, row in results.tail(10).iterrows():
            result_style = "green" if row["won"] else "red"
            result_text = "W" if row["won"] else "L"
            recent.add_row(
                str(row["date"]),
                row["bet_type"],
                row["side"],
                f"{row['model_prob']:.0%}",
                f"{row['ev']:+.1%}",
                f"{row['units']:.1f}",
                f"[{result_style}]{result_text}[/{result_style}]",
                f"[{result_style}]${row['pnl']:+.2f}[/{result_style}]",
            )

        console.print(recent)
        console.print()
