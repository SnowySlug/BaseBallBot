"""Performance tracking — accuracy, ROI, calibration analysis."""

from datetime import date

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from bbbot.db.models import Game, Prediction


def render_performance_report(session: Session, start: date, end: date,
                              console: Console | None = None) -> dict:
    """Generate and display performance metrics for a date range.

    Returns a summary dict with computed metrics.
    """
    if console is None:
        console = Console()

    # Fetch predictions that have been resolved (game is final)
    predictions = list(session.execute(
        select(Prediction, Game).join(Game, Prediction.game_id == Game.id).where(
            and_(
                Game.game_date >= start,
                Game.game_date <= end,
                Game.status == "final",
                Game.home_score.isnot(None),
            )
        )
    ).all())

    if not predictions:
        console.print("[yellow]No resolved predictions found in this range.[/yellow]")
        return {}

    # Calculate metrics
    total = len(predictions)
    correct_ml = 0
    total_ml_ev = 0.0
    tier_records = {"A": [0, 0], "B": [0, 0], "C": [0, 0], "D": [0, 0]}  # [wins, total]
    total_over_correct = 0
    total_over = 0

    for pred, game in predictions:
        # Win prediction accuracy
        home_favored = (pred.home_win_prob or 0.5) > 0.5
        home_won = game.home_score > game.away_score

        if home_favored == home_won:
            correct_ml += 1

        # Track by tier
        tier = pred.confidence_tier or "D"
        if tier in tier_records:
            tier_records[tier][1] += 1
            if home_favored == home_won:
                tier_records[tier][0] += 1

        # O/U accuracy
        if pred.total_runs_pred and game.total_runs is not None:
            total_over += 1
            pred_over = pred.total_runs_pred > 8.5
            actual_over = game.total_runs > 8.5
            if pred_over == actual_over:
                total_over_correct += 1

    accuracy = correct_ml / total if total > 0 else 0
    ou_accuracy = total_over_correct / total_over if total_over > 0 else 0

    # Display
    console.print()
    console.print(Panel(
        f"[bold]Performance Report: {start} to {end}[/bold]\n"
        f"[dim]{total} resolved predictions[/dim]",
        border_style="blue",
    ))

    # Summary table
    summary = Table(show_header=False, padding=(0, 2))
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")

    summary.add_row("ML Accuracy", f"{accuracy:.1%} ({correct_ml}/{total})")
    summary.add_row("O/U Accuracy", f"{ou_accuracy:.1%} ({total_over_correct}/{total_over})")

    console.print(summary)

    # Tier breakdown
    tier_table = Table(title="By Confidence Tier")
    tier_table.add_column("Tier", style="bold")
    tier_table.add_column("Record", justify="center")
    tier_table.add_column("Win %", justify="center")

    for tier in ("A", "B", "C", "D"):
        wins, total_t = tier_records[tier]
        if total_t > 0:
            pct = f"{wins / total_t:.1%}"
        else:
            pct = "-"
        tier_table.add_row(tier, f"{wins}-{total_t - wins}", pct)

    console.print(tier_table)
    console.print()

    return {
        "total": total,
        "ml_accuracy": accuracy,
        "ou_accuracy": ou_accuracy,
        "tier_records": tier_records,
    }
