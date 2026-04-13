"""Daily prediction card — formatted CLI output."""

from datetime import date, datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def render_daily_card(predictions: list[dict], game_date: date,
                      console: Console | None = None) -> None:
    """Render the daily prediction card as a rich table.

    Each prediction dict should have:
        away_team, home_team, away_sp, home_sp, game_time,
        home_win_prob, away_win_prob,
        home_runs_pred, away_runs_pred, total_pred,
        confidence_tier, home_ml_ev, away_ml_ev,
        over_prob, under_prob,
        recommended_bet, recommended_units
    """
    if console is None:
        console = Console()

    if not predictions:
        console.print("[yellow]No predictions for this date.[/yellow]")
        return

    # Header
    console.print()
    console.print(Panel(
        f"[bold white]MLB Predictions -- {game_date}[/bold white]\n"
        f"[dim]Generated at {datetime.now().strftime('%H:%M:%S')} | "
        f"Baseline Model v0.1[/dim]",
        border_style="blue",
    ))

    # Main prediction table
    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Time", style="dim", width=5)
    table.add_column("Matchup", width=12)
    table.add_column("Pitchers", width=28)
    table.add_column("Win %", justify="center", width=11)
    table.add_column("Pred Score", justify="center", width=10)
    table.add_column("Total", justify="center", width=5)
    table.add_column("Tier", justify="center", width=4)
    table.add_column("Best Bet", width=18)
    table.add_column("Units", justify="right", width=5)

    for pred in predictions:
        time_str = pred.get("game_time", "TBD")
        away = pred.get("away_team", "?")
        home = pred.get("home_team", "?")
        matchup = f"{away} @ {home}"

        away_sp = pred.get("away_sp", "TBD")
        home_sp = pred.get("home_sp", "TBD")
        pitchers = f"{away_sp} vs {home_sp}"

        hwp = pred.get("home_win_prob", 0.5)
        awp = pred.get("away_win_prob", 0.5)
        win_pct = f"{awp:.0%}/{hwp:.0%}"

        hr = pred.get("home_runs_pred", 4.5)
        ar = pred.get("away_runs_pred", 4.5)
        pred_score = f"{ar:.1f}-{hr:.1f}"

        total = pred.get("total_pred", 9.0)

        tier = pred.get("confidence_tier", "D")
        tier_colors = {"A": "green bold", "B": "yellow", "C": "dim", "D": "dim"}
        tier_style = tier_colors.get(tier, "dim")

        rec_bet = pred.get("recommended_bet", "-")
        units = pred.get("recommended_units", 0)
        units_str = f"{units:.1f}" if units > 0 else "-"

        table.add_row(
            time_str, matchup, pitchers, win_pct,
            pred_score, f"{total:.1f}",
            f"[{tier_style}]{tier}[/{tier_style}]",
            rec_bet, units_str,
        )

    console.print(table)

    # Summary of top picks
    top_picks = [p for p in predictions if p.get("confidence_tier") in ("A", "B")]
    if top_picks:
        console.print()
        console.print("[bold green]Top Picks:[/bold green]")
        for pick in top_picks:
            bet = pick.get("recommended_bet", "")
            units = pick.get("recommended_units", 0)
            ev = pick.get("best_ev", 0)
            console.print(
                f"  [bold]{bet}[/bold] — "
                f"{units:.1f}u — "
                f"EV: {ev:+.1%}"
            )
    else:
        console.print("\n[dim]No high-confidence picks today.[/dim]")

    console.print()
