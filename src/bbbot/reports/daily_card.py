"""Daily prediction card — formatted CLI output matching dashboard style."""

from datetime import date, datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def render_daily_card(predictions: list[dict], game_date: date,
                      console: Console | None = None) -> None:
    """Render the daily prediction card matching the dashboard layout.

    Each prediction dict should have:
        away_team, home_team, away_sp, home_sp, game_time,
        home_win_prob, away_win_prob,
        home_runs_pred, away_runs_pred, total_pred,
        pick_team, pick_prob, confidence,
        ou_pick, ou_prob, ou_line,
        kalshi_edge (dict or None),
        status, away_score, home_score
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
        f"[dim]Generated at {datetime.now().strftime('%H:%M:%S')}[/dim]",
        border_style="blue",
    ))

    # Summary stats
    high_conf = sum(1 for p in predictions if p.get("confidence") == "HIGH")
    edges = sum(1 for p in predictions if p.get("kalshi_edge"))
    console.print(
        f"  [bold]{len(predictions)}[/bold] games  |  "
        f"[bold green]{high_conf}[/bold green] high confidence  |  "
        f"[bold yellow]{edges}[/bold yellow] Kalshi edges"
    )
    console.print()

    # Render each game card
    for pred in predictions:
        away = pred.get("away_team", "?")
        home = pred.get("home_team", "?")
        away_sp = pred.get("away_sp", "TBD")
        home_sp = pred.get("home_sp", "TBD")
        time_str = pred.get("game_time", "TBD")

        # Header with status
        status = pred.get("status", "scheduled")
        header = f"[bold white]{away} @ {home}[/bold white]"
        if status == "final" and pred.get("away_score") is not None:
            actual_winner = home if pred.get("home_score", 0) > pred.get("away_score", 0) else away
            if actual_winner == pred.get("pick_team"):
                header += f" — Final: {pred['away_score']}-{pred['home_score']} [bold green]Model Correct[/bold green]"
            else:
                header += f" — Final: {pred['away_score']}-{pred['home_score']} [bold red]Model Wrong[/bold red]"
        elif status == "live":
            header += " — [bold red]LIVE[/bold red]"
        else:
            header += f" — {time_str}"

        console.print(f"  {header}")
        console.print(f"  [dim]{away_sp} vs {home_sp}[/dim]")

        # Predicted Winner
        pick_team = pred.get("pick_team", "?")
        pick_prob = pred.get("pick_prob", 0.5)
        confidence = pred.get("confidence", "LOW")

        conf_colors = {"HIGH": "green", "MED": "yellow", "LOW": "red"}
        conf_color = conf_colors.get(confidence, "red")

        console.print(
            f"  Predicted Winner: [{conf_color} bold]{pick_team}[/{conf_color} bold]  "
            f"{pick_prob:.0%} win probability — [bold]{confidence}[/bold] confidence"
        )

        # Predicted Score
        hr = pred.get("home_runs_pred", 0)
        ar = pred.get("away_runs_pred", 0)
        total = pred.get("total_pred", 0)
        console.print(
            f"  Predicted Score:  [bold]{ar:.1f} - {hr:.1f}[/bold]  ({away} - {home})"
        )

        # Run Total + O/U
        ou_pick = pred.get("ou_pick", "Over")
        ou_prob = pred.get("ou_prob", 0.5)
        ou_line = pred.get("ou_line", 8.5)
        console.print(
            f"  Run Total:        [bold]{total:.1f} runs[/bold]  "
            f"Model says [bold]{ou_pick}[/bold] {ou_line} ({ou_prob:.0%})"
        )

        # Kalshi edge callout (only when >2% edge)
        kalshi_edge = pred.get("kalshi_edge")
        if kalshi_edge:
            k_odds = kalshi_edge["odds"]
            k_odds_str = f"+{k_odds:.0f}" if k_odds > 0 else f"{k_odds:.0f}"
            console.print(
                f"  [bold green]Kalshi Edge:[/bold green] Model gives {pick_team} a "
                f"{kalshi_edge['model_prob']:.0%} chance, but Kalshi implies "
                f"{kalshi_edge['implied_prob']:.0%} ({k_odds_str}). "
                f"That's a [bold]+{kalshi_edge['ev']:.1%}[/bold] edge."
            )

        console.print("  " + "-" * 60)
        console.print()

    console.print()
