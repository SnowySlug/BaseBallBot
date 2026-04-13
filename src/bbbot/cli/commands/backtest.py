"""CLI commands for backtesting."""

from datetime import date, datetime

import typer
from rich.console import Console

from bbbot.backtest.engine import BacktestEngine

console = Console()
app = typer.Typer(help="Backtesting commands")


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


@app.command()
def run(
    from_date: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD)"),
    bankroll: float = typer.Option(1000.0, "--bankroll", "-b", help="Starting bankroll"),
    kelly: float = typer.Option(0.25, "--kelly", "-k", help="Kelly fraction"),
    unit_size: float = typer.Option(100.0, "--unit", "-u", help="Unit size in dollars"),
    min_edge: float = typer.Option(0.03, "--min-edge", help="Minimum EV to bet"),
):
    """Run a backtest simulation over a date range."""
    start = _parse_date(from_date)
    end = _parse_date(to_date)

    console.print(f"\n[bold]Running backtest: {start} to {end}[/bold]")
    console.print(f"[dim]Bankroll: ${bankroll:,.0f} | Kelly: {kelly} | "
                  f"Unit: ${unit_size:.0f} | Min edge: {min_edge:.0%}[/dim]\n")

    engine = BacktestEngine(
        kelly_fraction=kelly,
        starting_bankroll=bankroll,
        unit_size=unit_size,
        min_edge=min_edge,
    )

    with console.status("[bold green]Running backtest..."):
        results = engine.run(start, end)

    engine.render_report(results, console)
