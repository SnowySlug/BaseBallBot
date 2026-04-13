"""CLI commands for data ingestion."""

from datetime import date, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from bbbot.db.engine import get_session, init_db
from bbbot.db.queries import get_games_by_date
from bbbot.ingest.boxscore import BoxScoreIngestor
from bbbot.ingest.schedule import DailyIngestor

console = Console()
app = typer.Typer(help="Data ingestion commands")


def _parse_date(date_str: str | None) -> date:
    if date_str is None:
        return date.today()
    return datetime.strptime(date_str, "%Y-%m-%d").date()


@app.command()
def daily(
    date_str: str = typer.Option(None, "--date", "-d", help="Date (YYYY-MM-DD), default today"),
):
    """Fetch today's MLB schedule and store in database."""
    game_date = _parse_date(date_str)
    console.print(f"\n[bold]Fetching MLB schedule for {game_date}...[/bold]\n")

    ingestor = DailyIngestor()
    count = ingestor.ingest_schedule(game_date)

    console.print(f"[green]Ingested {count} games.[/green]\n")

    # Show the games
    _show_games(game_date)


@app.command()
def scores(
    date_str: str = typer.Option(None, "--date", "-d", help="Date (YYYY-MM-DD), default yesterday"),
):
    """Update final scores for games on a date."""
    if date_str is None:
        game_date = date.today() - timedelta(days=1)
    else:
        game_date = _parse_date(date_str)

    console.print(f"\n[bold]Updating scores for {game_date}...[/bold]\n")

    ingestor = DailyIngestor()
    # First make sure games exist
    ingestor.ingest_schedule(game_date)
    scored = ingestor.ingest_scores(game_date)

    console.print(f"[green]Updated {scored} game scores.[/green]\n")

    _show_games(game_date)


@app.command()
def backfill(
    from_date: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD)"),
):
    """Backfill games and scores for a date range."""
    start = _parse_date(from_date)
    end = _parse_date(to_date)

    console.print(f"\n[bold]Backfilling games from {start} to {end}...[/bold]\n")

    ingestor = DailyIngestor()
    total_games = 0
    current = start

    with console.status("[bold green]Ingesting...") as status:
        while current <= end:
            status.update(f"[bold green]Ingesting {current}...")
            count = ingestor.ingest_schedule(current)
            total_games += count
            current += timedelta(days=1)

    console.print(f"\n[green]Backfilled {total_games} games across {(end - start).days + 1} days.[/green]\n")


@app.command()
def boxscores(
    from_date: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option(None, "--to", help="End date (YYYY-MM-DD), default same as --from"),
):
    """Fetch box scores for completed games in a date range."""
    start = _parse_date(from_date)
    end = _parse_date(to_date) if to_date else start

    console.print(f"\n[bold]Fetching box scores from {start} to {end}...[/bold]\n")

    # Make sure schedules are loaded first
    sched_ingestor = DailyIngestor()
    box_ingestor = BoxScoreIngestor()
    total = 0
    current = start

    with console.status("[bold green]Fetching box scores...") as status:
        while current <= end:
            status.update(f"[bold green]Box scores for {current}...")
            sched_ingestor.ingest_schedule(current)
            count = box_ingestor.ingest_boxscores(current)
            total += count
            current += timedelta(days=1)

    console.print(f"[green]Ingested {total} box scores.[/green]\n")


@app.command()
def odds():
    """Fetch current MLB odds from The Odds API."""
    from bbbot.ingest.odds_ingest import ingest_odds

    console.print("\n[bold]Fetching live MLB odds...[/bold]\n")
    count = ingest_odds()
    if count > 0:
        console.print(f"[green]Stored {count} odds snapshots.[/green]\n")
    else:
        console.print("[yellow]No odds fetched. Check your ODDS_API__API_KEY in .env[/yellow]\n")


@app.command()
def statcast(
    season: int = typer.Option(2025, "--season", "-s", help="Season year"),
    force: bool = typer.Option(False, "--force", "-f", help="Force refresh from FanGraphs"),
):
    """Ingest Statcast/FanGraphs pitching and batting leaderboards."""
    from bbbot.ingest.statcast import ingest_pitcher_statcast, ingest_batter_statcast

    console.print(f"\n[bold]Ingesting FanGraphs/Statcast data for {season}...[/bold]\n")

    with console.status("[bold green]Fetching pitching leaderboard..."):
        pitchers = ingest_pitcher_statcast(season, force_refresh=force)
    console.print(f"[green]Ingested {pitchers} pitcher Statcast records.[/green]")

    with console.status("[bold green]Fetching batting leaderboard..."):
        batters = ingest_batter_statcast(season, force_refresh=force)
    console.print(f"[green]Ingested {batters} batter Statcast records.[/green]\n")


def _show_games(game_date: date):
    """Display games for a date in a rich table."""
    init_db()
    session = get_session()
    try:
        games = get_games_by_date(session, game_date)
        if not games:
            console.print("[yellow]No games found.[/yellow]")
            return

        table = Table(title=f"MLB Games — {game_date}")
        table.add_column("Time (UTC)", style="dim")
        table.add_column("Away", style="bold")
        table.add_column("Home", style="bold")
        table.add_column("Away SP")
        table.add_column("Home SP")
        table.add_column("Score", justify="center")
        table.add_column("Status")

        for game in games:
            time_str = game.game_time_utc.strftime("%H:%M") if game.game_time_utc else "TBD"
            away_name = game.away_team.abbreviation if game.away_team else "?"
            home_name = game.home_team.abbreviation if game.home_team else "?"
            away_sp = game.away_sp.name if game.away_sp else "TBD"
            home_sp = game.home_sp.name if game.home_sp else "TBD"

            if game.status == "final" and game.home_score is not None:
                score = f"{game.away_score}-{game.home_score}"
            else:
                score = "-"

            status_styles = {
                "final": "green",
                "live": "red bold",
                "scheduled": "dim",
                "postponed": "yellow",
            }
            status_style = status_styles.get(game.status)
            if status_style:
                status_text = f"[{status_style}]{game.status}[/{status_style}]"
            else:
                status_text = game.status

            table.add_row(
                time_str, away_name, home_name,
                away_sp, home_sp, score, status_text
            )

        console.print(table)
    finally:
        session.close()
