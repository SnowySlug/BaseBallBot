"""CLI commands for reports and analysis."""

from datetime import date, datetime, timedelta

import typer
from rich.console import Console

from bbbot.db.engine import get_session, init_db
from bbbot.reports.performance import render_performance_report

console = Console()
app = typer.Typer(help="Report and analysis commands")


def _parse_date(date_str: str | None) -> date:
    if date_str is None:
        return date.today()
    return datetime.strptime(date_str, "%Y-%m-%d").date()


@app.command()
def performance(
    from_date: str = typer.Option(None, "--from", help="Start date (default: 30 days ago)"),
    to_date: str = typer.Option(None, "--to", help="End date (default: today)"),
):
    """Show prediction performance report."""
    end = _parse_date(to_date) if to_date else date.today()
    start = _parse_date(from_date) if from_date else end - timedelta(days=30)

    init_db()
    session = get_session()
    try:
        render_performance_report(session, start, end, console)
    finally:
        session.close()


@app.command()
def standings(
    date_str: str = typer.Option(None, "--date", "-d", help="Date (default: today)"),
):
    """Show current MLB standings with model context."""
    from rich.table import Table
    from bbbot.db.models import Game, Team
    from sqlalchemy import func, case, and_

    game_date = _parse_date(date_str)
    init_db()
    session = get_session()

    try:
        # Calculate W-L for each team from game results
        teams = list(session.query(Team).order_by(Team.division, Team.name).all())

        # Group by division
        divisions: dict[str, list] = {}
        for team in teams:
            div = team.division
            if div not in divisions:
                divisions[div] = []

            # Count wins/losses
            wins = session.query(func.count()).filter(
                Game.winning_team_id == team.id,
                Game.status == "final",
                Game.season == game_date.year,
            ).scalar() or 0

            losses_home = session.query(func.count()).filter(
                Game.home_team_id == team.id,
                Game.winning_team_id != team.id,
                Game.winning_team_id.isnot(None),
                Game.status == "final",
                Game.season == game_date.year,
            ).scalar() or 0

            losses_away = session.query(func.count()).filter(
                Game.away_team_id == team.id,
                Game.winning_team_id != team.id,
                Game.winning_team_id.isnot(None),
                Game.status == "final",
                Game.season == game_date.year,
            ).scalar() or 0

            losses = losses_home + losses_away
            pct = wins / (wins + losses) if (wins + losses) > 0 else 0

            divisions[div].append((team, wins, losses, pct))

        # Sort each division by win pct
        for div in divisions:
            divisions[div].sort(key=lambda x: x[3], reverse=True)

        # Render
        for div_name in sorted(divisions.keys()):
            table = Table(title=div_name)
            table.add_column("Team", style="bold", width=6)
            table.add_column("W", justify="right", width=4)
            table.add_column("L", justify="right", width=4)
            table.add_column("Pct", justify="right", width=6)

            for team, w, l, pct in divisions[div_name]:
                table.add_row(
                    team.abbreviation,
                    str(w), str(l),
                    f"{pct:.3f}",
                )
            console.print(table)

    finally:
        session.close()
