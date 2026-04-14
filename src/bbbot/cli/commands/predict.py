"""CLI commands for generating predictions — matches dashboard format."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from bbbot.betting.odds_math import american_to_decimal, calculate_ev
from bbbot.db.engine import get_session, init_db
from bbbot.db.models import OddsSnapshot
from bbbot.db.queries import get_games_by_date
from bbbot.features.builder import build_game_features, create_default_registry
from bbbot.ingest.schedule import DailyIngestor
from bbbot.models.baseline import BaselineRunsModel, BaselineWinModel
from bbbot.reports.daily_card import render_daily_card
from bbbot.reports.exporters import export_csv, export_html, export_json

console = Console()
app = typer.Typer(help="Prediction commands")


def _parse_date(date_str: str | None) -> date:
    if date_str is None:
        return date.today()
    return datetime.strptime(date_str, "%Y-%m-%d").date()


@app.command()
def today(
    date_str: str = typer.Option(None, "--date", "-d", help="Date (YYYY-MM-DD), default today"),
    export: str = typer.Option(None, "--export", "-e", help="Export format: csv, json, html, or all"),
):
    """Generate predictions for today's MLB games."""
    game_date = _parse_date(date_str)
    console.print(f"\n[bold]Generating predictions for {game_date}...[/bold]")

    # Make sure schedule is loaded
    ingestor = DailyIngestor()
    ingestor.ingest_schedule(game_date)

    init_db()
    session = get_session()

    try:
        games = get_games_by_date(session, game_date)
        if not games:
            console.print("[yellow]No games found for this date.[/yellow]")
            return

        # Build features
        registry = create_default_registry()

        # Try to load trained models; fall back to baseline
        from bbbot.models.training import load_trained_model
        win_model = load_trained_model("win_probability")
        runs_model_trained = load_trained_model("run_total")

        if win_model:
            console.print("[dim]Using trained win probability model[/dim]")
        else:
            win_model = BaselineWinModel()
            console.print("[dim]Using baseline win model (run `bbbot train all` for better predictions)[/dim]")

        if runs_model_trained:
            runs_model = runs_model_trained
            console.print("[dim]Using trained run total model[/dim]")
        else:
            runs_model = BaselineRunsModel()

        predictions = []
        for game in games:
            try:
                features = build_game_features(session, game, registry)
                feature_df = pd.DataFrame([features])

                # Win probability
                home_win_prob = float(win_model.predict_proba(feature_df)[0])
                away_win_prob = 1.0 - home_win_prob

                # Run totals
                run_preds = runs_model.predict(feature_df)[0]
                home_runs = float(run_preds[0])
                away_runs = float(run_preds[1])
                total = home_runs + away_runs

                # Model's winner
                if home_win_prob >= 0.5:
                    pick_team = game.home_team.abbreviation
                    pick_prob = home_win_prob
                    pick_side = "home"
                else:
                    pick_team = game.away_team.abbreviation
                    pick_prob = away_win_prob
                    pick_side = "away"

                # Confidence
                edge = abs(home_win_prob - 0.5)
                if edge >= 0.15:
                    confidence = "HIGH"
                elif edge >= 0.08:
                    confidence = "MED"
                else:
                    confidence = "LOW"

                # O/U default
                real_total_line = 8.5
                if hasattr(runs_model, 'predict_over_under'):
                    over_probs, under_probs = runs_model.predict_over_under(feature_df, line=real_total_line)
                elif hasattr(runs_model, 'predict_total_probs'):
                    over_probs, under_probs = runs_model.predict_total_probs(feature_df, line=real_total_line)
                else:
                    over_probs = np.array([0.5])
                    under_probs = np.array([0.5])
                ou_pick = "Over" if float(over_probs[0]) > 0.5 else "Under"
                ou_prob = float(over_probs[0]) if ou_pick == "Over" else float(under_probs[0])

                # Check for Kalshi edge — pregame odds only (first snapshot captured)
                kalshi_edge = None
                kalshi_snaps = session.query(OddsSnapshot).filter(
                    OddsSnapshot.game_id == game.id,
                    OddsSnapshot.sportsbook == "kalshi",
                    OddsSnapshot.market_type == "h2h",
                ).order_by(OddsSnapshot.captured_at.asc()).first()  # first = pregame

                if kalshi_snaps:
                    kalshi_odds = kalshi_snaps.home_line if pick_side == "home" else kalshi_snaps.away_line
                    if kalshi_odds is not None:
                        kalshi_implied = american_to_decimal(kalshi_odds)
                        kalshi_ev = calculate_ev(pick_prob, kalshi_implied)
                        if kalshi_ev > 0.02:  # only flag if >2% edge
                            kalshi_edge = {
                                "ev": kalshi_ev,
                                "odds": kalshi_odds,
                                "model_prob": pick_prob,
                                "implied_prob": 1 / kalshi_implied,
                            }

                    # Also get Kalshi total line if available (pregame)
                    kalshi_totals = session.query(OddsSnapshot).filter(
                        OddsSnapshot.game_id == game.id,
                        OddsSnapshot.sportsbook == "kalshi",
                        OddsSnapshot.market_type == "totals",
                    ).order_by(OddsSnapshot.captured_at.asc()).first()  # first = pregame
                    if kalshi_totals and kalshi_totals.total_line:
                        real_total_line = kalshi_totals.total_line
                        # Recompute O/U with Kalshi's line
                        if hasattr(runs_model, 'predict_over_under'):
                            over_probs, under_probs = runs_model.predict_over_under(feature_df, line=real_total_line)
                        elif hasattr(runs_model, 'predict_total_probs'):
                            over_probs, under_probs = runs_model.predict_total_probs(feature_df, line=real_total_line)
                        ou_pick = "Over" if float(over_probs[0]) > 0.5 else "Under"
                        ou_prob = float(over_probs[0]) if ou_pick == "Over" else float(under_probs[0])

                time_str = (game.game_time_utc.strftime("%I:%M %p UTC")
                           if game.game_time_utc else "TBD")

                predictions.append({
                    "game_id": game.id,
                    "game_time": time_str,
                    "away_team": game.away_team.abbreviation,
                    "home_team": game.home_team.abbreviation,
                    "away_sp": game.away_sp.name if game.away_sp else "TBD",
                    "home_sp": game.home_sp.name if game.home_sp else "TBD",
                    "home_win_prob": home_win_prob,
                    "away_win_prob": away_win_prob,
                    "home_runs_pred": home_runs,
                    "away_runs_pred": away_runs,
                    "total_pred": total,
                    "pick_team": pick_team,
                    "pick_prob": pick_prob,
                    "confidence": confidence,
                    "ou_pick": ou_pick,
                    "ou_prob": ou_prob,
                    "ou_line": real_total_line,
                    "kalshi_edge": kalshi_edge,
                    "status": game.status,
                    "home_score": game.home_score,
                    "away_score": game.away_score,
                })
            except Exception as e:
                console.print(f"[yellow]Error predicting {game.away_team.abbreviation} @ {game.home_team.abbreviation}: {e}[/yellow]")
                continue

        # Sort by confidence then probability
        conf_order = {"HIGH": 0, "MED": 1, "LOW": 2}
        predictions.sort(key=lambda p: (conf_order[p["confidence"]], -p["pick_prob"]))

        render_daily_card(predictions, game_date, console)

        # Export if requested
        if export:
            formats = export.lower().split(",") if export != "all" else ["csv", "json", "html"]
            for fmt in formats:
                fmt = fmt.strip()
                if fmt == "csv":
                    path = export_csv(predictions, game_date)
                    console.print(f"[dim]Exported CSV: {path}[/dim]")
                elif fmt == "json":
                    path = export_json(predictions, game_date)
                    console.print(f"[dim]Exported JSON: {path}[/dim]")
                elif fmt == "html":
                    path = export_html(predictions, game_date)
                    console.print(f"[dim]Exported HTML: {path}[/dim]")

    finally:
        session.close()
