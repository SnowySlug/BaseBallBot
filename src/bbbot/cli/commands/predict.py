"""CLI commands for generating predictions."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from bbbot.betting.kelly import fractional_kelly, kelly_to_units
from bbbot.betting.odds_math import american_to_decimal, calculate_ev
from bbbot.db.engine import get_session, init_db
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
    bankroll: float = typer.Option(1000.0, "--bankroll", "-b", help="Bankroll in dollars"),
    kelly: float = typer.Option(0.25, "--kelly", "-k", help="Kelly fraction (0.25 = quarter)"),
    unit_size: float = typer.Option(100.0, "--unit", "-u", help="Unit size in dollars"),
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

            # Over/under probabilities (use 8.5 as default line)
            if hasattr(runs_model, 'predict_over_under'):
                over_probs, under_probs = runs_model.predict_over_under(
                    feature_df, line=8.5
                )
            elif hasattr(runs_model, 'predict_total_probs'):
                over_probs, under_probs = runs_model.predict_total_probs(
                    feature_df, line=8.5
                )
            else:
                over_probs = np.array([0.5])
                under_probs = np.array([0.5])

            # Get real odds if available
            from bbbot.ingest.odds_ingest import get_best_odds_for_game
            odds_info = get_best_odds_for_game(session, game.id)

            # Calculate EV against real or synthetic odds
            best_bet = "-"
            best_ev = 0.0
            rec_units = 0.0
            candidates = []

            # Home ML
            home_odds_am = odds_info.get("home_ml") or (-150 if home_win_prob > 0.5 else 130)
            home_dec = american_to_decimal(home_odds_am)
            home_ev = calculate_ev(home_win_prob, home_dec)
            if home_ev > 0:
                kf = fractional_kelly(home_win_prob, home_dec, kelly)
                units = kelly_to_units(kf, bankroll, unit_size)
                book = odds_info.get("home_ml_book", "")
                label = f"{game.home_team.abbreviation} ML"
                if book:
                    label += f" ({book})"
                candidates.append((home_ev, label, units, home_odds_am))

            # Away ML
            away_odds_am = odds_info.get("away_ml") or (130 if home_win_prob > 0.5 else -150)
            away_dec = american_to_decimal(away_odds_am)
            away_ev = calculate_ev(away_win_prob, away_dec)
            if away_ev > 0:
                kf = fractional_kelly(away_win_prob, away_dec, kelly)
                units = kelly_to_units(kf, bankroll, unit_size)
                book = odds_info.get("away_ml_book", "")
                label = f"{game.away_team.abbreviation} ML"
                if book:
                    label += f" ({book})"
                candidates.append((away_ev, label, units, away_odds_am))

            # Over/Under
            real_total = odds_info.get("total_line") or 8.5
            over_odds_am = odds_info.get("over_odds") or -110
            under_odds_am = odds_info.get("under_odds") or -110

            # Recompute O/U with actual line
            if hasattr(runs_model, 'predict_over_under'):
                over_probs, under_probs = runs_model.predict_over_under(
                    feature_df, line=real_total)
            elif hasattr(runs_model, 'predict_total_probs'):
                over_probs, under_probs = runs_model.predict_total_probs(
                    feature_df, line=real_total)

            over_dec = american_to_decimal(over_odds_am)
            over_ev = calculate_ev(float(over_probs[0]), over_dec)
            if over_ev > 0:
                kf = fractional_kelly(float(over_probs[0]), over_dec, kelly)
                units = kelly_to_units(kf, bankroll, unit_size)
                book = odds_info.get("over_book", "")
                label = f"Over {real_total}"
                if book:
                    label += f" ({book})"
                candidates.append((over_ev, label, units, over_odds_am))

            under_dec = american_to_decimal(under_odds_am)
            under_ev = calculate_ev(float(under_probs[0]), under_dec)
            if under_ev > 0:
                kf = fractional_kelly(float(under_probs[0]), under_dec, kelly)
                units = kelly_to_units(kf, bankroll, unit_size)
                book = odds_info.get("under_book", "")
                label = f"Under {real_total}"
                if book:
                    label += f" ({book})"
                candidates.append((under_ev, label, units, under_odds_am))

            # Pick the best EV bet
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_ev, best_bet, rec_units, _ = candidates[0]

            # Confidence tier
            if best_ev >= 0.05:
                tier = "A"
            elif best_ev >= 0.03:
                tier = "B"
            elif best_ev > 0:
                tier = "C"
            else:
                tier = "D"

            time_str = (game.game_time_utc.strftime("%H:%M")
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
                "over_prob": float(over_probs[0]),
                "under_prob": float(under_probs[0]),
                "confidence_tier": tier,
                "recommended_bet": best_bet,
                "best_ev": best_ev,
                "recommended_units": rec_units,
                "home_ml_ev": calculate_ev(home_win_prob, american_to_decimal(-150))
                              if home_win_prob > 0.5 else 0,
                "away_ml_ev": calculate_ev(away_win_prob, american_to_decimal(-150))
                              if away_win_prob > 0.5 else 0,
            })

        # Sort by confidence tier then EV
        tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
        predictions.sort(key=lambda p: (tier_order.get(p["confidence_tier"], 9),
                                        -p.get("best_ev", 0)))

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
