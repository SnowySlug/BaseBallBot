"""CLI commands for model training."""

from datetime import date

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Model training commands")


@app.command("all")
def train_all_cmd(
    season: int = typer.Option(None, "--season", "-s",
                               help="Season year (default: current year)"),
):
    """Train all models (win probability + run total) on a season's data."""
    if season is None:
        season = date.today().year

    console.print(f"\n[bold]Training all models for {season} season...[/bold]\n")

    from bbbot.models.training import train_all

    with console.status("[bold green]Training models..."):
        results = train_all(season)

    if "error" in results:
        console.print(f"[red]Error: {results['error']}[/red]")
        console.print(f"[dim]Samples available: {results.get('samples', 0)}[/dim]")
        console.print("[dim]Run `bbbot ingest backfill` to load more data first.[/dim]")
        return

    # Display results
    if "win_model" in results:
        wm = results["win_model"]
        console.print("[green]Win Probability Model:[/green]")
        console.print(f"  Accuracy:    {wm.get('accuracy', 0):.1%}")
        console.print(f"  Log Loss:    {wm.get('log_loss', 0):.4f}")
        console.print(f"  Brier Score: {wm.get('brier_score', 0):.4f}")

    if "run_model" in results:
        rm = results["run_model"]
        console.print(f"\n[green]Run Total Model:[/green]")
        console.print(f"  Home MAE: {rm.get('home_mae', 0):.2f} runs")
        console.print(f"  Away MAE: {rm.get('away_mae', 0):.2f} runs")
        console.print(f"  Total MAE: {rm.get('total_mae', 0):.2f} runs")

    console.print(f"\n[green]Models saved to data/models/[/green]\n")


@app.command("evaluate")
def evaluate_cmd(
    season: int = typer.Option(None, "--season", "-s"),
):
    """Evaluate trained models without retraining."""
    from bbbot.models.training import load_trained_model

    if season is None:
        season = date.today().year

    console.print(f"\n[bold]Evaluating models for {season}...[/bold]\n")

    win_model = load_trained_model("win_probability")
    run_model = load_trained_model("run_total")

    if win_model is None and run_model is None:
        console.print("[yellow]No trained models found. Run `bbbot train all` first.[/yellow]")
        return

    if win_model:
        importance = win_model.get_feature_importance()
        if importance is not None:
            console.print("[bold]Top 15 Feature Importance (Win Model):[/bold]")
            for feat, imp in importance.head(15).items():
                bar = "=" * int(imp * 200)
                console.print(f"  {feat:40s} {imp:.4f} {bar}")
    console.print()
