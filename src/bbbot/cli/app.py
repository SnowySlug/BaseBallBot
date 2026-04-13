"""bbbot CLI — MLB Prediction Engine."""

import structlog
import typer

from bbbot.cli.commands.backtest import app as backtest_app
from bbbot.cli.commands.ingest import app as ingest_app
from bbbot.cli.commands.predict import app as predict_app
from bbbot.cli.commands.report import app as report_app
from bbbot.cli.commands.train import app as train_app

def _level_filter(logger, method_name, event_dict):
    """Drop debug messages."""
    if method_name == "debug":
        raise structlog.DropEvent
    return event_dict


structlog.configure(
    processors=[
        _level_filter,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

app = typer.Typer(
    name="bbbot",
    help="MLB Game Outcome & Run Total Prediction Engine",
    no_args_is_help=True,
)

app.add_typer(backtest_app, name="backtest")
app.add_typer(ingest_app, name="ingest")
app.add_typer(predict_app, name="predict")
app.add_typer(report_app, name="report")
app.add_typer(train_app, name="train")


@app.command()
def version():
    """Show version."""
    from bbbot import __version__
    typer.echo(f"bbbot v{__version__}")


if __name__ == "__main__":
    app()
