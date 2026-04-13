"""Export predictions to CSV, JSON, and HTML."""

import json
from datetime import date
from pathlib import Path

import structlog

log = structlog.get_logger()

EXPORT_DIR = Path("data/exports")


def _ensure_export_dir():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def export_csv(predictions: list[dict], game_date: date,
               output_path: Path | None = None) -> Path:
    """Export predictions to CSV."""
    import csv

    _ensure_export_dir()
    path = output_path or EXPORT_DIR / f"predictions_{game_date}.csv"

    fieldnames = [
        "game_time", "away_team", "home_team", "away_sp", "home_sp",
        "home_win_prob", "away_win_prob", "home_runs_pred", "away_runs_pred",
        "total_pred", "over_prob", "under_prob", "confidence_tier",
        "recommended_bet", "best_ev", "recommended_units",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for pred in predictions:
            row = {}
            for key in fieldnames:
                val = pred.get(key, "")
                if isinstance(val, float):
                    row[key] = f"{val:.4f}"
                else:
                    row[key] = val
            writer.writerow(row)

    log.info("exported_csv", path=str(path))
    return path


def export_json(predictions: list[dict], game_date: date,
                output_path: Path | None = None) -> Path:
    """Export predictions to JSON."""
    _ensure_export_dir()
    path = output_path or EXPORT_DIR / f"predictions_{game_date}.json"

    output = {
        "date": game_date.isoformat(),
        "model_version": "baseline_v0.1",
        "predictions": predictions,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    log.info("exported_json", path=str(path))
    return path


def export_html(predictions: list[dict], game_date: date,
                output_path: Path | None = None) -> Path:
    """Export predictions as a styled HTML report."""
    _ensure_export_dir()
    path = output_path or EXPORT_DIR / f"predictions_{game_date}.html"

    tier_colors = {"A": "#22c55e", "B": "#eab308", "C": "#f97316", "D": "#6b7280"}

    rows_html = ""
    for pred in predictions:
        tier = pred.get("confidence_tier", "D")
        color = tier_colors.get(tier, "#6b7280")
        hwp = pred.get("home_win_prob", 0.5)
        awp = pred.get("away_win_prob", 0.5)
        hr = pred.get("home_runs_pred", 0)
        ar = pred.get("away_runs_pred", 0)
        ev = pred.get("best_ev", 0)
        units = pred.get("recommended_units", 0)

        rows_html += f"""
        <tr>
            <td>{pred.get('game_time', 'TBD')}</td>
            <td><strong>{pred.get('away_team', '?')} @ {pred.get('home_team', '?')}</strong></td>
            <td>{pred.get('away_sp', 'TBD')} vs {pred.get('home_sp', 'TBD')}</td>
            <td>{awp:.0%} / {hwp:.0%}</td>
            <td>{ar:.1f} - {hr:.1f}</td>
            <td>{pred.get('total_pred', 0):.1f}</td>
            <td style="color: {color}; font-weight: bold;">{tier}</td>
            <td>{pred.get('recommended_bet', '-')}</td>
            <td>{ev:+.1%}</td>
            <td>{units:.1f}u</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MLB Predictions - {game_date}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #0f172a; color: #e2e8f0; }}
        h1 {{ color: #38bdf8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #1e293b; color: #94a3b8; padding: 12px 8px; text-align: left;
              font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        td {{ padding: 10px 8px; border-bottom: 1px solid #1e293b; }}
        tr:hover {{ background: #1e293b; }}
        .meta {{ color: #64748b; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <h1>MLB Predictions &mdash; {game_date}</h1>
    <p class="meta">Baseline Model v0.1 | Generated {game_date}</p>
    <table>
        <thead>
            <tr>
                <th>Time</th><th>Matchup</th><th>Pitchers</th>
                <th>Win %</th><th>Score</th><th>Total</th>
                <th>Tier</th><th>Bet</th><th>EV</th><th>Units</th>
            </tr>
        </thead>
        <tbody>{rows_html}
        </tbody>
    </table>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    log.info("exported_html", path=str(path))
    return path
