"""BBBot MLB Prediction Dashboard — Streamlit app."""

import os
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Load Streamlit Cloud secrets into env vars (so pydantic-settings picks them up)
# ---------------------------------------------------------------------------
try:
    for key in st.secrets:
        os.environ.setdefault(key, str(st.secrets[key]))
except Exception:
    pass  # No secrets configured

from bbbot.db.engine import get_session, init_db
from bbbot.db.models import (
    Game, Team, Player, OddsSnapshot, TeamBattingDaily,
    PitcherGameLog, StatcastPitcherMetrics, StatcastBatterMetrics,
)
from bbbot.betting.odds_math import american_to_decimal, calculate_ev
from bbbot.betting.kelly import fractional_kelly, kelly_to_units
from bbbot.features.builder import build_game_features, create_default_registry
from bbbot.ingest.odds_ingest import get_best_odds_for_game, get_kalshi_odds_for_game

# ---------------------------------------------------------------------------
# Team colors for charts and cards
# ---------------------------------------------------------------------------
TEAM_COLORS = {
    "AZ": "#A71930", "ATL": "#CE1141", "BAL": "#DF4601", "BOS": "#BD3039",
    "CHC": "#0E3386", "CWS": "#27251F", "CIN": "#C6011F", "CLE": "#00385D",
    "COL": "#333366", "DET": "#0C2340", "HOU": "#002D62", "KC": "#004687",
    "LAA": "#BA0021", "LAD": "#005A9C", "MIA": "#00A3E0", "MIL": "#FFC52F",
    "MIN": "#002B5C", "NYM": "#002D72", "NYY": "#003087", "OAK": "#003831",
    "PHI": "#E81828", "PIT": "#FDB827", "SD": "#2F241D", "SF": "#FD5A1E",
    "SEA": "#0C2C56", "STL": "#C41E3A", "TB": "#092C5C", "TEX": "#003278",
    "TOR": "#134A8E", "WSH": "#AB0003",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BBBot - MLB Predictions",
    page_icon="baseball",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp { background-color: #0e1117; }

    .game-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .game-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }

    .tier-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1px;
    }
    .tier-A { background: #00c853; color: #000; }
    .tier-B { background: #ffd600; color: #000; }
    .tier-C { background: #ff9100; color: #000; }
    .tier-D { background: #616161; color: #fff; }

    .stat-box {
        background: #1e2433;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2d3548;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #e0e0e0;
    }
    .stat-label {
        font-size: 12px;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .positive-ev { color: #00e676; }
    .negative-ev { color: #ff5252; }

    .matchup-vs {
        font-size: 14px;
        color: #5c6878;
        font-weight: 400;
    }
    .team-name {
        font-size: 22px;
        font-weight: 700;
    }

    .bet-rec {
        background: linear-gradient(135deg, #1b5e20 0%, #0d3311 100%);
        border: 1px solid #2e7d32;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
    }

    .header-gradient {
        background: linear-gradient(90deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }

    div[data-testid="stMetric"] {
        background: #1e2433;
        border: 1px solid #2d3548;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session init — auto-seed and auto-ingest on first load
# ---------------------------------------------------------------------------
@st.cache_resource
def get_db():
    init_db()
    return True

get_db()


@st.cache_data(ttl=600, show_spinner="Loading MLB data...")
def ensure_data(game_date: date):
    """Seed teams/parks and ingest schedule + odds if DB is empty."""
    session = get_session()
    try:
        team_count = session.query(Team).count()
        if team_count == 0:
            from bbbot.db.seed import seed_all
            seed_all(session)

        game_count = session.query(Game).filter(Game.game_date == game_date).count()
        if game_count == 0:
            from bbbot.ingest.schedule import DailyIngestor
            ingestor = DailyIngestor()
            ingestor.ingest_schedule(game_date)

        odds_count = session.query(OddsSnapshot).join(Game).filter(
            Game.game_date == game_date
        ).count()
        if odds_count == 0:
            try:
                from bbbot.ingest.odds_ingest import ingest_odds
                ingest_odds()
            except Exception:
                pass  # Odds API key may not be configured
    finally:
        session.close()


def get_sess():
    return get_session()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## BBBot")
st.sidebar.markdown("*MLB Prediction Engine*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Today's Predictions", "Game Explorer", "Model Performance", "Team Stats"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    "<div style='text-align:center; color:#5c6878; font-size:12px;'>"
    "Powered by XGBoost + LightGBM<br>Baseball Savant + MLB Stats API"
    "</div>",
    unsafe_allow_html=True,
)


# ===================================================================
# PAGE: Today's Predictions
# ===================================================================
if page == "Today's Predictions":
    st.markdown(
        "<div class='header-gradient'>"
        "<h1 style='margin:0; color:white;'>Today's Game Predictions</h1>"
        "<p style='margin:4px 0 0 0; color:#90caf9;'>Who wins, predicted scores, and run totals</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    game_date = st.date_input("Game Date", value=date.today())

    # Auto-fetch data for selected date
    ensure_data(game_date)

    session = get_sess()
    try:
        games = session.query(Game).filter(
            Game.game_date == game_date,
        ).order_by(Game.game_time_utc).all()

        if not games:
            st.warning(f"No games found for {game_date}. There may be no MLB games scheduled.")
        else:
            from bbbot.models.training import load_trained_model
            from bbbot.models.baseline import BaselineWinModel, BaselineRunsModel

            win_model = load_trained_model("win_probability")
            runs_model = load_trained_model("run_total")
            using_trained = win_model is not None

            if not win_model:
                win_model = BaselineWinModel()
            if not runs_model:
                runs_model = BaselineRunsModel()

            if using_trained:
                st.caption("Model: XGBoost/LightGBM ensemble trained on 2,400+ games")
            else:
                st.info("Using baseline model — run `bbbot train all` for better predictions.")

            registry = create_default_registry()
            predictions = []

            for game in games:
                try:
                    features = build_game_features(session, game, registry)
                    feature_df = pd.DataFrame([features])

                    home_win_prob = float(win_model.predict_proba(feature_df)[0])
                    away_win_prob = 1.0 - home_win_prob

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

                    # O/U
                    real_total_line = 8.5  # default
                    if hasattr(runs_model, 'predict_over_under'):
                        over_probs, under_probs = runs_model.predict_over_under(feature_df, line=real_total_line)
                    elif hasattr(runs_model, 'predict_total_probs'):
                        over_probs, under_probs = runs_model.predict_total_probs(feature_df, line=real_total_line)
                    else:
                        over_probs = np.array([0.5])
                        under_probs = np.array([0.5])
                    ou_pick = "Over" if float(over_probs[0]) > 0.5 else "Under"
                    ou_prob = float(over_probs[0]) if ou_pick == "Over" else float(under_probs[0])

                    # Check for Kalshi edge (only Kalshi)
                    kalshi_edge = None
                    kalshi_snaps = session.query(OddsSnapshot).filter(
                        OddsSnapshot.game_id == game.id,
                        OddsSnapshot.sportsbook == "kalshi",
                        OddsSnapshot.market_type == "h2h",
                    ).order_by(OddsSnapshot.captured_at.desc()).first()

                    if kalshi_snaps:
                        # Use the line for the side the model picks
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
                        # Also get Kalshi total line if available
                        kalshi_totals = session.query(OddsSnapshot).filter(
                            OddsSnapshot.game_id == game.id,
                            OddsSnapshot.sportsbook == "kalshi",
                            OddsSnapshot.market_type == "totals",
                        ).order_by(OddsSnapshot.captured_at.desc()).first()
                        if kalshi_totals and kalshi_totals.total_line:
                            real_total_line = kalshi_totals.total_line
                            # Recompute O/U with Kalshi's line
                            if hasattr(runs_model, 'predict_over_under'):
                                over_probs, under_probs = runs_model.predict_over_under(feature_df, line=real_total_line)
                            elif hasattr(runs_model, 'predict_total_probs'):
                                over_probs, under_probs = runs_model.predict_total_probs(feature_df, line=real_total_line)
                            ou_pick = "Over" if float(over_probs[0]) > 0.5 else "Under"
                            ou_prob = float(over_probs[0]) if ou_pick == "Over" else float(under_probs[0])

                    predictions.append({
                        "game": game,
                        "home_win_prob": home_win_prob,
                        "away_win_prob": away_win_prob,
                        "home_runs": home_runs,
                        "away_runs": away_runs,
                        "total": total,
                        "pick_team": pick_team,
                        "pick_prob": pick_prob,
                        "confidence": confidence,
                        "ou_pick": ou_pick,
                        "ou_prob": ou_prob,
                        "ou_line": real_total_line,
                        "kalshi_edge": kalshi_edge,
                    })
                except Exception as e:
                    st.warning(f"Error predicting {game.away_team.abbreviation} @ {game.home_team.abbreviation}: {e}")
                    continue

            # Sort by confidence
            conf_order = {"HIGH": 0, "MED": 1, "LOW": 2}
            predictions.sort(key=lambda p: (conf_order[p["confidence"]], -p["pick_prob"]))

            # Summary
            high_conf = sum(1 for p in predictions if p["confidence"] == "HIGH")
            edges = sum(1 for p in predictions if p["kalshi_edge"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Games Today", len(predictions))
            m2.metric("High Confidence Picks", high_conf)
            m3.metric("Kalshi Edges Found", edges)

            st.divider()

            # Render each game
            for pred in predictions:
                game = pred["game"]
                away = game.away_team.abbreviation
                home = game.home_team.abbreviation
                away_sp = game.away_sp.name if game.away_sp else "TBD"
                home_sp = game.home_sp.name if game.home_sp else "TBD"
                time_str = game.game_time_utc.strftime("%I:%M %p UTC") if game.game_time_utc else "TBD"

                # Header row
                hdr = f"### {away} @ {home}"
                if game.status == "final" and game.away_score is not None:
                    actual_winner = home if game.home_score > game.away_score else away
                    if actual_winner == pred["pick_team"]:
                        hdr += f" — Final: {game.away_score}-{game.home_score} :green[Model Correct]"
                    else:
                        hdr += f" — Final: {game.away_score}-{game.home_score} :red[Model Wrong]"
                elif game.status == "live":
                    hdr += " — :red[LIVE]"
                else:
                    hdr += f" — {time_str}"
                st.markdown(hdr)
                st.caption(f"{away_sp} vs {home_sp}")

                # Main columns
                c1, c2, c3 = st.columns(3)

                with c1:
                    conf_tag = {"HIGH": ":green", "MED": ":orange", "LOW": ":red"}[pred["confidence"]]
                    st.markdown(f"**Predicted Winner**")
                    st.markdown(f"## {conf_tag}[{pred['pick_team']}]")
                    st.markdown(f"{pred['pick_prob']:.0%} win probability — **{pred['confidence']}** confidence")

                with c2:
                    st.markdown("**Predicted Score**")
                    st.markdown(f"## {pred['away_runs']:.1f} - {pred['home_runs']:.1f}")
                    st.caption(f"{away} - {home}")

                with c3:
                    st.markdown("**Run Total**")
                    st.markdown(f"## {pred['total']:.1f} runs")
                    st.markdown(f"Model says **{pred['ou_pick']}** {pred['ou_line']} ({pred['ou_prob']:.0%})")

                # Kalshi edge callout — only shows when there's a real edge
                if pred["kalshi_edge"]:
                    e = pred["kalshi_edge"]
                    k_odds = e["odds"]
                    k_odds_str = f"+{k_odds:.0f}" if k_odds > 0 else f"{k_odds:.0f}"
                    st.success(
                        f"**Kalshi Edge:** Model gives {pred['pick_team']} a "
                        f"{e['model_prob']:.0%} chance, but Kalshi implies "
                        f"{e['implied_prob']:.0%} ({k_odds_str}). "
                        f"That's a **{e['ev']:+.1%}** edge."
                    )

                st.divider()

    finally:
        session.close()


# ===================================================================
# PAGE: Game Explorer
# ===================================================================
elif page == "Game Explorer":
    st.markdown(
        "<div class='header-gradient'>"
        "<h1 style='margin:0; color:white;'>Game Explorer</h1>"
        "<p style='margin:4px 0 0 0; color:#90caf9;'>Browse historical games, scores, and results</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=date.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("To", value=date.today())

    session = get_sess()
    try:
        games = session.query(Game).filter(
            Game.game_date >= start_date,
            Game.game_date <= end_date,
        ).order_by(Game.game_date.desc(), Game.game_time_utc).all()

        if not games:
            st.info("No games found in this range.")
        else:
            st.metric("Games Found", len(games))

            rows = []
            for g in games:
                rows.append({
                    "Date": g.game_date,
                    "Away": g.away_team.abbreviation if g.away_team else "?",
                    "Home": g.home_team.abbreviation if g.home_team else "?",
                    "Away SP": g.away_sp.name if g.away_sp else "TBD",
                    "Home SP": g.home_sp.name if g.home_sp else "TBD",
                    "Away Score": g.away_score if g.away_score is not None else "",
                    "Home Score": g.home_score if g.home_score is not None else "",
                    "Total": g.total_runs if g.total_runs is not None else "",
                    "Status": g.status,
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=500)

            # Runs distribution
            final_games = [g for g in games if g.status == "final" and g.total_runs is not None]
            if final_games:
                totals = [g.total_runs for g in final_games]
                fig = px.histogram(
                    x=totals, nbins=20,
                    labels={"x": "Total Runs", "y": "Games"},
                    title="Run Distribution",
                    color_discrete_sequence=["#2196f3"],
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
    finally:
        session.close()


# ===================================================================
# PAGE: Model Performance
# ===================================================================
elif page == "Model Performance":
    st.markdown(
        "<div class='header-gradient'>"
        "<h1 style='margin:0; color:white;'>Model Performance</h1>"
        "<p style='margin:4px 0 0 0; color:#90caf9;'>Tracking accuracy from today onwards</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Load model metadata
    import json
    from pathlib import Path

    win_meta_path = Path("data/models/win_probability/latest/metadata.json")
    run_meta_path = Path("data/models/run_total/latest/metadata.json")

    if win_meta_path.exists():
        with open(win_meta_path) as f:
            win_meta = json.load(f)

        st.subheader("Win Probability Model")
        m1, m2, m3, m4 = st.columns(4)
        metrics = win_meta.get("metrics", {})
        m1.metric("Training Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        m2.metric("Log Loss", f"{metrics.get('log_loss', 0):.4f}")
        m3.metric("Brier Score", f"{metrics.get('brier_score', 0):.4f}")
        m4.metric("Training Samples", f"{win_meta.get('n_samples', 0):,}")

        # Feature importance
        fi_path = Path("data/models/win_probability/latest/feature_importance.csv")
        if fi_path.exists():
            fi_df = pd.read_csv(fi_path)
            if "feature" in fi_df.columns and "importance" in fi_df.columns:
                top20 = fi_df.nlargest(20, "importance")
                fig = px.bar(
                    top20, x="importance", y="feature", orientation="h",
                    title="Top 20 Feature Importance",
                    color="importance",
                    color_continuous_scale="blues",
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
    else:
        st.warning("No trained win model found. Run `bbbot train all --season 2025`.")

    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)

        st.subheader("Run Total Model")
        metrics = run_meta.get("metrics", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Home MAE", f"{metrics.get('home_mae', 0):.2f} runs")
        m2.metric("Away MAE", f"{metrics.get('away_mae', 0):.2f} runs")
        m3.metric("Total MAE", f"{metrics.get('total_mae', 0):.2f} runs")
    else:
        st.warning("No trained run model found.")

    # Live accuracy — only games from today (2026-04-13) onwards
    st.divider()
    st.subheader("Live Accuracy (2026 Season — Today Onwards)")
    st.caption("Tracking model predictions against actual results starting April 13, 2026")

    tracking_start = date(2026, 4, 13)

    session = get_sess()
    try:
        # Load models for retroactive predictions on completed games
        from bbbot.models.training import load_trained_model
        from bbbot.models.baseline import BaselineWinModel, BaselineRunsModel

        win_model = load_trained_model("win_probability")
        runs_model = load_trained_model("run_total")
        if not win_model:
            win_model = BaselineWinModel()
        if not runs_model:
            runs_model = BaselineRunsModel()

        registry = create_default_registry()

        final_games = session.query(Game).filter(
            Game.game_date >= tracking_start,
            Game.status == "final",
            Game.home_score.isnot(None),
        ).order_by(Game.game_date).all()

        if not final_games:
            st.info("No completed games since tracking started. Results will appear here as games finish.")
        else:
            correct = 0
            wrong = 0
            total_run_error = []
            results_by_date = {}

            for game in final_games:
                try:
                    features = build_game_features(session, game, registry)
                    feature_df = pd.DataFrame([features])

                    home_win_prob = float(win_model.predict_proba(feature_df)[0])
                    predicted_winner = game.home_team.abbreviation if home_win_prob >= 0.5 else game.away_team.abbreviation
                    actual_winner = game.home_team.abbreviation if game.home_score > game.away_score else game.away_team.abbreviation

                    if predicted_winner == actual_winner:
                        correct += 1
                    else:
                        wrong += 1

                    run_preds = runs_model.predict(feature_df)[0]
                    pred_total = float(run_preds[0]) + float(run_preds[1])
                    actual_total = game.home_score + game.away_score
                    total_run_error.append(abs(pred_total - actual_total))

                    d = game.game_date
                    if d not in results_by_date:
                        results_by_date[d] = {"correct": 0, "wrong": 0}
                    if predicted_winner == actual_winner:
                        results_by_date[d]["correct"] += 1
                    else:
                        results_by_date[d]["wrong"] += 1
                except Exception:
                    continue

            total_games = correct + wrong
            accuracy = correct / total_games if total_games > 0 else 0
            avg_run_err = np.mean(total_run_error) if total_run_error else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games Tracked", total_games)
            m2.metric("Correct Picks", correct)
            m3.metric("Win Accuracy", f"{accuracy:.1%}")
            m4.metric("Avg Run Error", f"{avg_run_err:.1f}")

            # Accuracy over time chart
            if results_by_date:
                chart_data = []
                running_correct = 0
                running_total = 0
                for d in sorted(results_by_date.keys()):
                    running_correct += results_by_date[d]["correct"]
                    running_total += results_by_date[d]["correct"] + results_by_date[d]["wrong"]
                    chart_data.append({
                        "date": d,
                        "daily_accuracy": results_by_date[d]["correct"] / (results_by_date[d]["correct"] + results_by_date[d]["wrong"]),
                        "cumulative_accuracy": running_correct / running_total,
                    })

                chart_df = pd.DataFrame(chart_data)
                fig = px.line(
                    chart_df, x="date", y="cumulative_accuracy",
                    title="Cumulative Win Prediction Accuracy",
                    labels={"cumulative_accuracy": "Accuracy", "date": "Date"},
                    color_discrete_sequence=["#00e676"],
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="#ff5252",
                              annotation_text="Coin Flip (50%)")
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(tickformat=".0%", range=[0, 1]),
                )
                st.plotly_chart(fig, use_container_width=True)
    finally:
        session.close()


# ===================================================================
# PAGE: Team Stats
# ===================================================================
elif page == "Team Stats":
    current_season = date.today().year

    st.markdown(
        "<div class='header-gradient'>"
        f"<h1 style='margin:0; color:white;'>{current_season} Team Stats</h1>"
        "<p style='margin:4px 0 0 0; color:#90caf9;'>Current season records and batting stats</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    session = get_sess()
    try:
        teams = session.query(Team).order_by(Team.division, Team.name).all()

        team_stats = []
        for team in teams:
            wins = session.query(Game).filter(
                Game.winning_team_id == team.id,
                Game.status == "final",
                Game.season == current_season,
            ).count()
            losses = session.query(Game).filter(
                Game.status == "final",
                Game.season == current_season,
                ((Game.home_team_id == team.id) | (Game.away_team_id == team.id)),
                Game.winning_team_id != team.id,
                Game.winning_team_id.isnot(None),
            ).count()

            # This season's batting only
            season_batting = session.query(TeamBattingDaily).filter(
                TeamBattingDaily.team_id == team.id,
                TeamBattingDaily.game_date >= date(current_season, 1, 1),
            ).order_by(TeamBattingDaily.game_date.desc()).all()

            avg_runs = np.mean([b.runs for b in season_batting if b.runs is not None]) if season_batting else 0
            avg_hits = np.mean([b.hits for b in season_batting if b.hits is not None]) if season_batting else 0
            avg_hr = np.mean([b.home_runs for b in season_batting if b.home_runs is not None]) if season_batting else 0
            avg_k = np.mean([b.strikeouts for b in season_batting if b.strikeouts is not None]) if season_batting else 0

            pct = wins / (wins + losses) if (wins + losses) > 0 else 0

            team_stats.append({
                "Team": team.abbreviation,
                "Division": team.division,
                "W": wins,
                "L": losses,
                "PCT": pct,
                "R/G": round(avg_runs, 1),
                "H/G": round(avg_hits, 1),
                "HR/G": round(avg_hr, 2),
                "K/G": round(avg_k, 1),
                "Games": len(season_batting),
            })

        df = pd.DataFrame(team_stats).sort_values("PCT", ascending=False)

        # Division tabs
        divisions = sorted(df["Division"].unique())
        tabs = st.tabs(["All Teams"] + divisions)

        with tabs[0]:
            st.dataframe(
                df.style.format({"PCT": "{:.3f}"}),
                use_container_width=True,
                height=700,
            )

        for i, div in enumerate(divisions):
            with tabs[i + 1]:
                div_df = df[df["Division"] == div].reset_index(drop=True)
                st.dataframe(
                    div_df.style.format({"PCT": "{:.3f}"}),
                    use_container_width=True,
                )

        # Runs per game chart
        st.divider()
        fig = px.bar(
            df.sort_values("R/G", ascending=True),
            x="R/G", y="Team", orientation="h",
            title=f"{current_season} Runs Per Game",
            color="R/G",
            color_continuous_scale="reds",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=700,
        )
        st.plotly_chart(fig, use_container_width=True)

    finally:
        session.close()
