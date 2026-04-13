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
    ["Today's Predictions", "Game Explorer", "Model Performance", "Team Stats", "Odds Dashboard"],
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
    # Header
    st.markdown(
        "<div class='header-gradient'>"
        "<h1 style='margin:0; color:white;'>Today's Picks</h1>"
        "<p style='margin:4px 0 0 0; color:#90caf9;'>Model predictions with Kalshi odds</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_date, col_bankroll, col_kelly = st.columns([2, 1, 1])
    with col_date:
        game_date = st.date_input("Game Date", value=date.today())
    with col_bankroll:
        bankroll = st.number_input("Bankroll ($)", value=1000, step=100)
    with col_kelly:
        kelly_frac = st.selectbox("Kelly Fraction", [0.125, 0.25, 0.5, 1.0], index=1,
                                  format_func=lambda x: f"{x:.0%} Kelly")

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
            # Load models
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
                st.success("Using trained XGBoost/LightGBM ensemble")
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

                    # Model's winner pick
                    if home_win_prob >= 0.5:
                        pick_team = game.home_team.abbreviation
                        pick_prob = home_win_prob
                        pick_side = "home"
                    else:
                        pick_team = game.away_team.abbreviation
                        pick_prob = away_win_prob
                        pick_side = "away"

                    # Confidence level based on probability edge
                    edge = abs(home_win_prob - 0.5)
                    if edge >= 0.15:
                        confidence = "HIGH"
                        conf_color = "#00e676"
                    elif edge >= 0.08:
                        confidence = "MEDIUM"
                        conf_color = "#ffd600"
                    else:
                        confidence = "LOW"
                        conf_color = "#ff9100"

                    # Get Kalshi-first odds
                    odds_info = get_kalshi_odds_for_game(session, game.id)

                    # Compute EV against Kalshi odds
                    pick_odds_am = odds_info.get("home_ml") if pick_side == "home" else odds_info.get("away_ml")
                    pick_book = odds_info.get("home_ml_book") if pick_side == "home" else odds_info.get("away_ml_book")

                    pick_ev = None
                    pick_units = 0.0
                    if pick_odds_am is not None:
                        pick_dec = american_to_decimal(pick_odds_am)
                        pick_ev = calculate_ev(pick_prob, pick_dec)
                        if pick_ev > 0:
                            kf = fractional_kelly(pick_prob, pick_dec, kelly_frac)
                            pick_units = kelly_to_units(kf, bankroll, 100)

                    # O/U analysis
                    real_total_line = odds_info.get("total_line") or 8.5
                    over_odds_am = odds_info.get("over_odds") or -110
                    under_odds_am = odds_info.get("under_odds") or -110

                    if hasattr(runs_model, 'predict_over_under'):
                        over_probs, under_probs = runs_model.predict_over_under(feature_df, line=real_total_line)
                    elif hasattr(runs_model, 'predict_total_probs'):
                        over_probs, under_probs = runs_model.predict_total_probs(feature_df, line=real_total_line)
                    else:
                        over_probs = np.array([0.5])
                        under_probs = np.array([0.5])

                    ou_pick = "Over" if float(over_probs[0]) > 0.5 else "Under"
                    ou_prob = float(over_probs[0]) if ou_pick == "Over" else float(under_probs[0])
                    ou_odds_am = over_odds_am if ou_pick == "Over" else under_odds_am
                    ou_book = odds_info.get("over_book", "") if ou_pick == "Over" else odds_info.get("under_book", "")

                    ou_ev = None
                    ou_units = 0.0
                    if ou_odds_am is not None:
                        ou_dec = american_to_decimal(ou_odds_am)
                        ou_ev = calculate_ev(ou_prob, ou_dec)
                        if ou_ev > 0:
                            kf = fractional_kelly(ou_prob, ou_dec, kelly_frac)
                            ou_units = kelly_to_units(kf, bankroll, 100)

                    # Tier based on best EV
                    best_ev = max(pick_ev or 0, ou_ev or 0)
                    if best_ev >= 0.05:
                        tier = "A"
                    elif best_ev >= 0.03:
                        tier = "B"
                    elif best_ev > 0:
                        tier = "C"
                    else:
                        tier = "D"

                    predictions.append({
                        "game": game,
                        "home_win_prob": home_win_prob,
                        "away_win_prob": away_win_prob,
                        "home_runs": home_runs,
                        "away_runs": away_runs,
                        "total": total,
                        "pick_team": pick_team,
                        "pick_prob": pick_prob,
                        "pick_side": pick_side,
                        "confidence": confidence,
                        "conf_color": conf_color,
                        "pick_odds_am": pick_odds_am,
                        "pick_book": pick_book or "",
                        "pick_ev": pick_ev,
                        "pick_units": pick_units,
                        "ou_pick": ou_pick,
                        "ou_prob": ou_prob,
                        "ou_line": real_total_line,
                        "ou_odds_am": ou_odds_am,
                        "ou_book": ou_book,
                        "ou_ev": ou_ev,
                        "ou_units": ou_units,
                        "over_prob": float(over_probs[0]),
                        "under_prob": float(under_probs[0]),
                        "tier": tier,
                        "best_ev": best_ev,
                        "odds_info": odds_info,
                    })
                except Exception as e:
                    st.warning(f"Error predicting {game.away_team.abbreviation} @ {game.home_team.abbreviation}: {e}")
                    continue

            # Sort: highest confidence first, then by EV
            tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
            predictions.sort(key=lambda p: (tier_order[p["tier"]], -p["best_ev"]))

            # Summary metrics
            total_bets = sum(1 for p in predictions if (p["pick_ev"] or 0) > 0 or (p["ou_ev"] or 0) > 0)
            a_picks = sum(1 for p in predictions if p["tier"] == "A")
            positive_ev = [p for p in predictions if (p["pick_ev"] or 0) > 0]
            avg_ev = np.mean([p["pick_ev"] for p in positive_ev]) if positive_ev else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games", len(predictions))
            m2.metric("+EV Bets", total_bets)
            m3.metric("High Confidence", sum(1 for p in predictions if p["confidence"] == "HIGH"))
            m4.metric("Avg Pick EV", f"{avg_ev:+.1%}" if avg_ev else "N/A")

            st.divider()

            # ---- Render game cards ----
            for pred in predictions:
                game = pred["game"]
                away = game.away_team.abbreviation
                home = game.home_team.abbreviation
                away_color = TEAM_COLORS.get(away, "#666")
                home_color = TEAM_COLORS.get(home, "#666")
                pick_color = TEAM_COLORS.get(pred["pick_team"], "#00e676")

                time_str = game.game_time_utc.strftime("%I:%M %p UTC") if game.game_time_utc else "TBD"
                status_map = {
                    "final": ("FINAL", "#4caf50"),
                    "live": ("LIVE", "#f44336"),
                    "pregame": ("PRE", "#ff9800"),
                    "scheduled": (time_str, "#2196f3"),
                }
                status_text, status_color = status_map.get(game.status, (game.status, "#666"))

                # Score display for final games
                score_html = ""
                result_html = ""
                if game.status == "final" and game.away_score is not None:
                    score_html = (
                        f"<span style='color:#8892a4; font-size:14px; margin-left:12px;'>"
                        f"Final: {game.away_score} - {game.home_score}</span>"
                    )
                    # Check if model was right
                    actual_winner = home if game.home_score > game.away_score else away
                    if actual_winner == pred["pick_team"]:
                        result_html = "<span style='color:#00e676; font-weight:700; margin-left:8px;'>CORRECT</span>"
                    else:
                        result_html = "<span style='color:#ff5252; font-weight:700; margin-left:8px;'>WRONG</span>"

                # Odds display
                pick_odds_str = ""
                if pred["pick_odds_am"] is not None:
                    o = pred["pick_odds_am"]
                    pick_odds_str = f"+{o:.0f}" if o > 0 else f"{o:.0f}"

                ev_html = ""
                if pred["pick_ev"] is not None and pred["pick_ev"] > 0:
                    ev_html = (
                        f"<span style='color:#00e676; font-size:13px;'>"
                        f"EV: {pred['pick_ev']:+.1%} | {pred['pick_units']:.1f}u</span>"
                    )
                elif pred["pick_ev"] is not None:
                    ev_html = f"<span style='color:#ff9100; font-size:13px;'>EV: {pred['pick_ev']:+.1%} (no bet)</span>"

                # O/U display
                ou_odds_str = ""
                if pred["ou_odds_am"] is not None:
                    o = pred["ou_odds_am"]
                    ou_odds_str = f"+{o:.0f}" if o > 0 else f"{o:.0f}"

                ou_ev_html = ""
                if pred["ou_ev"] is not None and pred["ou_ev"] > 0:
                    ou_ev_html = (
                        f"<span style='color:#00e676; font-size:13px;'>"
                        f"EV: {pred['ou_ev']:+.1%} | {pred['ou_units']:.1f}u</span>"
                    )

                book_tag = f" @ {pred['pick_book']}" if pred["pick_book"] else ""
                ou_book_tag = f" @ {pred['ou_book']}" if pred["ou_book"] else ""

                away_sp = game.away_sp.name if game.away_sp else "TBD"
                home_sp = game.home_sp.name if game.home_sp else "TBD"

                st.markdown(f"""
                <div class="game-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                        <div>
                            <span style="color:{away_color}; font-size:24px; font-weight:700;">{away}</span>
                            <span class="matchup-vs" style="margin:0 10px;">@</span>
                            <span style="color:{home_color}; font-size:24px; font-weight:700;">{home}</span>
                            {score_html}{result_html}
                        </div>
                        <div style="text-align:right;">
                            <span class="tier-badge tier-{pred['tier']}">TIER {pred['tier']}</span>
                            <span style="color:{status_color}; font-weight:600; font-size:13px; margin-left:12px;">{status_text}</span>
                        </div>
                    </div>

                    <div style="color:#8892a4; font-size:13px; margin-bottom:16px;">
                        {away_sp} vs {home_sp}
                    </div>

                    <div style="display:flex; gap:20px; flex-wrap:wrap;">
                        <!-- MODEL PICK -->
                        <div style="flex:1; min-width:220px; background:#0d1520; border:2px solid {pick_color}; border-radius:10px; padding:16px;">
                            <div style="font-size:11px; color:#8892a4; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Model Pick</div>
                            <div style="font-size:28px; font-weight:800; color:{pick_color};">{pred['pick_team']} WINS</div>
                            <div style="font-size:15px; color:#e0e0e0; margin-top:4px;">
                                {pred['pick_prob']:.0%} probability
                                <span style="color:{pred['conf_color']}; font-weight:700; margin-left:8px;">{pred['confidence']}</span>
                            </div>
                            <div style="margin-top:8px; font-size:13px; color:#b0bec5;">
                                {pick_odds_str}{book_tag}
                            </div>
                            <div style="margin-top:4px;">{ev_html}</div>
                        </div>

                        <!-- PREDICTED SCORE -->
                        <div style="flex:0.6; min-width:160px; background:#0d1520; border:1px solid #2d3548; border-radius:10px; padding:16px; text-align:center;">
                            <div style="font-size:11px; color:#8892a4; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Predicted Score</div>
                            <div style="font-size:28px; font-weight:700; color:#e0e0e0;">
                                <span style="color:{away_color};">{pred['away_runs']:.1f}</span>
                                <span style="color:#5c6878;"> - </span>
                                <span style="color:{home_color};">{pred['home_runs']:.1f}</span>
                            </div>
                            <div style="font-size:13px; color:#8892a4; margin-top:4px;">Total: {pred['total']:.1f}</div>
                        </div>

                        <!-- O/U PICK -->
                        <div style="flex:0.8; min-width:180px; background:#0d1520; border:1px solid #2d3548; border-radius:10px; padding:16px;">
                            <div style="font-size:11px; color:#8892a4; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Over/Under</div>
                            <div style="font-size:22px; font-weight:700; color:#e0e0e0;">
                                {pred['ou_pick']} {pred['ou_line']}
                            </div>
                            <div style="font-size:13px; color:#b0bec5; margin-top:4px;">
                                {pred['ou_prob']:.0%} prob | {ou_odds_str}{ou_book_tag}
                            </div>
                            <div style="margin-top:4px;">{ou_ev_html}</div>
                        </div>

                        <!-- WIN PROBS -->
                        <div style="flex:0.6; min-width:160px; background:#0d1520; border:1px solid #2d3548; border-radius:10px; padding:16px; text-align:center;">
                            <div style="font-size:11px; color:#8892a4; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Win Probability</div>
                            <div style="font-size:16px; color:{away_color}; font-weight:600;">{away} {pred['away_win_prob']:.0%}</div>
                            <div style="font-size:16px; color:{home_color}; font-weight:600;">{home} {pred['home_win_prob']:.0%}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

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
        "<p style='margin:4px 0 0 0; color:#90caf9;'>Training metrics, feature importance, and calibration</p>"
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
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
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

    # Show recent prediction accuracy if we have final games
    st.divider()
    st.subheader("Recent Prediction Accuracy")

    session = get_sess()
    try:
        recent_final = session.query(Game).filter(
            Game.status == "final",
            Game.home_score.isnot(None),
        ).order_by(Game.game_date.desc()).limit(200).all()

        if recent_final:
            home_wins = sum(1 for g in recent_final if g.home_score > g.away_score)
            away_wins = len(recent_final) - home_wins

            fig = go.Figure(data=[
                go.Pie(
                    labels=["Home Wins", "Away Wins"],
                    values=[home_wins, away_wins],
                    marker=dict(colors=["#2196f3", "#ff9800"]),
                    hole=0.4,
                )
            ])
            fig.update_layout(
                title=f"Home vs Away Wins (Last {len(recent_final)} Games)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Scoring trends
            dates = sorted(set(g.game_date for g in recent_final))[-30:]
            daily_avg = []
            for d in dates:
                day_games = [g for g in recent_final if g.game_date == d]
                avg_total = np.mean([g.total_runs for g in day_games if g.total_runs])
                daily_avg.append({"date": d, "avg_total_runs": avg_total})

            if daily_avg:
                trend_df = pd.DataFrame(daily_avg)
                fig = px.line(
                    trend_df, x="date", y="avg_total_runs",
                    title="Average Total Runs Per Game (Last 30 Days)",
                    labels={"avg_total_runs": "Avg Runs", "date": "Date"},
                    color_discrete_sequence=["#00e676"],
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
# PAGE: Team Stats
# ===================================================================
elif page == "Team Stats":
    st.markdown(
        "<div class='header-gradient'>"
        "<h1 style='margin:0; color:white;'>Team Stats</h1>"
        "<p style='margin:4px 0 0 0; color:#90caf9;'>Season records, batting averages, and pitching performance</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    session = get_sess()
    try:
        teams = session.query(Team).order_by(Team.division, Team.name).all()

        # Build team records
        team_stats = []
        for team in teams:
            wins = session.query(Game).filter(
                Game.winning_team_id == team.id,
                Game.status == "final",
            ).count()
            losses = session.query(Game).filter(
                Game.status == "final",
                ((Game.home_team_id == team.id) | (Game.away_team_id == team.id)),
                Game.winning_team_id != team.id,
                Game.winning_team_id.isnot(None),
            ).count()

            # Recent batting
            recent_batting = session.query(TeamBattingDaily).filter(
                TeamBattingDaily.team_id == team.id,
            ).order_by(TeamBattingDaily.game_date.desc()).limit(30).all()

            avg_runs = np.mean([b.runs for b in recent_batting if b.runs is not None]) if recent_batting else 0
            avg_hits = np.mean([b.hits for b in recent_batting if b.hits is not None]) if recent_batting else 0
            avg_hr = np.mean([b.home_runs for b in recent_batting if b.home_runs is not None]) if recent_batting else 0

            pct = wins / (wins + losses) if (wins + losses) > 0 else 0

            team_stats.append({
                "Team": team.abbreviation,
                "Division": team.division,
                "W": wins,
                "L": losses,
                "PCT": pct,
                "R/G (30d)": round(avg_runs, 1),
                "H/G (30d)": round(avg_hits, 1),
                "HR/G (30d)": round(avg_hr, 2),
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
            df.sort_values("R/G (30d)", ascending=True),
            x="R/G (30d)", y="Team", orientation="h",
            title="Runs Per Game (Last 30 Games)",
            color="R/G (30d)",
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


# ===================================================================
# PAGE: Odds Dashboard
# ===================================================================
elif page == "Odds Dashboard":
    st.markdown(
        "<div class='header-gradient'>"
        "<h1 style='margin:0; color:white;'>Odds Dashboard</h1>"
        "<p style='margin:4px 0 0 0; color:#90caf9;'>Live odds comparison across sportsbooks</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    session = get_sess()
    try:
        # Get today's games with odds
        today_games = session.query(Game).filter(
            Game.game_date == date.today(),
        ).order_by(Game.game_time_utc).all()

        if not today_games:
            st.info("No games today. Try ingesting the schedule first.")
        else:
            for game in today_games:
                away = game.away_team.abbreviation
                home = game.home_team.abbreviation

                with st.expander(f"{away} @ {home}", expanded=True):
                    # Get all odds for this game
                    h2h_odds = session.query(OddsSnapshot).filter(
                        OddsSnapshot.game_id == game.id,
                        OddsSnapshot.market_type == "h2h",
                    ).order_by(OddsSnapshot.captured_at.desc()).all()

                    total_odds = session.query(OddsSnapshot).filter(
                        OddsSnapshot.game_id == game.id,
                        OddsSnapshot.market_type == "totals",
                    ).order_by(OddsSnapshot.captured_at.desc()).all()

                    if h2h_odds:
                        st.markdown("**Moneyline Odds**")
                        seen_books = set()
                        ml_rows = []
                        for snap in h2h_odds:
                            if snap.sportsbook in seen_books:
                                continue
                            seen_books.add(snap.sportsbook)
                            h = snap.home_line
                            a = snap.away_line
                            ml_rows.append({
                                "Sportsbook": snap.sportsbook,
                                f"{home} ML": f"+{h:.0f}" if h and h > 0 else (f"{h:.0f}" if h else ""),
                                f"{away} ML": f"+{a:.0f}" if a and a > 0 else (f"{a:.0f}" if a else ""),
                            })
                        if ml_rows:
                            st.dataframe(pd.DataFrame(ml_rows), use_container_width=True, hide_index=True)

                    if total_odds:
                        st.markdown("**Totals**")
                        seen_books = set()
                        tot_rows = []
                        for snap in total_odds:
                            if snap.sportsbook in seen_books:
                                continue
                            seen_books.add(snap.sportsbook)
                            tot_rows.append({
                                "Sportsbook": snap.sportsbook,
                                "Line": snap.total_line,
                                "Over": f"+{snap.over_odds:.0f}" if snap.over_odds and snap.over_odds > 0 else (f"{snap.over_odds:.0f}" if snap.over_odds else ""),
                                "Under": f"+{snap.under_odds:.0f}" if snap.under_odds and snap.under_odds > 0 else (f"{snap.under_odds:.0f}" if snap.under_odds else ""),
                            })
                        if tot_rows:
                            st.dataframe(pd.DataFrame(tot_rows), use_container_width=True, hide_index=True)

                    if not h2h_odds and not total_odds:
                        st.caption("No odds data available for this game.")

        # Overall odds stats
        st.divider()
        total_snaps = session.query(OddsSnapshot).count()
        unique_books = session.query(OddsSnapshot.sportsbook).distinct().count()
        st.metric("Total Odds Snapshots", f"{total_snaps:,}")
        st.metric("Sportsbooks Tracked", unique_books)

    finally:
        session.close()
