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
from bbbot.ingest.odds_ingest import get_best_odds_for_game

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
        "<h1 style='margin:0; color:white;'>Today's Predictions</h1>"
        "<p style='margin:4px 0 0 0; color:#90caf9;'>ML-powered game forecasts with real sportsbook odds</p>"
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
                st.info("Using baseline model. Run `bbbot train all` for better predictions.")

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

                    odds_info = get_best_odds_for_game(session, game.id)

                    # Build candidates
                    candidates = []

                    # Home ML
                    home_odds_am = odds_info.get("home_ml") or (-150 if home_win_prob > 0.5 else 130)
                    home_dec = american_to_decimal(home_odds_am)
                    home_ev = calculate_ev(home_win_prob, home_dec)
                    if home_ev > 0:
                        kf = fractional_kelly(home_win_prob, home_dec, kelly_frac)
                        units = kelly_to_units(kf, bankroll, 100)
                        book = odds_info.get("home_ml_book", "")
                        candidates.append({
                            "label": f"{game.home_team.abbreviation} ML",
                            "ev": home_ev, "units": units, "odds": home_odds_am, "book": book,
                        })

                    # Away ML
                    away_odds_am = odds_info.get("away_ml") or (130 if home_win_prob > 0.5 else -150)
                    away_dec = american_to_decimal(away_odds_am)
                    away_ev = calculate_ev(away_win_prob, away_dec)
                    if away_ev > 0:
                        kf = fractional_kelly(away_win_prob, away_dec, kelly_frac)
                        units = kelly_to_units(kf, bankroll, 100)
                        book = odds_info.get("away_ml_book", "")
                        candidates.append({
                            "label": f"{game.away_team.abbreviation} ML",
                            "ev": away_ev, "units": units, "odds": away_odds_am, "book": book,
                        })

                    # Over/Under
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

                    over_dec = american_to_decimal(over_odds_am)
                    over_ev = calculate_ev(float(over_probs[0]), over_dec)
                    if over_ev > 0:
                        kf = fractional_kelly(float(over_probs[0]), over_dec, kelly_frac)
                        units = kelly_to_units(kf, bankroll, 100)
                        book = odds_info.get("over_book", "")
                        candidates.append({
                            "label": f"Over {real_total_line}",
                            "ev": over_ev, "units": units, "odds": over_odds_am, "book": book,
                        })

                    under_dec = american_to_decimal(under_odds_am)
                    under_ev = calculate_ev(float(under_probs[0]), under_dec)
                    if under_ev > 0:
                        kf = fractional_kelly(float(under_probs[0]), under_dec, kelly_frac)
                        units = kelly_to_units(kf, bankroll, 100)
                        book = odds_info.get("under_book", "")
                        candidates.append({
                            "label": f"Under {real_total_line}",
                            "ev": under_ev, "units": units, "odds": under_odds_am, "book": book,
                        })

                    candidates.sort(key=lambda x: x["ev"], reverse=True)
                    best = candidates[0] if candidates else None

                    if best and best["ev"] >= 0.05:
                        tier = "A"
                    elif best and best["ev"] >= 0.03:
                        tier = "B"
                    elif best and best["ev"] > 0:
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
                        "over_prob": float(over_probs[0]),
                        "under_prob": float(under_probs[0]),
                        "tier": tier,
                        "best": best,
                        "candidates": candidates,
                        "odds_info": odds_info,
                        "total_line": real_total_line,
                    })
                except Exception as e:
                    st.warning(f"Error predicting {game.away_team.abbreviation} @ {game.home_team.abbreviation}: {e}")
                    continue

            # Sort by tier
            tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
            predictions.sort(key=lambda p: (tier_order[p["tier"]], -(p["best"]["ev"] if p["best"] else 0)))

            # Summary metrics
            total_picks = sum(1 for p in predictions if p["best"])
            a_picks = sum(1 for p in predictions if p["tier"] == "A")
            avg_ev = np.mean([p["best"]["ev"] for p in predictions if p["best"]]) if total_picks else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games", len(predictions))
            m2.metric("Actionable Picks", total_picks)
            m3.metric("A-Tier Picks", a_picks)
            m4.metric("Avg EV", f"{avg_ev:+.1%}")

            st.divider()

            # Render game cards
            for pred in predictions:
                game = pred["game"]
                away = game.away_team.abbreviation
                home = game.home_team.abbreviation
                away_color = TEAM_COLORS.get(away, "#666")
                home_color = TEAM_COLORS.get(home, "#666")

                time_str = game.game_time_utc.strftime("%I:%M %p UTC") if game.game_time_utc else "TBD"

                # Status badge
                status_map = {
                    "final": ("FINAL", "#4caf50"),
                    "live": ("LIVE", "#f44336"),
                    "pregame": ("PRE", "#ff9800"),
                    "scheduled": (time_str, "#2196f3"),
                }
                status_text, status_color = status_map.get(game.status, (game.status, "#666"))

                with st.container():
                    st.markdown(f"""
                    <div class="game-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                            <span class="tier-badge tier-{pred['tier']}">TIER {pred['tier']}</span>
                            <span style="color:{status_color}; font-weight:600; font-size:13px;">{status_text}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns([3, 2, 2, 3])

                    with c1:
                        # Matchup
                        score_str = ""
                        if game.status == "final" and game.away_score is not None:
                            score_str = f"  ({game.away_score} - {game.home_score})"
                        st.markdown(
                            f"<span style='color:{away_color}; font-size:22px; font-weight:700;'>{away}</span>"
                            f"  <span class='matchup-vs'>@</span>  "
                            f"<span style='color:{home_color}; font-size:22px; font-weight:700;'>{home}</span>"
                            f"<span style='color:#8892a4; font-size:14px;'>{score_str}</span>",
                            unsafe_allow_html=True,
                        )
                        away_sp = game.away_sp.name if game.away_sp else "TBD"
                        home_sp = game.home_sp.name if game.home_sp else "TBD"
                        st.caption(f"{away_sp}  vs  {home_sp}")

                    with c2:
                        # Win probabilities
                        st.markdown(
                            f"<div class='stat-box'>"
                            f"<div class='stat-value'>{pred['away_win_prob']:.0%} - {pred['home_win_prob']:.0%}</div>"
                            f"<div class='stat-label'>Win Probability</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    with c3:
                        # Predicted score
                        st.markdown(
                            f"<div class='stat-box'>"
                            f"<div class='stat-value'>{pred['away_runs']:.1f} - {pred['home_runs']:.1f}</div>"
                            f"<div class='stat-label'>Predicted Score</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    with c4:
                        # Best bet
                        if pred["best"]:
                            b = pred["best"]
                            ev_class = "positive-ev" if b["ev"] > 0 else "negative-ev"
                            odds_str = f"+{b['odds']:.0f}" if b["odds"] > 0 else f"{b['odds']:.0f}"
                            book_str = f" @ {b['book']}" if b["book"] else ""
                            st.markdown(
                                f"<div class='bet-rec'>"
                                f"<div style='font-weight:700; font-size:16px; color:#e0e0e0;'>{b['label']}</div>"
                                f"<div style='font-size:13px; color:#a5d6a7;'>"
                                f"EV: <span class='{ev_class}'>{b['ev']:+.1%}</span> | "
                                f"{odds_str}{book_str} | {b['units']:.1f}u</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<div class='stat-box'>"
                                "<div style='color:#616161;'>No +EV bet found</div>"
                                "</div>",
                                unsafe_allow_html=True,
                            )

                    # Expandable details
                    with st.expander("Details"):
                        d1, d2 = st.columns(2)
                        with d1:
                            st.markdown("**All +EV Opportunities:**")
                            if pred["candidates"]:
                                for c in pred["candidates"]:
                                    odds_s = f"+{c['odds']:.0f}" if c["odds"] > 0 else f"{c['odds']:.0f}"
                                    book_s = f" ({c['book']})" if c["book"] else ""
                                    st.markdown(
                                        f"- **{c['label']}** {odds_s}{book_s} — "
                                        f"EV: {c['ev']:+.1%}, {c['units']:.1f}u"
                                    )
                            else:
                                st.markdown("*No positive EV bets found*")

                        with d2:
                            st.markdown("**O/U Analysis:**")
                            st.markdown(f"- Line: **{pred['total_line']}**")
                            st.markdown(f"- Predicted Total: **{pred['total']:.1f}**")
                            st.markdown(f"- Over Prob: **{pred['over_prob']:.0%}**")
                            st.markdown(f"- Under Prob: **{pred['under_prob']:.0%}**")

                            if pred["odds_info"].get("home_ml"):
                                st.markdown("**Market Odds:**")
                                hml = pred["odds_info"]["home_ml"]
                                aml = pred["odds_info"]["away_ml"]
                                hml_s = f"+{hml:.0f}" if hml > 0 else f"{hml:.0f}"
                                aml_s = f"+{aml:.0f}" if aml > 0 else f"{aml:.0f}"
                                st.markdown(f"- {home} ML: {hml_s} ({pred['odds_info'].get('home_ml_book', '')})")
                                st.markdown(f"- {away} ML: {aml_s} ({pred['odds_info'].get('away_ml_book', '')})")

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
