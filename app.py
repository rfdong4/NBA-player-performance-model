"""
NBA Player Performance Predictor for Sports Betting
A comprehensive Streamlit application for analyzing player props.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predictor import NBAPredictor, PredictionFormatter
from src.data_fetcher import NBADataFetcher, get_current_season
from src.features import FeatureEngineer
from src.prizepicks import PrizePicksAnalyzer, PropPick


# Page configuration
st.set_page_config(
    page_title="NBA Player Props Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability
st.markdown("""
<style>
    /* Main metric cards */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px 20px;
        border-radius: 10px;
    }

    div[data-testid="stMetric"] label {
        color: #495057 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #212529 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #198754 !important;
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .prediction-box h2 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
    }

    .prediction-box p {
        margin: 5px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Over/Under picks */
    .over-pick {
        background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
        color: white;
        padding: 12px 25px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 3px 10px rgba(40, 167, 69, 0.3);
    }

    .under-pick {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 12px 25px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 3px 10px rgba(220, 53, 69, 0.3);
    }

    /* Info cards */
    .info-card {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }

    .info-card h4 {
        color: #1565c0;
        margin: 0 0 8px 0;
    }

    .info-card p {
        color: #333;
        margin: 0;
        line-height: 1.5;
    }

    /* Insight cards */
    .insight-positive {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #155724;
    }

    .insight-negative {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #721c24;
    }

    .insight-neutral {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #856404;
    }

    /* Stats table */
    .stats-header {
        color: #495057;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stats-value {
        color: #212529;
        font-size: 1.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the predictor with cached models."""
    predictor = NBAPredictor()
    try:
        predictor.load_models()
        return predictor, True
    except Exception as e:
        return predictor, False


@st.cache_data(ttl=300)
def fetch_player_data(player_name: str, season: str):
    """Fetch player data with caching."""
    fetcher = NBADataFetcher()
    player = fetcher.find_player(player_name)
    if player is None:
        return None, None

    df = fetcher.get_player_game_log(player['id'], season)
    return player, df


def create_performance_chart(df: pd.DataFrame, stat: str, stat_name: str, line: float = None):
    """Create a performance trend chart."""
    fig = go.Figure()

    # Main performance line
    fig.add_trace(go.Scatter(
        x=df['GAME_DATE'],
        y=df[stat],
        mode='lines+markers',
        name=stat_name,
        line=dict(color='#2196F3', width=3),
        marker=dict(size=8, color='#2196F3')
    ))

    # Rolling average
    rolling_avg = df[stat].rolling(5, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df['GAME_DATE'],
        y=rolling_avg,
        mode='lines',
        name='5-Game Avg',
        line=dict(color='#FF9800', width=2, dash='dash')
    ))

    # Add betting line if provided
    if line is not None:
        fig.add_hline(
            y=line,
            line_dash="dot",
            line_color="#E91E63",
            line_width=2,
            annotation_text=f"Line: {line}",
            annotation_position="right",
            annotation_font_color="#E91E63"
        )

    fig.update_layout(
        title=f"{stat_name} Performance Trend",
        xaxis_title="Game Date",
        yaxis_title=stat_name,
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(color="#333")
    )

    return fig


def create_hit_rate_chart(df: pd.DataFrame, stat: str, lines: list):
    """Create a hit rate chart for various lines."""
    hit_rates = []
    for line in lines:
        rate = (df[stat] > line).mean() * 100
        hit_rates.append(rate)

    colors = ['#28a745' if r > 55 else '#dc3545' if r < 45 else '#ffc107' for r in hit_rates]

    fig = go.Figure(data=[
        go.Bar(
            x=[f"O {l}" for l in lines],
            y=hit_rates,
            marker_color=colors,
            text=[f"{r:.0f}%" for r in hit_rates],
            textposition='auto',
            textfont=dict(size=14, color='white')
        )
    ])

    fig.update_layout(
        title="Hit Rate Analysis (Last 20 Games)",
        xaxis_title="Line",
        yaxis_title="Hit Rate %",
        template="plotly_white",
        height=350,
        yaxis=dict(range=[0, 100]),
        font=dict(color="#333")
    )

    return fig


def generate_prediction_insights(df: pd.DataFrame, prediction: float, prop_type: str, line: float = None):
    """Generate insights explaining the prediction."""
    insights = []

    stat_map = {
        'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST',
        'steals': 'STL', 'blocks': 'BLK', 'threes': 'FG3M'
    }

    stat = stat_map.get(prop_type.lower(), 'PTS')

    if stat not in df.columns:
        return insights

    # Calculate key metrics
    last_5_avg = df.tail(5)[stat].mean()
    last_10_avg = df.tail(10)[stat].mean()
    season_avg = df[stat].mean()
    last_game = df.iloc[-1][stat]

    # Trend analysis
    if last_5_avg > last_10_avg * 1.1:
        insights.append({
            'type': 'positive',
            'title': 'Hot Streak',
            'text': f"Player is trending UP - Last 5 games avg ({last_5_avg:.1f}) is {((last_5_avg/last_10_avg - 1) * 100):.0f}% higher than last 10 ({last_10_avg:.1f})"
        })
    elif last_5_avg < last_10_avg * 0.9:
        insights.append({
            'type': 'negative',
            'title': 'Cold Streak',
            'text': f"Player is trending DOWN - Last 5 games avg ({last_5_avg:.1f}) is {((1 - last_5_avg/last_10_avg) * 100):.0f}% lower than last 10 ({last_10_avg:.1f})"
        })
    else:
        insights.append({
            'type': 'neutral',
            'title': 'Stable Performance',
            'text': f"Consistent recent production - Last 5 avg ({last_5_avg:.1f}) is close to last 10 avg ({last_10_avg:.1f})"
        })

    # Last game analysis
    if last_game > season_avg * 1.3:
        insights.append({
            'type': 'positive',
            'title': 'Strong Last Game',
            'text': f"Last game ({int(last_game)}) was well above season average ({season_avg:.1f}) - momentum could carry forward"
        })
    elif last_game < season_avg * 0.7:
        insights.append({
            'type': 'negative',
            'title': 'Poor Last Game',
            'text': f"Last game ({int(last_game)}) was well below season average ({season_avg:.1f}) - potential bounce-back candidate"
        })

    # Home/Away analysis
    if 'MATCHUP' in df.columns:
        is_home = df.iloc[-1]['MATCHUP']
        if ' vs. ' in str(is_home):
            home_games = df[df['MATCHUP'].str.contains('vs.', na=False)]
            if len(home_games) >= 5:
                home_avg = home_games.tail(10)[stat].mean()
                if home_avg > season_avg * 1.05:
                    insights.append({
                        'type': 'positive',
                        'title': 'Home Court Advantage',
                        'text': f"Performs better at home - Home avg ({home_avg:.1f}) vs Season avg ({season_avg:.1f})"
                    })
        else:
            away_games = df[df['MATCHUP'].str.contains('@', na=False)]
            if len(away_games) >= 5:
                away_avg = away_games.tail(10)[stat].mean()
                if away_avg < season_avg * 0.95:
                    insights.append({
                        'type': 'negative',
                        'title': 'Road Struggles',
                        'text': f"Tends to underperform on the road - Away avg ({away_avg:.1f}) vs Season avg ({season_avg:.1f})"
                    })

    # Minutes analysis
    if 'MIN' in df.columns:
        recent_min = df.tail(5)['MIN'].mean()
        season_min = df['MIN'].mean()
        if recent_min < season_min * 0.85:
            insights.append({
                'type': 'negative',
                'title': 'Reduced Minutes',
                'text': f"Recent minutes ({recent_min:.1f}) are down {((1 - recent_min/season_min) * 100):.0f}% from season avg ({season_min:.1f}) - could limit production"
            })
        elif recent_min > season_min * 1.1:
            insights.append({
                'type': 'positive',
                'title': 'Increased Role',
                'text': f"Recent minutes ({recent_min:.1f}) are up {((recent_min/season_min - 1) * 100):.0f}% from season avg ({season_min:.1f}) - more opportunity"
            })

    # Consistency analysis
    std_dev = df.tail(10)[stat].std()
    cv = std_dev / last_10_avg if last_10_avg > 0 else 0
    if cv > 0.35:
        insights.append({
            'type': 'neutral',
            'title': 'High Variance',
            'text': f"Inconsistent performer - scores fluctuate significantly game-to-game (std dev: {std_dev:.1f})"
        })
    elif cv < 0.2:
        insights.append({
            'type': 'positive',
            'title': 'Consistent Scorer',
            'text': f"Very predictable output - low variance in recent games (std dev: {std_dev:.1f})"
        })

    # Line comparison
    if line is not None:
        hit_rate = (df.tail(10)[stat] > line).mean() * 100
        if hit_rate >= 70:
            insights.append({
                'type': 'positive',
                'title': f'Strong History vs Line',
                'text': f"Hit OVER {line} in {hit_rate:.0f}% of last 10 games - line may be too low"
            })
        elif hit_rate <= 30:
            insights.append({
                'type': 'negative',
                'title': f'Struggles vs Line',
                'text': f"Hit OVER {line} in only {hit_rate:.0f}% of last 10 games - line may be too high"
            })

    return insights


def display_prediction_section(result: dict, df: pd.DataFrame, prop_type: str, line: float = None):
    """Display the main prediction with context."""

    prediction = result['prediction']

    # Main prediction display
    st.markdown(f"""
    <div class="prediction-box">
        <h2>{prediction:.1f}</h2>
        <p>Predicted {prop_type.upper()}</p>
    </div>
    """, unsafe_allow_html=True)

    # Key stats row
    col1, col2, col3, col4 = st.columns(4)

    stats = result.get('recent_stats', {})
    with col1:
        st.metric("Last Game", stats.get('last_game', 'N/A'))
    with col2:
        st.metric("Last 5 Avg", stats.get('last_5_avg', 'N/A'))
    with col3:
        st.metric("Last 10 Avg", stats.get('last_10_avg', 'N/A'))
    with col4:
        st.metric("Season Avg", f"{df[get_stat_column(prop_type, df)].mean():.1f}" if get_stat_column(prop_type, df) in df.columns else 'N/A')

    # Betting analysis if line provided
    if line is not None and 'betting_analysis' in result:
        st.markdown("---")
        ba = result['betting_analysis']

        col1, col2 = st.columns([1, 2])

        with col1:
            # Recommendation
            rec = ba['recommendation']
            edge = max(ba['over_edge'], ba['under_edge']) * 100

            if rec == 'OVER':
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <div class="over-pick">OVER {line}</div>
                    <p style="margin-top: 15px; color: #333; font-size: 1.1rem;">
                        <strong>Edge: {edge:.1f}%</strong> ({ba['bet_strength'].title()})
                    </p>
                    <p style="color: #666;">
                        Model predicts <strong>{prediction:.1f}</strong>, which is <strong>{prediction - line:+.1f}</strong> vs the line
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <div class="under-pick">UNDER {line}</div>
                    <p style="margin-top: 15px; color: #333; font-size: 1.1rem;">
                        <strong>Edge: {edge:.1f}%</strong> ({ba['bet_strength'].title()})
                    </p>
                    <p style="color: #666;">
                        Model predicts <strong>{prediction:.1f}</strong>, which is <strong>{prediction - line:+.1f}</strong> vs the line
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # Probability breakdown
            st.markdown("#### Probability Breakdown")

            prob_over = ba['probability_over'] * 100
            prob_under = ba['probability_under'] * 100

            # Progress bar style probability display
            st.markdown(f"""
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #28a745; font-weight: bold;">OVER: {prob_over:.1f}%</span>
                    <span style="color: #dc3545; font-weight: bold;">UNDER: {prob_under:.1f}%</span>
                </div>
                <div style="background: #dc3545; border-radius: 10px; height: 25px; overflow: hidden;">
                    <div style="background: #28a745; width: {prob_over}%; height: 100%;"></div>
                </div>
                <p style="text-align: center; color: #666; margin-top: 8px; font-size: 0.9rem;">
                    Break-even at -110 odds requires 52.4% win rate
                </p>
            </div>
            """, unsafe_allow_html=True)


def get_stat_column(prop_type: str, df: pd.DataFrame) -> str:
    """Get the stat column name for a prop type."""
    stat_map = {
        'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST',
        'steals': 'STL', 'blocks': 'BLK', 'threes': 'FG3M',
        'turnovers': 'TOV'
    }

    if prop_type in ['pts_reb_ast', 'pts_reb', 'pts_ast']:
        return 'COMBINED'

    return stat_map.get(prop_type.lower(), 'PTS')


def prizepicks_page():
    """PrizePicks comparison page."""
    st.title("🎯 PrizePicks Analyzer")
    st.markdown("*Paste your PrizePicks props to find the best plays*")

    predictor, models_loaded = load_predictor()

    if not models_loaded:
        st.error("Models not loaded. Please train models first: `python train_models.py`")
        return

    # Initialize analyzer
    analyzer = PrizePicksAnalyzer(predictor)
    analyzer._models_loaded = models_loaded

    # Input section
    st.subheader("Enter Props")

    col1, col2 = st.columns([2, 1])

    with col1:
        props_input = st.text_area(
            "Paste props (one per line)",
            placeholder="LeBron James, points, 25.5\nStephen Curry, threes, 4.5\nNikola Jokic, rebounds, 12.5\nLuka Doncic, assists, 8.5",
            height=200,
            help="Format: Player Name, prop_type, line"
        )

    with col2:
        st.markdown("**Supported prop types:**")
        st.markdown("""
        - `points`
        - `rebounds`
        - `assists`
        - `steals`
        - `blocks`
        - `threes`
        - `turnovers`
        """)

        min_edge = st.slider("Minimum Edge %", 0, 15, 5) / 100
        min_confidence = st.selectbox("Minimum Confidence", ["weak", "moderate", "strong"], index=1)

    analyze_btn = st.button("🔍 Analyze Props", type="primary", use_container_width=True)

    if analyze_btn and props_input.strip():
        # Parse props
        props = []
        for line in props_input.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                try:
                    props.append({
                        'player': parts[0],
                        'prop_type': parts[1].lower(),
                        'line': float(parts[2])
                    })
                except ValueError:
                    st.warning(f"Could not parse line: {line}")

        if not props:
            st.error("No valid props found. Check the format.")
            return

        with st.spinner(f"Analyzing {len(props)} props..."):
            picks = analyzer.analyze_slate(props)

        if not picks:
            st.warning("Could not analyze any props. Check player names and try again.")
            return

        # Summary metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        strong_picks = [p for p in picks if p.confidence == 'strong' and p.edge >= min_edge]
        moderate_picks = [p for p in picks if p.confidence == 'moderate' and p.edge >= min_edge]
        over_picks = [p for p in picks if p.recommendation == 'OVER' and p.edge >= min_edge]
        under_picks = [p for p in picks if p.recommendation == 'UNDER' and p.edge >= min_edge]

        with col1:
            st.metric("Total Props", len(picks))
        with col2:
            st.metric("Strong Plays", len(strong_picks))
        with col3:
            st.metric("Overs", len(over_picks))
        with col4:
            st.metric("Unders", len(under_picks))

        # Best picks section
        st.markdown("---")
        st.subheader("🏆 Best Plays")

        best_picks = [p for p in picks if p.edge >= min_edge and p.confidence in (['strong', 'moderate'] if min_confidence != 'weak' else ['strong', 'moderate', 'weak'])]

        if not best_picks:
            st.info(f"No picks meet the {min_edge*100:.0f}% edge and {min_confidence} confidence criteria.")
        else:
            for pick in best_picks[:5]:
                trend_emoji = "🔥" if pick.trend == 'hot' else "❄️" if pick.trend == 'cold' else "➡️"
                conf_color = "#28a745" if pick.confidence == 'strong' else "#ffc107" if pick.confidence == 'moderate' else "#6c757d"

                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 2])

                    with col1:
                        st.markdown(f"### {pick.player_name}")
                        st.caption(f"{pick.prop_type.upper()} | Line: {pick.line}")

                    with col2:
                        rec_color = "#28a745" if pick.recommendation == 'OVER' else "#dc3545"
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <span style="background: {rec_color}; color: white; padding: 8px 20px; border-radius: 5px; font-weight: bold; font-size: 1.2rem;">
                                {pick.recommendation} {pick.line}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.metric("Prediction", f"{pick.prediction:.1f}", delta=f"{pick.diff_from_line:+.1f}")

                    # Details row
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Edge", f"{pick.edge_pct:.1f}%")
                    with col2:
                        st.metric("Probability", f"{pick.probability*100:.0f}%")
                    with col3:
                        st.metric("L5 Avg", f"{pick.last_5_avg:.1f}")
                    with col4:
                        st.metric("Hit Rate (L10)", f"{pick.recent_hit_rate*100:.0f}%")
                    with col5:
                        st.markdown(f"**Trend:** {trend_emoji} {pick.trend.upper()}")

                    # Reasoning
                    with st.expander("Analysis Details"):
                        for reason in pick.reasoning:
                            st.markdown(f"• {reason}")
                        if pick.minutes_concern:
                            st.warning("⚠️ Minutes concern - recent minutes down")

                    st.markdown("---")

        # All picks table
        st.subheader("📊 All Props Ranked")

        table_data = []
        for pick in picks:
            table_data.append({
                'Player': pick.player_name,
                'Prop': pick.prop_type,
                'Line': pick.line,
                'Prediction': round(pick.prediction, 1),
                'Diff': round(pick.diff_from_line, 1),
                'Edge %': round(pick.edge_pct, 1),
                'Pick': pick.recommendation,
                'Confidence': pick.confidence,
                'Hit Rate': f"{pick.recent_hit_rate*100:.0f}%",
                'Trend': pick.trend
            })

        df = pd.DataFrame(table_data)

        # Style the dataframe
        def highlight_pick(val):
            if val == 'OVER':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'UNDER':
                return 'background-color: #f8d7da; color: #721c24'
            return ''

        def highlight_confidence(val):
            if val == 'strong':
                return 'background-color: #28a745; color: white'
            elif val == 'moderate':
                return 'background-color: #ffc107; color: black'
            return ''

        styled_df = df.style.applymap(highlight_pick, subset=['Pick']).applymap(highlight_confidence, subset=['Confidence'])
        st.dataframe(styled_df, use_container_width=True, height=400)

        # Parlay builder
        st.markdown("---")
        st.subheader("🎲 Suggested Parlay")

        parlay_legs = st.slider("Number of legs", 2, 5, 3)

        parlay_picks, parlay_prob = analyzer.build_parlay(props, legs=parlay_legs)

        if parlay_picks:
            st.markdown(f"**Combined Probability: {parlay_prob*100:.1f}%**")

            for i, pick in enumerate(parlay_picks, 1):
                col1, col2, col3 = st.columns([1, 3, 2])
                with col1:
                    st.markdown(f"### Leg {i}")
                with col2:
                    st.markdown(f"**{pick.player_name}**")
                    st.caption(f"{pick.prop_type.upper()}")
                with col3:
                    rec_color = "#28a745" if pick.recommendation == 'OVER' else "#dc3545"
                    st.markdown(f"""
                    <span style="background: {rec_color}; color: white; padding: 5px 15px; border-radius: 5px; font-weight: bold;">
                        {pick.recommendation} {pick.line}
                    </span>
                    <span style="margin-left: 10px;">Edge: {pick.edge_pct:.1f}%</span>
                    """, unsafe_allow_html=True)


def single_player_page():
    """Single player analysis page (original functionality)."""
    st.title("🏀 Single Player Analysis")
    st.markdown("*Deep dive into a single player's props*")

    predictor, models_loaded = load_predictor()

    # Sidebar inputs
    with st.sidebar:
        st.header("Settings")

        player_name = st.text_input(
            "Player Name",
            placeholder="e.g., LeBron James",
            help="Enter the full name of an NBA player"
        )

        prop_type = st.selectbox(
            "Prop Type",
            options=["Points", "Rebounds", "Assists", "Steals", "Blocks", "Threes", "Pts+Reb+Ast", "Pts+Reb", "Pts+Ast"],
            index=0
        )

        prop_map = {
            "Points": "points", "Rebounds": "rebounds", "Assists": "assists",
            "Steals": "steals", "Blocks": "blocks", "Threes": "threes",
            "Pts+Reb+Ast": "pts_reb_ast", "Pts+Reb": "pts_reb", "Pts+Ast": "pts_ast"
        }

        line = st.number_input(
            "Betting Line (optional)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            help="Enter the sportsbook line for analysis"
        )

        if line == 0.0:
            line = None

        current_season = get_current_season()
        season = st.selectbox(
            "Season",
            options=[current_season, '2023-24', '2022-23'],
            index=0
        )

        analyze_button = st.button("Analyze Player", type="primary", use_container_width=True)

        st.markdown("---")
        if not models_loaded:
            st.warning("Models not loaded. Train models first:\n```\npython train_models.py\n```")

    # Main content
    if analyze_button and player_name:
        with st.spinner(f"Fetching data for {player_name}..."):
            player, df = fetch_player_data(player_name, season)

        if player is None:
            st.error(f"Player not found: {player_name}")
            st.info("Try using the player's full name (e.g., 'Stephen Curry' instead of 'Steph')")
            return

        if df is None or df.empty:
            st.error(f"No game data found for {player_name} in {season}")
            return

        # Player header
        st.header(f"{player['full_name']}")
        st.caption(f"Season: {season} | Games Played: {len(df)}")

        selected_prop = prop_map[prop_type]
        stat_col = get_stat_column(selected_prop, df)

        # Combined stats handling
        if selected_prop == "pts_reb_ast":
            df['COMBINED'] = df['PTS'] + df['REB'] + df['AST']
        elif selected_prop == "pts_reb":
            df['COMBINED'] = df['PTS'] + df['REB']
        elif selected_prop == "pts_ast":
            df['COMBINED'] = df['PTS'] + df['AST']

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Prediction", "Performance", "Game Log"])

        with tab1:
            if models_loaded:
                with st.spinner("Generating prediction..."):
                    result = predictor.get_player_prediction(
                        player_name,
                        prop_type=selected_prop,
                        line=line,
                        include_analysis=True
                    )

                if 'error' in result:
                    st.warning(result['error'])

                    # Show basic stats
                    st.subheader("Recent Statistics")
                    actual_col = stat_col if stat_col in df.columns else 'PTS'
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Last Game", int(df.iloc[-1][actual_col]))
                    with col2:
                        st.metric("Last 5 Avg", f"{df.tail(5)[actual_col].mean():.1f}")
                    with col3:
                        st.metric("Last 10 Avg", f"{df.tail(10)[actual_col].mean():.1f}")
                    with col4:
                        st.metric("Season Avg", f"{df[actual_col].mean():.1f}")
                else:
                    # Display prediction
                    display_prediction_section(result, df, selected_prop, line)

                    # Insights section
                    st.markdown("---")
                    st.subheader("Why This Prediction?")

                    insights = generate_prediction_insights(df, result['prediction'], selected_prop, line)

                    for insight in insights:
                        if insight['type'] == 'positive':
                            st.markdown(f"""
                            <div class="insight-positive">
                                <strong>{insight['title']}:</strong> {insight['text']}
                            </div>
                            """, unsafe_allow_html=True)
                        elif insight['type'] == 'negative':
                            st.markdown(f"""
                            <div class="insight-negative">
                                <strong>{insight['title']}:</strong> {insight['text']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="insight-neutral">
                                <strong>{insight['title']}:</strong> {insight['text']}
                            </div>
                            """, unsafe_allow_html=True)

                    # Workload section
                    if 'MIN' in df.columns:
                        st.markdown("---")
                        st.subheader("Workload Monitor")

                        recent_min = df.tail(5)['MIN'].mean()
                        season_min = df['MIN'].mean()
                        last_game_min = df.iloc[-1]['MIN']

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            delta = last_game_min - season_min
                            st.metric("Last Game MIN", f"{last_game_min:.0f}", delta=f"{delta:+.1f} vs avg")
                        with col2:
                            st.metric("L5 Avg MIN", f"{recent_min:.1f}")
                        with col3:
                            st.metric("Season Avg MIN", f"{season_min:.1f}")

                        # Workload warnings
                        if recent_min < season_min * 0.85:
                            st.warning(f"Minutes trending down - could indicate load management or reduced role")

                        heavy_games = (df.tail(5)['MIN'] >= 35).sum()
                        if heavy_games >= 3:
                            st.info(f"Heavy recent workload ({heavy_games}/5 games with 35+ min) - watch for fatigue")
            else:
                st.info("Train models to see predictions. Showing basic statistics:")
                actual_col = stat_col if stat_col in df.columns else 'PTS'
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Last Game", int(df.iloc[-1][actual_col]))
                with col2:
                    st.metric("Last 5 Avg", f"{df.tail(5)[actual_col].mean():.1f}")
                with col3:
                    st.metric("Last 10 Avg", f"{df.tail(10)[actual_col].mean():.1f}")
                with col4:
                    st.metric("Season Avg", f"{df[actual_col].mean():.1f}")

        with tab2:
            st.subheader("Performance Trends")

            actual_stat_col = stat_col if stat_col in df.columns else 'PTS'
            fig = create_performance_chart(df, actual_stat_col, prop_type, line)
            st.plotly_chart(fig, use_container_width=True)

            if line:
                st.subheader("Hit Rate Analysis")

                lines_to_check = [
                    max(0, line - 2),
                    max(0, line - 1),
                    line,
                    line + 1,
                    line + 2
                ]

                fig_hit = create_hit_rate_chart(df.tail(20), actual_stat_col, lines_to_check)
                st.plotly_chart(fig_hit, use_container_width=True)

                # Last 10 games visual
                st.subheader("Last 10 Games vs Line")
                recent = df.tail(10).copy()

                cols = st.columns(10)
                for idx, (_, row) in enumerate(recent.iterrows()):
                    with cols[idx]:
                        val = row[actual_stat_col]
                        is_over = val > line
                        color = "#28a745" if is_over else "#dc3545"
                        result_text = "O" if is_over else "U"
                        st.markdown(f"""
                        <div style="text-align:center; padding: 10px; background: {color}; border-radius: 8px; color: white;">
                            <div style="font-size: 1.2rem; font-weight: bold;">{int(val)}</div>
                            <div style="font-size: 0.8rem;">{result_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(row['GAME_DATE'].strftime('%m/%d') if hasattr(row['GAME_DATE'], 'strftime') else str(row['GAME_DATE'])[:5])

        with tab3:
            st.subheader("Game Log")

            display_cols = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST',
                          'STL', 'BLK', 'FG_PCT', 'FG3M', 'FG3A', 'FTM', 'FTA', 'PLUS_MINUS']
            available_cols = [c for c in display_cols if c in df.columns]

            display_df = df[available_cols].copy()
            display_df = display_df.sort_values('GAME_DATE', ascending=False)
            display_df['GAME_DATE'] = pd.to_datetime(display_df['GAME_DATE']).dt.strftime('%Y-%m-%d')

            st.dataframe(display_df, use_container_width=True, height=500)

    elif not player_name and analyze_button:
        st.warning("Please enter a player name")


def main():
    """Main app with page navigation."""

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🎯 PrizePicks Analyzer", "🏀 Single Player"],
        index=0
    )

    if page == "🎯 PrizePicks Analyzer":
        prizepicks_page()
    else:
        single_player_page()

    # Footer
    st.markdown("---")
    st.caption(
        "Data sourced from NBA Stats API. Predictions are for entertainment purposes only. "
        "Please gamble responsibly."
    )


if __name__ == "__main__":
    main()
