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


# Page configuration
st.set_page_config(
    page_title="NBA Player Props Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .over-pick {
        background-color: #28a745;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .under-pick {
        background-color: #dc3545;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
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
        st.warning(f"Models not loaded: {e}. Train models first using the training script.")
        return predictor, False


@st.cache_data(ttl=300)  # Cache for 5 minutes
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
        line=dict(color='#667eea', width=2),
        marker=dict(size=8)
    ))

    # Rolling average
    rolling_avg = df[stat].rolling(5, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df['GAME_DATE'],
        y=rolling_avg,
        mode='lines',
        name='5-Game Avg',
        line=dict(color='#ffa726', width=2, dash='dash')
    ))

    # Add betting line if provided
    if line is not None:
        fig.add_hline(
            y=line,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Line: {line}",
            annotation_position="right"
        )

    fig.update_layout(
        title=f"{stat_name} Performance Trend",
        xaxis_title="Game Date",
        yaxis_title=stat_name,
        template="plotly_dark",
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_hit_rate_chart(df: pd.DataFrame, stat: str, lines: list):
    """Create a hit rate chart for various lines."""
    hit_rates = []
    for line in lines:
        rate = (df[stat] > line).mean() * 100
        hit_rates.append(rate)

    fig = go.Figure(data=[
        go.Bar(
            x=[f"Over {l}" for l in lines],
            y=hit_rates,
            marker_color=['#28a745' if r > 50 else '#dc3545' for r in hit_rates],
            text=[f"{r:.1f}%" for r in hit_rates],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="Hit Rate Analysis",
        xaxis_title="Line",
        yaxis_title="Hit Rate %",
        template="plotly_dark",
        height=350,
        yaxis=dict(range=[0, 100])
    )

    return fig


def display_prediction_card(prediction: dict, line: float = None):
    """Display a prediction card with styling."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prediction",
            f"{prediction['prediction']:.1f}",
            delta=f"vs Line: {prediction['prediction'] - line:+.1f}" if line else None
        )

    with col2:
        if prediction.get('std_dev'):
            st.metric(
                "Confidence Range",
                f"{prediction['prediction'] - prediction['std_dev']:.1f} - {prediction['prediction'] + prediction['std_dev']:.1f}"
            )
        else:
            st.metric("Confidence", "N/A")

    with col3:
        if 'betting_analysis' in prediction:
            ba = prediction['betting_analysis']
            recommendation = ba['recommendation']
            edge = max(ba['over_edge'], ba['under_edge']) * 100

            if recommendation == 'OVER':
                st.markdown(f"<span class='over-pick'>OVER {line}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='under-pick'>UNDER {line}</span>", unsafe_allow_html=True)
            st.caption(f"Edge: {edge:.1f}% ({ba['bet_strength']})")


def main():
    st.title("🏀 NBA Player Props Analyzer")
    st.markdown("*AI-powered predictions for sports betting analysis*")

    # Load predictor
    predictor, models_loaded = load_predictor()
    fetcher = NBADataFetcher()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Player search
        player_name = st.text_input(
            "Player Name",
            placeholder="e.g., LeBron James",
            help="Enter the full name of an NBA player"
        )

        # Prop type selection
        prop_type = st.selectbox(
            "Prop Type",
            options=[
                "Points", "Rebounds", "Assists", "Steals", "Blocks",
                "Threes", "Pts+Reb+Ast", "Pts+Reb", "Pts+Ast"
            ],
            index=0
        )

        prop_map = {
            "Points": "points",
            "Rebounds": "rebounds",
            "Assists": "assists",
            "Steals": "steals",
            "Blocks": "blocks",
            "Threes": "threes",
            "Pts+Reb+Ast": "pts_reb_ast",
            "Pts+Reb": "pts_reb",
            "Pts+Ast": "pts_ast"
        }

        # Betting line input
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

        # Season selector
        current_season = get_current_season()
        season = st.selectbox(
            "Season",
            options=[current_season, '2023-24', '2022-23'],
            index=0
        )

        analyze_button = st.button("🔍 Analyze Player", type="primary", use_container_width=True)

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

        # Get stat column
        stat_map_col = {
            "points": "PTS",
            "rebounds": "REB",
            "assists": "AST",
            "steals": "STL",
            "blocks": "BLK",
            "threes": "FG3M"
        }

        selected_prop = prop_map[prop_type]
        stat_col = stat_map_col.get(selected_prop, "PTS")

        # Combined stats handling
        if selected_prop == "pts_reb_ast":
            df['COMBINED'] = df['PTS'] + df['REB'] + df['AST']
            stat_col = 'COMBINED'
        elif selected_prop == "pts_reb":
            df['COMBINED'] = df['PTS'] + df['REB']
            stat_col = 'COMBINED'
        elif selected_prop == "pts_ast":
            df['COMBINED'] = df['PTS'] + df['AST']
            stat_col = 'COMBINED'

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Performance", "📋 Game Log"])

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

                    # Show basic stats instead
                    st.subheader("Recent Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Last Game", int(df.iloc[-1][stat_col] if stat_col in df.columns else df.iloc[-1]['PTS']))
                    with col2:
                        st.metric("Last 5 Avg", f"{df.tail(5)[stat_col if stat_col in df.columns else 'PTS'].mean():.1f}")
                    with col3:
                        st.metric("Last 10 Avg", f"{df.tail(10)[stat_col if stat_col in df.columns else 'PTS'].mean():.1f}")
                    with col4:
                        st.metric("Season Avg", f"{df[stat_col if stat_col in df.columns else 'PTS'].mean():.1f}")
                else:
                    # Display prediction
                    st.subheader("Model Prediction")
                    display_prediction_card(result, line)

                    # Recent stats
                    if 'recent_stats' in result:
                        st.subheader("Recent Performance Context")
                        stats = result['recent_stats']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Last Game", stats.get('last_game', 'N/A'))
                        with col2:
                            st.metric("Last 5 Avg", stats.get('last_5_avg', 'N/A'))
                        with col3:
                            st.metric("Last 10 Avg", stats.get('last_10_avg', 'N/A'))
                        with col4:
                            st.metric("L10 Range", f"{stats.get('last_10_min', 'N/A')} - {stats.get('last_10_max', 'N/A')}")

                    # Betting analysis details
                    if 'betting_analysis' in result:
                        st.subheader("Betting Analysis")
                        ba = result['betting_analysis']

                        col1, col2 = st.columns(2)
                        with col1:
                            # Probability gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=ba['probability_over'] * 100,
                                title={'text': "Probability Over"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#28a745" if ba['probability_over'] > 0.5 else "#dc3545"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "#ffcdd2"},
                                        {'range': [50, 100], 'color': "#c8e6c9"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 52.38  # Break-even at -110
                                    }
                                }
                            ))
                            fig.update_layout(height=300, template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.markdown("**Analysis Details:**")
                            st.write(f"- **Line:** {ba['line']}")
                            st.write(f"- **Prediction vs Line:** {ba['edge_over_line']:+.1f}")
                            st.write(f"- **Over Probability:** {ba['probability_over']*100:.1f}%")
                            st.write(f"- **Under Probability:** {ba['probability_under']*100:.1f}%")
                            st.write(f"- **Over Edge:** {ba['over_edge']*100:+.1f}%")
                            st.write(f"- **Under Edge:** {ba['under_edge']*100:+.1f}%")
                            st.write(f"- **Confidence:** {ba['confidence']*100:.0f}%")
            else:
                st.warning("Models not trained. Showing basic statistics only.")
                st.subheader("Recent Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Last Game", int(df.iloc[-1][stat_col] if stat_col in df.columns else df.iloc[-1]['PTS']))
                with col2:
                    st.metric("Last 5 Avg", f"{df.tail(5)[stat_col if stat_col in df.columns else 'PTS'].mean():.1f}")
                with col3:
                    st.metric("Last 10 Avg", f"{df.tail(10)[stat_col if stat_col in df.columns else 'PTS'].mean():.1f}")
                with col4:
                    st.metric("Season Avg", f"{df[stat_col if stat_col in df.columns else 'PTS'].mean():.1f}")

        with tab2:
            st.subheader("Performance Trends")

            # Main trend chart
            actual_stat_col = stat_col if stat_col in df.columns else 'PTS'
            fig = create_performance_chart(df, actual_stat_col, prop_type, line)
            st.plotly_chart(fig, use_container_width=True)

            # Hit rate analysis
            if line:
                st.subheader("Hit Rate Analysis")

                # Calculate various hit rates
                avg_val = df[actual_stat_col].mean()
                lines_to_check = [
                    max(0, line - 2),
                    max(0, line - 1),
                    line,
                    line + 1,
                    line + 2
                ]

                fig_hit = create_hit_rate_chart(df.tail(20), actual_stat_col, lines_to_check)
                st.plotly_chart(fig_hit, use_container_width=True)

                # Recent game breakdown
                st.subheader("Last 10 Games Breakdown")
                recent = df.tail(10).copy()
                recent['Over Line'] = recent[actual_stat_col] > line

                cols = st.columns(10)
                for idx, (_, row) in enumerate(recent.iterrows()):
                    with cols[idx]:
                        val = row[actual_stat_col]
                        is_over = val > line
                        color = "green" if is_over else "red"
                        st.markdown(f"<div style='text-align:center;color:{color};font-weight:bold;'>{int(val)}</div>", unsafe_allow_html=True)
                        st.caption(row['GAME_DATE'].strftime('%m/%d'))

        with tab3:
            st.subheader("Game Log")

            # Prepare display dataframe
            display_cols = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST',
                          'STL', 'BLK', 'FG_PCT', 'FG3M', 'FG3A', 'FTM', 'FTA', 'PLUS_MINUS']
            available_cols = [c for c in display_cols if c in df.columns]

            display_df = df[available_cols].copy()
            display_df = display_df.sort_values('GAME_DATE', ascending=False)
            display_df['GAME_DATE'] = display_df['GAME_DATE'].dt.strftime('%Y-%m-%d')

            st.dataframe(
                display_df,
                use_container_width=True,
                height=500
            )

    elif not player_name and analyze_button:
        st.warning("Please enter a player name")

    # Footer
    st.markdown("---")
    st.caption(
        "Data sourced from NBA Stats API. Predictions are for entertainment purposes only. "
        "Please gamble responsibly."
    )


if __name__ == "__main__":
    main()
