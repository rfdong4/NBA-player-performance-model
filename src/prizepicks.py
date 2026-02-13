"""
PrizePicks Integration Module.
Analyze and compare model predictions against PrizePicks lines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .predictor import NBAPredictor
from .data_fetcher import NBADataFetcher


@dataclass
class PropPick:
    """Represents a single prop pick with analysis."""
    player_name: str
    prop_type: str
    line: float
    prediction: float
    edge: float
    probability: float
    recommendation: str  # 'OVER' or 'UNDER'
    confidence: str  # 'strong', 'moderate', 'weak'
    recent_hit_rate: float
    last_5_avg: float
    trend: str  # 'hot', 'cold', 'stable'
    minutes_concern: bool
    reasoning: List[str]

    @property
    def edge_pct(self) -> float:
        return self.edge * 100

    @property
    def diff_from_line(self) -> float:
        return self.prediction - self.line


class PrizePicksAnalyzer:
    """Analyze PrizePicks props and find the best opportunities."""

    def __init__(self, predictor: NBAPredictor = None):
        """
        Initialize the analyzer.

        Args:
            predictor: NBAPredictor instance (will create one if not provided)
        """
        self.predictor = predictor or NBAPredictor()
        self.fetcher = NBADataFetcher()
        self._models_loaded = False

    def load_models(self):
        """Load prediction models."""
        try:
            self.predictor.load_models()
            self._models_loaded = True
        except Exception as e:
            print(f"Failed to load models: {e}")
            self._models_loaded = False

    def analyze_prop(
        self,
        player_name: str,
        prop_type: str,
        line: float
    ) -> Optional[PropPick]:
        """
        Analyze a single PrizePicks prop.

        Args:
            player_name: Player's full name
            prop_type: Type of prop ('points', 'rebounds', 'assists', etc.)
            line: PrizePicks line

        Returns:
            PropPick object with analysis or None if analysis fails
        """
        if not self._models_loaded:
            self.load_models()

        # Get prediction
        result = self.predictor.get_player_prediction(
            player_name,
            prop_type=prop_type,
            line=line,
            include_analysis=True
        )

        if 'error' in result:
            return None

        # Get player data for additional analysis
        player = self.fetcher.find_player(player_name)
        if not player:
            return None

        from .data_fetcher import get_current_season
        df = self.fetcher.get_player_game_log(player['id'], get_current_season())

        if df is None or df.empty:
            return None

        # Map prop type to stat column
        stat_map = {
            'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST',
            'steals': 'STL', 'blocks': 'BLK', 'threes': 'FG3M',
            'turnovers': 'TOV'
        }
        stat_col = stat_map.get(prop_type.lower(), 'PTS')

        if stat_col not in df.columns:
            return None

        # Calculate additional metrics
        last_5_avg = df.tail(5)[stat_col].mean()
        last_10_avg = df.tail(10)[stat_col].mean()
        recent_hit_rate = (df.tail(10)[stat_col] > line).mean()

        # Determine trend
        if last_5_avg > last_10_avg * 1.1:
            trend = 'hot'
        elif last_5_avg < last_10_avg * 0.9:
            trend = 'cold'
        else:
            trend = 'stable'

        # Check minutes concern
        minutes_concern = False
        if 'MIN' in df.columns:
            recent_min = df.tail(5)['MIN'].mean()
            season_min = df['MIN'].mean()
            minutes_concern = recent_min < season_min * 0.85

        # Build reasoning
        reasoning = []
        prediction = result['prediction']
        diff = prediction - line

        if abs(diff) > 3:
            reasoning.append(f"Model predicts {prediction:.1f}, significantly {'above' if diff > 0 else 'below'} the {line} line")
        else:
            reasoning.append(f"Model predicts {prediction:.1f}, close to the {line} line")

        if recent_hit_rate >= 0.7:
            reasoning.append(f"Strong recent form: hit over {line} in {recent_hit_rate*100:.0f}% of last 10 games")
        elif recent_hit_rate <= 0.3:
            reasoning.append(f"Struggling vs line: only hit over {line} in {recent_hit_rate*100:.0f}% of last 10 games")

        if trend == 'hot':
            reasoning.append(f"Player is HOT: L5 avg ({last_5_avg:.1f}) up from L10 ({last_10_avg:.1f})")
        elif trend == 'cold':
            reasoning.append(f"Player is COLD: L5 avg ({last_5_avg:.1f}) down from L10 ({last_10_avg:.1f})")

        if minutes_concern:
            reasoning.append("WARNING: Recent minutes down - possible load management")

        # Get betting analysis
        ba = result.get('betting_analysis', {})

        return PropPick(
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            prediction=prediction,
            edge=max(ba.get('over_edge', 0), ba.get('under_edge', 0)),
            probability=ba.get('probability_over', 0.5) if ba.get('recommendation') == 'OVER' else ba.get('probability_under', 0.5),
            recommendation=ba.get('recommendation', 'OVER' if diff > 0 else 'UNDER'),
            confidence=ba.get('bet_strength', 'weak'),
            recent_hit_rate=recent_hit_rate,
            last_5_avg=last_5_avg,
            trend=trend,
            minutes_concern=minutes_concern,
            reasoning=reasoning
        )

    def analyze_slate(
        self,
        props: List[Dict[str, Any]]
    ) -> List[PropPick]:
        """
        Analyze a full slate of PrizePicks props.

        Args:
            props: List of dicts with 'player', 'prop_type', and 'line' keys

        Returns:
            List of PropPick objects sorted by edge
        """
        picks = []

        for prop in props:
            pick = self.analyze_prop(
                player_name=prop['player'],
                prop_type=prop['prop_type'],
                line=prop['line']
            )
            if pick:
                picks.append(pick)

        # Sort by edge (best picks first)
        picks.sort(key=lambda x: x.edge, reverse=True)

        return picks

    def get_best_picks(
        self,
        props: List[Dict[str, Any]],
        min_edge: float = 0.05,
        min_confidence: str = 'moderate',
        max_picks: int = 5
    ) -> List[PropPick]:
        """
        Get the best picks from a slate.

        Args:
            props: List of props to analyze
            min_edge: Minimum edge to include (default 5%)
            min_confidence: Minimum confidence level ('weak', 'moderate', 'strong')
            max_picks: Maximum number of picks to return

        Returns:
            List of best PropPick objects
        """
        all_picks = self.analyze_slate(props)

        confidence_order = {'weak': 0, 'moderate': 1, 'strong': 2}
        min_conf_value = confidence_order.get(min_confidence, 1)

        # Filter by criteria
        filtered = [
            p for p in all_picks
            if p.edge >= min_edge
            and confidence_order.get(p.confidence, 0) >= min_conf_value
            and not p.minutes_concern  # Exclude players with minutes concerns
        ]

        return filtered[:max_picks]

    def build_parlay(
        self,
        props: List[Dict[str, Any]],
        legs: int = 3,
        correlation_aware: bool = True
    ) -> Tuple[List[PropPick], float]:
        """
        Build an optimal parlay from available props.

        Args:
            props: List of props to choose from
            legs: Number of legs in parlay
            correlation_aware: Avoid correlated picks (same game)

        Returns:
            Tuple of (list of picks, estimated probability)
        """
        all_picks = self.analyze_slate(props)

        # Filter to strong/moderate picks only
        candidates = [p for p in all_picks if p.confidence in ['strong', 'moderate'] and p.edge > 0.03]

        if len(candidates) < legs:
            candidates = [p for p in all_picks if p.edge > 0][:legs]

        # Sort by edge
        candidates.sort(key=lambda x: x.edge, reverse=True)

        # Select picks (avoiding correlation if requested)
        selected = []
        used_players = set()

        for pick in candidates:
            if len(selected) >= legs:
                break

            # Skip if we already have this player (correlation)
            if correlation_aware and pick.player_name in used_players:
                continue

            selected.append(pick)
            used_players.add(pick.player_name)

        # Calculate combined probability
        combined_prob = 1.0
        for pick in selected:
            combined_prob *= pick.probability

        return selected, combined_prob

    def format_picks_table(self, picks: List[PropPick]) -> str:
        """Format picks as a readable table."""
        if not picks:
            return "No picks found matching criteria."

        lines = []
        lines.append("=" * 100)
        lines.append(f"{'PLAYER':<20} {'PROP':<10} {'LINE':>6} {'PRED':>6} {'DIFF':>6} {'EDGE':>6} {'REC':<6} {'CONF':<8} {'TREND':<6}")
        lines.append("-" * 100)

        for pick in picks:
            diff_str = f"{pick.diff_from_line:+.1f}"
            edge_str = f"{pick.edge_pct:.1f}%"
            trend_emoji = "🔥" if pick.trend == 'hot' else "❄️" if pick.trend == 'cold' else "➡️"

            lines.append(
                f"{pick.player_name:<20} {pick.prop_type:<10} {pick.line:>6.1f} {pick.prediction:>6.1f} "
                f"{diff_str:>6} {edge_str:>6} {pick.recommendation:<6} {pick.confidence:<8} {trend_emoji}"
            )

        lines.append("=" * 100)

        return "\n".join(lines)

    def format_detailed_pick(self, pick: PropPick) -> str:
        """Format a single pick with full details."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  {pick.player_name} - {pick.prop_type.upper()}")
        lines.append(f"{'='*60}")
        lines.append(f"  Line: {pick.line}  |  Prediction: {pick.prediction:.1f}  |  Diff: {pick.diff_from_line:+.1f}")
        lines.append(f"  Recommendation: {pick.recommendation} ({pick.confidence})")
        lines.append(f"  Edge: {pick.edge_pct:.1f}%  |  Probability: {pick.probability*100:.1f}%")
        lines.append(f"  L5 Avg: {pick.last_5_avg:.1f}  |  Hit Rate (L10): {pick.recent_hit_rate*100:.0f}%")
        lines.append(f"  Trend: {pick.trend.upper()}")

        if pick.minutes_concern:
            lines.append(f"  ⚠️  MINUTES CONCERN - Recent minutes down")

        lines.append(f"\n  Analysis:")
        for reason in pick.reasoning:
            lines.append(f"    • {reason}")

        return "\n".join(lines)


def quick_analyze(props_text: str) -> str:
    """
    Quick analysis from text input.

    Args:
        props_text: Text with player props in format:
                   "Player Name, prop_type, line" per line

    Returns:
        Formatted analysis string
    """
    analyzer = PrizePicksAnalyzer()
    analyzer.load_models()

    props = []
    for line in props_text.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            props.append({
                'player': parts[0],
                'prop_type': parts[1].lower(),
                'line': float(parts[2])
            })

    if not props:
        return "No valid props found. Use format: 'Player Name, prop_type, line'"

    picks = analyzer.analyze_slate(props)

    output = []
    output.append("\n" + analyzer.format_picks_table(picks))

    # Show detailed analysis for top 3
    output.append("\n\nTOP PICKS - DETAILED ANALYSIS:")
    for pick in picks[:3]:
        output.append(analyzer.format_detailed_pick(pick))

    # Suggest a parlay
    if len(picks) >= 3:
        parlay_picks, parlay_prob = analyzer.build_parlay([
            {'player': p.player_name, 'prop_type': p.prop_type, 'line': p.line}
            for p in picks
        ], legs=3)

        output.append(f"\n\nSUGGESTED 3-LEG PARLAY (Combined Prob: {parlay_prob*100:.1f}%):")
        for i, pick in enumerate(parlay_picks, 1):
            output.append(f"  {i}. {pick.player_name} {pick.recommendation} {pick.line} {pick.prop_type}")

    return "\n".join(output)
