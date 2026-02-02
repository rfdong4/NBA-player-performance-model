"""
PrizePicks Analyzer - Analyzes props and ranks by probability of hitting.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from collections import defaultdict
import logging

from .predictor import NBAPredictor

logger = logging.getLogger(__name__)


@dataclass
class PropEntry:
    """Represents a single PrizePicks prop entry."""
    player_name: str
    prop_type: str  # Our model's prop type (e.g., 'points', 'rebounds')
    line: float
    prop_type_display: str = ""  # PrizePicks display name
    team: str = ""
    position: str = ""
    start_time: str = ""


class PrizePicksAnalyzer:
    """Analyzes multiple PrizePicks props and ranks by probability of hitting."""

    def __init__(self, predictor: NBAPredictor):
        """
        Initialize the analyzer.

        Args:
            predictor: NBAPredictor instance with loaded models
        """
        self.predictor = predictor

    def analyze_props(
        self,
        props: List[PropEntry],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple props and return ranked results.

        For each prop, calculates the probability of hitting OVER and UNDER,
        then recommends the direction with higher probability.

        Args:
            props: List of PropEntry objects to analyze
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of analysis results sorted by best probability (descending).
            Each result contains:
            - player_name: str
            - prop_type: str
            - prop_type_display: str
            - line: float
            - prediction: float (model's predicted value)
            - probability_over: float
            - probability_under: float
            - best_pick: str ('OVER' or 'UNDER')
            - best_probability: float
            - edge: float (probability - break-even)
            - confidence: str ('high', 'medium', 'low')
            - model_std: float (model uncertainty)
            - error: str (if analysis failed)
        """
        results = []
        total = len(props)

        for i, prop in enumerate(props):
            if progress_callback:
                progress_callback(i + 1, total)

            result = self._analyze_single_prop(prop)
            results.append(result)

        # Sort by best probability (descending)
        results.sort(key=lambda x: x.get('best_probability', 0), reverse=True)

        # Add rank
        for i, result in enumerate(results):
            result['rank'] = i + 1

        return results

    def _analyze_single_prop(self, prop: PropEntry) -> Dict[str, Any]:
        """Analyze a single prop entry."""
        base_result = {
            'player_name': prop.player_name,
            'prop_type': prop.prop_type,
            'prop_type_display': prop.prop_type_display or prop.prop_type.title(),
            'line': prop.line,
            'team': prop.team,
            'position': prop.position,
            'start_time': prop.start_time,
        }

        try:
            # Get prediction from model
            prediction = self.predictor.get_player_prediction(
                player_name=prop.player_name,
                prop_type=prop.prop_type,
                line=prop.line
            )

            if 'error' in prediction:
                return {
                    **base_result,
                    'error': prediction['error'],
                    'best_probability': 0,
                }

            # Extract betting analysis
            betting = prediction.get('betting_analysis', {})

            prob_over = betting.get('probability_over', 0.5)
            prob_under = betting.get('probability_under', 0.5)

            # Determine best pick
            if prob_over >= prob_under:
                best_pick = 'OVER'
                best_prob = prob_over
            else:
                best_pick = 'UNDER'
                best_prob = prob_under

            # Calculate edge over break-even (-110 odds = 52.4%)
            breakeven = 0.524
            edge = best_prob - breakeven

            # Determine confidence level
            if best_prob >= 0.60:
                confidence = 'high'
            elif best_prob >= 0.55:
                confidence = 'medium'
            else:
                confidence = 'low'

            return {
                **base_result,
                'prediction': prediction.get('prediction', 0),
                'probability_over': prob_over,
                'probability_under': prob_under,
                'best_pick': best_pick,
                'best_probability': best_prob,
                'edge': edge,
                'confidence': confidence,
                'model_std': prediction.get('std_dev', 0),
                'games_analyzed': prediction.get('games_analyzed', 0),
            }

        except Exception as e:
            logger.warning(f"Failed to analyze prop for {prop.player_name}: {e}")
            return {
                **base_result,
                'error': str(e),
                'best_probability': 0,
            }

    def get_correlation_warnings(self, props: List[PropEntry]) -> List[str]:
        """
        Identify correlated props that may affect parlay probability.

        Args:
            props: List of PropEntry objects

        Returns:
            List of warning messages
        """
        warnings = []

        # Group props by player
        player_props = defaultdict(list)
        for prop in props:
            player_props[prop.player_name.lower()].append(prop)

        # Check for same-player multiple props
        for player, player_prop_list in player_props.items():
            if len(player_prop_list) > 1:
                prop_types = [p.prop_type_display or p.prop_type for p in player_prop_list]
                warnings.append(
                    f"Multiple props for {player_prop_list[0].player_name}: "
                    f"{', '.join(prop_types)}. These are highly correlated."
                )

        return warnings

    def filter_results(
        self,
        results: List[Dict],
        min_probability: float = 0.0,
        prop_types: Optional[List[str]] = None,
        confidence_levels: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Filter analysis results.

        Args:
            results: List of analysis results
            min_probability: Minimum best_probability threshold
            prop_types: List of prop types to include (None = all)
            confidence_levels: List of confidence levels to include (None = all)

        Returns:
            Filtered list of results
        """
        filtered = []

        for result in results:
            # Skip errored results
            if 'error' in result:
                continue

            # Check probability threshold
            if result.get('best_probability', 0) < min_probability:
                continue

            # Check prop type filter
            if prop_types and result.get('prop_type') not in prop_types:
                continue

            # Check confidence filter
            if confidence_levels and result.get('confidence') not in confidence_levels:
                continue

            filtered.append(result)

        return filtered

    def get_summary_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate summary statistics for analysis results.

        Args:
            results: List of analysis results

        Returns:
            Dict with summary stats
        """
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            return {
                'total_props': len(results),
                'analyzed_props': 0,
                'failed_props': len(results),
            }

        probs = [r['best_probability'] for r in valid_results]
        edges = [r['edge'] for r in valid_results]

        # Count by confidence
        high_conf = sum(1 for r in valid_results if r.get('confidence') == 'high')
        med_conf = sum(1 for r in valid_results if r.get('confidence') == 'medium')
        low_conf = sum(1 for r in valid_results if r.get('confidence') == 'low')

        # Best pick
        best = max(valid_results, key=lambda x: x['best_probability'])

        return {
            'total_props': len(results),
            'analyzed_props': len(valid_results),
            'failed_props': len(results) - len(valid_results),
            'avg_probability': sum(probs) / len(probs),
            'max_probability': max(probs),
            'min_probability': min(probs),
            'avg_edge': sum(edges) / len(edges),
            'positive_edge_count': sum(1 for e in edges if e > 0),
            'high_confidence_count': high_conf,
            'medium_confidence_count': med_conf,
            'low_confidence_count': low_conf,
            'best_pick': {
                'player': best['player_name'],
                'prop': best.get('prop_type_display', best['prop_type']),
                'line': best['line'],
                'pick': best['best_pick'],
                'probability': best['best_probability'],
            },
        }
