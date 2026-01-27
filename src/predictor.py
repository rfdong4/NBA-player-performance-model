"""
Prediction Service for NBA Player Performance.
High-level API for making predictions for sports betting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .data_fetcher import NBADataFetcher, get_current_season
from .features import FeatureEngineer, get_betting_features
from .models import ModelTrainer, BettingAnalyzer
from .config import MIN_GAMES_FOR_PREDICTION, MODEL_DIR


class NBAPredictor:
    """High-level prediction service for NBA player performance."""

    def __init__(self, model_path: str = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved models (default: MODEL_DIR)
        """
        self.fetcher = NBADataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.betting_analyzer = None

        self.model_path = model_path or MODEL_DIR
        self._models_loaded = False

    def load_models(self):
        """Load trained models."""
        try:
            self.model_trainer.load(self.model_path)
            self.betting_analyzer = BettingAnalyzer(self.model_trainer)
            self._models_loaded = True
            print(f"Models loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"No models found at {self.model_path}. Please train models first.")
            self._models_loaded = False

    def _ensure_models_loaded(self):
        """Ensure models are loaded before prediction."""
        if not self._models_loaded:
            self.load_models()
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Please train models first.")

    def get_player_prediction(
        self,
        player_name: str,
        prop_type: str = 'points',
        line: float = None,
        include_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Get prediction for a player's next game.

        Args:
            player_name: Name of the player
            prop_type: Type of prop ('points', 'rebounds', 'assists', etc.)
            line: Optional betting line for analysis
            include_analysis: Include betting analysis

        Returns:
            Dict with prediction and analysis
        """
        self._ensure_models_loaded()

        # Find player
        player = self.fetcher.find_player(player_name)
        if player is None:
            return {'error': f"Player not found: {player_name}"}

        # Get recent game data
        current_season = get_current_season()
        df = self.fetcher.get_player_game_log(player['id'], current_season)

        if df.empty:
            return {'error': f"No game data found for {player_name}"}

        if len(df) < MIN_GAMES_FOR_PREDICTION:
            return {
                'error': f"Not enough games ({len(df)}) for reliable prediction. "
                        f"Need at least {MIN_GAMES_FOR_PREDICTION} games."
            }

        # Add player ID for feature engineering
        df['PLAYER_ID'] = player['id']
        df['SEASON'] = current_season

        # Create features
        df_features = self.feature_engineer.create_all_features(df, include_target=False)

        # Get latest row (most recent features)
        latest = df_features.iloc[[-1]]

        # Map prop type to target
        target_map = {
            'points': 'TARGET_PTS',
            'rebounds': 'TARGET_REB',
            'assists': 'TARGET_AST',
            'steals': 'TARGET_STL',
            'blocks': 'TARGET_BLK',
            'turnovers': 'TARGET_TOV',
            'threes': 'TARGET_FG3M',
            'pts_reb_ast': 'TARGET_PRA',
            'pts_reb': 'TARGET_PR',
            'pts_ast': 'TARGET_PA',
            'reb_ast': 'TARGET_RA'
        }

        target = target_map.get(prop_type.lower())
        if target is None:
            return {'error': f"Unknown prop type: {prop_type}"}

        if target not in self.model_trainer.models:
            return {'error': f"No model available for {prop_type}"}

        # Make prediction
        try:
            prediction, std = self.model_trainer.predict_with_confidence(
                latest[self.model_trainer.feature_columns],
                target
            )
        except Exception as e:
            return {'error': f"Prediction failed: {str(e)}"}

        result = {
            'player': player['full_name'],
            'player_id': player['id'],
            'prop_type': prop_type,
            'prediction': float(prediction[0]),
            'std_dev': float(std[0]) if std[0] > 0 else None,
            'games_analyzed': len(df),
            'last_game_date': df['GAME_DATE'].max().strftime('%Y-%m-%d'),
            'recent_stats': self._get_recent_stats(df, prop_type)
        }

        # Add betting analysis if line provided
        if line is not None and include_analysis and self.betting_analyzer:
            analysis = self.betting_analyzer.analyze_line(
                latest[self.model_trainer.feature_columns],
                target,
                line
            )
            result['betting_analysis'] = analysis

        return result

    def get_multi_prop_prediction(
        self,
        player_name: str,
        props: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Get predictions for multiple props for a player.

        Args:
            player_name: Name of the player
            props: Dict of prop_type -> line (optional)

        Returns:
            Dict with predictions for all props
        """
        self._ensure_models_loaded()

        # Find player
        player = self.fetcher.find_player(player_name)
        if player is None:
            return {'error': f"Player not found: {player_name}"}

        # Get data
        current_season = get_current_season()
        df = self.fetcher.get_player_game_log(player['id'], current_season)

        if df.empty or len(df) < MIN_GAMES_FOR_PREDICTION:
            return {'error': "Not enough game data"}

        df['PLAYER_ID'] = player['id']
        df['SEASON'] = current_season

        # Create features
        df_features = self.feature_engineer.create_all_features(df, include_target=False)
        latest = df_features.iloc[[-1]]

        # Get all available predictions
        results = {
            'player': player['full_name'],
            'predictions': {},
            'value_bets': []
        }

        target_names = {
            'TARGET_PTS': 'points',
            'TARGET_REB': 'rebounds',
            'TARGET_AST': 'assists',
            'TARGET_STL': 'steals',
            'TARGET_BLK': 'blocks',
            'TARGET_FG3M': 'threes',
            'TARGET_PRA': 'pts_reb_ast',
        }

        for target, name in target_names.items():
            if target in self.model_trainer.models:
                try:
                    pred, std = self.model_trainer.predict_with_confidence(
                        latest[self.model_trainer.feature_columns],
                        target
                    )
                    results['predictions'][name] = {
                        'prediction': float(pred[0]),
                        'std_dev': float(std[0]) if std[0] > 0 else None
                    }
                except Exception:
                    continue

        # Add betting analysis if lines provided
        if props and self.betting_analyzer:
            lines_mapped = {
                f"TARGET_{k.upper().replace('PTS_REB_AST', 'PRA').replace('PTS_REB', 'PR').replace('PTS_AST', 'PA').replace('REB_AST', 'RA')}": v
                for k, v in props.items()
            }
            results['value_bets'] = self.betting_analyzer.get_value_bets(
                latest[self.model_trainer.feature_columns],
                lines_mapped
            )

        return results

    def _get_recent_stats(self, df: pd.DataFrame, prop_type: str) -> Dict:
        """Get recent statistics for context."""
        stat_map = {
            'points': 'PTS',
            'rebounds': 'REB',
            'assists': 'AST',
            'steals': 'STL',
            'blocks': 'BLK',
            'turnovers': 'TOV',
            'threes': 'FG3M'
        }

        stat = stat_map.get(prop_type.lower())
        if stat is None or stat not in df.columns:
            return {}

        recent = df.tail(10)
        return {
            'last_5_avg': round(df.tail(5)[stat].mean(), 1),
            'last_10_avg': round(recent[stat].mean(), 1),
            'last_10_max': int(recent[stat].max()),
            'last_10_min': int(recent[stat].min()),
            'last_game': int(df.iloc[-1][stat]),
            'hit_rate_over_median': round(
                (recent[stat] > recent[stat].median()).mean() * 100, 1
            )
        }

    def get_today_predictions(
        self,
        player_names: List[str],
        prop_type: str = 'points'
    ) -> List[Dict]:
        """
        Get predictions for multiple players.

        Args:
            player_names: List of player names
            prop_type: Prop type to predict

        Returns:
            List of predictions
        """
        results = []
        for name in player_names:
            result = self.get_player_prediction(name, prop_type)
            results.append(result)
        return results

    def compare_to_line(
        self,
        player_name: str,
        prop_type: str,
        line: float
    ) -> Dict[str, Any]:
        """
        Compare prediction to a betting line.

        Args:
            player_name: Player name
            prop_type: Type of prop
            line: Betting line

        Returns:
            Dict with comparison analysis
        """
        return self.get_player_prediction(
            player_name,
            prop_type,
            line=line,
            include_analysis=True
        )


class PredictionFormatter:
    """Formats predictions for display."""

    @staticmethod
    def format_prediction(result: Dict) -> str:
        """Format a single prediction for display."""
        if 'error' in result:
            return f"Error: {result['error']}"

        output = []
        output.append(f"\n{'='*50}")
        output.append(f"Player: {result['player']}")
        output.append(f"Prop: {result['prop_type'].upper()}")
        output.append(f"{'='*50}")
        output.append(f"\nPrediction: {result['prediction']:.1f}")

        if result.get('std_dev'):
            output.append(f"Confidence Range: {result['prediction'] - result['std_dev']:.1f} - {result['prediction'] + result['std_dev']:.1f}")

        if 'recent_stats' in result:
            stats = result['recent_stats']
            output.append(f"\nRecent Performance:")
            output.append(f"  Last Game: {stats.get('last_game', 'N/A')}")
            output.append(f"  Last 5 Avg: {stats.get('last_5_avg', 'N/A')}")
            output.append(f"  Last 10 Avg: {stats.get('last_10_avg', 'N/A')}")
            output.append(f"  Last 10 Range: {stats.get('last_10_min', 'N/A')} - {stats.get('last_10_max', 'N/A')}")

        if 'betting_analysis' in result:
            ba = result['betting_analysis']
            output.append(f"\nBetting Analysis (Line: {ba['line']}):")
            output.append(f"  Prediction vs Line: {ba['edge_over_line']:+.1f}")
            output.append(f"  Probability Over: {ba['probability_over']*100:.1f}%")
            output.append(f"  Probability Under: {ba['probability_under']*100:.1f}%")
            output.append(f"  Recommendation: {ba['recommendation']} ({ba['bet_strength']})")
            output.append(f"  Edge: {max(ba['over_edge'], ba['under_edge'])*100:.1f}%")

        return '\n'.join(output)

    @staticmethod
    def format_value_bets(value_bets: List[Dict]) -> str:
        """Format value bets for display."""
        if not value_bets:
            return "No value bets found."

        output = ["\nValue Bets Found:"]
        output.append("-" * 40)

        for bet in value_bets:
            edge = max(bet['over_edge'], bet['under_edge']) * 100
            output.append(
                f"{bet['prop']:12} | Line: {bet['line']:5.1f} | "
                f"Pred: {bet['prediction']:5.1f} | "
                f"{bet['recommendation']:5} | Edge: {edge:+.1f}%"
            )

        return '\n'.join(output)
