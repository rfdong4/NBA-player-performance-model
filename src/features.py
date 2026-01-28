"""
Feature Engineering Module for NBA Player Performance Prediction.
Creates comprehensive features for sports betting analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from .config import ROLLING_WINDOWS, BASIC_STATS, SHOOTING_STATS, ADVANCED_STATS


class FeatureEngineer:
    """Creates features for NBA player performance prediction."""

    def __init__(self, rolling_windows: List[int] = None):
        """
        Initialize the feature engineer.

        Args:
            rolling_windows: List of window sizes for rolling averages
        """
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.feature_columns = []

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        stats: List[str] = None,
        min_periods: int = 1
    ) -> pd.DataFrame:
        """
        Create rolling average features for specified stats.

        Args:
            df: DataFrame with player game logs (should be sorted by date)
            stats: Stats columns to create rolling features for
            min_periods: Minimum periods required for rolling calculation

        Returns:
            DataFrame with rolling features added
        """
        if stats is None:
            # Use all available stats
            stats = [c for c in BASIC_STATS + SHOOTING_STATS + ADVANCED_STATS
                     if c in df.columns]

        df = df.copy()

        for window in self.rolling_windows:
            for stat in stats:
                if stat not in df.columns:
                    continue

                col_name = f'{stat}_L{window}'
                # Shift by 1 to avoid data leakage (use only past games)
                df[col_name] = (
                    df.groupby('PLAYER_ID')[stat]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean())
                )
                self.feature_columns.append(col_name)

        return df

    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend features showing recent performance direction.

        Args:
            df: DataFrame with rolling features already created

        Returns:
            DataFrame with trend features added
        """
        df = df.copy()

        # Compare short-term to long-term averages for trend detection
        for stat in BASIC_STATS:
            short_col = f'{stat}_L3'
            long_col = f'{stat}_L10'

            if short_col in df.columns and long_col in df.columns:
                # Trend: positive means player is trending up
                trend_col = f'{stat}_TREND'
                df[trend_col] = df[short_col] - df[long_col]
                self.feature_columns.append(trend_col)

                # Momentum: percentage change from long-term average
                momentum_col = f'{stat}_MOMENTUM'
                df[momentum_col] = (df[short_col] / df[long_col].replace(0, np.nan) - 1) * 100
                self.feature_columns.append(momentum_col)

        return df

    def create_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features measuring player consistency.

        Args:
            df: DataFrame with player game logs

        Returns:
            DataFrame with consistency features added
        """
        df = df.copy()

        for stat in ['PTS', 'REB', 'AST', 'FG3M']:
            if stat not in df.columns:
                continue

            for window in [5, 10]:
                # Standard deviation of recent performance
                std_col = f'{stat}_STD_L{window}'
                df[std_col] = (
                    df.groupby('PLAYER_ID')[stat]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=3).std())
                )
                self.feature_columns.append(std_col)

                # Coefficient of variation (std / mean) - relative consistency
                mean_col = f'{stat}_L{window}'
                if mean_col in df.columns:
                    cv_col = f'{stat}_CV_L{window}'
                    df[cv_col] = df[std_col] / df[mean_col].replace(0, np.nan)
                    self.feature_columns.append(cv_col)

        return df

    def create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create efficiency-based features.

        Args:
            df: DataFrame with player game logs

        Returns:
            DataFrame with efficiency features added
        """
        df = df.copy()

        # True Shooting Percentage approximation
        if all(c in df.columns for c in ['PTS', 'FGA', 'FTA']):
            df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
            df['TS_PCT'] = df['TS_PCT'].replace([np.inf, -np.inf], np.nan)

            for window in self.rolling_windows:
                col_name = f'TS_PCT_L{window}'
                df[col_name] = (
                    df.groupby('PLAYER_ID')['TS_PCT']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                self.feature_columns.append(col_name)

        # Points per minute
        if 'MIN' in df.columns and 'PTS' in df.columns:
            # Convert MIN to numeric if it's a string
            if df['MIN'].dtype == 'object':
                df['MIN_NUMERIC'] = df['MIN'].apply(self._parse_minutes)
            else:
                df['MIN_NUMERIC'] = df['MIN']

            df['PTS_PER_MIN'] = df['PTS'] / df['MIN_NUMERIC'].replace(0, np.nan)

            for window in self.rolling_windows:
                col_name = f'PTS_PER_MIN_L{window}'
                df[col_name] = (
                    df.groupby('PLAYER_ID')['PTS_PER_MIN']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                self.feature_columns.append(col_name)

        # Usage-adjusted stats (points per field goal attempt)
        if 'PTS' in df.columns and 'FGA' in df.columns:
            df['PTS_PER_FGA'] = df['PTS'] / df['FGA'].replace(0, np.nan)

            for window in self.rolling_windows:
                col_name = f'PTS_PER_FGA_L{window}'
                df[col_name] = (
                    df.groupby('PLAYER_ID')['PTS_PER_FGA']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                self.feature_columns.append(col_name)

        return df

    def create_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rest-related features.

        Args:
            df: DataFrame with GAME_DATE column

        Returns:
            DataFrame with rest features added
        """
        df = df.copy()

        if 'GAME_DATE' not in df.columns:
            return df

        # Days since last game
        df['DAYS_REST'] = (
            df.groupby('PLAYER_ID')['GAME_DATE']
            .transform(lambda x: x.diff().dt.days)
        )

        # Cap at reasonable values
        df['DAYS_REST'] = df['DAYS_REST'].clip(0, 14)

        # Back-to-back indicator
        df['IS_B2B'] = (df['DAYS_REST'] == 1).astype(int)

        # Extended rest indicator (3+ days)
        df['EXTENDED_REST'] = (df['DAYS_REST'] >= 3).astype(int)

        self.feature_columns.extend(['DAYS_REST', 'IS_B2B', 'EXTENDED_REST'])

        return df

    def create_home_away_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create home/away features.

        Args:
            df: DataFrame with MATCHUP column

        Returns:
            DataFrame with home/away features added
        """
        df = df.copy()

        if 'MATCHUP' in df.columns:
            # Parse home/away from matchup string (e.g., "LAL vs. GSW" = home)
            df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
            self.feature_columns.append('IS_HOME')

            # Rolling home/away splits - using expanding mean as simpler alternative
            for stat in ['PTS', 'REB', 'AST']:
                if stat not in df.columns:
                    continue

                # Create masked columns for home/away stats
                home_stat = df[stat].where(df['IS_HOME'] == 1)
                away_stat = df[stat].where(df['IS_HOME'] == 0)

                # Home performance - use transform with expanding mean (shifted)
                home_col = f'{stat}_HOME_L10'
                df[home_col] = (
                    home_stat.groupby(df['PLAYER_ID'])
                    .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
                )
                self.feature_columns.append(home_col)

                # Away performance
                away_col = f'{stat}_AWAY_L10'
                df[away_col] = (
                    away_stat.groupby(df['PLAYER_ID'])
                    .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
                )
                self.feature_columns.append(away_col)

        return df

    def create_game_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create game context features.

        Args:
            df: DataFrame with game log data

        Returns:
            DataFrame with context features added
        """
        df = df.copy()

        # Games played in season (fatigue indicator)
        df['GAMES_PLAYED_SEASON'] = df.groupby(['PLAYER_ID', 'SEASON']).cumcount()
        self.feature_columns.append('GAMES_PLAYED_SEASON')

        # Season progress (0-1)
        df['SEASON_PROGRESS'] = df['GAMES_PLAYED_SEASON'] / 82
        self.feature_columns.append('SEASON_PROGRESS')

        return df

    def create_minutes_workload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create minutes and workload-related features for injury/load management detection.

        Args:
            df: DataFrame with player game logs

        Returns:
            DataFrame with minutes/workload features added
        """
        df = df.copy()

        if 'MIN' not in df.columns:
            return df

        # Convert MIN to numeric if needed
        if df['MIN'].dtype == 'object':
            df['MIN_NUMERIC'] = df['MIN'].apply(self._parse_minutes)
        else:
            df['MIN_NUMERIC'] = df['MIN']

        # Rolling minutes averages (already in rolling features, but we need specific ones)
        # Minutes trend - comparing recent to longer term
        df['MIN_L3'] = (
            df.groupby('PLAYER_ID')['MIN_NUMERIC']
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        df['MIN_L10'] = (
            df.groupby('PLAYER_ID')['MIN_NUMERIC']
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )

        # Minutes trend (positive = increasing minutes, negative = decreasing/load management)
        df['MIN_TREND'] = df['MIN_L3'] - df['MIN_L10']
        self.feature_columns.append('MIN_TREND')

        # Minutes volatility - high volatility might indicate injury concerns
        df['MIN_STD_L5'] = (
            df.groupby('PLAYER_ID')['MIN_NUMERIC']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
        )
        self.feature_columns.append('MIN_STD_L5')

        # Low minutes games in recent stretch (potential load management indicator)
        df['LOW_MIN_GAMES_L5'] = (
            df.groupby('PLAYER_ID')['MIN_NUMERIC']
            .transform(lambda x: (x.shift(1) < 20).rolling(5, min_periods=1).sum())
        )
        self.feature_columns.append('LOW_MIN_GAMES_L5')

        # Did not play (DNP) or very low minutes in last game
        df['LAST_GAME_LOW_MIN'] = (
            df.groupby('PLAYER_ID')['MIN_NUMERIC']
            .transform(lambda x: (x.shift(1) < 15).astype(int))
        )
        self.feature_columns.append('LAST_GAME_LOW_MIN')

        # Minutes consistency (coefficient of variation)
        df['MIN_CV_L10'] = df['MIN_STD_L5'] / df['MIN_L10'].replace(0, np.nan)
        self.feature_columns.append('MIN_CV_L10')

        # Season high minutes percentage (are they at peak workload?)
        df['MIN_SEASON_MAX'] = (
            df.groupby(['PLAYER_ID', 'SEASON'])['MIN_NUMERIC']
            .transform(lambda x: x.shift(1).expanding().max())
        )
        df['MIN_VS_SEASON_MAX'] = df['MIN_L3'] / df['MIN_SEASON_MAX'].replace(0, np.nan)
        self.feature_columns.append('MIN_VS_SEASON_MAX')

        # Cumulative minutes load (fatigue indicator)
        df['CUMULATIVE_MIN_L5'] = (
            df.groupby('PLAYER_ID')['MIN_NUMERIC']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        )
        self.feature_columns.append('CUMULATIVE_MIN_L5')

        # Heavy workload indicator (played 35+ min in 3+ of last 5 games)
        df['HEAVY_WORKLOAD_L5'] = (
            df.groupby('PLAYER_ID')['MIN_NUMERIC']
            .transform(lambda x: (x.shift(1) >= 35).rolling(5, min_periods=1).sum())
        )
        self.feature_columns.append('HEAVY_WORKLOAD_L5')

        return df

    def create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create opponent-related features based on matchup data.

        Args:
            df: DataFrame with player game logs including MATCHUP column

        Returns:
            DataFrame with opponent features added
        """
        df = df.copy()

        if 'MATCHUP' not in df.columns:
            return df

        # Extract opponent team abbreviation from matchup
        # Format: "LAL vs. GSW" (home) or "LAL @ GSW" (away)
        def extract_opponent(matchup):
            if pd.isna(matchup):
                return None
            if ' vs. ' in matchup:
                return matchup.split(' vs. ')[1]
            elif ' @ ' in matchup:
                return matchup.split(' @ ')[1]
            return None

        df['OPPONENT'] = df['MATCHUP'].apply(extract_opponent)

        # Performance against this opponent (historical)
        for stat in ['PTS', 'REB', 'AST']:
            if stat not in df.columns:
                continue

            # Average against this specific opponent
            opp_col = f'{stat}_VS_OPP_AVG'
            df[opp_col] = (
                df.groupby(['PLAYER_ID', 'OPPONENT'])[stat]
                .transform(lambda x: x.shift(1).expanding().mean())
            )
            self.feature_columns.append(opp_col)

        # Games played vs this opponent (familiarity)
        df['GAMES_VS_OPP'] = (
            df.groupby(['PLAYER_ID', 'OPPONENT']).cumcount()
        )
        self.feature_columns.append('GAMES_VS_OPP')

        # Opponent strength proxy - using player's performance variance against them
        # Higher variance might indicate tougher/more variable opponent
        if 'PTS' in df.columns:
            df['PTS_VS_OPP_STD'] = (
                df.groupby(['PLAYER_ID', 'OPPONENT'])['PTS']
                .transform(lambda x: x.shift(1).expanding().std())
            )
            # Fill NaN with overall std
            overall_std = df.groupby('PLAYER_ID')['PTS'].transform(
                lambda x: x.shift(1).expanding().std()
            )
            df['PTS_VS_OPP_STD'] = df['PTS_VS_OPP_STD'].fillna(overall_std)
            self.feature_columns.append('PTS_VS_OPP_STD')

        # Recent form against division/conference could be added with more data

        return df

    def create_scoring_opportunity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to scoring opportunities and usage.

        Args:
            df: DataFrame with player game logs

        Returns:
            DataFrame with scoring opportunity features
        """
        df = df.copy()

        # Field goal attempts trend (usage indicator)
        if 'FGA' in df.columns:
            df['FGA_L3'] = (
                df.groupby('PLAYER_ID')['FGA']
                .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            )
            df['FGA_L10'] = (
                df.groupby('PLAYER_ID')['FGA']
                .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            )
            df['FGA_TREND'] = df['FGA_L3'] - df['FGA_L10']
            self.feature_columns.append('FGA_TREND')

            # Usage stability
            df['FGA_STD_L5'] = (
                df.groupby('PLAYER_ID')['FGA']
                .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
            )
            self.feature_columns.append('FGA_STD_L5')

        # Three-point attempt rate trend
        if 'FG3A' in df.columns and 'FGA' in df.columns:
            df['THREE_RATE'] = df['FG3A'] / df['FGA'].replace(0, np.nan)
            df['THREE_RATE_L5'] = (
                df.groupby('PLAYER_ID')['THREE_RATE']
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )
            self.feature_columns.append('THREE_RATE_L5')

        # Free throw attempts (getting to the line indicator)
        if 'FTA' in df.columns:
            df['FTA_L5'] = (
                df.groupby('PLAYER_ID')['FTA']
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )
            self.feature_columns.append('FTA_L5')

            # Free throw rate (FTA/FGA)
            if 'FGA' in df.columns:
                df['FT_RATE'] = df['FTA'] / df['FGA'].replace(0, np.nan)
                df['FT_RATE_L5'] = (
                    df.groupby('PLAYER_ID')['FT_RATE']
                    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
                )
                self.feature_columns.append('FT_RATE_L5')

        return df

    def create_all_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Create all features for the model.

        Args:
            df: Raw DataFrame with player game logs
            include_target: Whether to create target variables

        Returns:
            DataFrame with all features
        """
        # Reset feature columns list
        self.feature_columns = []

        # Ensure data is sorted
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        # Create all feature types
        df = self.create_rolling_features(df)
        df = self.create_trend_features(df)
        df = self.create_consistency_features(df)
        df = self.create_efficiency_features(df)
        df = self.create_rest_features(df)
        df = self.create_home_away_features(df)
        df = self.create_game_context_features(df)
        df = self.create_minutes_workload_features(df)
        df = self.create_opponent_features(df)
        df = self.create_scoring_opportunity_features(df)

        if include_target:
            df = self.create_target_variables(df)

        # Remove duplicates from feature columns
        self.feature_columns = list(dict.fromkeys(self.feature_columns))

        return df

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for prediction.

        Args:
            df: DataFrame with player game logs

        Returns:
            DataFrame with target variables added
        """
        df = df.copy()

        # Single stat targets
        for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']:
            if stat in df.columns:
                df[f'TARGET_{stat}'] = df[stat]

        # Combined prop targets
        if all(c in df.columns for c in ['PTS', 'REB', 'AST']):
            df['TARGET_PRA'] = df['PTS'] + df['REB'] + df['AST']
            df['TARGET_PR'] = df['PTS'] + df['REB']
            df['TARGET_PA'] = df['PTS'] + df['AST']
            df['TARGET_RA'] = df['REB'] + df['AST']

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns

    def prepare_features_for_prediction(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for model prediction.

        Args:
            df: DataFrame with features
            feature_cols: Specific feature columns to use

        Returns:
            Tuple of (feature DataFrame, list of valid feature columns)
        """
        if feature_cols is None:
            feature_cols = self.feature_columns

        # Filter to existing columns
        valid_cols = [c for c in feature_cols if c in df.columns]

        # Get feature matrix
        X = df[valid_cols].copy()

        # Handle missing values
        X = X.fillna(X.median())

        return X, valid_cols

    @staticmethod
    def _parse_minutes(min_str) -> float:
        """Parse minutes string (MM:SS) to float."""
        if pd.isna(min_str):
            return 0.0
        if isinstance(min_str, (int, float)):
            return float(min_str)
        try:
            if ':' in str(min_str):
                parts = str(min_str).split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_str)
        except (ValueError, IndexError):
            return 0.0


def get_betting_features(
    player_stats: pd.DataFrame,
    opponent_stats: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Get key features relevant for betting decisions.

    Args:
        player_stats: Player's recent statistics
        opponent_stats: Optional opponent defensive stats

    Returns:
        Dict of key betting features
    """
    features = {}

    if player_stats.empty:
        return features

    latest = player_stats.iloc[-1]

    # Recent averages
    for stat in ['PTS', 'REB', 'AST', 'FG3M']:
        for window in [3, 5, 10]:
            col = f'{stat}_L{window}'
            if col in latest.index:
                features[col] = latest[col]

    # Trends
    for stat in ['PTS', 'REB', 'AST']:
        trend_col = f'{stat}_TREND'
        if trend_col in latest.index:
            features[trend_col] = latest[trend_col]

    # Consistency
    for stat in ['PTS', 'REB', 'AST']:
        std_col = f'{stat}_STD_L10'
        if std_col in latest.index:
            features[std_col] = latest[std_col]

    # Context
    if 'DAYS_REST' in latest.index:
        features['DAYS_REST'] = latest['DAYS_REST']

    if 'IS_HOME' in latest.index:
        features['IS_HOME'] = latest['IS_HOME']

    return features
