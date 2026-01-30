#!/usr/bin/env python3
"""
Model Training Script for NBA Player Performance Prediction.
Fetches data from NBA API and trains models for various betting props.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import NBADataFetcher, get_current_season
from src.features import FeatureEngineer
from src.models import ModelTrainer
from src.config import MODEL_DIR, TRAINING_SEASONS, DATA_DIR


def fetch_training_data(
    n_players: int = 100,
    seasons: list = None,
    save_raw: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch training data from NBA API.

    Args:
        n_players: Number of players to fetch data for
        seasons: List of seasons to fetch
        save_raw: Save raw data to CSV
        verbose: Print progress

    Returns:
        DataFrame with all game logs
    """
    if seasons is None:
        seasons = TRAINING_SEASONS

    fetcher = NBADataFetcher()

    if verbose:
        print(f"Fetching data for {n_players} players across {len(seasons)} seasons...")
        print(f"Seasons: {seasons}")

    # Get active player IDs
    player_ids = fetcher.get_active_players_sample(n_players)

    if verbose:
        print(f"Selected {len(player_ids)} players")

    # Fetch data
    df = fetcher.fetch_multi_player_data(player_ids, seasons, verbose=verbose)

    if df.empty:
        raise ValueError("No data fetched!")

    if verbose:
        print(f"\nTotal games fetched: {len(df)}")
        print(f"Unique players: {df['PLAYER_ID'].nunique()}")
        print(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")

    # Save raw data
    if save_raw:
        os.makedirs(DATA_DIR, exist_ok=True)
        raw_path = os.path.join(DATA_DIR, 'raw_game_logs.csv')
        df.to_csv(raw_path, index=False)
        if verbose:
            print(f"Raw data saved to {raw_path}")

    return df


def load_existing_data(path: str = None) -> pd.DataFrame:
    """
    Load existing training data from CSV.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with game logs
    """
    if path is None:
        path = os.path.join(DATA_DIR, 'raw_game_logs.csv')

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    return df


def prepare_training_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Prepare training data with features.

    Args:
        df: Raw game log DataFrame
        verbose: Print progress

    Returns:
        DataFrame with features and targets
    """
    if verbose:
        print("\nEngineering features...")

    fe = FeatureEngineer()
    df_features = fe.create_all_features(df, include_target=True)

    if verbose:
        print(f"Created {len(fe.get_feature_columns())} features")
        print(f"Sample features: {fe.get_feature_columns()[:10]}...")

    # Drop rows with missing features
    feature_cols = fe.get_feature_columns()
    target_cols = [c for c in df_features.columns if c.startswith('TARGET_')]

    initial_len = len(df_features)
    df_features = df_features.dropna(subset=feature_cols + target_cols)

    if verbose:
        print(f"Dropped {initial_len - len(df_features)} rows with missing values")
        print(f"Final training samples: {len(df_features)}")

    return df_features, feature_cols


def train_models(
    df: pd.DataFrame,
    feature_cols: list,
    model_type: str = 'ensemble',
    test_size: float = 0.2,
    use_feature_selection: bool = True,
    verbose: bool = True
):
    """
    Train models for all targets.

    Args:
        df: DataFrame with features and targets
        feature_cols: List of feature column names
        model_type: Type of model to train
        test_size: Fraction of data for testing
        use_feature_selection: Whether to use automatic feature selection
        verbose: Print progress

    Returns:
        Trained ModelTrainer instance
    """
    if verbose:
        print(f"\nTraining {model_type} models...")
        if use_feature_selection:
            print("Feature selection: ENABLED")
        else:
            print("Feature selection: DISABLED")

    trainer = ModelTrainer(model_type=model_type, use_feature_selection=use_feature_selection)

    # Get target columns
    targets = [c for c in df.columns if c.startswith('TARGET_')]

    if verbose:
        print(f"Training models for {len(targets)} targets:")
        for t in targets:
            print(f"  - {t}")

    # Train models
    metrics = trainer.train(
        df,
        feature_cols,
        targets=targets,
        test_size=test_size,
        use_time_split=True,
        verbose=verbose
    )

    return trainer, metrics


def print_metrics_summary(metrics: dict):
    """Print a summary of model metrics."""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)

    for target, m in metrics.items():
        target_name = target.replace('TARGET_', '')
        print(f"\n{target_name}:")
        print(f"  MAE:  {m['mae']:.2f}")
        print(f"  RMSE: {m['rmse']:.2f}")
        print(f"  R²:   {m['r2']:.3f}")
        print(f"  Mean Actual: {m['mean_actual']:.1f} ± {m['std_actual']:.1f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Train NBA player performance prediction models'
    )
    parser.add_argument(
        '--players', '-p',
        type=int,
        default=100,
        help='Number of players to fetch data for (default: 100)'
    )
    parser.add_argument(
        '--model-type', '-m',
        type=str,
        choices=['random_forest', 'gradient_boosting', 'xgboost', 'ensemble'],
        default='ensemble',
        help='Model type to train (default: ensemble)'
    )
    parser.add_argument(
        '--no-feature-selection',
        action='store_true',
        help='Disable automatic feature selection'
    )
    parser.add_argument(
        '--use-existing', '-e',
        action='store_true',
        help='Use existing data file instead of fetching new data'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to existing data file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=MODEL_DIR,
        help=f'Output directory for models (default: {MODEL_DIR})'
    )
    parser.add_argument(
        '--test-size', '-t',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    print("=" * 60)
    print("NBA PLAYER PERFORMANCE MODEL TRAINER")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model type: {args.model_type}")
    print(f"Output directory: {args.output_dir}")

    # Fetch or load data
    if args.use_existing:
        if verbose:
            print("\nLoading existing data...")
        df = load_existing_data(args.data_path)
    else:
        df = fetch_training_data(
            n_players=args.players,
            save_raw=True,
            verbose=verbose
        )

    # Prepare features
    df_features, feature_cols = prepare_training_data(df, verbose=verbose)

    # Train models
    trainer, metrics = train_models(
        df_features,
        feature_cols,
        model_type=args.model_type,
        test_size=args.test_size,
        use_feature_selection=not args.no_feature_selection,
        verbose=verbose
    )

    # Print summary
    print_metrics_summary(metrics)

    # Save models
    trainer.save(args.output_dir)

    # Print feature importance for main target
    if verbose:
        print("\nTop 10 Features for Points Prediction:")
        importance = trainer.get_feature_importance('TARGET_PTS')
        if not importance.empty:
            print(importance.head(10).to_string(index=False))

    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
