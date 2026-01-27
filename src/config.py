"""
Configuration settings for the NBA Player Performance Model.
"""

# Rolling window sizes for feature engineering
ROLLING_WINDOWS = [3, 5, 10, 15]

# Feature groups
BASIC_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
SHOOTING_STATS = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT']
ADVANCED_STATS = ['MIN', 'PLUS_MINUS', 'OREB', 'DREB']

# All stats to fetch
ALL_STATS = BASIC_STATS + SHOOTING_STATS + ADVANCED_STATS

# Prediction targets for betting
PREDICTION_TARGETS = {
    'points': 'PTS',
    'rebounds': 'REB',
    'assists': 'AST',
    'steals': 'STL',
    'blocks': 'BLK',
    'turnovers': 'TOV',
    'threes': 'FG3M',
    'pts_reb_ast': ['PTS', 'REB', 'AST'],  # Combined prop
    'pts_reb': ['PTS', 'REB'],
    'pts_ast': ['PTS', 'AST'],
    'reb_ast': ['REB', 'AST'],
}

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Data paths
MODEL_DIR = 'models'
DATA_DIR = 'data'

# API settings
NBA_API_DELAY = 0.6  # Seconds between API calls to avoid rate limiting

# Seasons to fetch for training (most recent seasons)
TRAINING_SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25']

# Minimum games required for prediction
MIN_GAMES_FOR_PREDICTION = 10
