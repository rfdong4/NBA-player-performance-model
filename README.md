# NBA Player Props Analyzer

An AI-powered NBA player performance prediction system designed for sports betting analysis. Predicts multiple player props including points, rebounds, assists, and combined statistics.

## Features

- **Multiple Prediction Targets**: Points, rebounds, assists, steals, blocks, threes, and combined props (PRA, PR, PA)
- **Advanced Feature Engineering**: 50+ features including:
  - Multi-window rolling averages (3, 5, 10, 15 games)
  - Performance trends and momentum indicators
  - Consistency metrics (standard deviation, coefficient of variation)
  - Efficiency stats (true shooting %, points per minute)
  - Context features (rest days, home/away, season progress)
- **Betting Analysis**:
  - Over/under probability calculations
  - Edge detection against betting lines
  - Value bet identification
  - Confidence scoring
- **Interactive Dashboard**: Streamlit app with:
  - Real-time predictions
  - Performance trend charts
  - Hit rate analysis
  - Complete game logs

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd NBA-player-performance-model

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Models

Before making predictions, you need to train the models:

```bash
# Train with default settings (100 players, random forest)
python train_models.py

# Train with more players for better accuracy
python train_models.py --players 200

# Use XGBoost for potentially better performance
python train_models.py --model-type xgboost --players 150

# Use existing data (if you've already fetched it)
python train_models.py --use-existing
```

Training options:
- `--players, -p`: Number of players to fetch data for (default: 100)
- `--model-type, -m`: Model type - random_forest, gradient_boosting, or xgboost
- `--use-existing, -e`: Use existing data file instead of fetching new data
- `--test-size, -t`: Fraction of data for testing (default: 0.2)
- `--output-dir, -o`: Output directory for models

### 2. Run the Web App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### 3. Make Predictions

In the web app:
1. Enter a player's full name (e.g., "LeBron James")
2. Select the prop type (Points, Rebounds, Assists, etc.)
3. Optionally enter a betting line for analysis
4. Click "Analyze Player"

## Project Structure

```
NBA-player-performance-model/
├── app.py                    # Streamlit web application
├── train_models.py           # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration settings
│   ├── data_fetcher.py      # NBA API data fetching
│   ├── features.py          # Feature engineering
│   ├── models.py            # Model training and prediction
│   └── predictor.py         # High-level prediction service
├── models/                   # Trained model files (created after training)
└── data/                     # Cached data files (created after training)
```

## Programmatic Usage

```python
from src.predictor import NBAPredictor

# Initialize and load models
predictor = NBAPredictor()
predictor.load_models()

# Get a prediction
result = predictor.get_player_prediction(
    player_name="Stephen Curry",
    prop_type="points",
    line=28.5
)

print(f"Prediction: {result['prediction']:.1f}")
print(f"Recommendation: {result['betting_analysis']['recommendation']}")
print(f"Edge: {max(result['betting_analysis']['over_edge'], result['betting_analysis']['under_edge'])*100:.1f}%")

# Get predictions for multiple props
multi_result = predictor.get_multi_prop_prediction(
    player_name="LeBron James",
    props={
        "points": 25.5,
        "rebounds": 7.5,
        "assists": 8.5
    }
)
```

## Feature Details

### Rolling Statistics (per window: 3, 5, 10, 15 games)
- Points, rebounds, assists, steals, blocks, turnovers
- Field goals made/attempted, FG%
- Three-pointers made/attempted, 3P%
- Free throws made/attempted, FT%
- Minutes played, plus/minus

### Derived Features
- **Trends**: Short-term vs long-term performance comparison
- **Momentum**: Percentage change from baseline
- **Consistency**: Standard deviation and coefficient of variation
- **Efficiency**: True shooting %, points per minute, points per FGA
- **Context**: Days rest, back-to-back games, home/away splits

### Target Variables
- Individual stats: PTS, REB, AST, STL, BLK, TOV, FG3M
- Combined props: PTS+REB+AST, PTS+REB, PTS+AST, REB+AST

## Model Performance

Performance varies by target, but typical metrics:
- **Points MAE**: ~4-5 points
- **Rebounds MAE**: ~1.5-2 rebounds
- **Assists MAE**: ~1.5-2 assists
- **R-squared**: 0.55-0.70 depending on stat

## Data Sources

- **NBA Stats API**: Real-time player game logs and statistics
- **Historical Data**: Multiple seasons for training (configurable)

## Disclaimer

This tool is for entertainment and educational purposes only. Predictions are based on historical data and statistical models, which cannot account for all factors affecting player performance. Always gamble responsibly and within your means.

## License

MIT License - see LICENSE file for details.
