# NBA Player Performance Predictor

A machine learning model that predicts NBA player performance (points scored) for upcoming games using historical game data and rolling averages.

## 🏀 Overview

This project uses a Random Forest regression model to predict how many points an NBA player will score in their next game based on their recent performance trends. The model analyzes rolling averages of key performance metrics over the last 10 games to make predictions.

## ✨ Features

- **Real-time Data Fetching**: Uses the NBA API to fetch current player game logs
- **Rolling Average Analysis**: Calculates 10-game rolling averages for key metrics:
  - Minutes played
  - Field goals made/attempted
  - Field goal percentage
  - Points scored
  - Rebounds
  - Assists
- **Interactive Web Interface**: Streamlit-based web app for easy user interaction
- **Machine Learning Model**: Random Forest regressor trained on 2023-24 season data
- **Performance Metrics**: Model achieves R² score of 0.66 with MAE of 3.94 points

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd NBA-player-performance-model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib nba_api
   ```

## 📊 Model Performance

The model was trained on NBA 2023-24 season data and achieves:
- **Mean Absolute Error (MAE)**: 3.94 points
- **Root Mean Squared Error (RMSE)**: 5.23 points  
- **R² Score**: 0.66

## 🚀 Usage

### Web Application

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the provided URL (usually `http://localhost:8501`)

3. **Enter a player's full name** (e.g., "LeBron James", "Stephen Curry")

4. **Click "Predict"** to get the predicted points for their next game

### Programmatic Usage

```python
from app import predict_player_performance

# Predict performance for a player
predicted_points = predict_player_performance("LeBron James")
print(f"Predicted points: {predicted_points:.2f}")
```

## 📁 Project Structure

```
NBA-player-performance-model/
├── app.py                          # Streamlit web application
├── nbaPerformanceModel.ipynb       # Jupyter notebook with model training
├── nbaPerformanceModel.joblib      # Trained Random Forest model
├── model_feature_names.joblib      # Feature names for model input
└── README.md                       # This file
```

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: Random Forest Regressor
- **Features**: 7 rolling average metrics over 10-game windows
- **Training Data**: 10,185 games (before 2024-01-01)
- **Test Data**: 12,902 games (after 2024-01-01)

### Feature Engineering
The model uses rolling averages of the following metrics:
- `ROLLING_MIN`: Average minutes played
- `ROLLING_FGM`: Average field goals made
- `ROLLING_FGA`: Average field goal attempts
- `ROLLING_FG_PCT`: Average field goal percentage
- `ROLLING_PTS`: Average points scored
- `ROLLING_REB`: Average rebounds
- `ROLLING_AST`: Average assists

### Data Sources
- **NBA API**: Real-time player game logs
- **Historical Data**: 2023-24 season game statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.


## 🔮 Future Improvements

- [ ] Add more features (opponent strength, home/away games, rest days)
- [ ] Implement ensemble methods for better accuracy
- [ ] Add confidence intervals to predictions
- [ ] Support for predicting other statistics (rebounds, assists, etc.)
- [ ] Real-time model retraining with new data
- [ ] Mobile app version

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.