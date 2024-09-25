import joblib
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import streamlit as st

# Load the trained model
model = joblib.load('nbaPerformanceModel.joblib')

# Load the feature names
model_feature_names = joblib.load('model_feature_names.joblib')

def get_player_data(player_name):
    # Find the player's ID
    player_dict = players.get_players()
    player_info = [player for player in player_dict if player['full_name'].lower() == player_name.lower()]
    
    if not player_info:
        st.error("Player not found.")
        return None
    
    player_id = player_info[0]['id']
    
    # Fetch the player's game log
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2023')  # Update season as needed
        df = gamelog.get_data_frames()[0]
    except Exception as e:
        st.error(f"Error fetching data for {player_name}: {e}")
        return None
    
    # Convert GAME_DATE to datetime and sort
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df.sort_values('GAME_DATE', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def compute_features(player_df):
    # Ensure the required columns are present
    required_columns = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'PTS', 'REB', 'AST', 'GAME_DATE']
    for col in required_columns:
        if col not in player_df.columns:
            st.error(f"Column {col} is missing from player data.")
            return None

    
    # Rolling averages
    rolling_columns = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'PTS', 'REB', 'AST']
    window_size = 10
    for col in rolling_columns:
        player_df[f'ROLLING_{col}'] = player_df[col].rolling(window=window_size, min_periods=1).mean()

    # Select the latest game's features
    latest_game_features = player_df.iloc[-1]

    # Create the feature vector expected by the model
    feature_vector = latest_game_features[[
        
        'ROLLING_MIN',
        'ROLLING_FGM',
        'ROLLING_FGA',
        'ROLLING_FG_PCT',
        'ROLLING_PTS',
        'ROLLING_REB',
        'ROLLING_AST'
    ]].values.reshape(1, -1)  # Reshape for model input

    # Convert to DataFrame for consistency
    feature_vector = pd.DataFrame(feature_vector, columns=[
        
        'ROLLING_MIN',
        'ROLLING_FGM',
        'ROLLING_FGA',
        'ROLLING_FG_PCT',
        'ROLLING_PTS',
        'ROLLING_REB',
        'ROLLING_AST'
    ])

    return feature_vector


def predict_player_performance(player_name):
    # Get player data
    player_df = get_player_data(player_name)
    if player_df is None or player_df.empty:
        return None

    # Compute features
    feature_vector = compute_features(player_df)
    if feature_vector is None:
        return None

    # Make prediction
    predicted_pts = model.predict(feature_vector)[0]

    return predicted_pts


def main():
    st.title("NBA Player Performance Predictor")
    player_name = st.text_input("Enter the NBA player's full name:")
    
    if st.button("Predict"):
        st.write(f"Fetching data for {player_name}...")
        predicted_pts = predict_player_performance(player_name)
        if predicted_pts is not None:
            st.success(f"Predicted points for {player_name} in the next game: {predicted_pts:.2f}")
        else:
            st.error(f"Could not predict performance for {player_name}.")

if __name__ == "__main__":
    main()
