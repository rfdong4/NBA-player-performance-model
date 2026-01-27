"""
NBA Data Fetcher - Comprehensive data retrieval from NBA Stats API.
Handles player data, game logs, team schedules, and opponent stats.
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import warnings

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playergamelog,
    commonplayerinfo,
    leaguegamefinder,
    teamgamelog,
    scoreboardv2,
    leaguedashteamstats,
    playerdashboardbygeneralsplits
)

from .config import NBA_API_DELAY, TRAINING_SEASONS

warnings.filterwarnings('ignore')


class NBADataFetcher:
    """Fetches and processes NBA data for player performance modeling."""

    def __init__(self, delay: float = NBA_API_DELAY):
        """
        Initialize the data fetcher.

        Args:
            delay: Seconds to wait between API calls to avoid rate limiting
        """
        self.delay = delay
        self._players_cache = None
        self._teams_cache = None

    def _api_call_with_delay(self, func, **kwargs):
        """Execute API call with delay to avoid rate limiting."""
        time.sleep(self.delay)
        try:
            return func(**kwargs)
        except Exception as e:
            print(f"API call failed: {e}")
            time.sleep(2)  # Extra delay on failure
            return func(**kwargs)

    @property
    def all_players(self) -> List[Dict]:
        """Get all NBA players (cached)."""
        if self._players_cache is None:
            self._players_cache = players.get_players()
        return self._players_cache

    @property
    def all_teams(self) -> List[Dict]:
        """Get all NBA teams (cached)."""
        if self._teams_cache is None:
            self._teams_cache = teams.get_teams()
        return self._teams_cache

    def find_player(self, name: str) -> Optional[Dict]:
        """
        Find a player by name (partial match supported).

        Args:
            name: Player name to search for

        Returns:
            Player dict or None if not found
        """
        name_lower = name.lower()

        # Try exact match first
        for player in self.all_players:
            if player['full_name'].lower() == name_lower:
                return player

        # Try partial match
        matches = [p for p in self.all_players
                   if name_lower in p['full_name'].lower()]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # Prefer active players
            active = [p for p in matches if p['is_active']]
            if len(active) == 1:
                return active[0]
            return matches[0]  # Return first match

        return None

    def get_player_game_log(
        self,
        player_id: int,
        season: str = '2024-25',
        season_type: str = 'Regular Season'
    ) -> pd.DataFrame:
        """
        Get a player's game log for a season.

        Args:
            player_id: NBA player ID
            season: Season string (e.g., '2024-25')
            season_type: 'Regular Season' or 'Playoffs'

        Returns:
            DataFrame with game log data
        """
        try:
            gamelog = self._api_call_with_delay(
                playergamelog.PlayerGameLog,
                player_id=player_id,
                season=season,
                season_type_all_star=season_type
            )
            df = gamelog.get_data_frames()[0]

            if not df.empty:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
                df = df.sort_values('GAME_DATE').reset_index(drop=True)
                df['SEASON'] = season

            return df

        except Exception as e:
            print(f"Error fetching game log for player {player_id}: {e}")
            return pd.DataFrame()

    def get_player_career_stats(
        self,
        player_id: int,
        seasons: List[str] = None
    ) -> pd.DataFrame:
        """
        Get a player's game logs across multiple seasons.

        Args:
            player_id: NBA player ID
            seasons: List of seasons to fetch (default: TRAINING_SEASONS)

        Returns:
            DataFrame with all game logs concatenated
        """
        if seasons is None:
            seasons = TRAINING_SEASONS

        all_games = []

        for season in seasons:
            df = self.get_player_game_log(player_id, season)
            if not df.empty:
                all_games.append(df)

        if all_games:
            return pd.concat(all_games, ignore_index=True).sort_values('GAME_DATE')
        return pd.DataFrame()

    def get_player_info(self, player_id: int) -> Dict:
        """
        Get detailed player information.

        Args:
            player_id: NBA player ID

        Returns:
            Dict with player information
        """
        try:
            info = self._api_call_with_delay(
                commonplayerinfo.CommonPlayerInfo,
                player_id=player_id
            )
            df = info.get_data_frames()[0]

            if not df.empty:
                return df.iloc[0].to_dict()
        except Exception as e:
            print(f"Error fetching player info: {e}")

        return {}

    def get_team_game_log(
        self,
        team_id: int,
        season: str = '2024-25'
    ) -> pd.DataFrame:
        """
        Get a team's game log for a season.

        Args:
            team_id: NBA team ID
            season: Season string

        Returns:
            DataFrame with team game log
        """
        try:
            gamelog = self._api_call_with_delay(
                teamgamelog.TeamGameLog,
                team_id=team_id,
                season=season
            )
            df = gamelog.get_data_frames()[0]

            if not df.empty:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
                df = df.sort_values('GAME_DATE').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"Error fetching team game log: {e}")
            return pd.DataFrame()

    def get_team_stats(self, season: str = '2024-25') -> pd.DataFrame:
        """
        Get league-wide team statistics.

        Args:
            season: Season string

        Returns:
            DataFrame with team stats including defensive ratings
        """
        try:
            stats = self._api_call_with_delay(
                leaguedashteamstats.LeagueDashTeamStats,
                season=season,
                measure_type_detailed_defense='Base'
            )
            return stats.get_data_frames()[0]

        except Exception as e:
            print(f"Error fetching team stats: {e}")
            return pd.DataFrame()

    def get_today_games(self) -> pd.DataFrame:
        """
        Get today's NBA games.

        Returns:
            DataFrame with today's scheduled games
        """
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            scoreboard = self._api_call_with_delay(
                scoreboardv2.ScoreboardV2,
                game_date=today
            )

            games_df = scoreboard.get_data_frames()[0]
            return games_df

        except Exception as e:
            print(f"Error fetching today's games: {e}")
            return pd.DataFrame()

    def get_upcoming_opponent(
        self,
        player_id: int,
        team_id: int = None
    ) -> Optional[Dict]:
        """
        Get information about a player's upcoming opponent.

        Args:
            player_id: NBA player ID
            team_id: Optional team ID (will be fetched if not provided)

        Returns:
            Dict with opponent information or None
        """
        if team_id is None:
            player_info = self.get_player_info(player_id)
            team_id = player_info.get('TEAM_ID')

        if team_id is None:
            return None

        games_df = self.get_today_games()

        if games_df.empty:
            return None

        # Find game involving this team
        team_game = games_df[
            (games_df['HOME_TEAM_ID'] == team_id) |
            (games_df['VISITOR_TEAM_ID'] == team_id)
        ]

        if team_game.empty:
            return None

        game = team_game.iloc[0]
        is_home = game['HOME_TEAM_ID'] == team_id
        opponent_id = game['VISITOR_TEAM_ID'] if is_home else game['HOME_TEAM_ID']

        opponent_team = next(
            (t for t in self.all_teams if t['id'] == opponent_id),
            None
        )

        return {
            'opponent_id': opponent_id,
            'opponent_name': opponent_team['full_name'] if opponent_team else 'Unknown',
            'is_home': is_home,
            'game_id': game['GAME_ID']
        }

    def fetch_multi_player_data(
        self,
        player_ids: List[int],
        seasons: List[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Fetch game logs for multiple players.

        Args:
            player_ids: List of NBA player IDs
            seasons: Seasons to fetch
            verbose: Print progress

        Returns:
            DataFrame with all players' game logs
        """
        all_data = []
        total = len(player_ids)

        for i, player_id in enumerate(player_ids):
            if verbose:
                print(f"Fetching player {i+1}/{total} (ID: {player_id})")

            df = self.get_player_career_stats(player_id, seasons)

            if not df.empty:
                df['PLAYER_ID'] = player_id
                all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def get_active_players_sample(self, n: int = 100) -> List[int]:
        """
        Get a sample of active player IDs for training data.

        Args:
            n: Number of players to sample

        Returns:
            List of player IDs
        """
        active = [p for p in self.all_players if p['is_active']]

        if len(active) <= n:
            return [p['id'] for p in active]

        # Sample players
        import random
        sampled = random.sample(active, n)
        return [p['id'] for p in sampled]


def get_current_season() -> str:
    """Get the current NBA season string."""
    now = datetime.now()
    year = now.year
    month = now.month

    # NBA season runs Oct-June
    if month >= 10:
        return f"{year}-{str(year+1)[2:]}"
    else:
        return f"{year-1}-{str(year)[2:]}"
