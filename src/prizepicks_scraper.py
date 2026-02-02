"""
PrizePicks Scraper - Fetches current NBA props from PrizePicks API.
"""

import requests
from typing import List, Dict, Optional
import time


class PrizePicksScraper:
    """Fetches current NBA player props from PrizePicks."""

    BASE_URL = "https://api.prizepicks.com"
    NBA_LEAGUE_ID = 7

    # Map PrizePicks stat types to our model's prop types
    STAT_TYPE_MAPPING = {
        'Points': 'points',
        'Rebounds': 'rebounds',
        'Assists': 'assists',
        'Pts+Rebs+Asts': 'pts_reb_ast',
        '3-PT Made': 'threes',
        'Steals': 'steals',
        'Blocks': 'blocks',
        'Turnovers': 'turnovers',
        'Pts+Rebs': 'pts_reb',
        'Pts+Asts': 'pts_ast',
        'Rebs+Asts': 'reb_ast',
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
        })

    def fetch_nba_props(self) -> List[Dict]:
        """
        Fetch current NBA player props from PrizePicks.

        Returns:
            List of props, each containing:
            - player_name: str
            - player_id: str (PrizePicks ID)
            - prop_type: str (our model's prop type name)
            - prop_type_display: str (PrizePicks display name)
            - line: float
            - team: str (optional)
            - opponent: str (optional)
            - game_time: str (optional)
        """
        try:
            # Fetch projections for NBA
            url = f"{self.BASE_URL}/projections"
            params = {
                'league_id': self.NBA_LEAGUE_ID,
                'per_page': 250,
                'single_stat': 'true',
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            return self._parse_projections(data)

        except requests.RequestException as e:
            raise PrizePicksScraperError(f"Failed to fetch props: {str(e)}")
        except (KeyError, ValueError) as e:
            raise PrizePicksScraperError(f"Failed to parse response: {str(e)}")

    def _parse_projections(self, data: Dict) -> List[Dict]:
        """Parse the PrizePicks API response into our prop format."""
        props = []

        # The API returns data in a JSON:API format
        # Projections are in 'data', players are in 'included'
        projections = data.get('data', [])
        included = data.get('included', [])

        # Build lookup for players and stat types
        players = {}
        stat_types = {}

        for item in included:
            if item.get('type') == 'new_player':
                players[item['id']] = {
                    'name': item.get('attributes', {}).get('name', ''),
                    'team': item.get('attributes', {}).get('team', ''),
                    'position': item.get('attributes', {}).get('position', ''),
                }
            elif item.get('type') == 'stat_type':
                stat_types[item['id']] = item.get('attributes', {}).get('name', '')

        # Parse each projection
        for proj in projections:
            attrs = proj.get('attributes', {})
            relationships = proj.get('relationships', {})

            # Get player info
            player_rel = relationships.get('new_player', {}).get('data', {})
            player_id = player_rel.get('id')
            player_info = players.get(player_id, {})

            # Get stat type
            stat_type_rel = relationships.get('stat_type', {}).get('data', {})
            stat_type_id = stat_type_rel.get('id')
            stat_type_name = stat_types.get(stat_type_id, '')

            # Map to our prop type
            our_prop_type = self._map_stat_type(stat_type_name)

            # Skip props we don't have models for
            if our_prop_type is None:
                continue

            prop = {
                'player_name': player_info.get('name', ''),
                'player_id': player_id,
                'prop_type': our_prop_type,
                'prop_type_display': stat_type_name,
                'line': float(attrs.get('line_score', 0)),
                'team': player_info.get('team', ''),
                'position': player_info.get('position', ''),
                'start_time': attrs.get('start_time', ''),
                'description': attrs.get('description', ''),
            }

            # Only add if we have valid data
            if prop['player_name'] and prop['line'] > 0:
                props.append(prop)

        return props

    def _map_stat_type(self, pp_stat: str) -> Optional[str]:
        """
        Map PrizePicks stat names to our model's target names.

        Args:
            pp_stat: PrizePicks stat type name

        Returns:
            Our model's prop type name, or None if not supported
        """
        return self.STAT_TYPE_MAPPING.get(pp_stat)

    def get_supported_stat_types(self) -> List[str]:
        """Return list of PrizePicks stat types we support."""
        return list(self.STAT_TYPE_MAPPING.keys())


class PrizePicksScraperError(Exception):
    """Exception raised for PrizePicks scraper errors."""
    pass
