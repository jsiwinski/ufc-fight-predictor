#!/usr/bin/env python3
"""
UFC Moneyline Odds Scraper.

Scrapes moneyline odds from DraftKings and FanDuel via The Odds API.
Converts American odds to implied probabilities and matches fighter names.

Usage:
    from src.odds.scraper import UFCOddsScraper

    with UFCOddsScraper() as scraper:
        odds = scraper.scrape_upcoming_odds()
"""

import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
ODDS_DIR = DATA_DIR / 'odds'
LEDGER_PATH = DATA_DIR / 'ledger' / 'prediction_ledger.json'

# API Configuration
# The Odds API - aggregates DraftKings, FanDuel, and other books
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "mma_mixed_martial_arts"

# Target bookmakers (in priority order)
TARGET_BOOKMAKERS = ["draftkings", "fanduel"]

# Rate limiting
DEFAULT_RATE_LIMIT = 2.0  # seconds between requests


# =============================================================================
# Odds Conversion Functions
# =============================================================================

def american_to_implied_prob(odds: int) -> float:
    """
    Convert American moneyline odds to implied probability.

    Examples:
        -250 -> 250 / (250 + 100) = 0.714 (71.4%)
        +200 -> 100 / (200 + 100) = 0.333 (33.3%)

    Note: Raw implied probabilities sum to > 1.0 due to vig (book's margin).

    Args:
        odds: American moneyline odds (e.g., -250, +200)

    Returns:
        Implied probability as float between 0 and 1
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def remove_vig(f1_implied: float, f2_implied: float) -> Tuple[float, float]:
    """
    Normalize implied probabilities to remove the book's vig.

    Raw probabilities sum to > 1.0 due to vig. This normalizes them to sum to 1.0.

    Example:
        Raw: 71.4% + 33.3% = 104.7% (4.7% vig)
        Fair: 71.4/104.7 = 68.2%, 33.3/104.7 = 31.8%

    Args:
        f1_implied: Fighter 1 raw implied probability
        f2_implied: Fighter 2 raw implied probability

    Returns:
        Tuple of (f1_fair, f2_fair) probabilities
    """
    total = f1_implied + f2_implied
    if total == 0:
        return 0.5, 0.5
    return f1_implied / total, f2_implied / total


# =============================================================================
# Fighter Name Matching (copied from serve.py for consistency)
# =============================================================================

def normalize_name(name: str) -> str:
    """
    Normalize fighter name for matching.

    Handles:
    - Unicode/accent characters (José -> jose)
    - Transliteration variants (Sergey/Serghei)
    - Hyphens, apostrophes, periods
    - Common suffixes (Jr, Jr., II, III)
    """
    if not name:
        return ""

    # Unicode normalize — decompose accents
    name = unicodedata.normalize('NFKD', name)
    # Strip accent marks (combining characters)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # Lowercase
    name = name.lower()

    # Remove apostrophes (various unicode variants)
    name = name.replace("'", "").replace("'", "").replace("'", "").replace("`", "")

    # Replace hyphens and periods with spaces
    name = name.replace("-", " ").replace(".", " ")

    # Remove common suffixes that vary
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', name)

    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def fuzzy_match_score(name1: str, name2: str) -> float:
    """Calculate fuzzy match score between two names."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    return SequenceMatcher(None, n1, n2).ratio()


def find_fighter_match(
    fighter_name: str,
    known_fighters: List[str],
    threshold: float = 0.85
) -> Tuple[Optional[str], float]:
    """
    Find best matching fighter name from known fighters.

    Args:
        fighter_name: Name to match
        known_fighters: List of known fighter names
        threshold: Minimum match score required

    Returns:
        Tuple of (matched_name or None, match_score)
    """
    best_match = None
    best_score = 0.0

    normalized_target = normalize_name(fighter_name)

    for known in known_fighters:
        # Exact match (case insensitive)
        if normalize_name(known) == normalized_target:
            return known, 1.0

        # Fuzzy match
        score = fuzzy_match_score(fighter_name, known)
        if score > best_score:
            best_score = score
            best_match = known

    if best_score >= threshold:
        return best_match, best_score

    return None, best_score


# =============================================================================
# UFC Odds Scraper
# =============================================================================

class UFCOddsScraper:
    """
    Scrapes UFC moneyline odds from sportsbooks via The Odds API.

    Supports DraftKings and FanDuel as primary sources.

    Usage:
        with UFCOddsScraper() as scraper:
            odds = scraper.scrape_upcoming_odds()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: float = DEFAULT_RATE_LIMIT
    ):
        """
        Initialize the odds scraper.

        Args:
            api_key: The Odds API key. If None, reads from ODDS_API_KEY env var.
            rate_limit: Seconds between API requests
        """
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.rate_limit = rate_limit
        self.session = self._create_session()
        self._last_request_time = 0

        # Ensure odds directory exists
        ODDS_DIR.mkdir(parents=True, exist_ok=True)

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        session.headers.update({
            'User-Agent': 'UFC-Model-Odds-Scraper/1.0'
        })

        return session

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def close(self):
        """Close the session."""
        if self.session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _api_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make a request to The Odds API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response or None on error
        """
        if not self.api_key:
            logger.error("No API key provided. Set ODDS_API_KEY environment variable.")
            return None

        self._rate_limit()

        url = f"{ODDS_API_BASE}/{endpoint}"
        params['apiKey'] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Log remaining requests quota
            remaining = response.headers.get('x-requests-remaining')
            used = response.headers.get('x-requests-used')
            if remaining:
                logger.info(f"API quota: {remaining} remaining, {used} used")

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

    def scrape_upcoming_odds(self) -> List[Dict]:
        """
        Scrape moneyline odds for all upcoming UFC fights.

        Returns:
            List of fight odds dicts with structure:
            [
                {
                    "event_name": "UFC 326: ...",
                    "event_date": "2026-03-07",
                    "fighter_1": "Max Holloway",
                    "fighter_2": "Charles Oliveira",
                    "fighter_1_canonical": "Max Holloway",
                    "fighter_2_canonical": "Charles Oliveira",
                    "odds": {
                        "draftkings": {...},
                        "fanduel": {...},
                        "consensus": {...}
                    }
                }
            ]
        """
        logger.info("Fetching UFC odds from The Odds API...")

        # Request odds with American format
        params = {
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american',
            'bookmakers': ','.join(TARGET_BOOKMAKERS)
        }

        data = self._api_request(f"sports/{ODDS_API_SPORT}/odds", params)

        if not data:
            logger.warning("No odds data received from API")
            return []

        # Parse the response
        fights = []
        for event in data:
            fight_data = self._parse_event(event)
            if fight_data:
                fights.append(fight_data)

        logger.info(f"Scraped odds for {len(fights)} fights")
        return fights

    def _parse_event(self, event: Dict) -> Optional[Dict]:
        """
        Parse a single event from The Odds API response.

        Args:
            event: Event data from API

        Returns:
            Parsed fight odds dict or None
        """
        try:
            # Extract basic info
            event_id = event.get('id', '')
            sport_title = event.get('sport_title', 'UFC')
            commence_time = event.get('commence_time', '')

            # Parse date
            if commence_time:
                event_date = datetime.fromisoformat(
                    commence_time.replace('Z', '+00:00')
                ).strftime('%Y-%m-%d')
            else:
                event_date = ''

            # Fighter names (The Odds API uses home_team/away_team for MMA)
            fighter_1 = event.get('home_team', '')
            fighter_2 = event.get('away_team', '')

            if not fighter_1 or not fighter_2:
                return None

            # Parse bookmaker odds
            bookmakers_data = event.get('bookmakers', [])
            odds = {}

            for bookmaker in bookmakers_data:
                book_key = bookmaker.get('key', '')
                if book_key not in TARGET_BOOKMAKERS:
                    continue

                book_odds = self._parse_bookmaker_odds(
                    bookmaker, fighter_1, fighter_2
                )
                if book_odds:
                    odds[book_key] = book_odds

            # Calculate consensus from available books
            if odds:
                odds['consensus'] = self._calculate_consensus(odds)

            # Build event name from fighters if not provided
            event_name = f"{sport_title}: {fighter_1} vs {fighter_2}"

            return {
                'event_id': event_id,
                'event_name': event_name,
                'event_date': event_date,
                'fighter_1': fighter_1,
                'fighter_2': fighter_2,
                'fighter_1_canonical': None,  # To be filled by match_fighters
                'fighter_2_canonical': None,
                'odds': odds
            }

        except Exception as e:
            logger.warning(f"Error parsing event: {e}")
            return None

    def _parse_bookmaker_odds(
        self,
        bookmaker: Dict,
        fighter_1: str,
        fighter_2: str
    ) -> Optional[Dict]:
        """
        Parse odds from a single bookmaker.

        Args:
            bookmaker: Bookmaker data from API
            fighter_1: First fighter name
            fighter_2: Second fighter name

        Returns:
            Parsed odds dict or None
        """
        try:
            markets = bookmaker.get('markets', [])
            h2h_market = None

            for market in markets:
                if market.get('key') == 'h2h':
                    h2h_market = market
                    break

            if not h2h_market:
                return None

            outcomes = h2h_market.get('outcomes', [])
            f1_odds = None
            f2_odds = None

            for outcome in outcomes:
                name = outcome.get('name', '')
                price = outcome.get('price', 0)

                # Match outcome to fighter
                if normalize_name(name) == normalize_name(fighter_1):
                    f1_odds = price
                elif normalize_name(name) == normalize_name(fighter_2):
                    f2_odds = price

            if f1_odds is None or f2_odds is None:
                # Try fuzzy matching if exact match failed
                for outcome in outcomes:
                    name = outcome.get('name', '')
                    price = outcome.get('price', 0)

                    if f1_odds is None and fuzzy_match_score(name, fighter_1) > 0.8:
                        f1_odds = price
                    elif f2_odds is None and fuzzy_match_score(name, fighter_2) > 0.8:
                        f2_odds = price

            if f1_odds is None or f2_odds is None:
                return None

            # Calculate implied probabilities
            f1_implied = american_to_implied_prob(f1_odds)
            f2_implied = american_to_implied_prob(f2_odds)

            # Remove vig for fair probabilities
            f1_fair, f2_fair = remove_vig(f1_implied, f2_implied)

            last_update = bookmaker.get('last_update', '')

            return {
                'f1_moneyline': f1_odds,
                'f2_moneyline': f2_odds,
                'f1_implied': round(f1_implied, 4),
                'f2_implied': round(f2_implied, 4),
                'f1_fair': round(f1_fair, 4),
                'f2_fair': round(f2_fair, 4),
                'scraped_at': last_update or datetime.utcnow().isoformat() + 'Z'
            }

        except Exception as e:
            logger.warning(f"Error parsing bookmaker odds: {e}")
            return None

    def _calculate_consensus(self, odds: Dict) -> Dict:
        """
        Calculate consensus fair probabilities from all bookmakers.

        Averages the vig-removed fair probabilities from each book.

        Args:
            odds: Dict of bookmaker -> odds data

        Returns:
            Consensus odds dict
        """
        f1_fairs = []
        f2_fairs = []

        for book, book_odds in odds.items():
            if book == 'consensus':
                continue
            if book_odds and 'f1_fair' in book_odds:
                f1_fairs.append(book_odds['f1_fair'])
                f2_fairs.append(book_odds['f2_fair'])

        if not f1_fairs:
            return {}

        f1_consensus = sum(f1_fairs) / len(f1_fairs)
        f2_consensus = sum(f2_fairs) / len(f2_fairs)

        return {
            'f1_fair': round(f1_consensus, 4),
            'f2_fair': round(f2_consensus, 4),
            'sources': len(f1_fairs)
        }

    def match_fighters_to_canonical(
        self,
        fights: List[Dict],
        known_fighters: List[str]
    ) -> Tuple[List[Dict], int, int]:
        """
        Match odds fighter names to canonical names.

        Args:
            fights: List of fight odds dicts
            known_fighters: List of canonical fighter names

        Returns:
            Tuple of (updated_fights, matched_count, unmatched_count)
        """
        matched = 0
        unmatched = 0

        for fight in fights:
            # Match fighter 1
            f1_match, f1_score = find_fighter_match(
                fight['fighter_1'], known_fighters
            )
            if f1_match:
                fight['fighter_1_canonical'] = f1_match
                logger.debug(f"Matched '{fight['fighter_1']}' -> '{f1_match}' ({f1_score:.2f})")
            else:
                fight['fighter_1_canonical'] = fight['fighter_1']
                logger.warning(
                    f"Could not match fighter: '{fight['fighter_1']}' "
                    f"(best score: {f1_score:.2f})"
                )
                unmatched += 1

            # Match fighter 2
            f2_match, f2_score = find_fighter_match(
                fight['fighter_2'], known_fighters
            )
            if f2_match:
                fight['fighter_2_canonical'] = f2_match
                logger.debug(f"Matched '{fight['fighter_2']}' -> '{f2_match}' ({f2_score:.2f})")
            else:
                fight['fighter_2_canonical'] = fight['fighter_2']
                logger.warning(
                    f"Could not match fighter: '{fight['fighter_2']}' "
                    f"(best score: {f2_score:.2f})"
                )
                unmatched += 1

            if f1_match and f2_match:
                matched += 1

        return fights, matched, unmatched

    def save_odds(self, fights: List[Dict], filepath: Optional[Path] = None) -> Path:
        """
        Save scraped odds to JSON file.

        Args:
            fights: List of fight odds dicts
            filepath: Optional custom filepath

        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = ODDS_DIR / 'upcoming_odds.json'

        data = {
            'scraped_at': datetime.utcnow().isoformat() + 'Z',
            'sources': TARGET_BOOKMAKERS,
            'fight_count': len(fights),
            'fights': fights
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved odds to {filepath}")
        return filepath

    def append_to_history(self, fights: List[Dict]) -> Path:
        """
        Append scraped odds to history log.

        Args:
            fights: List of fight odds dicts

        Returns:
            Path to history file
        """
        history_path = ODDS_DIR / 'odds_history.jsonl'

        record = {
            'scraped_at': datetime.utcnow().isoformat() + 'Z',
            'fight_count': len(fights),
            'fights': fights
        }

        with open(history_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        logger.info(f"Appended to history: {history_path}")
        return history_path


def load_upcoming_odds() -> Optional[Dict]:
    """Load the most recent upcoming odds data."""
    odds_path = ODDS_DIR / 'upcoming_odds.json'
    if not odds_path.exists():
        return None

    with open(odds_path) as f:
        return json.load(f)


def load_ledger() -> Optional[Dict]:
    """Load the prediction ledger."""
    if not LEDGER_PATH.exists():
        return None

    with open(LEDGER_PATH) as f:
        return json.load(f)


def get_known_fighters_from_ledger() -> List[str]:
    """Extract all known fighter names from the prediction ledger."""
    ledger = load_ledger()
    if not ledger:
        return []

    fighters = set()
    for entry in ledger.get('entries', []):
        for fight in entry.get('fights', []):
            f1 = fight.get('fighter_1', '')
            f2 = fight.get('fighter_2', '')
            if f1:
                fighters.add(f1)
            if f2:
                fighters.add(f2)

    return list(fighters)


# TODO Phase 11: Display odds alongside model predictions on fight cards
# TODO Phase 11: Store odds snapshot in prediction_ledger.json at prediction time
# TODO Phase 11: Compute value metrics (model_prob - consensus_fair) for each fight
# TODO Phase 11: Add "Value Picks" section to upcoming page highlighting edge fights
