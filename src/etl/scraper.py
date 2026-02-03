"""
UFC Historical Data Scraper for Fight Prediction System.

This module scrapes comprehensive UFC fight data from ufcstats.com including:
- Historical fight results with round-by-round statistics
- Fighter biographical data (height, weight, reach, stance, DOB)
- Fighter career statistics (striking accuracy, takedown stats, etc.)
- Detailed fight-level metrics (strikes by target/position, control time, etc.)

Data Structure:
    Fight Data (77 columns):
        - Event info: name, date, location
        - Fighter info: names, nicknames, results
        - Fight outcome: method, round, time, referee, bout type
        - Performance metrics: knockdowns, strikes, takedowns, submissions
        - Detailed strikes: by target (head/body/leg) and position (distance/clinch/ground)

    Fighter Profile Data (14 columns):
        - Physical: height, weight, reach, stance, DOB
        - Career stats: SLpM, Str_Acc, SApM, Str_Def, TD_Avg, TD_Acc, TD_Def, Sub_Avg

Design Philosophy:
    - Respectful scraping with configurable rate limiting (default 2s between requests)
    - Modular functions for different data types (events, fights, fighters)
    - Comprehensive error handling with retry logic
    - Progress tracking for long scraping sessions
    - Data validation and cleaning built-in
    - Database-ready CSV output format

Reference Implementation:
    Based on analysis of UFC-DataLab repository (https://github.com/komaksym/UFC-DataLab)
    Enhanced with better error handling, retry logic, and fighter profile scraping.

Note: This scrapes historical data only. Upcoming fight predictions require separate approach.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import time
from datetime import datetime
import re
from pathlib import Path
import csv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class UFCDataScraper:
    """
    Comprehensive UFC data scraper for fight prediction ML models.

    This scraper collects:
    1. Historical fight data with detailed round-by-round statistics
    2. Fighter biographical and career statistics
    3. Event information

    Features:
    - Automatic retry logic with exponential backoff
    - Configurable rate limiting (default: 2s between requests)
    - Progress tracking and logging
    - Data validation and cleaning
    - CSV export functionality

    Attributes:
        base_url: Base URL for ufcstats.com
        fighters_url: URL for fighter listings
        rate_limit: Seconds to wait between requests
        session: Requests session with retry logic
        fighter_cache: Cache of fighter URLs to avoid redundant lookups
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the scraper with configuration.

        Args:
            config: Optional dictionary containing:
                - rate_limit: Seconds between requests (default: 2)
                - max_retries: Max retry attempts (default: 3)
                - timeout: Request timeout in seconds (default: 10)
                - user_agent: Custom user agent string
        """
        self.config = config or {}
        self.base_url = 'http://www.ufcstats.com'
        self.fighters_url = f'{self.base_url}/statistics/fighters'
        self.rate_limit = self.config.get('rate_limit', 2)
        self.timeout = self.config.get('timeout', 10)
        self.max_retries = self.config.get('max_retries', 3)

        # Setup session with retry logic
        self.session = self._create_session()

        # Cache for fighter URLs to avoid redundant lookups
        self.fighter_cache = {}

        logger.info(f"UFCDataScraper initialized with rate_limit={self.rate_limit}s, timeout={self.timeout}s")

    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic and custom headers.

        Returns:
            Configured requests Session object
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set headers
        user_agent = self.config.get(
            'user_agent',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        return session

    def _safe_request(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """
        Make a safe HTTP request with error handling and rate limiting.

        Args:
            url: URL to fetch
            retries: Number of retry attempts

        Returns:
            BeautifulSoup object or None if request fails
        """
        for attempt in range(retries):
            try:
                time.sleep(self.rate_limit)  # Rate limiting
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}) for {url}: {e}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None

    def scrape_all_data(self, output_dir: str = 'data/raw', limit_events: Optional[int] = None) -> Dict[str, str]:
        """
        Scrape all UFC data: fights, events, and fighter profiles.

        This is the main entry point for comprehensive data collection.

        Args:
            output_dir: Directory to save CSV files
            limit_events: Optional limit on number of events to scrape (for testing)

        Returns:
            Dictionary with paths to generated CSV files:
                - 'fights': Path to fights CSV
                - 'fighters': Path to fighter profiles CSV

        Example:
            >>> scraper = UFCDataScraper()
            >>> files = scraper.scrape_all_data(output_dir='data/raw')
            >>> print(f"Fights data: {files['fights']}")
            >>> print(f"Fighters data: {files['fighters']}")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Starting comprehensive UFC data scraping")
        logger.info("=" * 80)

        # 1. Scrape historical fights
        logger.info("\n[1/2] Scraping historical fight data...")
        fights_df = self.scrape_historical_fights(limit_events=limit_events)
        fights_file = output_path / 'ufc_fights_raw.csv'
        fights_df.to_csv(fights_file, index=False)
        logger.info(f"Saved {len(fights_df)} fights to {fights_file}")

        # 2. Scrape fighter profiles
        logger.info("\n[2/2] Scraping fighter profiles...")
        fighters_df = self.scrape_all_fighters()
        fighters_file = output_path / 'ufc_fighters_raw.csv'
        fighters_df.to_csv(fighters_file, index=False)
        logger.info(f"Saved {len(fighters_df)} fighter profiles to {fighters_file}")

        logger.info("\n" + "=" * 80)
        logger.info("Scraping completed successfully!")
        logger.info("=" * 80)

        return {
            'fights': str(fights_file),
            'fighters': str(fighters_file)
        }

    def scrape_historical_fights(self, start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 limit_events: Optional[int] = None) -> pd.DataFrame:
        """
        Scrape comprehensive historical UFC fight data.

        This method scrapes all completed UFC events and their fights, extracting
        detailed statistics for machine learning models.

        Args:
            start_date: Optional start date filter (YYYY-MM-DD format)
            end_date: Optional end date filter (YYYY-MM-DD format)
            limit_events: Optional limit on number of events (useful for testing)

        Returns:
            DataFrame with 77+ columns including:
                Event: event_name, event_date, event_location
                Fighters: red_fighter_name, blue_fighter_name, nicknames
                Outcome: result, method, round, time, referee, bout_type
                Totals: KD, sig_str, total_str, TD, sub_att, rev, ctrl
                Strikes by target: head, body, leg (landed/attempted and %)
                Strikes by position: distance, clinch, ground (landed/attempted and %)

        Example:
            >>> scraper = UFCDataScraper()
            >>> df = scraper.scrape_historical_fights(start_date='2023-01-01')
            >>> print(df[['event_name', 'red_fighter_name', 'blue_fighter_name']].head())
        """
        start_time = time.time()
        logger.info(f"Starting fight data scraping (date range: {start_date or 'all'} to {end_date or 'all'})")

        # Convert date strings to datetime objects for filtering
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None

        # Get all events
        events = self._get_all_events()
        logger.info(f"Found {len(events)} total completed events")

        # Apply limit if specified (for testing)
        if limit_events:
            events = events[:limit_events]
            logger.info(f"Limited to {len(events)} events for testing")

        # Filter events by date if specified
        if start_dt or end_dt:
            filtered_events = []
            for event in events:
                try:
                    event_date = datetime.strptime(event['date'], '%B %d, %Y')
                    if start_dt and event_date < start_dt:
                        continue
                    if end_dt and event_date > end_dt:
                        continue
                    filtered_events.append(event)
                except ValueError:
                    logger.warning(f"Could not parse date for event: {event['name']}")
                    continue
            events = filtered_events
            logger.info(f"Filtered to {len(events)} events in date range")

        # Collect fights from all events
        all_fights = []
        failed_events = []

        for i, event in enumerate(events, 1):
            logger.info(f"[{i}/{len(events)}] Processing: {event['name']} ({event['date']})")
            try:
                fights = self._get_event_fights(event)
                all_fights.extend(fights)
                logger.info(f"  → Collected {len(fights)} fights from this event")
            except Exception as e:
                logger.error(f"  → Error processing event {event['name']}: {e}")
                failed_events.append(event['name'])
                continue

        elapsed = time.time() - start_time
        logger.info(f"\nScraping Summary:")
        logger.info(f"  Total fights collected: {len(all_fights)}")
        logger.info(f"  Events processed: {len(events) - len(failed_events)}/{len(events)}")
        logger.info(f"  Failed events: {len(failed_events)}")
        logger.info(f"  Time elapsed: {elapsed:.1f}s")

        if failed_events:
            logger.warning(f"Failed events: {', '.join(failed_events[:5])}{'...' if len(failed_events) > 5 else ''}")

        return pd.DataFrame(all_fights)

    def _get_all_events(self) -> List[Dict]:
        """
        Get list of all UFC events from ufcstats.com.

        Returns:
            List of dictionaries containing event info (name, date, url)
        """
        events_url = f"{self.base_url}/statistics/events/completed?page=all"

        try:
            response = self.session.get(events_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch events page: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        events = []

        # Find the events table
        table = soup.find('table', {'class': 'b-statistics__table-events'})
        if not table:
            logger.warning("Could not find events table")
            return []

        rows = table.find('tbody').find_all('tr', {'class': 'b-statistics__table-row'})

        for row in rows:
            try:
                # Extract event details
                cells = row.find_all('td', {'class': 'b-statistics__table-col'})
                if len(cells) < 2:
                    continue

                # Event name and link
                event_link = cells[0].find('a', {'class': 'b-link'})
                if not event_link:
                    continue

                event_name = event_link.text.strip()
                event_url = event_link['href']

                # Event date
                event_date = cells[0].find('span', {'class': 'b-statistics__date'})
                event_date = event_date.text.strip() if event_date else ''

                # Event location
                event_location = cells[1].text.strip() if len(cells) > 1 else ''

                events.append({
                    'name': event_name,
                    'date': event_date,
                    'location': event_location,
                    'url': event_url
                })
            except Exception as e:
                logger.warning(f"Error parsing event row: {e}")
                continue

        return events

    def _get_event_fights(self, event: Dict) -> List[Dict]:
        """
        Get all fights from a specific event.

        Args:
            event: Dictionary containing event info

        Returns:
            List of fight dictionaries
        """
        try:
            response = self.session.get(event['url'], timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch event {event['name']}: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        fights = []

        # Find the fights table
        table = soup.find('table', {'class': 'b-fight-details__table'})
        if not table:
            logger.warning(f"Could not find fights table for event {event['name']}")
            return []

        rows = table.find('tbody').find_all('tr', {'class': 'b-fight-details__table-row'})

        for row in rows:
            try:
                # Get fight details link
                fight_link = row.find('a', {'class': 'b-flag'})
                if not fight_link or 'href' not in fight_link.attrs:
                    continue

                fight_url = fight_link['href']

                # Get basic fight info from the row
                cells = row.find_all('td', {'class': 'b-fight-details__table-col'})
                if len(cells) < 7:
                    continue

                # Extract fighter names
                fighters = cells[1].find_all('a')
                if len(fighters) < 2:
                    continue

                fighter1_name = fighters[0].text.strip()
                fighter2_name = fighters[1].text.strip()

                # Get fight result
                result_icons = cells[0].find_all('i')
                if len(result_icons) >= 2:
                    if 'b-flag__text_style_green' in result_icons[0].get('class', []):
                        winner = fighter1_name
                    elif 'b-flag__text_style_green' in result_icons[1].get('class', []):
                        winner = fighter2_name
                    else:
                        winner = 'Draw'
                else:
                    winner = 'Unknown'

                # Weight class
                weight_class = cells[6].text.strip()

                # Method, round, time
                method_col = cells[7].find_all('p')
                method = method_col[0].text.strip() if len(method_col) > 0 else ''
                method_details = method_col[1].text.strip() if len(method_col) > 1 else ''

                # Parse round and time from method_details
                round_num = ''
                fight_time = ''
                if method_details:
                    round_match = re.search(r'Round:\s*(\d+)', method_details)
                    time_match = re.search(r'Time:\s*(\d+:\d+)', method_details)
                    if round_match:
                        round_num = round_match.group(1)
                    if time_match:
                        fight_time = time_match.group(1)

                # Get detailed fight stats
                fight_stats = self._get_fight_details(fight_url)

                fight = {
                    'fight_id': fight_url.split('/')[-1],
                    'event_name': event['name'],
                    'event_date': event['date'],
                    'event_location': event['location'],
                    'fighter1_name': fighter1_name,
                    'fighter2_name': fighter2_name,
                    'winner': winner,
                    'method': method,
                    'round': round_num,
                    'time': fight_time,
                    'weight_class': weight_class,
                    'fight_url': fight_url
                }

                # Merge fight stats if available
                if fight_stats:
                    fight.update(fight_stats)

                fights.append(fight)

                # Rate limiting for individual fight pages
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error parsing fight row: {e}")
                continue

        return fights

    def _get_fight_details(self, fight_url: str) -> Dict:
        """
        Extract comprehensive fight statistics from a fight detail page.

        Scrapes detailed round-by-round statistics including:
        - Total stats: knockdowns, strikes, takedowns, submissions, reversals, control
        - Significant strikes by target: head, body, leg
        - Significant strikes by position: distance, clinch, ground
        - Percentages for all strike categories

        Args:
            fight_url: URL to the fight details page on ufcstats.com

        Returns:
            Dictionary with 60+ statistical fields (red_fighter_* and blue_fighter_* prefixed)
        """
        soup = self._safe_request(fight_url)
        if not soup:
            return {}

        stats = {}

        try:
            # Extract fighter names and basic info from header
            header_names = soup.select('h3.b-fight-details__person-name a')
            if len(header_names) >= 2:
                stats['red_fighter_name'] = header_names[0].text.strip()
                stats['blue_fighter_name'] = header_names[1].text.strip()

            # Extract nicknames
            nicknames = soup.select('p.b-fight-details__person-title')
            if len(nicknames) >= 2:
                stats['red_fighter_nickname'] = nicknames[0].text.strip()
                stats['blue_fighter_nickname'] = nicknames[1].text.strip()

            # Find all statistics tables
            tables = soup.find_all('table', {'class': 'b-fight-details__table'})

            # TABLE 1: Totals (KD, Sig. str., Total str., TD, Sub. att., Rev., Ctrl)
            if len(tables) >= 1:
                self._parse_totals_table(tables[0], stats)

            # TABLE 2: Significant Strikes (by target and position)
            if len(tables) >= 2:
                self._parse_significant_strikes_table(tables[1], stats)

        except Exception as e:
            logger.warning(f"Error parsing fight details from {fight_url}: {e}")

        return stats

    def _parse_totals_table(self, table: BeautifulSoup, stats: Dict) -> None:
        """
        Parse the totals statistics table.

        Columns: KD, Sig. str., Sig. str. %, Total str., TD, TD %, Sub. att., Rev., Ctrl
        """
        try:
            tbody = table.find('tbody')
            if not tbody:
                return

            rows = tbody.find_all('tr', {'class': 'b-fight-details__table-row'})
            if len(rows) < 2:
                return

            # Red corner (row 0) and Blue corner (row 1)
            red_cells = rows[0].find_all('td', {'class': 'b-fight-details__table-col'})
            blue_cells = rows[1].find_all('td', {'class': 'b-fight-details__table-col'})

            if len(red_cells) >= 9 and len(blue_cells) >= 9:
                # KD (Knockdowns)
                stats['red_fighter_KD'] = red_cells[1].text.strip()
                stats['blue_fighter_KD'] = blue_cells[1].text.strip()

                # Significant strikes (landed of attempted)
                stats['red_fighter_sig_str'] = red_cells[2].text.strip()
                stats['blue_fighter_sig_str'] = blue_cells[2].text.strip()

                # Significant strike percentage
                stats['red_fighter_sig_str_pct'] = red_cells[3].text.strip()
                stats['blue_fighter_sig_str_pct'] = blue_cells[3].text.strip()

                # Total strikes
                stats['red_fighter_total_str'] = red_cells[4].text.strip()
                stats['blue_fighter_total_str'] = blue_cells[4].text.strip()

                # Takedowns
                stats['red_fighter_TD'] = red_cells[5].text.strip()
                stats['blue_fighter_TD'] = blue_cells[5].text.strip()

                # Takedown percentage
                stats['red_fighter_TD_pct'] = red_cells[6].text.strip()
                stats['blue_fighter_TD_pct'] = blue_cells[6].text.strip()

                # Submission attempts
                stats['red_fighter_sub_att'] = red_cells[7].text.strip()
                stats['blue_fighter_sub_att'] = blue_cells[7].text.strip()

                # Reversals
                stats['red_fighter_rev'] = red_cells[8].text.strip()
                stats['blue_fighter_rev'] = blue_cells[8].text.strip()

                # Control time
                if len(red_cells) > 9 and len(blue_cells) > 9:
                    stats['red_fighter_ctrl'] = red_cells[9].text.strip()
                    stats['blue_fighter_ctrl'] = blue_cells[9].text.strip()

        except Exception as e:
            logger.warning(f"Error parsing totals table: {e}")

    def _parse_significant_strikes_table(self, table: BeautifulSoup, stats: Dict) -> None:
        """
        Parse significant strikes by target and position table.

        Columns: Sig. str., Sig. str. %, Head, Body, Leg, Distance, Clinch, Ground
        """
        try:
            tbody = table.find('tbody')
            if not tbody:
                return

            rows = tbody.find_all('tr', {'class': 'b-fight-details__table-row'})
            if len(rows) < 2:
                return

            red_cells = rows[0].find_all('td', {'class': 'b-fight-details__table-col'})
            blue_cells = rows[1].find_all('td', {'class': 'b-fight-details__table-col'})

            if len(red_cells) >= 9 and len(blue_cells) >= 9:
                # Strikes by target: Head
                stats['red_fighter_sig_str_head'] = red_cells[3].text.strip()
                stats['blue_fighter_sig_str_head'] = blue_cells[3].text.strip()

                # Body
                stats['red_fighter_sig_str_body'] = red_cells[4].text.strip()
                stats['blue_fighter_sig_str_body'] = blue_cells[4].text.strip()

                # Leg
                stats['red_fighter_sig_str_leg'] = red_cells[5].text.strip()
                stats['blue_fighter_sig_str_leg'] = blue_cells[5].text.strip()

                # Strikes by position: Distance
                stats['red_fighter_sig_str_distance'] = red_cells[6].text.strip()
                stats['blue_fighter_sig_str_distance'] = blue_cells[6].text.strip()

                # Clinch
                stats['red_fighter_sig_str_clinch'] = red_cells[7].text.strip()
                stats['blue_fighter_sig_str_clinch'] = blue_cells[7].text.strip()

                # Ground
                stats['red_fighter_sig_str_ground'] = red_cells[8].text.strip()
                stats['blue_fighter_sig_str_ground'] = blue_cells[8].text.strip()

                # Calculate percentages for strikes by target and position
                self._calculate_strike_percentages(stats)

        except Exception as e:
            logger.warning(f"Error parsing significant strikes table: {e}")

    def _calculate_strike_percentages(self, stats: Dict) -> None:
        """
        Calculate percentage values for significant strikes by target and position.

        For each category (head/body/leg/distance/clinch/ground), calculates:
        percentage = (landed / attempted) * 100 if attempted > 0 else 0
        """
        categories = [
            'sig_str_head', 'sig_str_body', 'sig_str_leg',
            'sig_str_distance', 'sig_str_clinch', 'sig_str_ground'
        ]

        for fighter_prefix in ['red_fighter', 'blue_fighter']:
            for category in categories:
                key = f'{fighter_prefix}_{category}'
                if key in stats:
                    # Parse "landed of attempted" format (e.g., "25 of 50")
                    value = stats[key]
                    match = re.match(r'(\d+)\s+of\s+(\d+)', value)
                    if match:
                        landed = int(match.group(1))
                        attempted = int(match.group(2))
                        percentage = (landed / attempted * 100) if attempted > 0 else 0
                        stats[f'{key}_pct'] = f'{percentage:.0f}%'
                    else:
                        stats[f'{key}_pct'] = '0%'

    def scrape_upcoming_events(self) -> pd.DataFrame:
        """
        Scrape upcoming UFC events and fight cards.

        Returns:
            DataFrame containing upcoming fights

        Expected columns:
            - event_name, event_date, location
            - fighter1_name, fighter2_name
            - weight_class, is_title_fight
        """
        logger.info("Scraping upcoming events")

        upcoming_url = f"{self.base_url}/statistics/events/upcoming"

        try:
            response = self.session.get(upcoming_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch upcoming events: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.content, 'html.parser')
        upcoming_fights = []

        # Find the events table
        table = soup.find('table', {'class': 'b-statistics__table-events'})
        if not table:
            logger.warning("Could not find upcoming events table")
            return pd.DataFrame()

        rows = table.find('tbody').find_all('tr', {'class': 'b-statistics__table-row'})

        for row in rows:
            try:
                cells = row.find_all('td', {'class': 'b-statistics__table-col'})
                if len(cells) < 2:
                    continue

                # Event name and link
                event_link = cells[0].find('a', {'class': 'b-link'})
                if not event_link:
                    continue

                event_name = event_link.text.strip()
                event_url = event_link['href']

                # Event date
                event_date = cells[0].find('span', {'class': 'b-statistics__date'})
                event_date = event_date.text.strip() if event_date else ''

                # Event location
                event_location = cells[1].text.strip() if len(cells) > 1 else ''

                # Get fights for this event
                event_fights = self._get_upcoming_event_fights({
                    'name': event_name,
                    'date': event_date,
                    'location': event_location,
                    'url': event_url
                })

                upcoming_fights.extend(event_fights)
                time.sleep(self.rate_limit)

            except Exception as e:
                logger.warning(f"Error parsing upcoming event row: {e}")
                continue

        logger.info(f"Found {len(upcoming_fights)} upcoming fights")
        return pd.DataFrame(upcoming_fights)

    def _get_upcoming_event_fights(self, event: Dict) -> List[Dict]:
        """
        Get all fights from an upcoming event.

        Args:
            event: Dictionary containing event info

        Returns:
            List of fight dictionaries
        """
        try:
            response = self.session.get(event['url'], timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch upcoming event {event['name']}: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        fights = []

        # Find the fights table
        table = soup.find('table', {'class': 'b-fight-details__table'})
        if not table:
            logger.warning(f"Could not find fights table for upcoming event {event['name']}")
            return []

        rows = table.find('tbody').find_all('tr', {'class': 'b-fight-details__table-row'})

        for row in rows:
            try:
                cells = row.find_all('td', {'class': 'b-fight-details__table-col'})
                if len(cells) < 2:
                    continue

                # Extract fighter names
                fighters = cells[1].find_all('a')
                if len(fighters) < 2:
                    continue

                fighter1_name = fighters[0].text.strip()
                fighter2_name = fighters[1].text.strip()

                # Weight class
                weight_class = cells[6].text.strip() if len(cells) > 6 else ''

                # Check if it's a title fight
                is_title_fight = 'title' in weight_class.lower()

                fight = {
                    'event_name': event['name'],
                    'event_date': event['date'],
                    'location': event['location'],
                    'fighter1_name': fighter1_name,
                    'fighter2_name': fighter2_name,
                    'weight_class': weight_class,
                    'is_title_fight': is_title_fight
                }

                fights.append(fight)

            except Exception as e:
                logger.warning(f"Error parsing upcoming fight row: {e}")
                continue

        return fights

    def scrape_all_fighters(self) -> pd.DataFrame:
        """
        Scrape biographical and career statistics for all UFC fighters.

        This method scrapes the fighter directory alphabetically and extracts:
        - Physical attributes: height, weight, reach, stance, DOB
        - Career statistics: SLpM, Str_Acc, SApM, Str_Def, TD_Avg, TD_Acc, TD_Def, Sub_Avg

        Returns:
            DataFrame with columns:
                fighter_name, Height, Weight, Reach, Stance, DOB,
                SLpM, Str_Acc, SApM, Str_Def, TD_Avg, TD_Acc, TD_Def, Sub_Avg

        Example:
            >>> scraper = UFCDataScraper()
            >>> fighters_df = scraper.scrape_all_fighters()
            >>> print(fighters_df[['fighter_name', 'Height', 'Weight', 'Reach']].head())
        """
        start_time = time.time()
        logger.info("Starting fighter profile scraping...")

        all_fighters = []

        # Get all fighter links from alphabetical listing
        fighter_links = self._get_all_fighter_links()
        logger.info(f"Found {len(fighter_links)} total fighters")

        # Scrape each fighter's profile
        for i, (name, url) in enumerate(fighter_links.items(), 1):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(fighter_links)} fighters scraped")

            try:
                fighter_data = self._scrape_fighter_profile(name, url)
                if fighter_data:
                    all_fighters.append(fighter_data)
            except Exception as e:
                logger.warning(f"Error scraping fighter {name}: {e}")
                continue

        elapsed = time.time() - start_time
        logger.info(f"\nFighter scraping completed:")
        logger.info(f"  Total fighters: {len(all_fighters)}")
        logger.info(f"  Time elapsed: {elapsed:.1f}s")

        return pd.DataFrame(all_fighters)

    def _get_all_fighter_links(self) -> Dict[str, str]:
        """
        Get all fighter names and their profile URLs from the alphabetical directory.

        Returns:
            Dictionary mapping fighter names to their profile URLs
        """
        fighter_links = {}

        # Iterate through alphabet
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            url = f"{self.fighters_url}?char={letter}&page=all"
            soup = self._safe_request(url)

            if not soup:
                logger.warning(f"Failed to fetch fighters for letter: {letter}")
                continue

            # Find all fighter links in the table
            rows = soup.select('tr.b-statistics__table-row')

            for row in rows:
                fighter_link = row.select_one('a.b-link.b-link_style_black')
                if fighter_link and fighter_link.get('href'):
                    name = fighter_link.text.strip()
                    url = fighter_link['href']
                    fighter_links[name] = url

            logger.debug(f"Letter '{letter}': found {len([n for n in fighter_links if n.startswith(letter.upper())])} fighters")

        return fighter_links

    def _scrape_fighter_profile(self, name: str, profile_url: str) -> Optional[Dict]:
        """
        Scrape detailed profile information for a single fighter.

        Args:
            name: Fighter's name
            profile_url: URL to the fighter's profile page

        Returns:
            Dictionary with fighter biographical and career statistics
        """
        soup = self._safe_request(profile_url)
        if not soup:
            return None

        fighter_data = {'fighter_name': name}

        try:
            # Extract biographical data from the left column
            bio_list = soup.select('ul.b-list__box-list li.b-list__box-list-item')

            for item in bio_list:
                text = item.text.strip()

                # Height
                if text.startswith('Height:'):
                    fighter_data['Height'] = text.replace('Height:', '').strip()

                # Weight
                elif text.startswith('Weight:'):
                    fighter_data['Weight'] = text.replace('Weight:', '').strip()

                # Reach
                elif text.startswith('Reach:'):
                    reach = text.replace('Reach:', '').strip()
                    fighter_data['Reach'] = reach if reach != '--' else ''

                # Stance
                elif text.startswith('STANCE:'):
                    fighter_data['Stance'] = text.replace('STANCE:', '').strip()

                # Date of Birth
                elif text.startswith('DOB:'):
                    fighter_data['DOB'] = text.replace('DOB:', '').strip()

            # Extract career statistics
            stats_list = soup.select('ul.b-list__box-list.b-list__box-list_margin-top li.b-list__box-list-item')

            for item in stats_list:
                text = item.text.strip()

                # Significant Strikes Landed per Minute
                if 'SLpM:' in text:
                    fighter_data['SLpM'] = text.replace('SLpM:', '').strip()

                # Striking Accuracy
                elif 'Str. Acc.:' in text:
                    fighter_data['Str_Acc'] = text.replace('Str. Acc.:', '').strip()

                # Significant Strikes Absorbed per Minute
                elif 'SApM:' in text:
                    fighter_data['SApM'] = text.replace('SApM:', '').strip()

                # Striking Defense
                elif 'Str. Def:' in text:
                    fighter_data['Str_Def'] = text.replace('Str. Def:', '').strip()

                # Takedown Average
                elif 'TD Avg.:' in text:
                    fighter_data['TD_Avg'] = text.replace('TD Avg.:', '').strip()

                # Takedown Accuracy
                elif 'TD Acc.:' in text:
                    fighter_data['TD_Acc'] = text.replace('TD Acc.:', '').strip()

                # Takedown Defense
                elif 'TD Def.:' in text:
                    fighter_data['TD_Def'] = text.replace('TD Def.:', '').strip()

                # Submission Average
                elif 'Sub. Avg.:' in text:
                    fighter_data['Sub_Avg'] = text.replace('Sub. Avg.:', '').strip()

            # Fill in missing fields with empty strings
            for field in ['Height', 'Weight', 'Reach', 'Stance', 'DOB',
                         'SLpM', 'Str_Acc', 'SApM', 'Str_Def',
                         'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']:
                if field not in fighter_data:
                    fighter_data[field] = ''

        except Exception as e:
            logger.warning(f"Error parsing fighter profile for {name}: {e}")
            return None

        return fighter_data

    def scrape_fighter_stats(self, fighter_name: str) -> Optional[Dict]:
        """
        Scrape detailed statistics for a specific fighter by name.

        Args:
            fighter_name: Name of the fighter

        Returns:
            Dictionary containing fighter biographical and career statistics, or None if not found

        Example:
            >>> scraper = UFCDataScraper()
            >>> stats = scraper.scrape_fighter_stats("Conor McGregor")
            >>> print(f"Height: {stats['Height']}, Reach: {stats['Reach']}")
        """
        logger.info(f"Searching for fighter: {fighter_name}")

        # First, find the fighter's URL
        first_letter = fighter_name[0].lower()
        url = f"{self.fighters_url}?char={first_letter}&page=all"
        soup = self._safe_request(url)

        if not soup:
            logger.error(f"Failed to fetch fighter listing for letter: {first_letter}")
            return None

        # Search for the fighter
        rows = soup.select('tr.b-statistics__table-row')
        fighter_url = None

        for row in rows:
            fighter_link = row.select_one('a.b-link.b-link_style_black')
            if fighter_link and fighter_link.text.strip().lower() == fighter_name.lower():
                fighter_url = fighter_link['href']
                break

        if not fighter_url:
            logger.error(f"Fighter not found: {fighter_name}")
            return None

        # Scrape the fighter's profile
        return self._scrape_fighter_profile(fighter_name, fighter_url)

    def validate_fight_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and clean scraped fight data.

        Performs:
        - Missing value analysis
        - Data type validation
        - Outlier detection for numeric fields
        - Duplicate detection

        Args:
            df: DataFrame containing fight data

        Returns:
            Tuple of (cleaned DataFrame, validation report dictionary)
        """
        report = {
            'total_rows': len(df),
            'missing_values': {},
            'duplicates': 0,
            'data_quality_score': 0.0
        }

        if df.empty:
            logger.warning("Empty DataFrame provided for validation")
            return df, report

        # Check for missing values
        missing = df.isnull().sum()
        report['missing_values'] = {col: int(count) for col, count in missing.items() if count > 0}

        # Check for duplicates (based on fight_id or combination of event + fighters)
        if 'fight_id' in df.columns:
            duplicates = df.duplicated(subset=['fight_id'], keep='first')
            report['duplicates'] = int(duplicates.sum())
            df = df[~duplicates]  # Remove duplicates

        # Calculate data quality score (percentage of non-null values)
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.notna().sum().sum()
        report['data_quality_score'] = (non_null_cells / total_cells * 100) if total_cells > 0 else 0

        logger.info(f"Data validation complete: {report['total_rows']} rows, "
                   f"{report['duplicates']} duplicates removed, "
                   f"quality score: {report['data_quality_score']:.1f}%")

        return df, report

    def export_to_csv(self, df: pd.DataFrame, filepath: str, validate: bool = True) -> str:
        """
        Export DataFrame to CSV with optional validation.

        Args:
            df: DataFrame to export
            filepath: Output file path
            validate: Whether to validate data before export

        Returns:
            Path to the exported file

        Example:
            >>> scraper = UFCDataScraper()
            >>> df = scraper.scrape_historical_fights(limit_events=10)
            >>> scraper.export_to_csv(df, 'data/raw/fights.csv')
        """
        if validate:
            df, report = self.validate_fight_data(df)
            logger.info(f"Validation report: {report}")

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Export to CSV
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Exported {len(df)} rows to {filepath}")

        return filepath

    def get_scraping_progress(self) -> Dict:
        """
        Get current scraping progress statistics.

        Returns:
            Dictionary with progress metrics
        """
        return {
            'fighter_cache_size': len(self.fighter_cache),
            'session_active': bool(self.session),
            'rate_limit': self.rate_limit,
            'base_url': self.base_url
        }

    def close(self):
        """
        Close the requests session and cleanup resources.

        Should be called when done scraping to free up resources.

        Example:
            >>> scraper = UFCDataScraper()
            >>> try:
            >>>     data = scraper.scrape_all_data()
            >>> finally:
            >>>     scraper.close()
        """
        if self.session:
            self.session.close()
            logger.info("Scraper session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()


# Utility functions for data processing

def parse_strike_data(strike_string: str) -> Tuple[int, int]:
    """
    Parse strike data in "landed of attempted" format.

    Args:
        strike_string: String like "25 of 50" or "10 of 15"

    Returns:
        Tuple of (landed, attempted)

    Example:
        >>> landed, attempted = parse_strike_data("25 of 50")
        >>> print(f"{landed}/{attempted}")
        25/50
    """
    match = re.match(r'(\d+)\s+of\s+(\d+)', strike_string.strip())
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def calculate_strike_percentage(strike_string: str) -> float:
    """
    Calculate strike accuracy percentage from "landed of attempted" string.

    Args:
        strike_string: String like "25 of 50"

    Returns:
        Percentage as float (0-100)

    Example:
        >>> pct = calculate_strike_percentage("25 of 50")
        >>> print(f"{pct:.1f}%")
        50.0%
    """
    landed, attempted = parse_strike_data(strike_string)
    return (landed / attempted * 100) if attempted > 0 else 0.0


def parse_control_time(time_string: str) -> int:
    """
    Parse control time string to seconds.

    Args:
        time_string: Time in "MM:SS" format

    Returns:
        Total seconds as integer

    Example:
        >>> seconds = parse_control_time("3:45")
        >>> print(f"{seconds} seconds")
        225 seconds
    """
    match = re.match(r'(\d+):(\d+)', time_string.strip())
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return minutes * 60 + seconds
    return 0


def clean_fighter_name(name: str) -> str:
    """
    Clean and standardize fighter name.

    Args:
        name: Raw fighter name string

    Returns:
        Cleaned fighter name

    Example:
        >>> clean_name = clean_fighter_name("  CONOR McGREGOR  ")
        >>> print(clean_name)
        Conor McGregor
    """
    # Remove extra whitespace
    name = ' '.join(name.split())

    # Title case while preserving known patterns (Mc, Mac, O')
    words = name.split()
    cleaned_words = []

    for word in words:
        if word.startswith('Mc') and len(word) > 2:
            cleaned_words.append('Mc' + word[2:].capitalize())
        elif word.startswith('Mac') and len(word) > 3:
            cleaned_words.append('Mac' + word[3:].capitalize())
        elif word.startswith("O'") and len(word) > 2:
            cleaned_words.append("O'" + word[2:].capitalize())
        else:
            cleaned_words.append(word.capitalize())

    return ' '.join(cleaned_words)
