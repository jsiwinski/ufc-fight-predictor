#!/usr/bin/env python3
"""
ESPN Fighter Headshot Scraper.

Scrapes fighter headshots from ESPN for use in the UFC predictions web app.
Uses ESPN's search API to find fighter IDs, then downloads headshots from ESPN CDN.

Usage:
    # Scrape headshots for next upcoming event
    python src/scraper/headshots.py

    # Scrape for specific fighters
    python src/scraper/headshots.py --fighters "Vinicius Oliveira" "Mario Bautista"

    # Scrape from predictions file
    python src/scraper/headshots.py --from-predictions data/predictions/upcoming.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ESPN_SEARCH_API = "https://site.web.api.espn.com/apis/common/v3/search"
ESPN_HEADSHOT_URL = "https://a.espncdn.com/i/headshots/mma/players/full/{espn_id}.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "web" / "static" / "fighters"
RATE_LIMIT = 1.5  # seconds between requests


def slugify(name: str) -> str:
    """
    Convert fighter name to URL-safe slug.

    Args:
        name: Fighter's full name

    Returns:
        Slugified name (lowercase, hyphens, no special chars)
    """
    slug = name.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')


def get_initials(name: str) -> str:
    """
    Get fighter initials for fallback display.

    Args:
        name: Fighter's full name

    Returns:
        Two-letter initials (e.g., "VO" for "Vinicius Oliveira")
    """
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    elif len(parts) == 1 and len(parts[0]) >= 2:
        return parts[0][:2].upper()
    return "??"


class HeadshotScraper:
    """
    ESPN fighter headshot scraper with caching.

    Searches ESPN for UFC fighters by name and downloads their headshots.
    Implements rate limiting and caching to be respectful of ESPN's servers.
    """

    def __init__(self, output_dir: str = None, rate_limit: float = RATE_LIMIT):
        """
        Initialize the headshot scraper.

        Args:
            output_dir: Directory to save headshot images
            rate_limit: Seconds to wait between requests
        """
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.session = self._create_session()

        logger.info(f"HeadshotScraper initialized, output: {self.output_dir}")

    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json,image/png,image/jpeg,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        })

        return session

    def search_espn_fighter(self, name: str) -> Optional[Dict]:
        """
        Search ESPN for a fighter by name.

        Args:
            name: Fighter's name to search for

        Returns:
            Dict with espn_id, url, image_url or None if not found
        """
        try:
            params = {
                'query': name,
                'type': 'player',
                'sport': 'mma',
                'limit': 5
            }

            time.sleep(self.rate_limit)
            response = self.session.get(ESPN_SEARCH_API, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Look through results for best match
            items = data.get('items', [])
            if not items:
                logger.warning(f"No ESPN results for: {name}")
                return None

            # Try to find exact or close match
            name_lower = name.lower().strip()

            for item in items:
                item_name = item.get('displayName', '').lower().strip()

                # Check for exact match or close match
                if item_name == name_lower:
                    return self._extract_fighter_info(item)

                # Check if names share significant parts
                name_parts = set(name_lower.split())
                item_parts = set(item_name.split())
                overlap = name_parts & item_parts

                if len(overlap) >= min(len(name_parts), len(item_parts)):
                    return self._extract_fighter_info(item)

            # If no good match, return first result as fallback
            first_result = items[0]
            logger.info(f"Fuzzy match for '{name}' -> '{first_result.get('displayName', 'Unknown')}'")
            return self._extract_fighter_info(first_result)

        except requests.RequestException as e:
            logger.error(f"ESPN search failed for {name}: {e}")
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse ESPN response for {name}: {e}")
            return None

    def _extract_fighter_info(self, item: Dict) -> Dict:
        """Extract fighter info from ESPN search result."""
        espn_id = item.get('id', '')
        display_name = item.get('displayName', '')

        # Get headshot URL - try from response first, fall back to constructed URL
        headshot = item.get('headshot', {})
        image_url = headshot.get('href') if headshot else None

        if not image_url and espn_id:
            image_url = ESPN_HEADSHOT_URL.format(espn_id=espn_id)

        return {
            'espn_id': espn_id,
            'espn_name': display_name,
            'image_url': image_url,
            'url': f"https://www.espn.com/mma/fighter/_/id/{espn_id}"
        }

    def download_headshot(self, espn_id: str, fighter_name: str) -> Optional[str]:
        """
        Download fighter headshot from ESPN.

        Args:
            espn_id: ESPN's fighter ID
            fighter_name: Fighter's name (for filename)

        Returns:
            Path to saved file or None if failed
        """
        slug = slugify(fighter_name)
        filepath = self.output_dir / f"{slug}.png"

        # Check cache
        if filepath.exists():
            logger.debug(f"Cache hit: {filepath}")
            return str(filepath)

        image_url = ESPN_HEADSHOT_URL.format(espn_id=espn_id)

        try:
            time.sleep(self.rate_limit)
            response = self.session.get(image_url, timeout=15)

            # ESPN returns 200 with empty/placeholder for missing images
            if response.status_code == 404:
                logger.warning(f"No headshot found for {fighter_name} (ID: {espn_id})")
                return None

            response.raise_for_status()

            # Verify we got an actual image (not a redirect page or error)
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type.lower():
                logger.warning(f"Non-image response for {fighter_name}: {content_type}")
                return None

            # Check minimum file size (ESPN placeholder images are tiny)
            if len(response.content) < 1000:
                logger.warning(f"Image too small for {fighter_name}, likely placeholder")
                return None

            # Save the image
            with open(filepath, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded: {fighter_name} -> {filepath.name}")
            return str(filepath)

        except requests.RequestException as e:
            logger.error(f"Failed to download headshot for {fighter_name}: {e}")
            return None

    def scrape_fighter(self, name: str) -> Dict:
        """
        Search for and download headshot for a single fighter.

        Args:
            name: Fighter's full name

        Returns:
            Dict with status, path, and metadata
        """
        slug = slugify(name)
        cached_path = self.output_dir / f"{slug}.png"

        # Check cache first
        if cached_path.exists():
            return {
                'name': name,
                'slug': slug,
                'status': 'cached',
                'path': str(cached_path),
                'initials': get_initials(name)
            }

        # Search ESPN
        fighter_info = self.search_espn_fighter(name)

        if not fighter_info or not fighter_info.get('espn_id'):
            return {
                'name': name,
                'slug': slug,
                'status': 'not_found',
                'path': None,
                'initials': get_initials(name)
            }

        # Download headshot
        filepath = self.download_headshot(fighter_info['espn_id'], name)

        if filepath:
            return {
                'name': name,
                'slug': slug,
                'status': 'downloaded',
                'path': filepath,
                'espn_id': fighter_info['espn_id'],
                'espn_name': fighter_info.get('espn_name'),
                'initials': get_initials(name)
            }
        else:
            return {
                'name': name,
                'slug': slug,
                'status': 'download_failed',
                'path': None,
                'espn_id': fighter_info['espn_id'],
                'initials': get_initials(name)
            }

    def scrape_fighters(self, names: List[str]) -> Dict[str, Dict]:
        """
        Scrape headshots for multiple fighters.

        Args:
            names: List of fighter names

        Returns:
            Dict mapping fighter names to their status/path info
        """
        results = {}

        for i, name in enumerate(names, 1):
            logger.info(f"[{i}/{len(names)}] Processing: {name}")
            results[name] = self.scrape_fighter(name)

        return results

    def close(self):
        """Close the requests session."""
        if self.session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def scrape_event_headshots(
    fighter_names: List[str],
    output_dir: str = None
) -> Dict[str, Dict]:
    """
    Download headshots for all fighters on a card.

    Args:
        fighter_names: List of fighter names
        output_dir: Directory to save images (default: src/web/static/fighters/)

    Returns:
        Dict mapping fighter names to status/path info
    """
    with HeadshotScraper(output_dir=output_dir) as scraper:
        return scraper.scrape_fighters(fighter_names)


def get_fighters_from_predictions(predictions_path: str) -> List[str]:
    """
    Extract fighter names from a predictions JSON file.

    Args:
        predictions_path: Path to predictions JSON file

    Returns:
        List of unique fighter names
    """
    try:
        with open(predictions_path, 'r') as f:
            data = json.load(f)

        fighters = set()
        predictions = data.get('predictions', [])

        for pred in predictions:
            if 'fighter1' in pred:
                fighters.add(pred['fighter1'])
            if 'fighter2' in pred:
                fighters.add(pred['fighter2'])

        return list(fighters)

    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to read predictions file: {e}")
        return []


def get_upcoming_event_fighters() -> List[str]:
    """
    Get fighter names from the next upcoming event.

    Returns:
        List of fighter names from upcoming event
    """
    try:
        from src.predict.serve import PredictionPipeline

        pipeline = PredictionPipeline(
            model_path=str(PROJECT_ROOT / 'data' / 'models' / 'ufc_model_v1.pkl'),
            feature_names_path=str(PROJECT_ROOT / 'data' / 'models' / 'feature_names_v1.json'),
            processed_data_path=str(PROJECT_ROOT / 'data' / 'processed' / 'ufc_fights_features_v1.csv'),
            raw_data_path=str(PROJECT_ROOT / 'data' / 'raw' / 'ufc_fights_v1.csv')
        )

        upcoming_df = pipeline.scrape_upcoming_event()

        # Handle DataFrame result
        if isinstance(upcoming_df, pd.DataFrame):
            if upcoming_df.empty:
                logger.warning("No upcoming fights found")
                return []

            fighters = set()

            # Check for different column naming conventions
            if 'fighter1_name' in upcoming_df.columns:
                fighters.update(upcoming_df['fighter1_name'].dropna().tolist())
            if 'fighter2_name' in upcoming_df.columns:
                fighters.update(upcoming_df['fighter2_name'].dropna().tolist())

            return list(fighters)
        else:
            logger.warning("Unexpected result type from scrape_upcoming_event")
            return []

    except Exception as e:
        logger.error(f"Failed to get upcoming event fighters: {e}")
        import traceback
        traceback.print_exc()
        return []


def print_results(results: Dict[str, Dict]):
    """Print formatted results summary."""
    print("\n" + "=" * 60)
    print("HEADSHOT SCRAPING RESULTS")
    print("=" * 60)

    cached = [r for r in results.values() if r['status'] == 'cached']
    downloaded = [r for r in results.values() if r['status'] == 'downloaded']
    not_found = [r for r in results.values() if r['status'] == 'not_found']
    failed = [r for r in results.values() if r['status'] == 'download_failed']

    print(f"\nTotal fighters: {len(results)}")
    print(f"  Cached:     {len(cached)}")
    print(f"  Downloaded: {len(downloaded)}")
    print(f"  Not found:  {len(not_found)}")
    print(f"  Failed:     {len(failed)}")

    if downloaded:
        print("\nNewly downloaded:")
        for r in downloaded:
            print(f"  + {r['name']}")

    if not_found:
        print("\nNot found on ESPN:")
        for r in not_found:
            print(f"  - {r['name']} (will show: {r['initials']})")

    if failed:
        print("\nDownload failed:")
        for r in failed:
            print(f"  ! {r['name']}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Scrape UFC fighter headshots from ESPN')
    parser.add_argument('--fighters', nargs='+', help='Fighter names to scrape')
    parser.add_argument('--from-predictions', type=str, help='Path to predictions JSON file')
    parser.add_argument('--output', type=str, help='Output directory for images')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine fighter list
    if args.fighters:
        fighters = args.fighters
        print(f"Scraping {len(fighters)} fighters from command line")
    elif args.from_predictions:
        fighters = get_fighters_from_predictions(args.from_predictions)
        print(f"Scraping {len(fighters)} fighters from {args.from_predictions}")
    else:
        print("Fetching upcoming event fighters...")
        fighters = get_upcoming_event_fighters()
        print(f"Found {len(fighters)} fighters in upcoming event")

    if not fighters:
        print("No fighters to scrape")
        return

    # Run scraper
    results = scrape_event_headshots(fighters, output_dir=args.output)
    print_results(results)


if __name__ == '__main__':
    main()
