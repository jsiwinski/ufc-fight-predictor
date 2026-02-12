#!/usr/bin/env python3
"""
Fighter Details Scraper for UFC Fight Predictor.

Scrapes comprehensive fighter details from ufcstats.com including:
- Total professional record (wins, losses, draws)
- Physical attributes (height, reach, stance, DOB)

The previous Phase 9.1 scraper had a bug where it only extracted first names
from the fighter listing page. This scraper fixes that by properly combining
first and last name columns, AND extracts physical attributes.

Usage:
    python src/etl/scrape_fighter_details.py
    python src/etl/scrape_fighter_details.py --rate-limit 1.0
    python src/etl/scrape_fighter_details.py --from-fights  # Get fighters from fight data
"""

import argparse
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FighterDetailsScraper:
    """
    Comprehensive fighter details scraper.

    Extracts:
    - Total professional record (W-L-D)
    - Height (in inches)
    - Reach (in inches)
    - Stance (orthodox/southpaw/switch)
    - Date of Birth
    """

    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize the scraper.

        Args:
            rate_limit: Seconds to wait between requests (default 1.0)
        """
        self.base_url = 'http://www.ufcstats.com'
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def _safe_request(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """Make a safe HTTP request with retry logic."""
        for attempt in range(retries):
            try:
                time.sleep(self.rate_limit)
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}) for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def get_all_fighters_from_listing(self) -> Dict[str, str]:
        """
        Get all fighter names and URLs from alphabetical listing pages.

        FIXED: Properly combines first name + last name from separate columns.

        Returns:
            Dictionary mapping full fighter names to their profile URLs.
        """
        fighter_links = {}

        for letter in 'abcdefghijklmnopqrstuvwxyz':
            url = f"{self.base_url}/statistics/fighters?char={letter}&page=all"
            soup = self._safe_request(url)

            if not soup:
                logger.warning(f"Failed to fetch fighters for letter: {letter}")
                continue

            # Find all fighter rows
            rows = soup.select('tr.b-statistics__table-row')

            for row in rows:
                # Get all links in this row (first name, last name, nickname columns)
                links = row.select('a.b-link.b-link_style_black')

                if len(links) < 2:
                    continue

                # First link is first name, second is last name
                first_name = links[0].text.strip()
                last_name = links[1].text.strip()

                # Skip if both are empty
                if not first_name and not last_name:
                    continue

                # Full name is "First Last"
                full_name = f"{first_name} {last_name}".strip()

                # Get URL from any link (they all point to same page)
                fighter_url = links[0].get('href')

                if fighter_url and full_name:
                    fighter_links[full_name] = fighter_url

            logger.info(f"Letter '{letter}': {len(fighter_links)} total fighters found")

        return fighter_links

    def get_fighters_from_fight_data(self, fights_path: str) -> Dict[str, str]:
        """
        Get unique fighter names from fight data and find their profile URLs.

        This is a backup approach if the listing page scraping fails.
        It uses the fighter names we already have and searches for their URLs.

        Args:
            fights_path: Path to the raw fights CSV

        Returns:
            Dictionary mapping fighter names to their profile URLs (may have gaps)
        """
        logger.info(f"Loading fighters from {fights_path}...")

        fights_df = pd.read_csv(fights_path)

        # Get unique fighter names
        f1_names = fights_df['fighter1_name'].dropna().unique()
        f2_names = fights_df['fighter2_name'].dropna().unique()
        all_fighters = set(f1_names) | set(f2_names)

        logger.info(f"Found {len(all_fighters)} unique fighters in fight data")

        # Get fighter URLs from listing page
        fighter_urls = self.get_all_fighters_from_listing()

        # Build mapping for fighters in our data
        matched_urls = {}
        unmatched = []

        for fighter in all_fighters:
            if fighter in fighter_urls:
                matched_urls[fighter] = fighter_urls[fighter]
            else:
                # Try case-insensitive match
                for listing_name, url in fighter_urls.items():
                    if listing_name.lower() == fighter.lower():
                        matched_urls[fighter] = url
                        break
                else:
                    unmatched.append(fighter)

        logger.info(f"Matched {len(matched_urls)}/{len(all_fighters)} fighters")
        if unmatched:
            logger.info(f"Unmatched fighters (first 10): {unmatched[:10]}")

        return matched_urls

    def scrape_fighter_details(self, name: str, url: str) -> Optional[Dict]:
        """
        Scrape detailed information for a single fighter.

        Args:
            name: Fighter's name
            url: Fighter's profile URL

        Returns:
            Dictionary with fighter details or None if scraping fails
        """
        soup = self._safe_request(url)
        if not soup:
            return None

        details = {
            'fighter_name': name,
            'fighter_url': url,
            'total_wins': 0,
            'total_losses': 0,
            'total_draws': 0,
            'total_fights': 0,
            'height_inches': None,
            'reach_inches': None,
            'stance': None,
            'dob': None,
            'age_years': None,
        }

        try:
            # Extract total professional record: "Record: W-L-D"
            record_span = soup.select_one('span.b-content__title-record')
            if record_span:
                record_text = record_span.text.strip()
                match = re.search(r'Record:\s*(\d+)-(\d+)-(\d+)', record_text)
                if match:
                    details['total_wins'] = int(match.group(1))
                    details['total_losses'] = int(match.group(2))
                    details['total_draws'] = int(match.group(3))
                    details['total_fights'] = (
                        details['total_wins'] +
                        details['total_losses'] +
                        details['total_draws']
                    )

            # Extract biographical info from the list
            bio_items = soup.select('ul.b-list__box-list li.b-list__box-list-item')

            for item in bio_items:
                text = item.text.strip()

                # Height: "5' 9"" or "5' 10""
                if text.startswith('Height:'):
                    height_text = text.replace('Height:', '').strip()
                    height_inches = self._parse_height(height_text)
                    if height_inches:
                        details['height_inches'] = height_inches

                # Reach: "72""
                elif text.startswith('Reach:'):
                    reach_text = text.replace('Reach:', '').strip()
                    reach_inches = self._parse_reach(reach_text)
                    if reach_inches:
                        details['reach_inches'] = reach_inches

                # Stance: "Orthodox", "Southpaw", "Switch"
                elif 'STANCE:' in text.upper():
                    stance = text.replace('STANCE:', '').replace('Stance:', '').strip()
                    if stance and stance != '--':
                        details['stance'] = stance

                # DOB: "Mar 17, 1988"
                elif text.startswith('DOB:'):
                    dob_text = text.replace('DOB:', '').strip()
                    if dob_text and dob_text != '--':
                        details['dob'] = dob_text
                        # Calculate age
                        age = self._calculate_age(dob_text)
                        if age:
                            details['age_years'] = age

        except Exception as e:
            logger.warning(f"Error parsing details for {name}: {e}")

        return details

    def _parse_height(self, height_text: str) -> Optional[int]:
        """Parse height string to inches. E.g., "5' 9"" -> 69"""
        if not height_text or height_text == '--':
            return None

        try:
            # Match "5' 9"" or "5' 10""
            match = re.search(r"(\d+)'\s*(\d+)\"?", height_text)
            if match:
                feet = int(match.group(1))
                inches = int(match.group(2))
                return feet * 12 + inches
        except:
            pass

        return None

    def _parse_reach(self, reach_text: str) -> Optional[int]:
        """Parse reach string to inches. E.g., '72"' -> 72"""
        if not reach_text or reach_text == '--':
            return None

        try:
            # Match digits followed by optional quote
            match = re.search(r'(\d+)', reach_text)
            if match:
                return int(match.group(1))
        except:
            pass

        return None

    def _calculate_age(self, dob_text: str) -> Optional[int]:
        """Calculate age from DOB string. E.g., "Mar 17, 1988" -> 36"""
        if not dob_text or dob_text == '--':
            return None

        try:
            # Parse various date formats
            for fmt in ['%b %d, %Y', '%B %d, %Y']:
                try:
                    dob = datetime.strptime(dob_text, fmt)
                    today = datetime.now()
                    age = today.year - dob.year
                    if (today.month, today.day) < (dob.month, dob.day):
                        age -= 1
                    return age
                except ValueError:
                    continue
        except:
            pass

        return None

    def scrape_all_details(
        self,
        output_path: str = 'data/raw/fighter_details.csv',
        from_fights: bool = False,
        fights_path: str = 'data/raw/ufc_fights_v1.csv'
    ) -> pd.DataFrame:
        """
        Scrape details for all fighters.

        Args:
            output_path: Path to save the output CSV
            from_fights: If True, only scrape fighters from fight data
            fights_path: Path to fight data (used if from_fights=True)

        Returns:
            DataFrame with all fighter details
        """
        logger.info("=" * 60)
        logger.info("FIGHTER DETAILS SCRAPER")
        logger.info("=" * 60)

        # Get fighter URLs
        if from_fights:
            fighter_urls = self.get_fighters_from_fight_data(fights_path)
        else:
            fighter_urls = self.get_all_fighters_from_listing()

        logger.info(f"\nScraping details for {len(fighter_urls)} fighters...")
        logger.info(f"Estimated time: {len(fighter_urls) * self.rate_limit / 60:.1f} minutes")

        all_details = []
        failed = []

        for i, (name, url) in enumerate(fighter_urls.items(), 1):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(fighter_urls)} ({i/len(fighter_urls)*100:.1f}%)")

            details = self.scrape_fighter_details(name, url)

            if details:
                all_details.append(details)
            else:
                failed.append(name)

        # Create DataFrame
        df = pd.DataFrame(all_details)

        # Save to CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SCRAPING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Fighters scraped: {len(df)}")
        logger.info(f"Fighters failed: {len(failed)}")
        logger.info(f"Output saved to: {output_path}")

        # Stats
        if len(df) > 0:
            logger.info(f"\nData coverage:")
            logger.info(f"  - Height: {df['height_inches'].notna().sum()}/{len(df)} ({df['height_inches'].notna().mean()*100:.1f}%)")
            logger.info(f"  - Reach: {df['reach_inches'].notna().sum()}/{len(df)} ({df['reach_inches'].notna().mean()*100:.1f}%)")
            logger.info(f"  - Stance: {df['stance'].notna().sum()}/{len(df)} ({df['stance'].notna().mean()*100:.1f}%)")
            logger.info(f"  - DOB: {df['dob'].notna().sum()}/{len(df)} ({df['dob'].notna().mean()*100:.1f}%)")

            logger.info(f"\nRecord stats:")
            logger.info(f"  - Avg total fights: {df['total_fights'].mean():.1f}")
            logger.info(f"  - Max total fights: {df['total_fights'].max()}")
            logger.info(f"  - Total with 0 fights: {(df['total_fights'] == 0).sum()}")

        return df


def main():
    parser = argparse.ArgumentParser(description='Scrape UFC fighter details')
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Seconds between requests (default: 1.0)'
    )
    parser.add_argument(
        '--from-fights',
        action='store_true',
        help='Only scrape fighters that appear in fight data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/fighter_details.csv',
        help='Output CSV path'
    )
    parser.add_argument(
        '--fights-path',
        type=str,
        default='data/raw/ufc_fights_v1.csv',
        help='Path to fights CSV (for --from-fights mode)'
    )

    args = parser.parse_args()

    scraper = FighterDetailsScraper(rate_limit=args.rate_limit)

    df = scraper.scrape_all_details(
        output_path=args.output,
        from_fights=args.from_fights,
        fights_path=args.fights_path
    )

    return df


if __name__ == '__main__':
    main()
