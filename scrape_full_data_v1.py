"""
Scrape all historical UFC fight data and save as v1 CSV.

This script will scrape all 760+ events from ufcstats.com (takes ~25-30 minutes).
"""

import logging
from datetime import datetime
from pathlib import Path
from src.etl.scraper import UFCDataScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    start_time = datetime.now()
    logger.info("Starting full UFC data scrape for v1...")
    logger.info("This will take approximately 25-30 minutes")

    # Ensure output directory exists
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize scraper with default rate limiting (2 seconds)
    with UFCDataScraper() as scraper:
        # Scrape all historical fights
        logger.info("Scraping all historical fights...")
        df = scraper.scrape_historical_fights()

        logger.info(f"Successfully scraped {len(df)} total fights from {df['event_name'].nunique()} events")

        # Save to CSV
        output_path = 'data/raw/ufc_fights_v1.csv'
        logger.info(f"Saving to {output_path}...")

        filepath = scraper.export_to_csv(df, output_path, validate=True)

        # Show summary
        logger.info("=" * 80)
        logger.info("SCRAPE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Output file: {filepath}")
        logger.info(f"Total fights: {len(df):,}")
        logger.info(f"Total events: {df['event_name'].nunique():,}")
        logger.info(f"Columns: {df.shape[1]}")
        logger.info(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")

        file_size = Path(filepath).stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size:.2f} MB")

        elapsed = datetime.now() - start_time
        logger.info(f"Time elapsed: {elapsed}")
        logger.info("=" * 80)

if __name__ == '__main__':
    main()
