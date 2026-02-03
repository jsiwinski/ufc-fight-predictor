"""
Example usage of the UFC Data Scraper.

This script demonstrates how to use the UFCDataScraper class to collect
historical fight data and fighter profiles for machine learning models.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.etl.scraper import UFCDataScraper
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_scrape_all_data():
    """
    Example 1: Scrape all historical fights and fighter profiles.

    This is the comprehensive approach - scrapes everything.
    Warning: This can take several hours for full UFC history!
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Scrape All UFC Data (Comprehensive)")
    print("=" * 80)

    # Initialize scraper with custom config
    config = {
        'rate_limit': 2,      # Wait 2 seconds between requests (be respectful!)
        'timeout': 10,        # 10 second timeout per request
        'max_retries': 3      # Retry failed requests 3 times
    }

    scraper = UFCDataScraper(config)

    try:
        # Scrape all data
        output_files = scraper.scrape_all_data(
            output_dir='data/raw',
            limit_events=None  # Set to small number for testing (e.g., 10)
        )

        print(f"\nFiles created:")
        print(f"  Fights: {output_files['fights']}")
        print(f"  Fighters: {output_files['fighters']}")

    finally:
        scraper.close()


def example_2_scrape_recent_fights():
    """
    Example 2: Scrape only recent fights (last year).

    More practical for regular updates.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Scrape Recent Fights Only")
    print("=" * 80)

    with UFCDataScraper() as scraper:
        # Scrape fights from 2024 onwards
        df = scraper.scrape_historical_fights(
            start_date='2024-01-01',
            end_date=None
        )

        print(f"\nScraped {len(df)} fights from 2024 onwards")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df[['event_name', 'event_date', 'red_fighter_name',
                  'blue_fighter_name', 'method']].head())

        # Export to CSV
        scraper.export_to_csv(df, 'data/raw/recent_fights.csv')


def example_3_scrape_specific_fighter():
    """
    Example 3: Get stats for a specific fighter.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Scrape Specific Fighter Profile")
    print("=" * 80)

    with UFCDataScraper() as scraper:
        # Get fighter stats
        fighter_name = "Conor McGregor"
        stats = scraper.scrape_fighter_stats(fighter_name)

        if stats:
            print(f"\nFighter: {stats['fighter_name']}")
            print(f"  Height: {stats.get('Height', 'N/A')}")
            print(f"  Weight: {stats.get('Weight', 'N/A')}")
            print(f"  Reach: {stats.get('Reach', 'N/A')}")
            print(f"  Stance: {stats.get('Stance', 'N/A')}")
            print(f"  Striking Accuracy: {stats.get('Str_Acc', 'N/A')}")
            print(f"  Takedown Accuracy: {stats.get('TD_Acc', 'N/A')}")
        else:
            print(f"Fighter '{fighter_name}' not found")


def example_4_test_scraping():
    """
    Example 4: Test with limited data (for development).

    Use this to test the scraper without waiting hours.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Test Scraping (Limited Events)")
    print("=" * 80)

    with UFCDataScraper({'rate_limit': 1}) as scraper:
        # Scrape only 5 most recent events
        df = scraper.scrape_historical_fights(limit_events=5)

        print(f"\nScraped {len(df)} fights from 5 events")
        print(f"Columns: {df.shape[1]}")

        # Validate data
        validated_df, report = scraper.validate_fight_data(df)

        print(f"\nValidation Report:")
        print(f"  Total rows: {report['total_rows']}")
        print(f"  Duplicates: {report['duplicates']}")
        print(f"  Data quality: {report['data_quality_score']:.1f}%")

        if report['missing_values']:
            print(f"\nColumns with missing values:")
            for col, count in list(report['missing_values'].items())[:10]:
                print(f"    {col}: {count}")


def example_5_scrape_fighters_only():
    """
    Example 5: Scrape only fighter profiles (no fight data).
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Scrape All Fighter Profiles")
    print("=" * 80)

    with UFCDataScraper() as scraper:
        # This scrapes ALL fighters alphabetically
        # Warning: Can take 30+ minutes depending on rate limit
        df = scraper.scrape_all_fighters()

        print(f"\nScraped {len(df)} fighter profiles")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df[['fighter_name', 'Height', 'Weight', 'Reach',
                  'Str_Acc', 'TD_Acc']].head(10))

        # Export
        scraper.export_to_csv(df, 'data/raw/all_fighters.csv')


if __name__ == '__main__':
    print("\nUFC Data Scraper - Usage Examples")
    print("=" * 80)
    print("\nChoose an example to run:")
    print("1. Scrape all data (fights + fighters) - COMPREHENSIVE")
    print("2. Scrape recent fights only (2024+)")
    print("3. Scrape specific fighter profile")
    print("4. Test scraping (5 events only)")
    print("5. Scrape all fighter profiles only")
    print("\nNote: Example 1 and 5 take significant time. Use Example 4 for testing.")

    choice = input("\nEnter choice (1-5) or press Enter to run Example 4: ").strip()

    examples = {
        '1': example_1_scrape_all_data,
        '2': example_2_scrape_recent_fights,
        '3': example_3_scrape_specific_fighter,
        '4': example_4_test_scraping,
        '5': example_5_scrape_fighters_only,
        '': example_4_test_scraping  # Default
    }

    example_func = examples.get(choice, example_4_test_scraping)

    try:
        example_func()
        print("\n" + "=" * 80)
        print("Example completed successfully!")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user")
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
