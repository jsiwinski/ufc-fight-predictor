#!/usr/bin/env python3
"""
Generate predictions for all upcoming and recent UFC events.

This script scrapes ALL upcoming events and recent completed events,
runs predictions using the Phase 8 model, and saves results to events.json.

Also writes predictions to the immutable ledger for historical tracking.

Usage:
    python src/predict/generate_all_events.py
    python src/predict/generate_all_events.py --completed-limit 10
    python src/predict/generate_all_events.py --write-ledger  # Also write to ledger
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.predict.serve import PredictionPipeline, MODEL_VERSION
from src.etl.scraper import UFCDataScraper
from src.predict.lock_results import record_predictions, get_ledger_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output path for events data
EVENTS_JSON_PATH = Path('data/predictions/events.json')


def slugify(text: str) -> str:
    """Convert event name to URL-safe slug."""
    return text.lower().replace(' ', '-').replace(':', '').replace('.', '')


def scrape_all_upcoming_events(scraper: UFCDataScraper) -> List[Dict]:
    """
    Scrape ALL upcoming events, not just the first one.

    Returns:
        List of event dicts, each containing event info and list of fights
    """
    logger.info("Scraping all upcoming events...")

    upcoming_url = f"{scraper.base_url}/statistics/events/upcoming"

    try:
        response = scraper.session.get(upcoming_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch upcoming events: {e}")
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    events = []

    # Find the events table
    table = soup.find('table', {'class': 'b-statistics__table-events'})
    if not table:
        logger.warning("Could not find upcoming events table")
        return []

    rows = table.find('tbody').find_all('tr', {'class': 'b-statistics__table-row'})

    for row in rows:
        try:
            cells = row.find_all('td', {'class': 'b-statistics__table-col'})
            if len(cells) < 2:
                continue

            event_link = cells[0].find('a', {'class': 'b-link'})
            if not event_link:
                continue

            event_name = event_link.text.strip()
            event_url = event_link['href']

            event_date = cells[0].find('span', {'class': 'b-statistics__date'})
            event_date = event_date.text.strip() if event_date else ''

            event_location = cells[1].text.strip() if len(cells) > 1 else ''

            events.append({
                'name': event_name,
                'date': event_date,
                'location': event_location,
                'url': event_url,
                'status': 'upcoming'
            })

        except Exception as e:
            logger.warning(f"Error parsing upcoming event row: {e}")
            continue

    # Now get fights for each event
    for event in events:
        logger.info(f"  Scraping fights for: {event['name']}")
        fights = scraper._get_upcoming_event_fights(event)
        event['fights'] = fights

    logger.info(f"Found {len(events)} upcoming events")
    return events


def scrape_recent_completed_events(scraper: UFCDataScraper, limit: int = 10) -> List[Dict]:
    """
    Scrape recent completed events.

    Args:
        scraper: UFCDataScraper instance
        limit: Maximum number of events to scrape

    Returns:
        List of event dicts with fight results
    """
    logger.info(f"Scraping last {limit} completed events...")

    events_url = f"{scraper.base_url}/statistics/events/completed?page=all"

    try:
        response = scraper.session.get(events_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch completed events: {e}")
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    events = []

    table = soup.find('table', {'class': 'b-statistics__table-events'})
    if not table:
        return []

    rows = table.find('tbody').find_all('tr', {'class': 'b-statistics__table-row'})

    count = 0
    for row in rows:
        if count >= limit:
            break

        try:
            cells = row.find_all('td', {'class': 'b-statistics__table-col'})
            if len(cells) < 2:
                continue

            event_link = cells[0].find('a', {'class': 'b-link'})
            if not event_link:
                continue

            event_name = event_link.text.strip()
            event_url = event_link['href']

            event_date = cells[0].find('span', {'class': 'b-statistics__date'})
            event_date = event_date.text.strip() if event_date else ''

            event_location = cells[1].text.strip() if len(cells) > 1 else ''

            events.append({
                'name': event_name,
                'date': event_date,
                'location': event_location,
                'url': event_url,
                'status': 'completed'
            })
            count += 1

        except Exception as e:
            logger.warning(f"Error parsing completed event row: {e}")
            continue

    # Get fights for each event
    for event in events:
        logger.info(f"  Scraping fights for: {event['name']}")
        fights = scraper._get_event_fights(event)
        event['fights_raw'] = fights

    logger.info(f"Found {len(events)} completed events")
    return events


def generate_upcoming_predictions(
    pipeline: PredictionPipeline,
    events: List[Dict]
) -> List[Dict]:
    """
    Generate predictions for all upcoming events.

    Returns:
        List of event dicts with predictions added
    """
    results = []

    for event in events:
        if not event.get('fights'):
            continue

        logger.info(f"Generating predictions for: {event['name']}")

        # Convert fights to DataFrame
        fights_df = pd.DataFrame(event['fights'])

        try:
            predictions = pipeline.predict_event(fights_df)
        except Exception as e:
            logger.error(f"Error predicting {event['name']}: {e}")
            continue

        # Format event data
        event_result = {
            'name': event['name'],
            'slug': slugify(event['name']),
            'date': event['date'],
            'location': event.get('location', ''),
            'status': 'upcoming',
            'fight_count': len(predictions),
            'fights': []
        }

        for pred in predictions:
            # Store top_factors for tornado chart display
            # Convert from list of tuples to list of dicts for JSON serialization
            top_factors = pred.get('top_factors', [])
            factors_for_json = [
                {'feature': f[0], 'value': f[1]}
                for f in top_factors if isinstance(f, (list, tuple))
            ]

            fight = {
                'fighter_1': pred['fighter1'],
                'fighter_2': pred['fighter2'],
                'weight_class': pred.get('weight_class', ''),
                'f1_win_prob': round(pred['f1_win_prob'], 3),
                'f2_win_prob': round(pred['f2_win_prob'], 3),
                'predicted_winner': pred['predicted_winner'],
                'confidence': pred['confidence'],
                'f1_elo': pred.get('f1_elo', 1500),
                'f2_elo': pred.get('f2_elo', 1500),
                'f1_exact_match': pred.get('f1_exact_match', True),
                'f2_exact_match': pred.get('f2_exact_match', True),
                'top_factors': factors_for_json,
            }
            event_result['fights'].append(fight)

        results.append(event_result)

    return results


def generate_backtest_predictions(
    pipeline: PredictionPipeline,
    events: List[Dict]
) -> List[Dict]:
    """
    Generate backtest predictions for completed events.

    Returns:
        List of event dicts with predictions and actual results
    """
    results = []

    for event in events:
        logger.info(f"Backtesting: {event['name']}")

        # Parse event date to use for backtest
        try:
            # Try multiple date formats
            for fmt in ['%B %d, %Y', '%Y-%m-%d']:
                try:
                    event_dt = datetime.strptime(event['date'], fmt)
                    break
                except ValueError:
                    continue
            else:
                logger.warning(f"Could not parse date for {event['name']}: {event['date']}")
                continue
        except Exception as e:
            logger.warning(f"Date parse error for {event['name']}: {e}")
            continue

        # Run backtest
        try:
            predictions = pipeline.backtest_event(event_dt.strftime('%Y-%m-%d'))
        except Exception as e:
            logger.error(f"Error backtesting {event['name']}: {e}")
            continue

        if not predictions:
            continue

        # Calculate accuracy
        correct = sum(1 for p in predictions if p.get('correct', False))
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0

        # Format event data
        event_result = {
            'name': event['name'],
            'slug': slugify(event['name']),
            'date': event['date'],
            'location': event.get('location', ''),
            'status': 'completed',
            'fight_count': total,
            'correct': correct,
            'accuracy': round(accuracy, 3),
            'accuracy_pct': f"{accuracy * 100:.1f}",
            'fights': []
        }

        for pred in predictions:
            # Store top_factors for tornado chart display
            # Convert from list of tuples to list of dicts for JSON serialization
            top_factors = pred.get('top_factors', [])
            factors_for_json = [
                {'feature': f[0], 'value': f[1]}
                for f in top_factors if isinstance(f, (list, tuple))
            ]

            fight = {
                'fighter_1': pred['fighter1'],
                'fighter_2': pred['fighter2'],
                'weight_class': pred.get('weight_class', ''),
                'f1_win_prob': round(pred['f1_win_prob'], 3),
                'f2_win_prob': round(pred['f2_win_prob'], 3),
                'predicted_winner': pred['predicted_winner'],
                'confidence': pred['confidence'],
                'f1_elo': pred.get('f1_elo', 1500),
                'f2_elo': pred.get('f2_elo', 1500),
                'actual_winner': pred.get('actual_winner'),
                'method': pred.get('method'),
                'correct': pred.get('correct', False),
                'top_factors': factors_for_json,
            }
            event_result['fights'].append(fight)

        results.append(event_result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions for all upcoming and recent UFC events'
    )
    parser.add_argument(
        '--completed-limit',
        type=int,
        default=5,
        help='Number of recent completed events to include (default: 5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(EVENTS_JSON_PATH),
        help='Output JSON file path'
    )
    parser.add_argument(
        '--write-ledger',
        action='store_true',
        help='Write predictions to the immutable ledger'
    )
    parser.add_argument(
        '--ledger-only',
        action='store_true',
        help='Only write to ledger (skip events.json)'
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("UFC MULTI-EVENT PREDICTION GENERATOR")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_VERSION}")
    print()

    # Initialize
    pipeline = PredictionPipeline()
    scraper = UFCDataScraper({'rate_limit': 1})

    try:
        # Scrape upcoming events
        upcoming_events = scrape_all_upcoming_events(scraper)

        # Scrape completed events
        completed_events = scrape_recent_completed_events(scraper, args.completed_limit)

        # Generate predictions for upcoming
        upcoming_results = generate_upcoming_predictions(pipeline, upcoming_events)

        # Generate backtest for completed
        completed_results = generate_backtest_predictions(pipeline, completed_events)

        # Combine and save to events.json (unless ledger-only)
        if not args.ledger_only:
            output_data = {
                'generated_at': datetime.now().isoformat(),
                'model_version': MODEL_VERSION,
                'upcoming_count': len(upcoming_results),
                'completed_count': len(completed_results),
                'events': upcoming_results + completed_results
            }

            # Ensure output directory exists
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)

        # Write to ledger if requested
        if args.write_ledger or args.ledger_only:
            print()
            print("-" * 70)
            print("Writing predictions to ledger...")
            print("-" * 70)

            # Model description based on version
            if MODEL_VERSION == 'phase8':
                model_desc = "Stacking Ensemble (HGB + XGB + LR), 51 features, Elo + Platt calibration"
            else:
                model_desc = "HistGradientBoosting, 145 features"

            # Write upcoming events as unlocked entries
            for event in upcoming_results:
                # Parse date to YYYY-MM-DD format
                try:
                    event_dt = datetime.strptime(event['date'], '%B %d, %Y')
                    date_str = event_dt.strftime('%Y-%m-%d')
                except ValueError:
                    date_str = event['date']

                success = record_predictions(
                    event_id=event['slug'],
                    event_name=event['name'],
                    event_date=date_str,
                    location=event.get('location', ''),
                    fights=event['fights'],
                    model_version=f"{MODEL_VERSION}_ensemble_v1" if MODEL_VERSION == 'phase8' else f"{MODEL_VERSION}_v1",
                    model_description=f"{model_desc} (live prediction)",
                    prediction_type='live'
                )
                if success:
                    print(f"  Recorded: {event['name']} ({len(event['fights'])} fights)")

            # Write completed events as locked entries (backtests)
            for event in completed_results:
                # Parse date to YYYY-MM-DD format
                try:
                    event_dt = datetime.strptime(event['date'], '%B %d, %Y')
                    date_str = event_dt.strftime('%Y-%m-%d')
                except ValueError:
                    date_str = event['date']

                success = record_predictions(
                    event_id=event['slug'],
                    event_name=event['name'],
                    event_date=date_str,
                    location=event.get('location', ''),
                    fights=event['fights'],
                    model_version=f"{MODEL_VERSION}_ensemble_v1" if MODEL_VERSION == 'phase8' else f"{MODEL_VERSION}_v1",
                    model_description=f"{model_desc} (backtest)",
                    prediction_type='backtest'
                )
                if success:
                    print(f"  Recorded: {event['name']} ({len(event['fights'])} fights) - backtest")

            # Show ledger stats
            stats = get_ledger_stats()
            print()
            print(f"Ledger: {stats['locked_entries']} locked, {stats['unlocked_entries']} pending")

        print()
        print("=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Upcoming events: {len(upcoming_results)}")
        print(f"Completed events: {len(completed_results)}")
        if not args.ledger_only:
            print(f"Output: {args.output}")
        if args.write_ledger or args.ledger_only:
            print(f"Ledger: data/ledger/prediction_ledger.json")

        # Show accuracy summary for completed events
        if completed_results:
            total_correct = sum(e['correct'] for e in completed_results)
            total_fights = sum(e['fight_count'] for e in completed_results)
            overall_acc = total_correct / total_fights if total_fights > 0 else 0
            print(f"Overall backtest accuracy: {total_correct}/{total_fights} ({overall_acc:.1%})")

        print()

    finally:
        pipeline.close()
        scraper.close()


if __name__ == '__main__':
    main()
