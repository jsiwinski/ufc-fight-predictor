#!/usr/bin/env python3
"""
Lock completed events in the prediction ledger.

This script:
1. Loads the prediction ledger
2. Finds all unlocked (pending) entries
3. Checks if event date has passed
4. Scrapes actual results from ufcstats.com
5. Fills in actual_winner, actual_method, actual_round
6. Computes accuracy (correct/total)
7. Sets locked = True, locked_at = now()
8. Saves the updated ledger

Usage:
    python src/predict/lock_results.py
    python src/predict/lock_results.py --dry-run        # Show what would be locked without saving
    python src/predict/lock_results.py --force EVENT_ID # Force lock specific event
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.etl.scraper import UFCDataScraper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
LEDGER_PATH = Path('data/ledger/prediction_ledger.json')
MODEL_REGISTRY_PATH = Path('data/models/model_registry.json')


def normalize_name(name: str) -> str:
    """Normalize fighter name for matching."""
    if not name:
        return ""
    name = ' '.join(name.lower().strip().split())
    name = name.replace("'", "").replace("-", " ").replace(".", "")
    return name


def fuzzy_match_score(name1: str, name2: str) -> float:
    """Calculate fuzzy match score between two names."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    return SequenceMatcher(None, n1, n2).ratio()


def find_fight_match(
    f1_name: str,
    f2_name: str,
    results: List[Dict],
    threshold: float = 0.85
) -> Optional[Dict]:
    """
    Find matching fight result from scraped results.

    Args:
        f1_name: Fighter 1 name from ledger
        f2_name: Fighter 2 name from ledger
        results: List of scraped fight results
        threshold: Minimum match score

    Returns:
        Matching result dict or None
    """
    best_match = None
    best_score = 0.0

    for result in results:
        r_f1 = result.get('fighter1_name', '')
        r_f2 = result.get('fighter2_name', '')

        # Try matching in both orders
        score_1 = (fuzzy_match_score(f1_name, r_f1) + fuzzy_match_score(f2_name, r_f2)) / 2
        score_2 = (fuzzy_match_score(f1_name, r_f2) + fuzzy_match_score(f2_name, r_f1)) / 2

        score = max(score_1, score_2)

        if score > best_score:
            best_score = score
            best_match = result
            # Store whether fighters were swapped
            best_match['_swapped'] = score_2 > score_1

    if best_score >= threshold:
        return best_match

    return None


def load_ledger() -> Dict:
    """Load the prediction ledger."""
    if not LEDGER_PATH.exists():
        logger.error(f"Ledger not found at {LEDGER_PATH}")
        return {'ledger_version': '1.0', 'entries': []}

    with open(LEDGER_PATH, 'r') as f:
        return json.load(f)


def save_ledger(ledger: Dict) -> None:
    """Save the prediction ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, 'w') as f:
        json.dump(ledger, f, indent=2, default=str)
    logger.info(f"Saved ledger to {LEDGER_PATH}")


def scrape_event_results(scraper: UFCDataScraper, event_date: str) -> List[Dict]:
    """
    Scrape fight results for a specific event date.

    Args:
        scraper: UFCDataScraper instance
        event_date: Event date string (YYYY-MM-DD)

    Returns:
        List of fight result dicts
    """
    logger.info(f"Scraping results for event on {event_date}")

    # Get list of completed events
    events_url = f"{scraper.base_url}/statistics/events/completed?page=all"

    try:
        from bs4 import BeautifulSoup
        response = scraper.session.get(events_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        logger.error(f"Failed to fetch completed events: {e}")
        return []

    # Find event matching the date
    table = soup.find('table', {'class': 'b-statistics__table-events'})
    if not table:
        logger.warning("Could not find events table")
        return []

    rows = table.find('tbody').find_all('tr', {'class': 'b-statistics__table-row'})

    target_date = datetime.strptime(event_date, '%Y-%m-%d')

    for row in rows:
        try:
            cells = row.find_all('td', {'class': 'b-statistics__table-col'})
            if len(cells) < 2:
                continue

            event_link = cells[0].find('a', {'class': 'b-link'})
            if not event_link:
                continue

            date_span = cells[0].find('span', {'class': 'b-statistics__date'})
            if not date_span:
                continue

            date_str = date_span.text.strip()

            # Parse date
            try:
                # Format: "January 31, 2026"
                parsed_date = datetime.strptime(date_str, '%B %d, %Y')
            except ValueError:
                continue

            # Check if date matches (within 2 days to handle timezone issues)
            if abs((parsed_date - target_date).days) <= 2:
                event_url = event_link['href']
                event_name = event_link.text.strip()

                logger.info(f"Found matching event: {event_name}")

                # Get fights for this event
                event_info = {
                    'name': event_name,
                    'date': date_str,
                    'url': event_url,
                    'location': cells[1].text.strip() if len(cells) > 1 else ''
                }

                return scraper._get_event_fights(event_info)

        except Exception as e:
            logger.warning(f"Error parsing event row: {e}")
            continue

    logger.warning(f"No event found matching date {event_date}")
    return []


def lock_event(
    entry: Dict,
    results: List[Dict],
    dry_run: bool = False
) -> Tuple[bool, Dict]:
    """
    Lock a single event entry with scraped results.

    Args:
        entry: Ledger entry to lock
        results: Scraped fight results
        dry_run: If True, don't modify entry

    Returns:
        Tuple of (success, updated_entry)
    """
    if entry.get('locked', False):
        logger.info(f"Event {entry['event_id']} is already locked")
        return False, entry

    event_name = entry.get('event_name', 'Unknown')
    fights = entry.get('fights', [])

    if not fights:
        logger.warning(f"No fights in entry for {event_name}")
        return False, entry

    if not results:
        logger.warning(f"No results available for {event_name}")
        return False, entry

    matched_count = 0
    correct_count = 0
    total_counted = 0  # Excludes no contests

    for fight in fights:
        f1 = fight.get('fighter_1', '')
        f2 = fight.get('fighter_2', '')

        # Find matching result
        result = find_fight_match(f1, f2, results)

        if result is None:
            logger.warning(f"No result found for {f1} vs {f2}")
            fight['actual_winner'] = None
            fight['actual_method'] = None
            fight['actual_round'] = None
            fight['correct'] = None
            continue

        matched_count += 1

        # Get winner name
        winner = result.get('winner', '')
        method = result.get('method', '')
        round_num = result.get('round', '')

        # Handle draw/no contest
        if winner.lower() in ['draw', 'no contest', 'nc', 'unknown']:
            fight['actual_winner'] = winner
            fight['actual_method'] = method
            fight['actual_round'] = int(round_num) if round_num else None
            fight['correct'] = None  # Exclude from accuracy
            continue

        # Determine actual winner name (handle swapped order)
        if result.get('_swapped', False):
            # Fighters were in opposite order
            if winner == result.get('fighter1_name'):
                actual_winner = f2
            else:
                actual_winner = f1
        else:
            if winner == result.get('fighter1_name'):
                actual_winner = f1
            else:
                actual_winner = f2

        fight['actual_winner'] = actual_winner
        fight['actual_method'] = method
        fight['actual_round'] = int(round_num) if round_num else None

        # Compute correct
        predicted = fight.get('predicted_winner', '')
        # Normalize names for comparison
        is_correct = normalize_name(predicted) == normalize_name(actual_winner)
        fight['correct'] = is_correct

        total_counted += 1
        if is_correct:
            correct_count += 1

    if matched_count == 0:
        logger.error(f"Could not match any fights for {event_name}")
        return False, entry

    logger.info(f"Matched {matched_count}/{len(fights)} fights for {event_name}")

    # Update entry with accuracy
    if not dry_run:
        entry['overall_correct'] = correct_count
        entry['overall_total'] = total_counted
        entry['overall_accuracy'] = round(correct_count / total_counted, 3) if total_counted > 0 else 0
        entry['locked'] = True
        entry['locked_at'] = datetime.utcnow().isoformat() + 'Z'

    logger.info(f"Accuracy for {event_name}: {correct_count}/{total_counted} ({correct_count/total_counted*100:.1f}%)" if total_counted > 0 else "No results to score")

    return True, entry


def lock_completed_events(dry_run: bool = False, force_event_id: Optional[str] = None) -> int:
    """
    Lock all completed events in the ledger.

    Args:
        dry_run: If True, show what would be done without saving
        force_event_id: Force lock specific event even if conditions not met

    Returns:
        Number of events locked
    """
    ledger = load_ledger()
    entries = ledger.get('entries', [])

    if not entries:
        logger.info("No entries in ledger")
        return 0

    # Find unlocked entries
    unlocked = [e for e in entries if not e.get('locked', False)]

    if force_event_id:
        unlocked = [e for e in entries if e.get('event_id') == force_event_id]
        if not unlocked:
            logger.error(f"Event {force_event_id} not found in ledger")
            return 0

    if not unlocked:
        logger.info("No unlocked entries to process")
        return 0

    logger.info(f"Found {len(unlocked)} unlocked entries")

    # Initialize scraper
    scraper = UFCDataScraper({'rate_limit': 2})
    locked_count = 0

    try:
        for entry in unlocked:
            event_id = entry.get('event_id', 'unknown')
            event_date = entry.get('event_date', '')
            event_name = entry.get('event_name', 'Unknown')

            logger.info(f"\nProcessing: {event_name} ({event_date})")

            # Check if event date has passed
            if event_date:
                try:
                    event_dt = datetime.strptime(event_date, '%Y-%m-%d')
                    if event_dt > datetime.now() and not force_event_id:
                        logger.info(f"Skipping {event_name} - event is in the future")
                        continue
                except ValueError:
                    logger.warning(f"Could not parse date {event_date}")

            # Scrape results
            results = scrape_event_results(scraper, event_date)

            if not results:
                logger.warning(f"No results found for {event_name}")
                continue

            # Lock the event
            success, updated_entry = lock_event(entry, results, dry_run=dry_run)

            if success:
                locked_count += 1
                # Update entry in place
                idx = entries.index(entry)
                entries[idx] = updated_entry

    finally:
        scraper.close()

    # Save updated ledger
    if not dry_run and locked_count > 0:
        ledger['entries'] = entries
        save_ledger(ledger)

    return locked_count


def record_predictions(
    event_id: str,
    event_name: str,
    event_date: str,
    location: str,
    fights: List[Dict],
    model_version: str,
    model_description: str,
    prediction_type: str = 'live'
) -> bool:
    """
    Record predictions for an upcoming event to the ledger.

    Creates an UNLOCKED entry that will be locked when results come in.
    If entry already exists and is unlocked, updates it.
    If entry is locked, does nothing.

    Args:
        event_id: URL-safe slug for event
        event_name: Full event name
        event_date: Date string (YYYY-MM-DD)
        location: Event location
        fights: List of fight prediction dicts
        model_version: Model version identifier
        model_description: Human-readable model description
        prediction_type: 'live' or 'backtest'

    Returns:
        True if entry was created/updated, False if already locked
    """
    ledger = load_ledger()
    entries = ledger.get('entries', [])

    # Check if entry exists
    existing_idx = None
    for i, entry in enumerate(entries):
        if entry.get('event_id') == event_id:
            if entry.get('locked', False):
                logger.info(f"Entry for {event_id} is already locked - not updating")
                return False
            existing_idx = i
            break

    # Format fights for ledger
    ledger_fights = []
    for i, fight in enumerate(fights):
        ledger_fight = {
            'fighter_1': fight.get('fighter_1', fight.get('fighter1', '')),
            'fighter_2': fight.get('fighter_2', fight.get('fighter2', '')),
            'weight_class': fight.get('weight_class', ''),
            'is_main_event': i == 0,
            'f1_win_prob': round(fight.get('f1_win_prob', 0.5), 3),
            'f2_win_prob': round(fight.get('f2_win_prob', 0.5), 3),
            'predicted_winner': fight.get('predicted_winner', ''),
            'confidence': fight.get('confidence', 'Low'),
            'f1_elo': fight.get('f1_elo', 1500),
            'f2_elo': fight.get('f2_elo', 1500),
            # Top factors for tornado chart (already in dict format from generate_all_events)
            'top_factors': fight.get('top_factors', []),
            # Results to be filled in at lock time
            'actual_winner': None,
            'actual_method': None,
            'actual_round': None,
            'correct': None
        }
        ledger_fights.append(ledger_fight)

    entry = {
        'event_id': event_id,
        'event_name': event_name,
        'event_date': event_date,
        'location': location,
        'locked': False,
        'locked_at': None,
        'prediction_type': prediction_type,
        'model_version': model_version,
        'model_description': model_description,
        'overall_correct': None,
        'overall_total': None,
        'overall_accuracy': None,
        'fights': ledger_fights
    }

    if existing_idx is not None:
        entries[existing_idx] = entry
        logger.info(f"Updated existing entry for {event_id}")
    else:
        entries.append(entry)
        logger.info(f"Created new entry for {event_id}")

    ledger['entries'] = entries
    save_ledger(ledger)

    return True


def get_ledger_stats() -> Dict:
    """Get summary statistics from the ledger."""
    ledger = load_ledger()
    entries = ledger.get('entries', [])

    locked = [e for e in entries if e.get('locked', False)]
    unlocked = [e for e in entries if not e.get('locked', False)]

    # Separate by prediction type
    live = [e for e in locked if e.get('prediction_type') == 'live']
    backtest = [e for e in locked if e.get('prediction_type') == 'backtest']

    # Calculate overall accuracy
    total_correct = sum(e.get('overall_correct', 0) for e in locked)
    total_fights = sum(e.get('overall_total', 0) for e in locked)

    live_correct = sum(e.get('overall_correct', 0) for e in live)
    live_fights = sum(e.get('overall_total', 0) for e in live)

    backtest_correct = sum(e.get('overall_correct', 0) for e in backtest)
    backtest_fights = sum(e.get('overall_total', 0) for e in backtest)

    return {
        'total_entries': len(entries),
        'locked_entries': len(locked),
        'unlocked_entries': len(unlocked),
        'total_fights_predicted': total_fights,
        'total_correct': total_correct,
        'overall_accuracy': total_correct / total_fights if total_fights > 0 else 0,
        'live_fights': live_fights,
        'live_correct': live_correct,
        'live_accuracy': live_correct / live_fights if live_fights > 0 else 0,
        'backtest_fights': backtest_fights,
        'backtest_correct': backtest_correct,
        'backtest_accuracy': backtest_correct / backtest_fights if backtest_fights > 0 else 0,
    }


def print_ledger_summary():
    """Print a summary of the ledger."""
    stats = get_ledger_stats()

    print()
    print("=" * 60)
    print("PREDICTION LEDGER SUMMARY")
    print("=" * 60)
    print()
    print(f"Total Events:       {stats['total_entries']}")
    print(f"  Locked:           {stats['locked_entries']}")
    print(f"  Pending:          {stats['unlocked_entries']}")
    print()
    print(f"Total Fights:       {stats['total_fights_predicted']}")
    print(f"Overall Accuracy:   {stats['total_correct']}/{stats['total_fights_predicted']} ({stats['overall_accuracy']:.1%})")
    print()

    if stats['live_fights'] > 0:
        print(f"Live Predictions:   {stats['live_correct']}/{stats['live_fights']} ({stats['live_accuracy']:.1%})")

    if stats['backtest_fights'] > 0:
        print(f"Backtests:          {stats['backtest_correct']}/{stats['backtest_fights']} ({stats['backtest_accuracy']:.1%})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Lock completed events in the prediction ledger'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without saving'
    )
    parser.add_argument(
        '--force',
        type=str,
        metavar='EVENT_ID',
        help='Force lock a specific event by ID'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show ledger summary only'
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("UFC PREDICTION LEDGER - LOCK RESULTS")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.summary:
        print_ledger_summary()
        return

    if args.dry_run:
        print("DRY RUN - No changes will be saved")
        print()

    # Lock completed events
    locked_count = lock_completed_events(
        dry_run=args.dry_run,
        force_event_id=args.force
    )

    print()
    print("-" * 60)
    print(f"Locked {locked_count} events")
    print("-" * 60)

    # Show updated summary
    print_ledger_summary()


if __name__ == '__main__':
    main()
