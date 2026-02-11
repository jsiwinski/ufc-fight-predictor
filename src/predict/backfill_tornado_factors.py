#!/usr/bin/env python3
"""
Backfill tornado plot factors for all existing ledger entries.

This script:
1. Loads the prediction ledger
2. For each event entry, re-generates top_factors using the prediction pipeline
3. Updates each fight with top_factors (preserves all other fields)
4. Saves the updated ledger

Usage:
    python src/predict/backfill_tornado_factors.py
    python src/predict/backfill_tornado_factors.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.predict.serve import PredictionPipeline, get_top_factors, build_matchup_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
LEDGER_PATH = Path('data/ledger/prediction_ledger.json')


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


def compute_fight_factors(
    pipeline: PredictionPipeline,
    fighter1: str,
    fighter2: str,
    weight_class: str,
    event_name: str,
    event_date: datetime
) -> List[Dict]:
    """
    Compute top factors for a single fight.

    Args:
        pipeline: PredictionPipeline instance
        fighter1: Fighter 1 name
        fighter2: Fighter 2 name
        weight_class: Weight class
        event_name: Event name
        event_date: Event date for temporal feature lookup

    Returns:
        List of factor dicts [{'feature': ..., 'value': ...}, ...]
    """
    try:
        # Get fighter features using the event date for temporal lookup
        f1_features, _, _ = pipeline.fighter_features.get_fighter_features(
            fighter1, event_date
        )
        f2_features, _, _ = pipeline.fighter_features.get_fighter_features(
            fighter2, event_date
        )

        # Build feature vector
        feature_vector = build_matchup_features(
            f1_features, f2_features,
            weight_class, event_date, event_name,
            pipeline.feature_names
        )

        # Get top factors
        top_factors = get_top_factors(feature_vector, pipeline.feature_names, top_n=5)

        # Convert to JSON-serializable format (ensure float conversion)
        return [
            {'feature': f[0], 'value': round(float(f[1]), 4)}
            for f in top_factors
        ]

    except Exception as e:
        logger.warning(f"Error computing factors for {fighter1} vs {fighter2}: {e}")
        return []


def backfill_ledger(dry_run: bool = False, force: bool = False) -> int:
    """
    Backfill top_factors for all ledger entries.

    Args:
        dry_run: If True, show what would be done without saving
        force: If True, re-compute factors even if already present

    Returns:
        Number of fights updated
    """
    ledger = load_ledger()
    entries = ledger.get('entries', [])

    if not entries:
        logger.info("No entries in ledger")
        return 0

    # Initialize pipeline
    logger.info("Loading prediction pipeline...")
    pipeline = PredictionPipeline()

    updated_fights = 0
    updated_events = 0

    for entry in entries:
        event_name = entry.get('event_name', 'Unknown')
        event_date_str = entry.get('event_date', '')
        fights = entry.get('fights', [])

        if not fights:
            continue

        # Parse event date
        try:
            event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
        except ValueError:
            logger.warning(f"Could not parse date for {event_name}: {event_date_str}")
            event_date = datetime.now()

        logger.info(f"Processing: {event_name} ({len(fights)} fights)")

        event_updated = False

        for fight in fights:
            fighter1 = fight.get('fighter_1', '')
            fighter2 = fight.get('fighter_2', '')
            weight_class = fight.get('weight_class', '')

            if not fighter1 or not fighter2:
                continue

            # Skip if already has factors (unless force mode)
            existing_factors = fight.get('top_factors', [])
            if existing_factors and len(existing_factors) > 0 and not force:
                logger.debug(f"  Skipping {fighter1} vs {fighter2} - already has factors")
                continue

            # Compute factors
            factors = compute_fight_factors(
                pipeline, fighter1, fighter2,
                weight_class, event_name, event_date
            )

            if factors:
                if not dry_run:
                    fight['top_factors'] = factors
                updated_fights += 1
                event_updated = True
                logger.info(f"  Updated: {fighter1} vs {fighter2} ({len(factors)} factors)")
            else:
                logger.warning(f"  No factors for: {fighter1} vs {fighter2}")

        if event_updated:
            updated_events += 1

    # Save updated ledger
    if not dry_run and updated_fights > 0:
        save_ledger(ledger)

    pipeline.close()

    return updated_fights


def main():
    parser = argparse.ArgumentParser(
        description='Backfill tornado plot factors for all ledger entries'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without saving'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-compute factors even if already present'
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("TORNADO FACTORS BACKFILL")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.dry_run:
        print("DRY RUN - No changes will be saved")
        print()

    if args.force:
        print("FORCE MODE - Re-computing all factors")
        print()

    updated = backfill_ledger(dry_run=args.dry_run, force=args.force)

    print()
    print("-" * 60)
    print(f"Updated {updated} fights with tornado factors")
    print("-" * 60)
    print()


if __name__ == '__main__':
    main()
