#!/usr/bin/env python3
"""
Populate Vegas Odds in Prediction Ledger.

Fetches current odds from The Odds API and attaches them to unlocked (upcoming)
ledger entries. For locked (completed) entries, odds must be captured before
the event happens.

Usage:
    python src/odds/populate_ledger_odds.py              # Dry run (show what would change)
    python src/odds/populate_ledger_odds.py --apply      # Apply changes to ledger
    python src/odds/populate_ledger_odds.py --force      # Re-fetch even if odds exist

Environment:
    ODDS_API_KEY: API key for The Odds API (https://the-odds-api.com)

Odds are fetched and attached to ledger entries during prediction generation.
To refresh odds independently: python src/odds/fetch_odds.py
Odds require ODDS_API_KEY environment variable to be set.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.odds.scraper import (
    UFCOddsScraper,
    normalize_name,
    fuzzy_match_score,
    load_ledger,
    get_known_fighters_from_ledger,
)

# Paths
LEDGER_PATH = PROJECT_ROOT / 'data' / 'ledger' / 'prediction_ledger.json'


def match_fight_to_odds(
    fighter1: str,
    fighter2: str,
    odds_fights: List[Dict],
) -> Optional[Dict]:
    """
    Find matching odds data for a fight.

    Args:
        fighter1: First fighter name
        fighter2: Second fighter name
        odds_fights: List of fight odds from the scraper

    Returns:
        Matched odds dict or None
    """
    fight_names = [normalize_name(fighter1), normalize_name(fighter2)]

    for odds_fight in odds_fights:
        f1 = odds_fight.get('fighter_1_canonical') or odds_fight.get('fighter_1', '')
        f2 = odds_fight.get('fighter_2_canonical') or odds_fight.get('fighter_2', '')
        odds_names = [normalize_name(f1), normalize_name(f2)]

        # Check exact match
        if set(fight_names) == set(odds_names):
            return odds_fight.get('odds', {})

        # Try fuzzy match
        matches = 0
        for fn in fight_names:
            for on in odds_names:
                if fuzzy_match_score(fn, on) > 0.85:
                    matches += 1
                    break

        if matches == 2:
            return odds_fight.get('odds', {})

    return None


def format_odds_for_ledger(raw_odds: Dict) -> Dict:
    """
    Format raw scraper odds for ledger storage.

    Args:
        raw_odds: Raw odds dict from scraper

    Returns:
        Formatted odds dict for ledger
    """
    result = {}

    # DraftKings
    dk = raw_odds.get('draftkings', {})
    if dk:
        result['draftkings'] = {
            'f1_ml': dk.get('f1_moneyline'),
            'f2_ml': dk.get('f2_moneyline'),
        }

    # FanDuel
    fd = raw_odds.get('fanduel', {})
    if fd:
        result['fanduel'] = {
            'f1_ml': fd.get('f1_moneyline'),
            'f2_ml': fd.get('f2_moneyline'),
        }

    # Consensus
    consensus = raw_odds.get('consensus', {})
    if consensus:
        result['consensus_f1_fair'] = consensus.get('f1_fair')
        result['consensus_f2_fair'] = consensus.get('f2_fair')

    result['scraped_at'] = datetime.utcnow().isoformat() + 'Z'

    return result


def populate_ledger_odds(
    apply: bool = False,
    force: bool = False,
    api_key: Optional[str] = None
) -> Tuple[int, int, int]:
    """
    Populate odds for unlocked ledger entries.

    Args:
        apply: If True, save changes to ledger. If False, dry run only.
        force: If True, re-fetch odds even if already present.
        api_key: Optional API key (else reads from env)

    Returns:
        Tuple of (events_updated, fights_matched, fights_unmatched)
    """
    # Load ledger
    ledger = load_ledger()
    if not ledger:
        print("ERROR: Could not load ledger")
        return 0, 0, 0

    # Check for API key
    key = api_key or os.getenv('ODDS_API_KEY')
    if not key:
        print()
        print("ERROR: No API key provided.")
        print()
        print("To use this tool, you need an API key from The Odds API.")
        print()
        print("1. Sign up at: https://the-odds-api.com")
        print("2. Get your free API key (500 requests/month)")
        print("3. Set the key: export ODDS_API_KEY='your-key-here'")
        print()
        return 0, 0, 0

    # Fetch current odds
    print("Fetching odds from The Odds API...")
    with UFCOddsScraper(api_key=key) as scraper:
        odds_fights = scraper.scrape_upcoming_odds()

        if not odds_fights:
            print("No odds data available from API.")
            return 0, 0, 0

        # Match fighters to canonical names
        known_fighters = get_known_fighters_from_ledger()
        if known_fighters:
            odds_fights, _, _ = scraper.match_fighters_to_canonical(
                odds_fights, known_fighters
            )

        print(f"Fetched odds for {len(odds_fights)} fights")

        # Also save to upcoming_odds.json
        scraper.save_odds(odds_fights)
        scraper.append_to_history(odds_fights)

    # Process ledger entries
    events_updated = 0
    fights_matched = 0
    fights_unmatched = 0

    for entry in ledger.get('entries', []):
        # Only process unlocked events
        if entry.get('locked', False):
            continue

        event_name = entry.get('event_name', '')
        event_updated = False

        print(f"\nProcessing: {event_name}")

        for fight in entry.get('fights', []):
            f1 = fight.get('fighter_1', '')
            f2 = fight.get('fighter_2', '')

            # Skip if odds already exist (unless force)
            if fight.get('odds') and not force:
                print(f"  {f1} vs {f2}: odds already exist (skipping)")
                continue

            # Find matching odds
            raw_odds = match_fight_to_odds(f1, f2, odds_fights)

            if raw_odds:
                formatted = format_odds_for_ledger(raw_odds)

                # Calculate edge
                f1_prob = fight.get('f1_win_prob', 0.5)
                f2_prob = fight.get('f2_win_prob', 0.5)
                consensus_f1 = formatted.get('consensus_f1_fair', 0.5)

                if f1_prob >= f2_prob:
                    edge = f1_prob - consensus_f1
                else:
                    edge = f2_prob - (1 - consensus_f1)

                formatted['edge_f1'] = round(edge, 4)

                fight['odds'] = formatted
                event_updated = True
                fights_matched += 1

                dk = formatted.get('draftkings', {})
                fd = formatted.get('fanduel', {})
                dk_str = f"DK: {dk.get('f1_ml')}/{dk.get('f2_ml')}" if dk else "DK: N/A"
                fd_str = f"FD: {fd.get('f1_ml')}/{fd.get('f2_ml')}" if fd else "FD: N/A"
                print(f"  {f1} vs {f2}: MATCHED - {dk_str}, {fd_str}")
            else:
                fights_unmatched += 1
                print(f"  {f1} vs {f2}: NO MATCH")

        if event_updated:
            events_updated += 1

    # Save ledger if applying
    if apply and events_updated > 0:
        print()
        print("Saving updated ledger...")
        with open(LEDGER_PATH, 'w') as f:
            json.dump(ledger, f, indent=2)
        print(f"Saved to {LEDGER_PATH}")
    elif not apply and events_updated > 0:
        print()
        print("DRY RUN - no changes saved. Use --apply to save changes.")

    return events_updated, fights_matched, fights_unmatched


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Populate Vegas odds in prediction ledger'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply changes to ledger (default is dry run)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-fetch odds even if already present'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='The Odds API key (or set ODDS_API_KEY env var)'
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("POPULATE LEDGER ODDS")
    print("=" * 60)
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Force re-fetch: {args.force}")
    print()

    events, matched, unmatched = populate_ledger_odds(
        apply=args.apply,
        force=args.force,
        api_key=args.api_key
    )

    print()
    print("-" * 60)
    print(f"Events updated: {events}")
    print(f"Fights matched: {matched}")
    print(f"Fights unmatched: {unmatched}")
    print("-" * 60)
    print()


if __name__ == '__main__':
    main()
