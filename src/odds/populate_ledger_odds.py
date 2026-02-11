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
    python src/odds/populate_ledger_odds.py --local-only # Re-match using existing data (no API call)
    python src/odds/populate_ledger_odds.py --local-only --apply --force  # Full re-match with save

Environment:
    ODDS_API_KEY: API key for The Odds API (https://the-odds-api.com)

Odds are fetched and attached to ledger entries during prediction generation.
To refresh odds independently: python src/odds/fetch_odds.py
Odds require ODDS_API_KEY environment variable to be set (unless using --local-only).
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
    load_upcoming_odds,
    get_known_fighters_from_ledger,
)

# Paths
LEDGER_PATH = PROJECT_ROOT / 'data' / 'ledger' / 'prediction_ledger.json'


def match_fight_to_odds(
    fighter1: str,
    fighter2: str,
    odds_fights: List[Dict],
) -> Optional[Tuple[Dict, bool]]:
    """
    Find matching odds data for a fight using pair-based scoring.

    Uses pair-based matching that:
    - Normalizes names (handles accents, hyphens, transliterations)
    - Tries both fighter orderings (straight and flipped)
    - Uses fuzzy matching with a combined pair score threshold

    Args:
        fighter1: First fighter name from ledger
        fighter2: Second fighter name from ledger
        odds_fights: List of fight odds from the scraper

    Returns:
        Tuple of (odds dict, is_flipped) or None if no match
        is_flipped=True means fighter order is reversed in odds vs ledger
    """
    l1_norm = normalize_name(fighter1)
    l2_norm = normalize_name(fighter2)

    best_match = None
    best_score = 0.0
    best_flipped = False

    for odds_fight in odds_fights:
        # Skip fights without actual odds data
        odds = odds_fight.get('odds', {})
        if not odds or (not odds.get('draftkings') and not odds.get('fanduel')):
            continue

        o1 = odds_fight.get('fighter_1_canonical') or odds_fight.get('fighter_1', '')
        o2 = odds_fight.get('fighter_2_canonical') or odds_fight.get('fighter_2', '')
        o1_norm = normalize_name(o1)
        o2_norm = normalize_name(o2)

        # Try straight matching (ledger F1 -> odds F1, ledger F2 -> odds F2)
        score_straight = fuzzy_match_score(l1_norm, o1_norm) + fuzzy_match_score(l2_norm, o2_norm)

        # Try flipped matching (ledger F1 -> odds F2, ledger F2 -> odds F1)
        score_flipped = fuzzy_match_score(l1_norm, o2_norm) + fuzzy_match_score(l2_norm, o1_norm)

        # Take the better orientation
        if score_straight >= score_flipped:
            score = score_straight
            flipped = False
        else:
            score = score_flipped
            flipped = True

        if score > best_score:
            best_score = score
            best_match = odds_fight
            best_flipped = flipped

    # Threshold: 1.7 out of 2.0 means both names must average ~85% similarity
    if best_score >= 1.7 and best_match:
        return best_match.get('odds', {}), best_flipped

    return None


def format_odds_for_ledger(raw_odds: Dict, flipped: bool = False) -> Dict:
    """
    Format raw scraper odds for ledger storage.

    Handles fighter order flipping: if the odds API has the fighters in
    opposite order compared to the ledger, swap the moneylines and probabilities.

    Args:
        raw_odds: Raw odds dict from scraper
        flipped: If True, swap F1/F2 values to match ledger order

    Returns:
        Formatted odds dict for ledger
    """
    result = {}

    # DraftKings
    dk = raw_odds.get('draftkings', {})
    if dk:
        if flipped:
            result['draftkings'] = {
                'f1_ml': dk.get('f2_moneyline'),
                'f2_ml': dk.get('f1_moneyline'),
            }
        else:
            result['draftkings'] = {
                'f1_ml': dk.get('f1_moneyline'),
                'f2_ml': dk.get('f2_moneyline'),
            }

    # FanDuel
    fd = raw_odds.get('fanduel', {})
    if fd:
        if flipped:
            result['fanduel'] = {
                'f1_ml': fd.get('f2_moneyline'),
                'f2_ml': fd.get('f1_moneyline'),
            }
        else:
            result['fanduel'] = {
                'f1_ml': fd.get('f1_moneyline'),
                'f2_ml': fd.get('f2_moneyline'),
            }

    # Consensus
    consensus = raw_odds.get('consensus', {})
    if consensus:
        if flipped:
            result['consensus_f1_fair'] = consensus.get('f2_fair')
            result['consensus_f2_fair'] = consensus.get('f1_fair')
        else:
            result['consensus_f1_fair'] = consensus.get('f1_fair')
            result['consensus_f2_fair'] = consensus.get('f2_fair')

    result['scraped_at'] = datetime.utcnow().isoformat() + 'Z'

    return result


def populate_ledger_odds(
    apply: bool = False,
    force: bool = False,
    api_key: Optional[str] = None,
    local_only: bool = False
) -> Tuple[int, int, int]:
    """
    Populate odds for unlocked ledger entries.

    Args:
        apply: If True, save changes to ledger. If False, dry run only.
        force: If True, re-fetch odds even if already present.
        api_key: Optional API key (else reads from env)
        local_only: If True, use existing scraped data instead of calling API

    Returns:
        Tuple of (events_updated, fights_matched, fights_unmatched)
    """
    # Load ledger
    ledger = load_ledger()
    if not ledger:
        print("ERROR: Could not load ledger")
        return 0, 0, 0

    if local_only:
        # Use existing scraped odds from disk
        print("Loading odds from local file (no API call)...")
        odds_data = load_upcoming_odds()
        if not odds_data:
            print("ERROR: No local odds data found at data/odds/upcoming_odds.json")
            print("Run 'python src/odds/fetch_odds.py' first to scrape odds data.")
            return 0, 0, 0

        odds_fights = odds_data.get('fights', [])
        print(f"Loaded odds for {len(odds_fights)} fights (scraped: {odds_data.get('scraped_at', 'unknown')})")
    else:
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
            print("Or use --local-only to re-match using existing scraped data.")
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
            match_result = match_fight_to_odds(f1, f2, odds_fights)

            if match_result:
                raw_odds, is_flipped = match_result
                formatted = format_odds_for_ledger(raw_odds, flipped=is_flipped)

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
    parser.add_argument(
        '--local-only',
        action='store_true',
        help='Re-match using existing scraped data (no API call)'
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("POPULATE LEDGER ODDS")
    print("=" * 60)
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Force re-fetch: {args.force}")
    print(f"Local only: {args.local_only}")
    print()

    events, matched, unmatched = populate_ledger_odds(
        apply=args.apply,
        force=args.force,
        api_key=args.api_key,
        local_only=args.local_only
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
