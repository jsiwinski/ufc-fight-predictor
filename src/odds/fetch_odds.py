#!/usr/bin/env python3
"""
UFC Moneyline Odds Fetcher CLI.

Scrapes moneyline odds from DraftKings and FanDuel for upcoming UFC fights.

Usage:
    python src/odds/fetch_odds.py                  # Fetch odds only
    python src/odds/fetch_odds.py --compare        # Fetch and compare to model
    python src/odds/fetch_odds.py --no-save        # Don't save to disk

Environment:
    ODDS_API_KEY: API key for The Odds API (https://the-odds-api.com)
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
    load_ledger,
    load_upcoming_odds,
    get_known_fighters_from_ledger,
    normalize_name,
    fuzzy_match_score,
)


def format_moneyline(odds: int) -> str:
    """Format moneyline odds for display."""
    if odds >= 0:
        return f"+{odds}"
    return str(odds)


def print_odds_table(fights: List[Dict]) -> None:
    """Print a formatted table of odds."""
    # Group fights by event date
    events = {}
    for fight in fights:
        event_date = fight.get('event_date', 'Unknown')
        event_name = fight.get('event_name', 'Unknown Event')
        key = (event_date, event_name)
        if key not in events:
            events[key] = []
        events[key].append(fight)

    # Sort events by date
    sorted_events = sorted(events.items(), key=lambda x: x[0][0])

    for (event_date, event_name), event_fights in sorted_events:
        print()
        print(f"{event_name} ({event_date})")
        print("=" * 70)
        print(f"{'Fight':<35} {'DK':>12} {'FD':>12} {'Consensus':>10}")
        print("-" * 70)

        for fight in event_fights:
            f1 = fight['fighter_1']
            f2 = fight['fighter_2']
            fight_name = f"{f1[:15]} vs {f2[:15]}"

            odds = fight.get('odds', {})

            # DraftKings
            dk = odds.get('draftkings', {})
            if dk:
                dk_str = f"{format_moneyline(dk['f1_moneyline'])}/{format_moneyline(dk['f2_moneyline'])}"
            else:
                dk_str = "N/A"

            # FanDuel
            fd = odds.get('fanduel', {})
            if fd:
                fd_str = f"{format_moneyline(fd['f1_moneyline'])}/{format_moneyline(fd['f2_moneyline'])}"
            else:
                fd_str = "N/A"

            # Consensus
            consensus = odds.get('consensus', {})
            if consensus:
                cons_str = f"{consensus['f1_fair']*100:.1f}%/{consensus['f2_fair']*100:.1f}%"
            else:
                cons_str = "N/A"

            print(f"{fight_name:<35} {dk_str:>12} {fd_str:>12} {cons_str:>10}")


def load_model_predictions() -> Dict[str, Dict]:
    """
    Load model predictions from the ledger.

    Returns dict mapping normalized fighter pair -> prediction data
    """
    ledger = load_ledger()
    if not ledger:
        return {}

    predictions = {}

    for entry in ledger.get('entries', []):
        # Only include unlocked (upcoming) events
        if entry.get('locked', False):
            continue

        for fight in entry.get('fights', []):
            f1 = fight.get('fighter_1', '')
            f2 = fight.get('fighter_2', '')

            if not f1 or not f2:
                continue

            # Create normalized key (alphabetical order for consistency)
            names = sorted([normalize_name(f1), normalize_name(f2)])
            key = f"{names[0]}|{names[1]}"

            predictions[key] = {
                'fighter_1': f1,
                'fighter_2': f2,
                'f1_win_prob': fight.get('f1_win_prob', 0.5),
                'f2_win_prob': fight.get('f2_win_prob', 0.5),
                'predicted_winner': fight.get('predicted_winner', ''),
                'confidence': fight.get('confidence', ''),
                'event_name': entry.get('event_name', ''),
                'event_date': entry.get('event_date', ''),
            }

    return predictions


def match_odds_to_prediction(
    fight: Dict,
    predictions: Dict[str, Dict]
) -> Optional[Dict]:
    """
    Find matching model prediction for an odds fight.

    Args:
        fight: Odds fight dict
        predictions: Dict of predictions keyed by normalized fighter pair

    Returns:
        Matched prediction or None
    """
    f1 = fight.get('fighter_1_canonical') or fight.get('fighter_1', '')
    f2 = fight.get('fighter_2_canonical') or fight.get('fighter_2', '')

    # Create normalized key
    names = sorted([normalize_name(f1), normalize_name(f2)])
    key = f"{names[0]}|{names[1]}"

    if key in predictions:
        return predictions[key]

    # Try fuzzy matching
    for pred_key, pred in predictions.items():
        pred_names = pred_key.split('|')
        fight_names = [normalize_name(f1), normalize_name(f2)]

        # Check if both fighters match with fuzzy logic
        matches = 0
        for fn in fight_names:
            for pn in pred_names:
                if fuzzy_match_score(fn, pn) > 0.85:
                    matches += 1
                    break

        if matches == 2:
            return pred

    return None


def print_comparison_table(fights: List[Dict], predictions: Dict[str, Dict]) -> None:
    """Print side-by-side comparison of model vs Vegas odds."""
    print()
    print("=" * 80)
    print("MODEL vs VEGAS COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Fight':<35} {'Model':>8} {'Vegas':>8} {'Edge':>8} {'Value?':>8}")
    print("-" * 80)

    value_fights = []
    matched_count = 0
    unmatched_count = 0

    for fight in fights:
        f1 = fight['fighter_1']
        f2 = fight['fighter_2']
        fight_name = f"{f1[:15]} vs {f2[:15]}"

        # Get consensus odds
        consensus = fight.get('odds', {}).get('consensus', {})
        if not consensus:
            continue

        vegas_f1 = consensus.get('f1_fair', 0.5)

        # Find matching prediction
        pred = match_odds_to_prediction(fight, predictions)

        if pred:
            matched_count += 1

            # Determine which fighter is f1 in the prediction
            pred_f1 = pred['fighter_1']
            pred_f2 = pred['fighter_2']

            # Check if prediction f1 matches odds f1
            if fuzzy_match_score(normalize_name(pred_f1), normalize_name(f1)) > 0.85:
                model_f1 = pred['f1_win_prob']
            elif fuzzy_match_score(normalize_name(pred_f2), normalize_name(f1)) > 0.85:
                model_f1 = pred['f2_win_prob']
            else:
                # Fallback
                model_f1 = pred['f1_win_prob']

            edge = (model_f1 - vegas_f1) * 100  # Edge in percentage points

            # Flag significant edges
            value_flag = ""
            if abs(edge) >= 5:
                value_flag = "<- MODEL DISAGREES" if edge < 0 else "VALUE"
                value_fights.append({
                    'fight': fight_name,
                    'edge': edge,
                    'model': model_f1,
                    'vegas': vegas_f1
                })

            print(f"{fight_name:<35} {model_f1*100:>7.1f}% {vegas_f1*100:>7.1f}% {edge:>+7.1f}pp {value_flag:>8}")
        else:
            unmatched_count += 1
            print(f"{fight_name:<35} {'N/A':>8} {vegas_f1*100:>7.1f}% {'N/A':>8} {'':>8}")

    print("-" * 80)
    print(f"Matched: {matched_count} | Unmatched: {unmatched_count}")
    print()

    # Highlight value bets
    if value_fights:
        print("POTENTIAL VALUE PICKS (|edge| >= 5pp):")
        print("-" * 40)
        for v in sorted(value_fights, key=lambda x: abs(x['edge']), reverse=True):
            direction = "OVER" if v['edge'] > 0 else "UNDER"
            print(f"  {v['fight']}: Model {direction} Vegas by {abs(v['edge']):.1f}pp")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch UFC moneyline odds from DraftKings and FanDuel'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare odds to model predictions'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save odds to disk'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='The Odds API key (or set ODDS_API_KEY env var)'
    )

    args = parser.parse_args()

    # Check for API key
    api_key = args.api_key or os.getenv('ODDS_API_KEY')
    if not api_key:
        print()
        print("ERROR: No API key provided.")
        print()
        print("To use this tool, you need an API key from The Odds API.")
        print()
        print("1. Sign up at: https://the-odds-api.com")
        print("2. Get your free API key (500 requests/month)")
        print("3. Set the key:")
        print("   export ODDS_API_KEY='your-key-here'")
        print()
        print("   Or pass via command line:")
        print("   python src/odds/fetch_odds.py --api-key 'your-key-here'")
        print()
        sys.exit(1)

    print()
    print("=" * 60)
    print("UFC MONEYLINE ODDS SCRAPER")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Scrape odds
    with UFCOddsScraper(api_key=api_key) as scraper:
        fights = scraper.scrape_upcoming_odds()

        if not fights:
            print("No UFC odds found. The API may not have upcoming fights listed yet.")
            print()
            print("This can happen if:")
            print("  - No UFC events are scheduled in the near future")
            print("  - Odds haven't been posted yet for upcoming events")
            print("  - API quota exceeded (check your usage)")
            print()
            sys.exit(0)

        # Match fighter names to canonical names
        known_fighters = get_known_fighters_from_ledger()
        if known_fighters:
            fights, matched, unmatched = scraper.match_fighters_to_canonical(
                fights, known_fighters
            )
            print(f"Fighter matching: {matched} fights matched, {unmatched} fighters unmatched")

        # Print odds table
        print_odds_table(fights)

        # Save odds
        if not args.no_save:
            scraper.save_odds(fights)
            scraper.append_to_history(fights)

        # Summary
        print()
        print("-" * 60)

        # Count bookmaker coverage
        dk_count = sum(1 for f in fights if f.get('odds', {}).get('draftkings'))
        fd_count = sum(1 for f in fights if f.get('odds', {}).get('fanduel'))

        print(f"Scraped {len(fights)} fights")
        print(f"DraftKings: {dk_count}/{len(fights)} matched")
        print(f"FanDuel: {fd_count}/{len(fights)} matched")

        missing_books = len(fights) - min(dk_count, fd_count)
        if missing_books > 0:
            print(f"{missing_books} fights missing odds from one or more books")

        print("-" * 60)

        # Compare mode
        if args.compare:
            predictions = load_model_predictions()
            if not predictions:
                print()
                print("No upcoming predictions found in ledger for comparison.")
                print("Run predictions first: python src/predict/generate_all_events.py")
            else:
                print_comparison_table(fights, predictions)

    print()


if __name__ == '__main__':
    main()
