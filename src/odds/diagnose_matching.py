#!/usr/bin/env python3
"""
Diagnose odds-to-ledger fighter name matching failures.

This script analyzes why some fights in the ledger don't have odds attached,
even when odds data exists in upcoming_odds.json.

Usage:
    python src/odds/diagnose_matching.py
"""

import json
import unicodedata
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ODDS_PATH = PROJECT_ROOT / 'data' / 'odds' / 'upcoming_odds.json'
LEDGER_PATH = PROJECT_ROOT / 'data' / 'ledger' / 'prediction_ledger.json'


def normalize_name_basic(name: str) -> str:
    """Basic normalization (current implementation)."""
    if not name:
        return ""
    name = ' '.join(name.lower().strip().split())
    name = name.replace("'", "").replace("-", " ").replace(".", "")
    return name


def normalize_name_improved(name: str) -> str:
    """
    Improved normalization that handles:
    - Unicode/accent characters (José -> jose)
    - Transliteration variants (Sergey/Serghei)
    - Hyphens, apostrophes, periods
    - Common suffixes (Jr, Jr., II, III)
    """
    if not name:
        return ""

    # Unicode normalize — decompose accents
    name = unicodedata.normalize('NFKD', name)
    # Strip accent marks (combining characters)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # Lowercase
    name = name.lower()

    # Remove apostrophes (various unicode variants)
    name = name.replace("'", "").replace("'", "").replace("'", "").replace("`", "")

    # Replace hyphens and periods with spaces
    name = name.replace("-", " ").replace(".", " ")

    # Remove common suffixes that vary
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', name)

    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def fuzzy_match_score(name1: str, name2: str) -> float:
    """Calculate fuzzy match score between two normalized names."""
    return SequenceMatcher(None, name1, name2).ratio()


def get_last_name(name: str) -> str:
    """Extract last name (last token) from normalized name."""
    parts = name.split()
    return parts[-1] if parts else ""


def match_fight_pair(
    odds_f1: str, odds_f2: str,
    ledger_f1: str, ledger_f2: str
) -> Tuple[float, bool]:
    """
    Calculate match score for a fight pair.

    Returns:
        Tuple of (score out of 2.0, is_flipped)
    """
    # Normalize all names
    o1 = normalize_name_improved(odds_f1)
    o2 = normalize_name_improved(odds_f2)
    l1 = normalize_name_improved(ledger_f1)
    l2 = normalize_name_improved(ledger_f2)

    # Try straight matching
    score_straight = fuzzy_match_score(o1, l1) + fuzzy_match_score(o2, l2)

    # Try flipped matching
    score_flipped = fuzzy_match_score(o1, l2) + fuzzy_match_score(o2, l1)

    if score_straight >= score_flipped:
        return score_straight, False
    else:
        return score_flipped, True


def diagnose():
    """Run matching diagnostics."""

    # Load data
    if not ODDS_PATH.exists():
        print(f"ERROR: Odds file not found: {ODDS_PATH}")
        return

    if not LEDGER_PATH.exists():
        print(f"ERROR: Ledger file not found: {LEDGER_PATH}")
        return

    with open(ODDS_PATH) as f:
        odds_data = json.load(f)

    with open(LEDGER_PATH) as f:
        ledger = json.load(f)

    # Get odds fights (only those with actual odds data)
    odds_fights = []
    for fight in odds_data.get('fights', []):
        if fight.get('odds') and (fight['odds'].get('draftkings') or fight['odds'].get('fanduel')):
            odds_fights.append({
                'f1': fight.get('fighter_1', ''),
                'f2': fight.get('fighter_2', ''),
                'f1_canonical': fight.get('fighter_1_canonical', ''),
                'f2_canonical': fight.get('fighter_2_canonical', ''),
                'event_date': fight.get('event_date', ''),
            })

    # Get unlocked ledger fights
    ledger_fights = []
    for entry in ledger.get('entries', []):
        if entry.get('locked', True):
            continue

        event_name = entry.get('event_name', '')
        event_date = entry.get('event_date', '')

        for fight in entry.get('fights', []):
            ledger_fights.append({
                'f1': fight.get('fighter_1', ''),
                'f2': fight.get('fighter_2', ''),
                'event_name': event_name,
                'event_date': event_date,
                'has_odds': 'odds' in fight and fight['odds'] is not None,
            })

    print("=" * 70)
    print("ODDS-TO-LEDGER MATCHING DIAGNOSTICS")
    print("=" * 70)
    print(f"\nOdds source: {ODDS_PATH}")
    print(f"Scraped at: {odds_data.get('scraped_at', 'unknown')}")
    print(f"Odds fights with data: {len(odds_fights)}")
    print(f"\nLedger unlocked fights: {len(ledger_fights)}")
    print(f"Ledger fights with odds: {sum(1 for f in ledger_fights if f['has_odds'])}")
    print(f"Ledger fights without odds: {sum(1 for f in ledger_fights if not f['has_odds'])}")

    # Match each ledger fight to odds
    matched = []
    unmatched_ledger = []

    for lf in ledger_fights:
        best_match = None
        best_score = 0.0
        best_flipped = False
        best_odds_fight = None

        for of in odds_fights:
            score, flipped = match_fight_pair(of['f1'], of['f2'], lf['f1'], lf['f2'])
            if score > best_score:
                best_score = score
                best_flipped = flipped
                best_odds_fight = of

        if best_score >= 1.7:  # Both names ~85% similar
            matched.append({
                'ledger': lf,
                'odds': best_odds_fight,
                'score': best_score,
                'flipped': best_flipped,
            })
        else:
            unmatched_ledger.append({
                'ledger': lf,
                'best_match': best_odds_fight,
                'best_score': best_score,
            })

    # Find unmatched odds fights
    matched_odds = set()
    for m in matched:
        matched_odds.add((m['odds']['f1'], m['odds']['f2']))

    unmatched_odds = []
    for of in odds_fights:
        if (of['f1'], of['f2']) not in matched_odds:
            unmatched_odds.append(of)

    # Print results
    print("\n" + "-" * 70)
    print("MATCHED (odds successfully attached):")
    print("-" * 70)
    for m in matched:
        lf = m['ledger']
        of = m['odds']
        status = "✓" if lf['has_odds'] else "⚠ (not in ledger yet)"
        flip = " [FLIPPED]" if m['flipped'] else ""
        print(f"  {lf['f1']} vs {lf['f2']}")
        print(f"    → {of['f1']} vs {of['f2']}{flip} (score: {m['score']:.2f}) {status}")

    print(f"\n  Total matched: {len(matched)}")

    print("\n" + "-" * 70)
    print("UNMATCHED LEDGER FIGHTS (no odds found):")
    print("-" * 70)
    for u in unmatched_ledger:
        lf = u['ledger']
        print(f"\n  {lf['f1']} vs {lf['f2']}")
        print(f"    Event: {lf['event_name']} ({lf['event_date']})")

        if u['best_match']:
            bm = u['best_match']
            print(f"    Closest match in odds: {bm['f1']} vs {bm['f2']} (score: {u['best_score']:.2f})")

            # Show name comparison
            l1_norm = normalize_name_improved(lf['f1'])
            l2_norm = normalize_name_improved(lf['f2'])
            o1_norm = normalize_name_improved(bm['f1'])
            o2_norm = normalize_name_improved(bm['f2'])

            score_11 = fuzzy_match_score(l1_norm, o1_norm)
            score_12 = fuzzy_match_score(l1_norm, o2_norm)
            score_21 = fuzzy_match_score(l2_norm, o1_norm)
            score_22 = fuzzy_match_score(l2_norm, o2_norm)

            print(f"    Name comparisons:")
            print(f"      '{l1_norm}' vs '{o1_norm}': {score_11:.2f}")
            print(f"      '{l1_norm}' vs '{o2_norm}': {score_12:.2f}")
            print(f"      '{l2_norm}' vs '{o1_norm}': {score_21:.2f}")
            print(f"      '{l2_norm}' vs '{o2_norm}': {score_22:.2f}")
        else:
            print(f"    No potential matches in odds data")

    print(f"\n  Total unmatched: {len(unmatched_ledger)}")

    print("\n" + "-" * 70)
    print("UNMATCHED ODDS FIGHTS (odds exist but no ledger fight):")
    print("-" * 70)
    for of in unmatched_odds:
        print(f"  {of['f1']} vs {of['f2']} ({of['event_date']})")

    print(f"\n  Total unmatched: {len(unmatched_odds)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Matched: {len(matched)}")
    print(f"Ledger fights missing odds: {len(unmatched_ledger)}")
    print(f"Odds fights without ledger match: {len(unmatched_odds)}")

    # Identify specific issues
    print("\n" + "-" * 70)
    print("IDENTIFIED ISSUES:")
    print("-" * 70)

    for u in unmatched_ledger:
        lf = u['ledger']
        if u['best_match']:
            bm = u['best_match']
            # Check if it's a name transliteration issue
            l1_norm = normalize_name_improved(lf['f1'])
            o1_norm = normalize_name_improved(bm['f1'])
            l2_norm = normalize_name_improved(lf['f2'])
            o2_norm = normalize_name_improved(bm['f2'])

            if l2_norm == o2_norm and l1_norm != o1_norm:
                score = fuzzy_match_score(l1_norm, o1_norm)
                if score > 0.6:
                    print(f"\n  NAME TRANSLITERATION: '{lf['f1']}' ↔ '{bm['f1']}'")
                    print(f"    Normalized: '{l1_norm}' vs '{o1_norm}' (score: {score:.2f})")
                else:
                    print(f"\n  DIFFERENT FIGHTER: Ledger has '{lf['f1']}', odds has '{bm['f1']}'")
                    print(f"    (Same opponent: {lf['f2']})")
            elif l1_norm == o1_norm and l2_norm != o2_norm:
                score = fuzzy_match_score(l2_norm, o2_norm)
                if score > 0.6:
                    print(f"\n  NAME TRANSLITERATION: '{lf['f2']}' ↔ '{bm['f2']}'")
                    print(f"    Normalized: '{l2_norm}' vs '{o2_norm}' (score: {score:.2f})")
                else:
                    print(f"\n  DIFFERENT FIGHTER: Ledger has '{lf['f2']}', odds has '{bm['f2']}'")
                    print(f"    (Same opponent: {lf['f1']})")
        else:
            print(f"\n  NO ODDS DATA: {lf['f1']} vs {lf['f2']}")
            print(f"    This fight is not in the odds API at all")


if __name__ == '__main__':
    diagnose()
