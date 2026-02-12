#!/usr/bin/env python3
"""
Flask web application for UFC fight predictions.

Routes:
    /                   - Next upcoming event predictions
    /event/<slug>       - Specific event predictions
    /archive            - Past predicted events with results
    /upcoming           - List of upcoming events
    /methodology        - Model methodology summary
    /api/predict/<slug> - JSON API for predictions
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, redirect, url_for

# Human-readable feature name mapping (shortened for bar chart display)
FEATURE_LABELS = {
    'diff_career_win_rate': 'Win %',
    'diff_career_ko_rate': 'KO %',
    'diff_career_sub_rate': 'Sub %',
    'diff_career_dec_rate': 'Dec %',
    'diff_career_finish_rate': 'Finish %',
    'diff_days_since_last_fight': 'Activity',
    'diff_win_rate_last_3': 'Recent Form',
    'diff_win_rate_last_5': 'Form (5)',
    'diff_win_rate_last_10': 'Form (10)',
    'diff_career_fights': 'Fights',
    'diff_weight_class_fights': 'Div Exp',
    'diff_win_streak': 'Streak',
    'diff_loss_streak': 'L Streak',
    'diff_fights_per_year': 'Frequency',
    'diff_momentum': 'Momentum',
    'diff_striking_volume': 'Volume',
    'diff_grappling_tendency': 'Grappling',
    'diff_avg_sig_strikes_landed_last_3': 'Strikes',
    'diff_avg_sig_strikes_landed_last_5': 'Strikes (5)',
    'diff_avg_sig_strikes_absorbed_last_3': 'Absorbed',
    'diff_avg_sig_strikes_absorbed_last_5': 'Absorbed (5)',
    'diff_avg_takedowns_last_3': 'Takedowns',
    'diff_avg_takedowns_last_5': 'TD (5)',
    'diff_finish_rate_last_3': 'Finish (3)',
    'diff_finish_rate_last_5': 'Finish (5)',
    'diff_avg_fight_time_last_3': 'Duration',
    'diff_avg_fight_time_last_5': 'Duration (5)',
    'diff_experience_ratio': 'Exp Ratio',
    'diff_data_completeness_score': 'Data',
    'diff_career_wins': 'Wins',
    'diff_career_losses': 'Losses',
    # Phase 8 Elo features
    'diff_elo': 'Elo Rating',
    'diff_elo_momentum': 'Elo Trend',
}

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict.serve import PredictionPipeline
from src.odds.scraper import (
    UFCOddsScraper,
    load_upcoming_odds,
    normalize_name,
    fuzzy_match_score,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PREDICTIONS_DIR = PROJECT_ROOT / 'data' / 'predictions'
LEDGER_PATH = PROJECT_ROOT / 'data' / 'ledger' / 'prediction_ledger.json'
MODEL_REGISTRY_PATH = PROJECT_ROOT / 'data' / 'models' / 'model_registry.json'
CACHE_MAX_AGE_HOURS = 24

# Global pipeline (loaded once)
_pipeline: Optional[PredictionPipeline] = None


def get_pipeline() -> PredictionPipeline:
    """Get or create the prediction pipeline (singleton)."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading prediction pipeline...")
        # Use defaults from serve.py which auto-detects Phase 9/8/v1 model
        _pipeline = PredictionPipeline()
        from src.predict.serve import MODEL_VERSION
        logger.info(f"Pipeline loaded successfully (model: {MODEL_VERSION})")
    return _pipeline


def get_model_version() -> str:
    """Get the current model version being used."""
    from src.predict.serve import MODEL_VERSION
    return MODEL_VERSION


def slugify(text: str) -> str:
    """Convert event name to URL-safe slug."""
    return text.lower().replace(' ', '-').replace(':', '').replace('.', '')


def slugify_fighter(name: str) -> str:
    """Convert fighter name to URL-safe slug for headshot filenames."""
    slug = name.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')


def get_initials(name: str) -> str:
    """Get fighter initials for fallback display."""
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    elif len(parts) == 1 and len(parts[0]) >= 2:
        return parts[0][:2].upper()
    return "??"


def get_last_name(full_name: str) -> str:
    """
    Extract last name from fighter's full name.

    Handles multi-word last names like "de la Cruz" by checking if name
    contains common prefixes (de, da, van, von, etc).
    """
    if not full_name:
        return ""

    parts = full_name.strip().split()
    if len(parts) <= 1:
        return full_name

    # Check for multi-word last name prefixes
    lower_parts = [p.lower() for p in parts]
    prefixes = {'de', 'da', 'do', 'van', 'von', 'la', 'le', 'del', 'dos'}

    # Find where the last name starts (after first name)
    last_name_start = 1
    for i in range(1, len(parts) - 1):
        if lower_parts[i] in prefixes:
            last_name_start = i
            break

    return ' '.join(parts[last_name_start:])


def get_headshot_url(fighter_name: str) -> Optional[str]:
    """
    Get headshot URL for a fighter if image exists.

    Args:
        fighter_name: Fighter's full name

    Returns:
        URL for the headshot image or None if not found
    """
    slug = slugify_fighter(fighter_name)
    filepath = Path(__file__).parent / 'static' / 'fighters' / f'{slug}.png'

    if filepath.exists():
        return f'/static/fighters/{slug}.png'
    return None


def get_body_photo_url(fighter_name: str) -> Optional[str]:
    """
    Get body photo URL for a fighter if image exists.

    Args:
        fighter_name: Fighter's full name

    Returns:
        URL for the body photo image or None if not found
    """
    slug = slugify_fighter(fighter_name)
    filepath = Path(__file__).parent / 'static' / 'fighters' / 'body' / f'{slug}_body.png'

    if filepath.exists():
        return f'/static/fighters/body/{slug}_body.png'
    return None


# Face directions cache (loaded once)
_face_directions: Optional[Dict[str, str]] = None
FACE_DIRECTIONS_FILE = Path(__file__).parent / 'static' / 'fighters' / 'body' / 'face_directions.json'


def get_face_directions() -> Dict[str, str]:
    """Load face directions from JSON file (cached)."""
    global _face_directions
    if _face_directions is None:
        if FACE_DIRECTIONS_FILE.exists():
            try:
                with open(FACE_DIRECTIONS_FILE, 'r') as f:
                    _face_directions = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load face directions: {e}")
                _face_directions = {}
        else:
            _face_directions = {}
    return _face_directions


def should_flip(fighter_name: str, position: str) -> bool:
    """
    Determine if a fighter's body photo should be horizontally flipped.

    Goal: Both fighters should face toward the center (like a staredown).
    - Left fighter (Fighter 1) should face RIGHT
    - Right fighter (Fighter 2) should face LEFT

    Args:
        fighter_name: Fighter's full name
        position: 'left' (Fighter 1) or 'right' (Fighter 2)

    Returns:
        True if the image should be flipped with scaleX(-1)
    """
    slug = slugify_fighter(fighter_name)
    directions = get_face_directions()
    direction = directions.get(slug, 'center')

    if position == 'left':
        # Left fighter should face RIGHT (toward center)
        # Flip if they're facing left (away from center)
        return direction == 'left'
    else:
        # Right fighter should face LEFT (toward center)
        # Flip if they're facing right (away from center)
        return direction == 'right'

    # If 'center', don't flip — straight-on works for both positions


def get_cached_predictions(event_slug: str) -> Optional[Dict]:
    """Load cached predictions if they exist and are fresh."""
    cache_file = PREDICTIONS_DIR / f'{event_slug}.json'

    if not cache_file.exists():
        return None

    # Check age
    mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
    age = datetime.now() - mtime

    if age > timedelta(hours=CACHE_MAX_AGE_HOURS):
        logger.info(f"Cache expired for {event_slug}")
        return None

    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return None


def save_predictions_cache(event_slug: str, data: Dict):
    """Save predictions to cache."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PREDICTIONS_DIR / f'{event_slug}.json'

    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Cached predictions to {cache_file}")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")


def get_feature_label(feature_name: str) -> str:
    """Get human-readable label for a feature name."""
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    # Fallback: clean up the name
    label = feature_name.replace('diff_', '').replace('_', ' ')
    return label.title()


def compute_tornado_bars(factors: List, top_n: int = 5) -> List[Dict]:
    """
    Format differential factors for tornado chart display.

    Tornado chart: labels in center, bars extend outward toward the fighter
    with the advantage. Red bars go LEFT (Fighter 1), blue bars go RIGHT (Fighter 2).

    Args:
        factors: List of (feature_name, value) tuples or dicts
                 Positive value = Fighter 1 advantage (red, left bar)
                 Negative value = Fighter 2 advantage (blue, right bar)
        top_n: Number of top factors to include

    Returns:
        List of tornado bar dicts with left_width and right_width percentages
    """
    parsed = []

    for factor in factors[:top_n]:
        if isinstance(factor, (list, tuple)):
            name, val = factor[0], factor[1]
        elif isinstance(factor, dict):
            name, val = factor.get('feature', ''), factor.get('value', 0)
        else:
            continue

        # Convert string values to float (handles JSON serialization issues)
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        if val == 0:
            continue

        parsed.append({
            'feature': name,
            'value': val,
            'label': get_feature_label(name),
        })

    if not parsed:
        return []

    # Rank-based normalization: sort by absolute value descending
    # Width percentages for ranks 1-5 (bars grow from center outward)
    width_steps = [85, 68, 52, 36, 22]
    sorted_by_abs = sorted(parsed, key=lambda f: abs(f['value']), reverse=True)

    # Assign widths based on rank
    for i, stat in enumerate(sorted_by_abs):
        pct = width_steps[min(i, len(width_steps) - 1)]
        val = stat['value']

        if val > 0:
            # Fighter 1 advantage: red bar extends LEFT
            stat['left_width'] = pct
            stat['right_width'] = 0
        else:
            # Fighter 2 advantage: blue bar extends RIGHT
            stat['left_width'] = 0
            stat['right_width'] = pct

    return parsed


def format_prediction(
    pred: Dict,
    position: int = 0,
    total_fights: int = 13,
    odds_lookup: Optional[Dict[str, Dict]] = None
) -> Dict:
    """
    Format a prediction dict for template rendering.

    Args:
        pred: Raw prediction dict
        position: Fight position in card (0 = main event)
        total_fights: Total fights on card
        odds_lookup: Optional dict of odds data keyed by normalized fighter pair
    """
    f1_prob = pred['f1_win_prob']
    f2_prob = pred['f2_win_prob']

    # Determine favorite
    if f1_prob > f2_prob:
        favorite = 1
    else:
        favorite = 2

    # Get all factors for tornado chart
    raw_factors = pred.get('top_factors', [])
    tornado_factors = compute_tornado_bars(raw_factors, top_n=5)

    # Legacy factors (kept for backwards compatibility)
    factors = []
    for factor in raw_factors[:3]:
        if isinstance(factor, (list, tuple)):
            name, val = factor[0], factor[1]
        elif isinstance(factor, dict):
            name, val = factor.get('feature', ''), factor.get('value', 0)
        else:
            continue
        # Convert string values to float
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        name = name.replace('diff_', '').replace('_', ' ')
        factors.append({
            'name': name,
            'value': f"+{val:.2f}" if val > 0 else f"{val:.2f}"
        })

    # Position labels and card section
    # Assume first 5 fights are main card, rest are prelims
    main_card_size = min(5, total_fights)

    if position == 0:
        position_label = "Main Event"
        card_section = "main"
    elif position == 1:
        position_label = "Co-Main"
        card_section = "main"
    elif position < main_card_size:
        position_label = None
        card_section = "main"
    else:
        position_label = None
        card_section = "prelim"

    # Get headshot info for both fighters
    f1_name = pred['fighter1']
    f2_name = pred['fighter2']

    # Match and format odds data
    odds_display = None
    if odds_lookup:
        raw_odds = match_fight_to_odds(f1_name, f2_name, odds_lookup)
        if raw_odds:
            odds_display = format_odds_for_display(
                raw_odds, f1_name, f2_name, f1_prob, f2_prob
            )

    # Also check if odds came directly in pred (from ledger)
    if not odds_display and pred.get('odds'):
        odds_display = format_odds_for_display(
            pred['odds'], f1_name, f2_name, f1_prob, f2_prob
        )

    return {
        'fighter1': f1_name,
        'fighter2': f2_name,
        'f1_prob': f1_prob,
        'f2_prob': f2_prob,
        'f1_pct': f"{f1_prob * 100:.1f}",
        'f2_pct': f"{f2_prob * 100:.1f}",
        'favorite': favorite,
        'confidence': pred.get('confidence', 'LOW'),
        'weight_class': pred.get('weight_class', ''),
        'factors': factors,
        'tornado_factors': tornado_factors,
        'position_label': position_label,
        'card_section': card_section,
        'f1_exact_match': pred.get('f1_exact_match', True),
        'f2_exact_match': pred.get('f2_exact_match', True),
        'fighter1_matched': pred.get('fighter1_matched', f1_name),
        'fighter2_matched': pred.get('fighter2_matched', f2_name),
        # Headshot fields
        'f1_headshot': get_headshot_url(f1_name),
        'f2_headshot': get_headshot_url(f2_name),
        'f1_initials': get_initials(f1_name),
        'f2_initials': get_initials(f2_name),
        # Body photo fields
        'f1_body_photo': get_body_photo_url(f1_name),
        'f2_body_photo': get_body_photo_url(f2_name),
        # Body photo flip (smart face direction)
        'f1_flip': should_flip(f1_name, 'left'),
        'f2_flip': should_flip(f2_name, 'right'),
        # Elo ratings (Phase 8)
        'f1_elo': pred.get('f1_elo', 1500),
        'f2_elo': pred.get('f2_elo', 1500),
        # Backtest fields
        'actual_winner': pred.get('actual_winner'),
        'correct': pred.get('correct'),
        'method': pred.get('method'),
        # Odds fields (may be None if no odds available)
        'odds': odds_display,
    }


def get_upcoming_predictions() -> Optional[Dict]:
    """Get predictions for the next upcoming event."""
    try:
        pipeline = get_pipeline()

        # Scrape upcoming event
        upcoming_fights = pipeline.scrape_upcoming_event()

        if len(upcoming_fights) == 0:
            return None

        # Generate predictions
        predictions = pipeline.predict_event(upcoming_fights)

        if not predictions:
            return None

        event_name = predictions[0].get('event_name', 'Unknown Event')
        event_date = predictions[0].get('event_date', '')
        event_slug = slugify(event_name)

        # Load odds data for matching
        odds_lookup = get_odds_for_fights(predictions)

        # Format predictions with odds
        total = len(predictions)
        formatted = [
            format_prediction(p, i, total, odds_lookup)
            for i, p in enumerate(predictions)
        ]

        # Count confidence levels
        confidence_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for p in predictions:
            conf = p.get('confidence', 'LOW')
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        result = {
            'event_name': event_name,
            'event_date': event_date,
            'event_slug': event_slug,
            'predictions': formatted,
            'fight_count': len(formatted),
            'confidence_counts': confidence_counts,
            'generated_at': datetime.now().isoformat()
        }

        # Cache it
        save_predictions_cache(event_slug, result)

        return result

    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_backtest_predictions(date_str: str) -> Optional[Dict]:
    """Get backtest predictions for a historical event."""
    try:
        pipeline = get_pipeline()
        predictions = pipeline.backtest_event(date_str)

        if not predictions:
            return None

        event_name = predictions[0].get('event_name', 'Unknown Event')
        event_date = predictions[0].get('event_date', '')
        event_slug = slugify(event_name)

        # Format predictions (no odds for backtests - historical data not available)
        total = len(predictions)
        formatted = [
            format_prediction(p, i, total, odds_lookup=None)
            for i, p in enumerate(predictions)
        ]

        # Calculate accuracy
        correct = sum(1 for p in predictions if p.get('correct', False))
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0

        # Count confidence levels
        confidence_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for p in predictions:
            conf = p.get('confidence', 'LOW')
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        return {
            'event_name': event_name,
            'event_date': event_date,
            'event_slug': event_slug,
            'predictions': formatted,
            'fight_count': len(formatted),
            'confidence_counts': confidence_counts,
            'is_backtest': True,
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'accuracy_pct': f"{accuracy * 100:.1f}",
            'generated_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting backtest predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_events_json() -> Optional[Dict]:
    """Load the events.json file with all event data."""
    events_file = PREDICTIONS_DIR / 'events.json'

    if not events_file.exists():
        return None

    try:
        with open(events_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading events.json: {e}")
        return None


def get_all_events() -> Tuple[List[Dict], List[Dict]]:
    """
    Get all events from events.json, split into upcoming and completed.

    Returns:
        Tuple of (upcoming_events, completed_events)
    """
    events_data = load_events_json()

    if not events_data or 'events' not in events_data:
        return [], []

    upcoming = []
    completed = []

    for event in events_data['events']:
        event_info = {
            'event_name': event.get('name', ''),
            'event_date': event.get('date', ''),
            'event_slug': event.get('slug', ''),
            'location': event.get('location', ''),
            'fight_count': event.get('fight_count', 0),
            'status': event.get('status', 'unknown'),
        }

        if event.get('status') == 'completed':
            event_info['is_backtest'] = True
            event_info['correct'] = event.get('correct', 0)
            event_info['total'] = event.get('fight_count', 0)
            event_info['accuracy'] = event.get('accuracy', 0)
            event_info['accuracy_pct'] = event.get('accuracy_pct', '0')
            completed.append(event_info)
        else:
            event_info['is_backtest'] = False
            upcoming.append(event_info)

    return upcoming, completed


def get_event_from_json(slug: str) -> Optional[Dict]:
    """Get a specific event from events.json by slug."""
    events_data = load_events_json()

    if not events_data or 'events' not in events_data:
        return None

    for event in events_data['events']:
        if event.get('slug') == slug:
            return event

    return None


def format_event_from_json(event_data: Dict) -> Optional[Dict]:
    """
    Format an event from events.json for template rendering.

    Converts the events.json format to the format expected by event.html template.

    Args:
        event_data: Raw event dict from events.json

    Returns:
        Formatted event dict for template, or None if invalid
    """
    if not event_data:
        return None

    fights = event_data.get('fights', [])
    if not fights:
        return None

    total_fights = len(fights)
    formatted_predictions = []

    # Load odds data for matching (only for upcoming events)
    is_completed = event_data.get('status') == 'completed'
    odds_lookup = get_odds_for_fights(fights) if not is_completed else {}

    for i, fight in enumerate(fights):
        # Convert events.json fight format to the format expected by format_prediction
        pred = {
            'fighter1': fight.get('fighter_1', ''),
            'fighter2': fight.get('fighter_2', ''),
            'f1_win_prob': fight.get('f1_win_prob', 0.5),
            'f2_win_prob': fight.get('f2_win_prob', 0.5),
            'confidence': fight.get('confidence', 'LOW'),
            'weight_class': fight.get('weight_class', ''),
            'f1_elo': fight.get('f1_elo', 1500),
            'f2_elo': fight.get('f2_elo', 1500),
            'f1_exact_match': fight.get('f1_exact_match', True),
            'f2_exact_match': fight.get('f2_exact_match', True),
            'top_factors': [],  # events.json doesn't store factors
            # Backtest fields
            'actual_winner': fight.get('actual_winner'),
            'correct': fight.get('correct'),
            'method': fight.get('method'),
            # Odds from event data if available
            'odds': fight.get('odds'),
        }
        formatted_predictions.append(format_prediction(pred, i, total_fights, odds_lookup))

    # Count confidence levels
    confidence_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for fight in fights:
        conf = fight.get('confidence', 'LOW')
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    is_backtest = event_data.get('status') == 'completed'

    result = {
        'event_name': event_data.get('name', ''),
        'event_date': event_data.get('date', ''),
        'event_slug': event_data.get('slug', ''),
        'predictions': formatted_predictions,
        'fight_count': total_fights,
        'confidence_counts': confidence_counts,
        'is_backtest': is_backtest,
    }

    if is_backtest:
        result['correct'] = event_data.get('correct', 0)
        result['total'] = total_fights
        result['accuracy'] = event_data.get('accuracy', 0)
        result['accuracy_pct'] = event_data.get('accuracy_pct', '0')

    return result


def get_archived_events() -> List[Dict]:
    """Get list of archived events with cached predictions."""
    # First try events.json
    upcoming, completed = get_all_events()

    if upcoming or completed:
        # Return all events (upcoming first, then completed)
        events = []
        for event in upcoming:
            events.append(event)
        for event in completed:
            events.append(event)
        return events

    # Fall back to individual cache files
    events = []

    if not PREDICTIONS_DIR.exists():
        return events

    for cache_file in sorted(PREDICTIONS_DIR.glob('*.json'), reverse=True):
        if cache_file.name == 'events.json':
            continue

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            events.append({
                'event_name': data.get('event_name', cache_file.stem),
                'event_date': data.get('event_date', ''),
                'event_slug': data.get('event_slug', cache_file.stem),
                'fight_count': data.get('fight_count', 0),
                'is_backtest': data.get('is_backtest', False),
                'correct': data.get('correct'),
                'total': data.get('total'),
                'accuracy_pct': data.get('accuracy_pct'),
            })
        except Exception as e:
            logger.warning(f"Error loading {cache_file}: {e}")
            continue

    return events


def load_ledger() -> Optional[Dict]:
    """Load the prediction ledger."""
    if not LEDGER_PATH.exists():
        return None

    try:
        with open(LEDGER_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading ledger: {e}")
        return None


def load_model_registry() -> Optional[Dict]:
    """Load the model registry."""
    if not MODEL_REGISTRY_PATH.exists():
        return None

    try:
        with open(MODEL_REGISTRY_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading model registry: {e}")
        return None


# =============================================================================
# Odds Integration Functions
# =============================================================================

def get_odds_for_fights(fights: List[Dict]) -> Dict[str, Dict]:
    """
    Load odds data and match to fights by fighter names.

    Args:
        fights: List of fight dicts with fighter_1/fighter_2 or fighter1/fighter2 keys

    Returns:
        Dict mapping normalized fight key to odds data
    """
    odds_data = load_upcoming_odds()
    if not odds_data:
        return {}

    # Build lookup dict by normalized fighter pair
    odds_lookup = {}
    for fight in odds_data.get('fights', []):
        f1 = fight.get('fighter_1', '') or fight.get('fighter_1_canonical', '')
        f2 = fight.get('fighter_2', '') or fight.get('fighter_2_canonical', '')
        if not f1 or not f2:
            continue

        # Create normalized key (alphabetical order for consistency)
        names = sorted([normalize_name(f1), normalize_name(f2)])
        key = f"{names[0]}|{names[1]}"
        odds_lookup[key] = fight.get('odds', {})

    return odds_lookup


def match_fight_to_odds(
    fighter1: str,
    fighter2: str,
    odds_lookup: Dict[str, Dict]
) -> Optional[Dict]:
    """
    Find matching odds data for a fight using pair-based scoring.

    Uses pair-based matching that:
    - Normalizes names (handles accents, hyphens, transliterations)
    - Tries both fighter orderings
    - Uses combined pair score threshold of 1.7/2.0

    Args:
        fighter1: First fighter name
        fighter2: Second fighter name
        odds_lookup: Dict of odds data keyed by normalized fighter pair

    Returns:
        Matched odds data or None
    """
    if not odds_lookup:
        return None

    # Create normalized key
    f1_norm = normalize_name(fighter1)
    f2_norm = normalize_name(fighter2)
    names = sorted([f1_norm, f2_norm])
    key = f"{names[0]}|{names[1]}"

    # Try exact match first
    if key in odds_lookup:
        return odds_lookup[key]

    # Try pair-based fuzzy matching
    best_match = None
    best_score = 0.0

    for odds_key, odds in odds_lookup.items():
        odds_names = odds_key.split('|')
        o1_norm = odds_names[0]
        o2_norm = odds_names[1] if len(odds_names) > 1 else ''

        # Try both orderings and take the better one
        # Straight: f1->o1, f2->o2
        score_straight = fuzzy_match_score(f1_norm, o1_norm) + fuzzy_match_score(f2_norm, o2_norm)
        # Flipped: f1->o2, f2->o1
        score_flipped = fuzzy_match_score(f1_norm, o2_norm) + fuzzy_match_score(f2_norm, o1_norm)

        score = max(score_straight, score_flipped)

        if score > best_score:
            best_score = score
            best_match = odds

    # Threshold: 1.7 out of 2.0 means both names must average ~85% similarity
    if best_score >= 1.7 and best_match:
        return best_match

    return None


def format_odds_for_display(
    odds: Optional[Dict],
    f1_name: str,
    f2_name: str,
    f1_model_prob: float,
    f2_model_prob: float
) -> Optional[Dict]:
    """
    Format odds data for template display.

    Handles two input formats:
    1. Scraper format: {'draftkings': {'f1_moneyline': X}, 'consensus': {'f1_fair': Y}}
    2. Ledger format: {'draftkings': {'f1_ml': X}, 'consensus_f1_fair': Y}

    Args:
        odds: Raw odds dict from scraper or ledger (may be None)
        f1_name: Fighter 1 name (for matching moneylines to correct fighter)
        f2_name: Fighter 2 name
        f1_model_prob: Model probability for fighter 1
        f2_model_prob: Model probability for fighter 2

    Returns:
        Formatted odds dict for template or None if no odds
    """
    if not odds:
        return None

    result = {}

    # DraftKings moneylines (handle both 'f1_moneyline' and 'f1_ml' formats)
    dk = odds.get('draftkings', {})
    if dk:
        f1_ml = dk.get('f1_moneyline') or dk.get('f1_ml')
        f2_ml = dk.get('f2_moneyline') or dk.get('f2_ml')
        if f1_ml is not None or f2_ml is not None:
            result['draftkings'] = {
                'f1_ml': f1_ml,
                'f2_ml': f2_ml,
            }

    # FanDuel moneylines (handle both formats)
    fd = odds.get('fanduel', {})
    if fd:
        f1_ml = fd.get('f1_moneyline') or fd.get('f1_ml')
        f2_ml = fd.get('f2_moneyline') or fd.get('f2_ml')
        if f1_ml is not None or f2_ml is not None:
            result['fanduel'] = {
                'f1_ml': f1_ml,
                'f2_ml': f2_ml,
            }

    # Consensus fair probabilities (handle both nested and flat formats)
    # Scraper format: {'consensus': {'f1_fair': X, 'f2_fair': Y}}
    # Ledger format: {'consensus_f1_fair': X, 'consensus_f2_fair': Y}
    consensus = odds.get('consensus', {})
    if consensus:
        consensus_f1 = consensus.get('f1_fair')
        consensus_f2 = consensus.get('f2_fair')
    else:
        # Try flat ledger format
        consensus_f1 = odds.get('consensus_f1_fair')
        consensus_f2 = odds.get('consensus_f2_fair')

    if consensus_f1 is not None and consensus_f2 is not None:
        result['consensus_f1_fair'] = consensus_f1
        result['consensus_f2_fair'] = consensus_f2

        # Calculate edge from predicted winner's perspective
        # Positive edge = model sees value the market is missing
        if f1_model_prob >= f2_model_prob:
            # Model picks F1
            edge = f1_model_prob - consensus_f1
        else:
            # Model picks F2
            edge = f2_model_prob - consensus_f2

        result['edge'] = edge
        result['edge_pct'] = abs(edge) * 100  # For threshold comparison in template

    return result if (result.get('draftkings') or result.get('fanduel') or result.get('consensus_f1_fair')) else None


def format_moneyline(ml: Optional[int]) -> str:
    """Format moneyline with +/- prefix."""
    if ml is None:
        return 'N/A'
    if ml >= 0:
        return f"+{ml}"
    return str(ml)


def get_ledger_stats() -> Dict:
    """Get summary statistics from the ledger."""
    ledger = load_ledger()
    if not ledger:
        return {
            'total_fights': 0,
            'total_correct': 0,
            'overall_accuracy': 0,
            'events_tracked': 0,
            'live_fights': 0,
            'live_correct': 0,
            'live_accuracy': 0,
            'backtest_fights': 0,
            'backtest_correct': 0,
            'backtest_accuracy': 0,
        }

    entries = ledger.get('entries', [])
    locked = [e for e in entries if e.get('locked', False)]

    # Separate by prediction type
    live = [e for e in locked if e.get('prediction_type') == 'live']
    backtest = [e for e in locked if e.get('prediction_type') == 'backtest']

    # Calculate overall accuracy
    total_correct = sum(e.get('overall_correct', 0) or 0 for e in locked)
    total_fights = sum(e.get('overall_total', 0) or 0 for e in locked)

    live_correct = sum(e.get('overall_correct', 0) or 0 for e in live)
    live_fights = sum(e.get('overall_total', 0) or 0 for e in live)

    backtest_correct = sum(e.get('overall_correct', 0) or 0 for e in backtest)
    backtest_fights = sum(e.get('overall_total', 0) or 0 for e in backtest)

    return {
        'total_fights': total_fights,
        'total_correct': total_correct,
        'overall_accuracy': total_correct / total_fights if total_fights > 0 else 0,
        'events_tracked': len(locked),
        'live_fights': live_fights,
        'live_correct': live_correct,
        'live_accuracy': live_correct / live_fights if live_fights > 0 else 0,
        'backtest_fights': backtest_fights,
        'backtest_correct': backtest_correct,
        'backtest_accuracy': backtest_correct / backtest_fights if backtest_fights > 0 else 0,
    }


def get_ledger_events() -> Tuple[List[Dict], List[Dict]]:
    """
    Get events from the ledger, split into upcoming (unlocked) and completed (locked).

    Returns:
        Tuple of (upcoming_events, completed_events)
    """
    ledger = load_ledger()
    if not ledger:
        return [], []

    entries = ledger.get('entries', [])
    upcoming = []
    completed = []

    for entry in entries:
        # Get main event fighter info (first fight in list)
        fights = entry.get('fights', [])
        main_event = fights[0] if fights else None

        event_info = {
            'event_id': entry.get('event_id', ''),
            'event_name': entry.get('event_name', ''),
            'event_date': entry.get('event_date', ''),
            'event_slug': entry.get('event_id', ''),  # Use event_id as slug
            'location': entry.get('location', ''),
            'fight_count': entry.get('overall_total') or len(fights),
            'locked': entry.get('locked', False),
            'locked_at': entry.get('locked_at'),
            'model_version': entry.get('model_version', ''),
            'model_description': entry.get('model_description', ''),
            'prediction_type': entry.get('prediction_type', 'unknown'),
        }

        # Add main event fighter data for preview cards
        if main_event:
            f1_name = main_event.get('fighter_1', '')
            f2_name = main_event.get('fighter_2', '')
            event_info['main_event_f1'] = f1_name
            event_info['main_event_f2'] = f2_name
            event_info['main_event_f1_photo'] = get_headshot_url(f1_name)
            event_info['main_event_f2_photo'] = get_headshot_url(f2_name)
            event_info['main_event_f1_last'] = get_last_name(f1_name)
            event_info['main_event_f2_last'] = get_last_name(f2_name)
            event_info['main_event_f1_initials'] = get_initials(f1_name)
            event_info['main_event_f2_initials'] = get_initials(f2_name)

        if entry.get('locked', False):
            event_info['status'] = 'completed'
            event_info['is_backtest'] = True
            event_info['correct'] = entry.get('overall_correct', 0)
            event_info['total'] = entry.get('overall_total', 0)
            event_info['accuracy'] = entry.get('overall_accuracy', 0)
            event_info['accuracy_pct'] = f"{(entry.get('overall_accuracy', 0) or 0) * 100:.1f}"
            completed.append(event_info)
        else:
            event_info['status'] = 'upcoming'
            event_info['is_backtest'] = False
            upcoming.append(event_info)

    # Sort completed by date (most recent first)
    completed.sort(key=lambda x: x.get('event_date', ''), reverse=True)

    # Sort upcoming by date (soonest first)
    upcoming.sort(key=lambda x: x.get('event_date', ''))

    return upcoming, completed


def get_ledger_event(event_id: str) -> Optional[Dict]:
    """Get a specific event from the ledger."""
    ledger = load_ledger()
    if not ledger:
        return None

    for entry in ledger.get('entries', []):
        if entry.get('event_id') == event_id:
            return entry

    return None


def format_ledger_event(entry: Dict) -> Optional[Dict]:
    """
    Format a ledger entry for template rendering.

    Args:
        entry: Raw entry from the ledger

    Returns:
        Formatted event dict for template
    """
    if not entry:
        return None

    fights = entry.get('fights', [])
    if not fights:
        return None

    total_fights = len(fights)
    formatted_predictions = []

    # Load odds data for matching (only for unlocked/upcoming events)
    is_locked = entry.get('locked', False)
    odds_lookup = get_odds_for_fights(fights) if not is_locked else {}

    for i, fight in enumerate(fights):
        # Convert ledger fight format to format_prediction input
        # top_factors from ledger is already in dict format [{'feature': ..., 'value': ...}]
        pred = {
            'fighter1': fight.get('fighter_1', ''),
            'fighter2': fight.get('fighter_2', ''),
            'f1_win_prob': fight.get('f1_win_prob', 0.5),
            'f2_win_prob': fight.get('f2_win_prob', 0.5),
            'confidence': fight.get('confidence', 'LOW'),
            'weight_class': fight.get('weight_class', ''),
            'f1_elo': fight.get('f1_elo', 1500),
            'f2_elo': fight.get('f2_elo', 1500),
            'f1_exact_match': True,
            'f2_exact_match': True,
            'top_factors': fight.get('top_factors', []),
            # Backtest fields from ledger
            'actual_winner': fight.get('actual_winner'),
            'correct': fight.get('correct'),
            'method': fight.get('actual_method'),  # Note: ledger uses actual_method
            # Odds from ledger if available
            'odds': fight.get('odds'),
        }
        formatted_predictions.append(format_prediction(pred, i, total_fights, odds_lookup))

    # Count confidence levels
    confidence_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for fight in fights:
        conf = fight.get('confidence', 'Low')
        # Normalize confidence to uppercase
        conf_upper = conf.upper() if conf else 'LOW'
        confidence_counts[conf_upper] = confidence_counts.get(conf_upper, 0) + 1

    is_locked = entry.get('locked', False)

    result = {
        'event_name': entry.get('event_name', ''),
        'event_date': entry.get('event_date', ''),
        'event_slug': entry.get('event_id', ''),
        'location': entry.get('location', ''),
        'predictions': formatted_predictions,
        'fight_count': total_fights,
        'confidence_counts': confidence_counts,
        'is_backtest': is_locked,
        'locked': is_locked,
        'locked_at': entry.get('locked_at'),
        'model_version': entry.get('model_version', ''),
        'model_description': entry.get('model_description', ''),
        'prediction_type': entry.get('prediction_type', ''),
    }

    if is_locked:
        result['correct'] = entry.get('overall_correct', 0)
        result['total'] = entry.get('overall_total', total_fights)
        result['accuracy'] = entry.get('overall_accuracy', 0)
        result['accuracy_pct'] = f"{(entry.get('overall_accuracy', 0) or 0) * 100:.1f}"

    return result


# Create Flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.context_processor
def inject_model_info():
    """Inject model version info into all templates."""
    from src.predict.serve import MODEL_VERSION
    if MODEL_VERSION == 'phase9':
        model_version = 'Phase 9'
        model_accuracy = '60.2%'
    elif MODEL_VERSION == 'phase8':
        model_version = 'Phase 8'
        model_accuracy = '60.5%'
    else:
        model_version = 'v1'
        model_accuracy = '58.9%'
    return {
        'model_version': model_version,
        'model_accuracy': model_accuracy,
    }


@app.route('/')
def index():
    """Homepage - next upcoming event predictions."""
    data = get_upcoming_predictions()

    # Get other upcoming events for sidebar
    upcoming_events, completed_events = get_all_events()

    # Filter out the current event from upcoming list
    if data and upcoming_events:
        current_slug = data.get('event_slug', '')
        upcoming_events = [e for e in upcoming_events if e.get('event_slug') != current_slug]

    if data is None:
        return render_template('index.html', error="No upcoming events found. Try checking back later.",
                               upcoming_events=upcoming_events[:5])

    return render_template('index.html', event=data, upcoming_events=upcoming_events[:5])


@app.route('/event/<event_slug>')
def event_detail(event_slug: str):
    """Display predictions for a specific event."""
    # First try the ledger
    ledger_entry = get_ledger_event(event_slug)

    if ledger_entry:
        formatted = format_ledger_event(ledger_entry)
        if formatted:
            return render_template('event.html', event=formatted)

    # Try events.json
    event_data = get_event_from_json(event_slug)

    if event_data:
        # Format for template
        formatted = format_event_from_json(event_data)
        if formatted:
            return render_template('event.html', event=formatted)

    # Fall back to cache file
    data = get_cached_predictions(event_slug)

    if data is None:
        return render_template('event.html', error=f"Event '{event_slug}' not found.")

    return render_template('event.html', event=data)


@app.route('/upcoming')
def upcoming():
    """
    List of all upcoming events with predictions.

    Shows all scheduled UFC events with predictions already generated,
    sorted by date (soonest first).
    """
    # Get upcoming events from ledger
    upcoming_events, _ = get_ledger_events()

    return render_template(
        'upcoming.html',
        upcoming=upcoming_events
    )


@app.route('/archive')
@app.route('/events')
def archive():
    """
    List of all completed events with results (track record).

    Shows:
    - Running stats banner (total accuracy, events tracked, current model)
    - Completed events only (locked with results)
    """
    # Try ledger first
    _, completed = get_ledger_events()
    stats = get_ledger_stats()

    # Get model info
    registry = load_model_registry()
    current_model = registry.get('current', 'unknown') if registry else 'unknown'

    # If ledger is empty, fall back to events.json
    if not completed:
        events = get_archived_events()
        return render_template('archive.html', events=events, stats=None)

    return render_template(
        'archive.html',
        completed=completed,
        stats=stats,
        current_model=current_model
    )


@app.route('/backtest/<date_str>')
def backtest(date_str: str):
    """Run backtest for a specific date."""
    data = get_backtest_predictions(date_str)

    if data is None:
        return render_template('event.html', error=f"No events found for date: {date_str}")

    # Cache the backtest results
    save_predictions_cache(data['event_slug'], data)

    return render_template('event.html', event=data)


@app.route('/methodology')
def methodology():
    """Methodology summary page."""
    return render_template('methodology.html')


@app.route('/about')
def about():
    """Redirect to methodology page."""
    return redirect('/methodology')


@app.route('/api/predict/next')
def api_predict_next():
    """API: Get predictions for next upcoming event."""
    data = get_upcoming_predictions()

    if data is None:
        return jsonify({'error': 'No upcoming events found'}), 404

    return jsonify(data)


@app.route('/api/predict/<event_slug>')
def api_predict_event(event_slug: str):
    """API: Get predictions for a specific event."""
    data = get_cached_predictions(event_slug)

    if data is None:
        return jsonify({'error': f"Event '{event_slug}' not found"}), 404

    return jsonify(data)


@app.route('/api/backtest/<date_str>')
def api_backtest(date_str: str):
    """API: Run backtest for a specific date."""
    data = get_backtest_predictions(date_str)

    if data is None:
        return jsonify({'error': f"No events found for date: {date_str}"}), 404

    return jsonify(data)


@app.template_filter('format_date')
def format_date_filter(date_str):
    """Template filter to format date strings."""
    if not date_str:
        return ''
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%B %d, %Y')
    except Exception:
        return date_str


@app.template_filter('format_date_long')
def format_date_long_filter(date_str):
    """Template filter to format date strings with day of week."""
    if not date_str:
        return ''
    try:
        # Handle various date formats
        if 'T' in str(date_str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = datetime.strptime(str(date_str), '%Y-%m-%d')
        # Use %d and strip leading zero manually for cross-platform compatibility
        day = dt.day
        return f"{dt.strftime('%A, %B')} {day}, {dt.year}"
    except Exception:
        try:
            # Try parsing as just date
            dt = datetime.strptime(str(date_str), '%Y-%m-%d')
            day = dt.day
            return f"{dt.strftime('%A, %B')} {day}, {dt.year}"
        except Exception:
            return date_str


@app.template_filter('format_ml')
def format_moneyline_filter(ml):
    """Template filter to format American moneyline odds with +/- prefix."""
    if ml is None:
        return 'N/A'
    try:
        ml = int(ml)
        if ml >= 0:
            return f"+{ml}"
        return str(ml)
    except (ValueError, TypeError):
        return 'N/A'


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))

    print("\n" + "=" * 60)
    print("UFC FIGHT PREDICTOR - Web Interface")
    print("=" * 60)
    print(f"Starting server at http://127.0.0.1:{port}")
    print("Press Ctrl+C to quit")
    print("=" * 60 + "\n")

    app.run(debug=True, host='127.0.0.1', port=port)
