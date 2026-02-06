#!/usr/bin/env python3
"""
Flask web application for UFC fight predictions.

Routes:
    /                   - Next upcoming event predictions
    /event/<slug>       - Specific event predictions
    /archive            - Past predicted events with results
    /about              - Link to methodology
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
}

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict.serve import PredictionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PREDICTIONS_DIR = PROJECT_ROOT / 'data' / 'predictions'
CACHE_MAX_AGE_HOURS = 24

# Global pipeline (loaded once)
_pipeline: Optional[PredictionPipeline] = None


def get_pipeline() -> PredictionPipeline:
    """Get or create the prediction pipeline (singleton)."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading prediction pipeline...")
        _pipeline = PredictionPipeline(
            model_path=str(PROJECT_ROOT / 'data' / 'models' / 'ufc_model_v1.pkl'),
            feature_names_path=str(PROJECT_ROOT / 'data' / 'models' / 'feature_names_v1.json'),
            processed_data_path=str(PROJECT_ROOT / 'data' / 'processed' / 'ufc_fights_features_v1.csv'),
            raw_data_path=str(PROJECT_ROOT / 'data' / 'raw' / 'ufc_fights_v1.csv')
        )
        logger.info("Pipeline loaded successfully")
    return _pipeline


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


def format_prediction(pred: Dict, position: int = 0, total_fights: int = 13) -> Dict:
    """Format a prediction dict for template rendering."""
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
        # Backtest fields
        'actual_winner': pred.get('actual_winner'),
        'correct': pred.get('correct'),
        'method': pred.get('method'),
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

        # Format predictions
        total = len(predictions)
        formatted = [format_prediction(p, i, total) for i, p in enumerate(predictions)]

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

        # Format predictions
        total = len(predictions)
        formatted = [format_prediction(p, i, total) for i, p in enumerate(predictions)]

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


def get_archived_events() -> List[Dict]:
    """Get list of archived events with cached predictions."""
    events = []

    if not PREDICTIONS_DIR.exists():
        return events

    for cache_file in sorted(PREDICTIONS_DIR.glob('*.json'), reverse=True):
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


# Create Flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index():
    """Homepage - next upcoming event predictions."""
    data = get_upcoming_predictions()

    if data is None:
        return render_template('index.html', error="No upcoming events found. Try checking back later.")

    return render_template('index.html', event=data)


@app.route('/event/<event_slug>')
def event_detail(event_slug: str):
    """Display predictions for a specific event."""
    data = get_cached_predictions(event_slug)

    if data is None:
        return render_template('event.html', error=f"Event '{event_slug}' not found in cache.")

    return render_template('event.html', event=data)


@app.route('/archive')
def archive():
    """List of past predicted events."""
    events = get_archived_events()
    return render_template('archive.html', events=events)


@app.route('/backtest/<date_str>')
def backtest(date_str: str):
    """Run backtest for a specific date."""
    data = get_backtest_predictions(date_str)

    if data is None:
        return render_template('event.html', error=f"No events found for date: {date_str}")

    # Cache the backtest results
    save_predictions_cache(data['event_slug'], data)

    return render_template('event.html', event=data)


@app.route('/about')
def about():
    """Redirect to methodology document."""
    return redirect('/static/methodology.html')


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
