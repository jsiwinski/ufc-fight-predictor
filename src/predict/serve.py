#!/usr/bin/env python3
"""
UFC Fight Prediction Serving Pipeline.

Generates win probability predictions for upcoming UFC events.

Usage:
    python src/predict/serve.py                    # Predict next upcoming event
    python src/predict/serve.py --event-url URL    # Predict specific event
    python src/predict/serve.py --no-scrape        # Use cached data only
    python src/predict/serve.py --backtest DATE    # Backtest against completed event
    python src/predict/serve.py --format json      # Output JSON instead of CSV

Example:
    python src/predict/serve.py
    python src/predict/serve.py --backtest 2026-01-31
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.etl.scraper import UFCDataScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Paths
MODEL_PATH = 'data/models/ufc_model_v1.pkl'
FEATURE_NAMES_PATH = 'data/models/feature_names_v1.json'
PROCESSED_DATA_PATH = 'data/processed/ufc_fights_features_v1.csv'
RAW_DATA_PATH = 'data/raw/ufc_fights_v1.csv'
PREDICTIONS_DIR = 'data/predictions'

# Metadata columns (not features)
METADATA_COLS = [
    'fight_id', 'fight_date', 'event_name', 'weight_class',
    'f1_name', 'f2_name', 'method', 'round'
]

# Target column
TARGET_COL = 'f1_is_winner'

# Confidence thresholds
CONFIDENCE_HIGH = 0.65
CONFIDENCE_MEDIUM = 0.55

# Weight classes for one-hot encoding
WEIGHT_CLASSES = [
    'Bantamweight', 'Catch Weight', 'Featherweight', 'Flyweight',
    'Heavyweight', 'Light Heavyweight', 'Lightweight', 'Middleweight',
    'Open Weight', 'Super Heavyweight', 'Welterweight',
    "Women's Bantamweight", "Women's Featherweight",
    "Women's Flyweight", "Women's Strawweight"
]

# Default imputation values (for debut fighters)
DEBUT_DEFAULTS = {
    'career_win_rate': 0.52,
    'career_ko_rate': 0.25,
    'career_sub_rate': 0.15,
    'career_dec_rate': 0.60,
    'career_finish_rate': 0.40,
    'days_since_last_fight': 365.0,
    'fights_per_year': 2.0,
}


# =============================================================================
# Fighter Matching
# =============================================================================

def normalize_name(name: str) -> str:
    """Normalize fighter name for matching."""
    if not name:
        return ""
    # Lowercase, strip whitespace, remove extra spaces
    name = ' '.join(name.lower().strip().split())
    # Remove common suffixes/prefixes
    name = name.replace("'", "").replace("-", " ").replace(".", "")
    return name


def fuzzy_match_score(name1: str, name2: str) -> float:
    """Calculate fuzzy match score between two names."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    return SequenceMatcher(None, n1, n2).ratio()


def find_fighter_match(
    fighter_name: str,
    known_fighters: List[str],
    threshold: float = 0.85
) -> Tuple[Optional[str], float]:
    """
    Find best matching fighter name from known fighters.

    Args:
        fighter_name: Name to match
        known_fighters: List of known fighter names
        threshold: Minimum match score required

    Returns:
        Tuple of (matched_name or None, match_score)
    """
    best_match = None
    best_score = 0.0

    normalized_target = normalize_name(fighter_name)

    for known in known_fighters:
        # Exact match (case insensitive)
        if normalize_name(known) == normalized_target:
            return known, 1.0

        # Fuzzy match
        score = fuzzy_match_score(fighter_name, known)
        if score > best_score:
            best_score = score
            best_match = known

    if best_score >= threshold:
        return best_match, best_score

    return None, best_score


# =============================================================================
# Feature Engineering for Predictions
# =============================================================================

class FighterFeatures:
    """Manages fighter feature lookup and computation."""

    def __init__(self, processed_data_path: str, raw_data_path: str):
        """Load historical data for feature computation."""
        logger.info(f"Loading processed data from {processed_data_path}")
        self.processed_df = pd.read_csv(processed_data_path)
        self.processed_df['fight_date'] = pd.to_datetime(self.processed_df['fight_date'])

        logger.info(f"Loading raw data from {raw_data_path}")
        self.raw_df = pd.read_csv(raw_data_path)

        # Build fighter index: fighter_name -> most recent features
        self._build_fighter_index()

        logger.info(f"Loaded {len(self.fighter_index)} fighters with historical data")

    def _build_fighter_index(self):
        """Build index of fighters to their most recent features."""
        self.fighter_index = {}
        self.known_fighters = set()

        # Get all unique fighters
        f1_names = self.processed_df['f1_name'].unique()
        f2_names = self.processed_df['f2_name'].unique()
        all_fighters = set(f1_names) | set(f2_names)
        self.known_fighters = list(all_fighters)

        for fighter in all_fighters:
            # Get most recent fight where this fighter was f1 or f2
            f1_fights = self.processed_df[self.processed_df['f1_name'] == fighter]
            f2_fights = self.processed_df[self.processed_df['f2_name'] == fighter]

            # Get most recent as f1
            if len(f1_fights) > 0:
                most_recent_f1 = f1_fights.sort_values('fight_date').iloc[-1]
                f1_features = self._extract_fighter_features(most_recent_f1, prefix='f1')
                f1_date = most_recent_f1['fight_date']
            else:
                f1_features = None
                f1_date = None

            # Get most recent as f2
            if len(f2_fights) > 0:
                most_recent_f2 = f2_fights.sort_values('fight_date').iloc[-1]
                f2_features = self._extract_fighter_features(most_recent_f2, prefix='f2')
                f2_date = most_recent_f2['fight_date']
            else:
                f2_features = None
                f2_date = None

            # Use the more recent one
            if f1_date is not None and f2_date is not None:
                if f1_date >= f2_date:
                    features = f1_features
                    last_fight_date = f1_date
                else:
                    features = f2_features
                    last_fight_date = f2_date
            elif f1_date is not None:
                features = f1_features
                last_fight_date = f1_date
            elif f2_date is not None:
                features = f2_features
                last_fight_date = f2_date
            else:
                features = None
                last_fight_date = None

            if features is not None:
                self.fighter_index[fighter] = {
                    'features': features,
                    'last_fight_date': last_fight_date,
                    'fight_count': len(f1_fights) + len(f2_fights)
                }

    def _extract_fighter_features(self, row: pd.Series, prefix: str) -> Dict:
        """
        Extract fighter-level features from a matchup row.

        Args:
            row: Row from processed data
            prefix: 'f1' or 'f2'

        Returns:
            Dictionary of feature_name -> value (without prefix)
        """
        features = {}

        # All f1_ or f2_ columns (excluding name and is_winner)
        for col in row.index:
            if col.startswith(f'{prefix}_') and col != f'{prefix}_name' and col != f'{prefix}_is_winner':
                # Store without prefix
                feature_name = col[3:]  # Remove 'f1_' or 'f2_'
                features[feature_name] = row[col]

        return features

    def get_fighter_features(
        self,
        fighter_name: str,
        prediction_date: Optional[datetime] = None
    ) -> Tuple[Dict, bool, str]:
        """
        Get features for a fighter.

        The features returned are from the fighter's most recent fight in the
        processed data, representing their stats going INTO that fight.

        For live predictions on future fights, the days_since_last_fight will
        be recalculated based on the prediction date. For backtests on historical
        fights that are in the processed data, the original value is preserved.

        Args:
            fighter_name: Fighter name
            prediction_date: Date of prediction (for days_since calculation)

        Returns:
            Tuple of (features dict, is_match_exact, matched_name)
        """
        if prediction_date is None:
            prediction_date = datetime.now()

        # Try exact match first
        if fighter_name in self.fighter_index:
            features = self.fighter_index[fighter_name]['features'].copy()
            last_fight = self.fighter_index[fighter_name]['last_fight_date']

            # Only recalculate days_since_last_fight for FUTURE predictions
            # (when prediction_date is after the most recent fight in our data)
            if last_fight is not None and prediction_date > last_fight:
                days_since = (prediction_date - last_fight).days
                features['days_since_last_fight'] = float(days_since)
            # Otherwise, keep the original value from processed data
            # (it correctly represents days since the PREVIOUS fight)

            return features, True, fighter_name

        # Try fuzzy match
        matched_name, score = find_fighter_match(fighter_name, self.known_fighters)

        if matched_name is not None:
            logger.warning(f"Fuzzy match: '{fighter_name}' -> '{matched_name}' (score: {score:.2f})")
            features = self.fighter_index[matched_name]['features'].copy()
            last_fight = self.fighter_index[matched_name]['last_fight_date']

            # Only recalculate days_since_last_fight for FUTURE predictions
            if last_fight is not None and prediction_date > last_fight:
                days_since = (prediction_date - last_fight).days
                features['days_since_last_fight'] = float(days_since)

            return features, False, matched_name

        # No match - return debut defaults
        logger.warning(f"No match found for '{fighter_name}' - using debut defaults")
        return self._get_debut_features(), False, fighter_name

    def _get_debut_features(self) -> Dict:
        """Get default features for debut fighters."""
        features = {}

        # Career stats with defaults
        features['career_fights'] = 0
        features['career_wins'] = 0
        features['career_losses'] = 0
        features['career_win_rate'] = DEBUT_DEFAULTS['career_win_rate']
        features['career_ko_rate'] = DEBUT_DEFAULTS['career_ko_rate']
        features['career_sub_rate'] = DEBUT_DEFAULTS['career_sub_rate']
        features['career_dec_rate'] = DEBUT_DEFAULTS['career_dec_rate']
        features['career_finish_rate'] = DEBUT_DEFAULTS['career_finish_rate']

        features['win_streak'] = 0
        features['loss_streak'] = 0
        features['days_since_last_fight'] = DEBUT_DEFAULTS['days_since_last_fight']
        features['fights_per_year'] = DEBUT_DEFAULTS['fights_per_year']
        features['weight_class_fights'] = 0
        features['fights_last_12mo'] = 0.0

        features['is_ufc_debut'] = 1
        features['data_completeness_score'] = 0.0
        features['has_detailed_stats'] = False

        # Rolling window features (all zeros/defaults)
        for window in [3, 5, 10]:
            features[f'win_rate_last_{window}'] = DEBUT_DEFAULTS['career_win_rate']
            features[f'finish_rate_last_{window}'] = DEBUT_DEFAULTS['career_finish_rate']
            features[f'avg_sig_strikes_landed_last_{window}'] = 0.0
            features[f'avg_sig_strikes_absorbed_last_{window}'] = 0.0
            features[f'avg_takedowns_last_{window}'] = 0.0
            features[f'avg_fight_time_last_{window}'] = 0.0

        features['striking_volume'] = 0.0
        features['grappling_tendency'] = 0.0
        features['experience_ratio'] = 0.0
        features['is_coming_off_layoff'] = 1
        features['is_new_to_weight_class'] = 1

        return features


def build_matchup_features(
    f1_features: Dict,
    f2_features: Dict,
    weight_class: str,
    fight_date: datetime,
    event_name: str,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Build a single-row feature vector for a matchup.

    Args:
        f1_features: Fighter 1 features (without prefix)
        f2_features: Fighter 2 features (without prefix)
        weight_class: Weight class of the fight
        fight_date: Date of the fight
        event_name: Name of the event
        feature_names: List of feature names model expects

    Returns:
        DataFrame with single row matching model's expected features
    """
    matchup = {}

    # Build f1_, f2_, and diff_ features
    all_feature_names = set()
    for key in f1_features.keys():
        all_feature_names.add(key)
    for key in f2_features.keys():
        all_feature_names.add(key)

    for feat in all_feature_names:
        f1_val = f1_features.get(feat, 0)
        f2_val = f2_features.get(feat, 0)

        # Handle NaN
        if pd.isna(f1_val):
            f1_val = 0
        if pd.isna(f2_val):
            f2_val = 0

        matchup[f'f1_{feat}'] = f1_val
        matchup[f'f2_{feat}'] = f2_val

        # Create differential for numeric features
        if isinstance(f1_val, (int, float, np.number)) and isinstance(f2_val, (int, float, np.number)):
            matchup[f'diff_{feat}'] = f1_val - f2_val

    # Calculate experience ratio
    total_fights = f1_features.get('career_fights', 0) + f2_features.get('career_fights', 0) + 1
    matchup['f1_experience_ratio'] = f1_features.get('career_fights', 0) / total_fights
    matchup['f2_experience_ratio'] = f2_features.get('career_fights', 0) / total_fights

    # Momentum differential
    f1_momentum = f1_features.get('win_streak', 0) - f1_features.get('loss_streak', 0)
    f2_momentum = f2_features.get('win_streak', 0) - f2_features.get('loss_streak', 0)
    matchup['diff_momentum'] = f1_momentum - f2_momentum

    # Temporal features
    matchup['is_title_fight'] = 1 if 'title' in event_name.lower() else 0
    matchup['fight_year'] = fight_date.year
    matchup['fight_month'] = fight_date.month

    # Fight time (placeholder - will be 0 for predictions)
    matchup['fight_time_seconds'] = 0

    # Weight class one-hot encoding
    for wc in WEIGHT_CLASSES:
        col_name = f'wc_{wc}'
        matchup[col_name] = 1 if wc.lower() in weight_class.lower() else 0

    # Handle "title bout" variants
    if 'title' in weight_class.lower():
        # Extract actual weight class from title bout
        for wc in WEIGHT_CLASSES:
            if wc.lower() in weight_class.lower().replace(' title bout', ''):
                matchup[f'wc_{wc}'] = 1
                break

    # Create DataFrame with exact feature names in order
    result = {}
    for feat in feature_names:
        if feat in matchup:
            result[feat] = matchup[feat]
        else:
            # Missing feature - default to 0
            result[feat] = 0

    return pd.DataFrame([result])


# =============================================================================
# Prediction Generation
# =============================================================================

def get_confidence_level(probability: float) -> str:
    """Determine confidence level from probability."""
    prob_away_from_50 = abs(probability - 0.5)
    if prob_away_from_50 >= (CONFIDENCE_HIGH - 0.5):
        return 'HIGH'
    elif prob_away_from_50 >= (CONFIDENCE_MEDIUM - 0.5):
        return 'MEDIUM'
    else:
        return 'LOW'


def get_top_factors(
    feature_vector: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Get top factors influencing the prediction.

    For now, uses the differential features with largest absolute values.
    SHAP would be better but may not be compatible with HistGradientBoosting.
    """
    diff_features = [f for f in feature_names if f.startswith('diff_')]

    factors = []
    for feat in diff_features:
        if feat in feature_vector.columns:
            val = feature_vector[feat].iloc[0]
            if not pd.isna(val) and val != 0:
                factors.append((feat, val))

    # Sort by absolute value
    factors.sort(key=lambda x: abs(x[1]), reverse=True)

    return factors[:top_n]


def format_factor(factor_name: str, value: float) -> str:
    """Format a factor for display."""
    # Clean up feature name
    name = factor_name.replace('diff_', '').replace('_', ' ')

    if value > 0:
        return f"+{value:.2f} {name}"
    else:
        return f"{value:.2f} {name}"


# =============================================================================
# Main Pipeline
# =============================================================================

class PredictionPipeline:
    """Main prediction pipeline."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        feature_names_path: str = FEATURE_NAMES_PATH,
        processed_data_path: str = PROCESSED_DATA_PATH,
        raw_data_path: str = RAW_DATA_PATH
    ):
        """Initialize pipeline with model and data."""
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        # Load feature names
        logger.info(f"Loading feature names from {feature_names_path}")
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
        logger.info(f"Model expects {len(self.feature_names)} features")

        # Load fighter features
        self.fighter_features = FighterFeatures(processed_data_path, raw_data_path)

        # Scraper for upcoming events
        self.scraper = None

    def _get_scraper(self) -> UFCDataScraper:
        """Get or create scraper instance."""
        if self.scraper is None:
            self.scraper = UFCDataScraper({'rate_limit': 2})
        return self.scraper

    def scrape_upcoming_event(self, event_url: Optional[str] = None) -> pd.DataFrame:
        """
        Scrape upcoming event data.

        Args:
            event_url: Specific event URL, or None for next upcoming

        Returns:
            DataFrame with upcoming fights
        """
        scraper = self._get_scraper()

        if event_url:
            logger.info(f"Scraping specific event: {event_url}")
            # Parse event URL to get event info
            # For now, scrape upcoming and filter
            upcoming_df = scraper.scrape_upcoming_events()
            # Filter by URL if possible
            if 'event_url' in upcoming_df.columns:
                event_fights = upcoming_df[upcoming_df['event_url'] == event_url]
                if len(event_fights) > 0:
                    return event_fights
            return upcoming_df.head(20)  # Return first event's fights
        else:
            logger.info("Scraping upcoming events")
            upcoming_df = scraper.scrape_upcoming_events()

            if len(upcoming_df) == 0:
                logger.warning("No upcoming events found")
                return pd.DataFrame()

            # Get the first (next) event
            first_event = upcoming_df['event_name'].iloc[0]
            event_fights = upcoming_df[upcoming_df['event_name'] == first_event]

            logger.info(f"Found {len(event_fights)} fights for {first_event}")
            return event_fights

    def predict_fight(
        self,
        fighter1_name: str,
        fighter2_name: str,
        weight_class: str,
        event_name: str,
        fight_date: Optional[datetime] = None
    ) -> Dict:
        """
        Generate prediction for a single fight.

        Returns:
            Dictionary with prediction results
        """
        if fight_date is None:
            fight_date = datetime.now()

        # Get fighter features
        f1_features, f1_exact, f1_matched = self.fighter_features.get_fighter_features(
            fighter1_name, fight_date
        )
        f2_features, f2_exact, f2_matched = self.fighter_features.get_fighter_features(
            fighter2_name, fight_date
        )

        # Build feature vector
        feature_vector = build_matchup_features(
            f1_features, f2_features,
            weight_class, fight_date, event_name,
            self.feature_names
        )

        # Generate prediction
        proba = self.model.predict_proba(feature_vector)[0]
        f1_win_prob = proba[1]  # Probability of f1 winning
        f2_win_prob = proba[0]  # Probability of f2 winning (1 - f1)

        # Determine predicted winner
        if f1_win_prob > 0.5:
            predicted_winner = fighter1_name
            win_prob = f1_win_prob
        else:
            predicted_winner = fighter2_name
            win_prob = f2_win_prob

        confidence = get_confidence_level(max(f1_win_prob, f2_win_prob))

        # Get top factors
        top_factors = get_top_factors(feature_vector, self.feature_names)

        return {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'fighter1_matched': f1_matched,
            'fighter2_matched': f2_matched,
            'f1_win_prob': f1_win_prob,
            'f2_win_prob': f2_win_prob,
            'predicted_winner': predicted_winner,
            'win_probability': win_prob,
            'confidence': confidence,
            'weight_class': weight_class,
            'top_factors': top_factors,
            'f1_exact_match': f1_exact,
            'f2_exact_match': f2_exact,
        }

    def predict_event(
        self,
        event_fights: pd.DataFrame,
        event_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Generate predictions for all fights in an event.

        Args:
            event_fights: DataFrame with event fights
            event_date: Date of event

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        # Parse event date if string
        if event_date is None:
            if 'event_date' in event_fights.columns and len(event_fights) > 0:
                date_str = event_fights['event_date'].iloc[0]
                try:
                    event_date = pd.to_datetime(date_str)
                except:
                    event_date = datetime.now()
            else:
                event_date = datetime.now()

        event_name = event_fights['event_name'].iloc[0] if 'event_name' in event_fights.columns else 'Unknown Event'

        for idx, row in event_fights.iterrows():
            fighter1 = row.get('fighter1_name', row.get('red_fighter_name', ''))
            fighter2 = row.get('fighter2_name', row.get('blue_fighter_name', ''))
            weight_class = row.get('weight_class', 'Unknown')

            if not fighter1 or not fighter2:
                logger.warning(f"Skipping fight with missing fighter names")
                continue

            try:
                pred = self.predict_fight(
                    fighter1, fighter2,
                    weight_class, event_name,
                    event_date
                )
                pred['event_name'] = event_name
                pred['event_date'] = event_date.strftime('%Y-%m-%d') if isinstance(event_date, datetime) else str(event_date)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting {fighter1} vs {fighter2}: {e}")
                continue

        return predictions

    def backtest_event(self, cutoff_date: str) -> List[Dict]:
        """
        Backtest predictions against a completed event.

        Args:
            cutoff_date: Date string (YYYY-MM-DD) for event to backtest

        Returns:
            List of predictions with actual results
        """
        logger.info(f"Backtesting against events on/after {cutoff_date}")

        # Load processed data and filter to cutoff date
        df = self.fighter_features.processed_df
        cutoff = pd.to_datetime(cutoff_date)

        # Get fights on or just after cutoff date
        test_fights = df[df['fight_date'] >= cutoff].sort_values('fight_date')

        if len(test_fights) == 0:
            logger.warning(f"No fights found on or after {cutoff_date}")
            return []

        # Get first event
        first_event = test_fights['event_name'].iloc[0]
        event_fights = test_fights[test_fights['event_name'] == first_event]
        event_date = event_fights['fight_date'].iloc[0]

        logger.info(f"Backtesting against {first_event} ({len(event_fights)} fights)")

        predictions = []
        for idx, row in event_fights.iterrows():
            # Get pre-fight features (need to recompute based on fights before this date)
            # For simplicity, we'll use the current features but this is slightly leaky
            # A proper backtest would recompute features excluding this fight

            fighter1 = row['f1_name']
            fighter2 = row['f2_name']
            weight_class = row['weight_class']
            actual_winner = row['f1_is_winner']

            try:
                pred = self.predict_fight(
                    fighter1, fighter2,
                    weight_class, first_event,
                    event_date
                )
                pred['event_name'] = first_event
                pred['event_date'] = event_date.strftime('%Y-%m-%d')
                pred['actual_f1_won'] = int(actual_winner)
                pred['actual_winner'] = fighter1 if actual_winner == 1 else fighter2
                pred['correct'] = (pred['predicted_winner'] == pred['actual_winner'])
                pred['method'] = row.get('method', 'Unknown')

                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting {fighter1} vs {fighter2}: {e}")
                continue

        return predictions

    def close(self):
        """Clean up resources."""
        if self.scraper:
            self.scraper.close()


# =============================================================================
# Output Formatting
# =============================================================================

def print_predictions(predictions: List[Dict], is_backtest: bool = False):
    """Print predictions in formatted output."""
    if not predictions:
        print("No predictions to display.")
        return

    event_name = predictions[0].get('event_name', 'Unknown Event')
    event_date = predictions[0].get('event_date', '')

    print()
    print("=" * 70)
    print(f"{event_name} — {event_date}")
    print("=" * 70)

    correct_count = 0
    total_count = 0

    for i, pred in enumerate(predictions):
        f1 = pred['fighter1']
        f2 = pred['fighter2']
        f1_prob = pred['f1_win_prob']
        f2_prob = pred['f2_win_prob']
        confidence = pred['confidence']
        factors = pred.get('top_factors', [])

        # Format probabilities
        f1_pct = f"{f1_prob*100:.1f}%"
        f2_pct = f"{f2_prob*100:.1f}%"

        # Determine favorite
        if f1_prob > f2_prob:
            line = f"{f1} ({f1_pct}) vs {f2} ({f2_pct}) [{confidence}]"
        else:
            line = f"{f1} ({f1_pct}) vs {f2} ({f2_pct}) [{confidence}]"

        # Add fight position indicator
        if i == 0:
            position = "Main Event"
        elif i == 1:
            position = "Co-Main"
        else:
            position = f"Fight {len(predictions) - i}"

        print(f"\n{position}: {line}")

        # Print top factors
        if factors:
            factor_strs = [format_factor(f[0], f[1]) for f in factors[:3]]
            print(f"  Key: {', '.join(factor_strs)}")

        # Print match warnings
        if not pred.get('f1_exact_match', True):
            print(f"  ⚠️  {f1} matched to '{pred['fighter1_matched']}'")
        if not pred.get('f2_exact_match', True):
            print(f"  ⚠️  {f2} matched to '{pred['fighter2_matched']}'")

        # Print backtest results
        if is_backtest and 'actual_winner' in pred:
            actual = pred['actual_winner']
            correct = pred['correct']
            method = pred.get('method', '')

            if correct:
                correct_count += 1
                status = "✓ CORRECT"
            else:
                status = "✗ WRONG"

            total_count += 1
            print(f"  {status} — Actual: {actual} ({method})")

    # Print backtest summary
    if is_backtest and total_count > 0:
        accuracy = correct_count / total_count
        print()
        print("-" * 70)
        print(f"BACKTEST RESULTS: {correct_count}/{total_count} correct ({accuracy:.1%})")
        print("-" * 70)


def save_predictions(predictions: List[Dict], output_path: str, format: str = 'csv'):
    """Save predictions to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        # Convert factors to string for JSON serialization
        for pred in predictions:
            if 'top_factors' in pred:
                pred['top_factors'] = [
                    {'feature': f[0], 'value': f[1]}
                    for f in pred['top_factors']
                ]
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
    else:
        # CSV format
        df = pd.DataFrame(predictions)

        # Flatten top_factors to string
        if 'top_factors' in df.columns:
            df['top_factors'] = df['top_factors'].apply(
                lambda x: '; '.join([f"{f[0]}:{f[1]:.3f}" for f in x]) if x else ''
            )

        df.to_csv(output_path, index=False)

    logger.info(f"Saved predictions to {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate UFC fight predictions'
    )
    parser.add_argument(
        '--event-url',
        type=str,
        help='Specific event URL to predict'
    )
    parser.add_argument(
        '--no-scrape',
        action='store_true',
        help='Skip scraping, use cached data only'
    )
    parser.add_argument(
        '--backtest',
        type=str,
        help='Backtest against events on/after this date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'json'],
        default='csv',
        help='Output format'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (auto-generated if not specified)'
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("UFC FIGHT PREDICTION PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: HistGradientBoostingClassifier")
    print()

    # Initialize pipeline
    pipeline = PredictionPipeline()

    try:
        if args.backtest:
            # Backtest mode
            predictions = pipeline.backtest_event(args.backtest)
            is_backtest = True
        elif args.no_scrape:
            # No scrape mode - just load most recent processed data
            logger.error("--no-scrape requires --backtest to specify which data to use")
            sys.exit(1)
        else:
            # Live prediction mode
            upcoming_fights = pipeline.scrape_upcoming_event(args.event_url)

            if len(upcoming_fights) == 0:
                print("No upcoming fights found. Try --backtest mode instead.")
                sys.exit(1)

            predictions = pipeline.predict_event(upcoming_fights)
            is_backtest = False

        # Print predictions
        print_predictions(predictions, is_backtest=is_backtest)

        # Save predictions
        if predictions:
            if args.output:
                output_path = args.output
            else:
                event_name = predictions[0].get('event_name', 'unknown').replace(' ', '_').replace(':', '')
                event_date = predictions[0].get('event_date', datetime.now().strftime('%Y%m%d'))
                ext = 'json' if args.format == 'json' else 'csv'
                output_path = f"{PREDICTIONS_DIR}/predictions_{event_name}_{event_date}.{ext}"

            save_predictions(predictions, output_path, args.format)
            print(f"\nPredictions saved to: {output_path}")

    finally:
        pipeline.close()


if __name__ == '__main__':
    main()
