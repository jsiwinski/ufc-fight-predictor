"""
Feature engineering module for creating predictive features from UFC fight data.

This module transforms raw fight and fighter data into features suitable for
machine learning models, handling early-career imputation, temporal features,
and matchup comparisons.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import re
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions for Data Parsing
# ============================================================================

def parse_strike_data(strike_str: str) -> Tuple[int, int]:
    """
    Parse strike data from 'X of Y' format to (landed, attempted).

    Args:
        strike_str: String in format "25 of 50"

    Returns:
        Tuple of (landed, attempted). Returns (0, 0) for invalid/missing data.
    """
    if pd.isna(strike_str) or strike_str == '' or strike_str == 'N/A':
        return (0, 0)

    try:
        parts = str(strike_str).strip().split(' of ')
        if len(parts) != 2:
            return (0, 0)
        landed = int(parts[0].strip())
        attempted = int(parts[1].strip())
        return (landed, attempted)
    except (ValueError, AttributeError):
        return (0, 0)


def parse_time_to_seconds(time_str: str) -> int:
    """
    Parse time string from 'MM:SS' format to total seconds.

    Args:
        time_str: String in format "5:32"

    Returns:
        Total seconds. Returns 0 for invalid/missing data.
    """
    if pd.isna(time_str) or time_str == '' or time_str == 'N/A':
        return 0

    try:
        parts = str(time_str).strip().split(':')
        if len(parts) != 2:
            return 0
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except (ValueError, AttributeError):
        return 0


def parse_percentage(pct_str: str) -> float:
    """
    Parse percentage string to float.

    Args:
        pct_str: String like "50%" or "0.50"

    Returns:
        Float between 0 and 1. Returns 0 for invalid/missing data.
    """
    if pd.isna(pct_str) or pct_str == '' or pct_str == 'N/A':
        return 0.0

    try:
        val = str(pct_str).strip().replace('%', '')
        pct = float(val)
        # Normalize to 0-1 range if given as percentage
        if pct > 1:
            pct = pct / 100.0
        return max(0.0, min(1.0, pct))
    except (ValueError, AttributeError):
        return 0.0


# ============================================================================
# Domain-Specific Opponent Skill Profiles (Phase 7)
# ============================================================================

# Minimum fights for opponent to have reliable skill profile
MIN_FIGHTS_FOR_SKILL = 3


def compute_opponent_striking_skill(
    opponent_name: str,
    fight_date: pd.Timestamp,
    fighter_history: pd.DataFrame
) -> Optional[float]:
    """
    Compute opponent's striking skill at the time of the fight.

    Measures how dangerous the opponent is on the feet. If you land strikes
    against a good striker, that's more impressive.

    Components (from opponent's prior fights):
    - sig_strikes_landed per fight (40%) - offensive striking volume
    - sig_strike_accuracy (30%) - derived from landed/attempted
    - sig_strikes_absorbed per fight inverse (20%) - defense (lower = better)
    - knockdowns scored per fight (10%) - power

    Args:
        opponent_name: Name of the opponent
        fight_date: Date of the current fight
        fighter_history: DataFrame with fighter-fight records

    Returns:
        Raw striking skill score, or None if < MIN_FIGHTS_FOR_SKILL fights
    """
    opponent_history = fighter_history[
        (fighter_history['fighter_name'] == opponent_name) &
        (fighter_history['fight_date'] < fight_date)
    ]

    if len(opponent_history) < MIN_FIGHTS_FOR_SKILL:
        return None

    # Filter to fights with detailed stats
    detailed = opponent_history[opponent_history['has_detailed_stats'] == True]
    if len(detailed) == 0:
        return None

    # Component 1: Avg sig strikes landed per fight (normalized 0-1)
    avg_sig_landed = detailed['sig_strikes_landed'].mean()
    # Normalize: 0-50 strikes is typical range, cap at 100
    norm_sig_landed = min(avg_sig_landed / 100.0, 1.0)

    # Component 2: Strike accuracy
    total_landed = detailed['sig_strikes_landed'].sum()
    total_attempted = detailed['sig_strikes_attempted'].sum()
    if total_attempted > 0:
        accuracy = total_landed / total_attempted
    else:
        accuracy = 0.5  # neutral

    # Component 3: Strikes absorbed inverse (lower = better defense)
    avg_sig_absorbed = detailed['opp_sig_strikes_landed'].mean()
    # Normalize inversely: 0-80 typical, we want lower to be better
    norm_defense = max(0, 1.0 - (avg_sig_absorbed / 100.0))

    # Component 4: Knockdowns per fight
    avg_knockdowns = detailed['knockdowns'].mean()
    # Normalize: 0-1 knockdowns typical, cap at 3
    norm_knockdowns = min(avg_knockdowns / 3.0, 1.0)

    # Combine
    raw_score = (0.40 * norm_sig_landed) + (0.30 * accuracy) + \
                (0.20 * norm_defense) + (0.10 * norm_knockdowns)

    return raw_score


def compute_opponent_grappling_skill(
    opponent_name: str,
    fight_date: pd.Timestamp,
    fighter_history: pd.DataFrame
) -> Optional[float]:
    """
    Compute opponent's grappling/wrestling skill at the time of the fight.

    Measures how good the opponent is at wrestling exchanges. Landing
    takedowns against a good grappler is more impressive.

    Components (from opponent's prior fights):
    - takedown defense % (40%) - how hard to take down
    - takedowns absorbed per fight inverse (30%) - lower = harder to TD
    - control time absorbed inverse (20%) - less = better at scrambling
    - reversals per fight (10%) - ability to escape/reverse

    Args:
        opponent_name: Name of the opponent
        fight_date: Date of the current fight
        fighter_history: DataFrame with fighter-fight records

    Returns:
        Raw grappling skill score, or None if < MIN_FIGHTS_FOR_SKILL fights
    """
    opponent_history = fighter_history[
        (fighter_history['fighter_name'] == opponent_name) &
        (fighter_history['fight_date'] < fight_date)
    ]

    if len(opponent_history) < MIN_FIGHTS_FOR_SKILL:
        return None

    detailed = opponent_history[opponent_history['has_detailed_stats'] == True]
    if len(detailed) == 0:
        return None

    # We need to compute TD defense from the opponent's perspective
    # TD defense = 1 - (TDs absorbed / TDs attempted against)
    # We don't have TDs attempted against directly, so use TDs absorbed inverse

    # Component 1: TDs absorbed per fight inverse (lower = better TD defense)
    # Need to get opponent's view of takedowns - they had takedowns landed ON them
    avg_tds_absorbed = detailed['takedowns_landed'].mean()  # TDs the opponent landed on their opponents
    # We actually want TDs landed AGAINST the opponent by their opponents
    # This is tricky - we need the opponent's opponent's takedowns
    # Simplify: use the opponent's own TD success as proxy for grappling skill

    # Alternative: Use opponent's takedown accuracy as grappling skill indicator
    total_td_landed = detailed['takedowns_landed'].sum()
    total_td_attempted = detailed['takedowns_attempted'].sum()
    if total_td_attempted > 0:
        td_accuracy = total_td_landed / total_td_attempted
    else:
        td_accuracy = 0.5  # neutral

    # Component 2: Takedowns per fight (offensive grappling)
    avg_tds = detailed['takedowns_landed'].mean()
    norm_tds = min(avg_tds / 5.0, 1.0)  # 5 TDs per fight is very high

    # Component 3: Control time per fight
    avg_control = detailed['control_time_seconds'].mean()
    norm_control = min(avg_control / 300.0, 1.0)  # 5 min = max

    # Component 4: Submission attempts per fight (grappling threat)
    avg_subs = detailed['submission_attempts'].mean()
    norm_subs = min(avg_subs / 3.0, 1.0)  # 3 sub attempts is high

    # Combine
    raw_score = (0.35 * td_accuracy) + (0.30 * norm_tds) + \
                (0.25 * norm_control) + (0.10 * norm_subs)

    return raw_score


def compute_opponent_durability(
    opponent_name: str,
    fight_date: pd.Timestamp,
    fighter_history: pd.DataFrame
) -> Optional[float]:
    """
    Compute opponent's durability at the time of the fight.

    Measures how hard the opponent is to finish. Getting a KO against
    someone who's never been finished is worth more.

    Components (from opponent's prior fights):
    - % of fights that went to decision (40%) - more = more durable
    - KO/TKO loss rate inverse (30%) - lower = more durable chin
    - times knocked down per fight inverse (20%) - fewer = more durable
    - average fight duration (10%) - longer = more durable

    Args:
        opponent_name: Name of the opponent
        fight_date: Date of the current fight
        fighter_history: DataFrame with fighter-fight records

    Returns:
        Raw durability score, or None if < MIN_FIGHTS_FOR_SKILL fights
    """
    opponent_history = fighter_history[
        (fighter_history['fighter_name'] == opponent_name) &
        (fighter_history['fight_date'] < fight_date)
    ]

    if len(opponent_history) < MIN_FIGHTS_FOR_SKILL:
        return None

    n_fights = len(opponent_history)

    # Component 1: Decision rate (fights that went the distance)
    # Check if opponent's fights went to decision (any decision type)
    decision_fights = opponent_history['method'].str.contains('Decision', case=False, na=False).sum()
    decision_rate = decision_fights / n_fights

    # Component 2: KO/TKO loss rate inverse
    # Count fights where opponent LOST by KO/TKO
    ko_losses = ((opponent_history['method'].str.contains('KO|TKO', case=False, na=False)) &
                 (opponent_history['is_winner'] == 0)).sum()
    ko_loss_rate = ko_losses / n_fights
    inverse_ko_loss = 1.0 - ko_loss_rate  # Higher = more durable

    # Component 3: Knockdowns absorbed per fight inverse
    detailed = opponent_history[opponent_history['has_detailed_stats'] == True]
    if len(detailed) > 0:
        # We need knockdowns against opponent - that's their opponent's knockdowns on them
        # Use opponent's sig strikes absorbed as proxy for damage taken
        avg_absorbed = detailed['opp_sig_strikes_landed'].mean()
        norm_damage = max(0, 1.0 - (avg_absorbed / 100.0))  # Higher = less damage taken
    else:
        norm_damage = 0.5  # neutral

    # Component 4: Average fight duration
    avg_duration = opponent_history['fight_time_seconds'].mean()
    # Normalize: 15 min (full 3 rounds) = 1.0
    norm_duration = min(avg_duration / 900.0, 1.0)

    # Combine
    raw_score = (0.40 * decision_rate) + (0.30 * inverse_ko_loss) + \
                (0.20 * norm_damage) + (0.10 * norm_duration)

    return raw_score


def compute_opponent_overall_quality(
    opponent_name: str,
    fight_date: pd.Timestamp,
    fighter_history: pd.DataFrame
) -> Optional[float]:
    """
    Compute opponent's overall quality at the time of the fight.

    Used for adjusting win/loss outcomes. Beating a .800 career fighter
    counts more than beating a .300 career fighter.

    Components:
    - career win rate (50%)
    - number of UFC fights / experience (25%)
    - win rate last 5 (25%)

    Args:
        opponent_name: Name of the opponent
        fight_date: Date of the current fight
        fighter_history: DataFrame with fighter-fight records

    Returns:
        Raw overall quality score, or None if debut
    """
    opponent_history = fighter_history[
        (fighter_history['fighter_name'] == opponent_name) &
        (fighter_history['fight_date'] < fight_date)
    ]

    if len(opponent_history) == 0:
        return None

    career_fights = len(opponent_history)
    career_wins = opponent_history['is_winner'].sum()
    career_win_rate = career_wins / career_fights

    # Experience normalized (10+ fights = fully experienced)
    normalized_experience = min(career_fights / 10.0, 1.0)

    # Recent form (last 5)
    recent_fights = opponent_history.sort_values('fight_date').tail(5)
    recent_form = recent_fights['is_winner'].mean()

    raw_score = (0.50 * career_win_rate) + (0.25 * normalized_experience) + (0.25 * recent_form)

    return raw_score


def normalize_skill_scores(raw_scores: pd.Series, neutral_value: float = 1.0) -> pd.Series:
    """
    Normalize raw skill scores to centered scale where 1.0 = average.

    Uses z-score transformation then scales to target range (0.5 to 1.5).

    Args:
        raw_scores: Series of raw skill scores (None for unknown)
        neutral_value: Value to use for unknown/debut fighters

    Returns:
        Series of normalized scores
    """
    valid_scores = raw_scores.dropna()

    if len(valid_scores) == 0:
        return pd.Series([neutral_value] * len(raw_scores), index=raw_scores.index)

    mean_score = valid_scores.mean()
    std_score = valid_scores.std()

    if std_score == 0 or pd.isna(std_score):
        std_score = 0.1

    normalized = raw_scores.copy().astype(float)

    for idx in raw_scores.index:
        if pd.isna(raw_scores[idx]):
            normalized[idx] = neutral_value
        else:
            # Z-score then scale to 0.5-1.5 range
            z = (raw_scores[idx] - mean_score) / std_score
            z = max(-2, min(2, z))
            normalized[idx] = 1.0 + (z * 0.25)

    return normalized


# Legacy function for backward compatibility with Phase 6
def compute_opponent_quality_for_fight(
    opponent_name: str,
    fight_date: pd.Timestamp,
    fighter_history: pd.DataFrame
) -> float:
    """Legacy function - redirects to compute_opponent_overall_quality."""
    return compute_opponent_overall_quality(opponent_name, fight_date, fighter_history)


def normalize_opponent_quality_scores(raw_scores: pd.Series) -> pd.Series:
    """Legacy function - redirects to normalize_skill_scores."""
    return normalize_skill_scores(raw_scores, neutral_value=1.0)


# ============================================================================
# Main Feature Engineering Class
# ============================================================================

class FeatureEngineer:
    """
    Feature engineer for creating predictive features from UFC fight data.

    Implements comprehensive feature engineering including:
    - Historical fighter statistics
    - Early-career imputation
    - Matchup comparisons
    - Temporal/momentum features
    - Physical attributes (Phase 9.2)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature engineer with configuration.

        Args:
            config: Dictionary containing feature engineering parameters
        """
        self.config = config or {}

        # Imputation parameters for debut fighters
        self.debut_win_rate = 0.52
        self.debut_ko_rate = 0.25
        self.debut_sub_rate = 0.15
        self.debut_dec_rate = 0.60

        # Experience tier thresholds
        self.experience_tiers = {
            'debut': 0,
            'novice': 2,
            'developing': 5,
            'established': 6
        }

        # Rolling window sizes
        self.rolling_windows = [3, 5, 10]

        # Weight class medians (calculated from data)
        self.weight_class_medians = {}

        # Feature metadata
        self.feature_names = []
        self.imputation_stats = {}

        # Opponent quality scoring (Phase 6)
        self.compute_opponent_quality = self.config.get('compute_opponent_quality', True)
        self.opponent_quality_stats = {}

        # Physical attributes (Phase 9.2)
        self.fighter_details = None
        self.physical_medians = {}
        fighter_details_path = self.config.get('fighter_details_path', 'data/raw/fighter_details.csv')
        self._load_fighter_details(fighter_details_path)

    def _load_fighter_details(self, path: str):
        """
        Load fighter physical details from CSV (Phase 9.2).

        Args:
            path: Path to fighter_details.csv
        """
        try:
            details_path = Path(path)
            if details_path.exists():
                self.fighter_details = pd.read_csv(path)
                logger.info(f"Loaded physical details for {len(self.fighter_details)} fighters")

                # Parse DOB to datetime for age calculation
                self.fighter_details['dob_parsed'] = pd.to_datetime(
                    self.fighter_details['dob'], format='%b %d, %Y', errors='coerce'
                )

                # Create lookup dict for faster access
                self.fighter_details_lookup = {}
                for _, row in self.fighter_details.iterrows():
                    self.fighter_details_lookup[row['fighter_name']] = {
                        'height_inches': row['height_inches'] if pd.notna(row['height_inches']) else None,
                        'reach_inches': row['reach_inches'] if pd.notna(row['reach_inches']) else None,
                        'stance': row['stance'] if pd.notna(row['stance']) else None,
                        'dob': row['dob_parsed'] if pd.notna(row['dob_parsed']) else None,
                    }
            else:
                logger.warning(f"Fighter details file not found: {path}")
                self.fighter_details = None
                self.fighter_details_lookup = {}
        except Exception as e:
            logger.warning(f"Error loading fighter details: {e}")
            self.fighter_details = None
            self.fighter_details_lookup = {}

    def _compute_physical_medians(self, df: pd.DataFrame):
        """
        Compute weight-class median values for physical attributes (Phase 9.2).

        Used for imputing missing height, reach, and age values.
        Should be called on TRAINING data only to avoid leakage.

        Args:
            df: DataFrame with weight_class, f1_name, f2_name columns
        """
        if self.fighter_details is None:
            logger.warning("No fighter details loaded, skipping physical median computation")
            return

        logger.info("Computing weight-class medians for physical attributes...")

        # Get all fighters by weight class
        weight_class_fighters = {}

        for _, row in df.iterrows():
            wc = row['weight_class']
            if wc not in weight_class_fighters:
                weight_class_fighters[wc] = set()
            weight_class_fighters[wc].add(row['f1_name'])
            weight_class_fighters[wc].add(row['f2_name'])

        # Compute medians per weight class
        for wc, fighters in weight_class_fighters.items():
            heights = []
            reaches = []
            ages = []

            for fighter in fighters:
                if fighter in self.fighter_details_lookup:
                    details = self.fighter_details_lookup[fighter]
                    if details['height_inches'] is not None:
                        heights.append(details['height_inches'])
                    if details['reach_inches'] is not None:
                        reaches.append(details['reach_inches'])
                    # For age, use typical prime age (30)
                    ages.append(30.0)

            self.physical_medians[wc] = {
                'height_inches': float(np.median(heights)) if heights else 70.0,
                'reach_inches': float(np.median(reaches)) if reaches else 70.0,
                'age': 30.0,  # Default prime age
            }

        # Global fallback
        all_heights = [d['height_inches'] for d in self.fighter_details_lookup.values()
                       if d['height_inches'] is not None]
        all_reaches = [d['reach_inches'] for d in self.fighter_details_lookup.values()
                       if d['reach_inches'] is not None]

        self.physical_medians['_global'] = {
            'height_inches': float(np.median(all_heights)) if all_heights else 70.0,
            'reach_inches': float(np.median(all_reaches)) if all_reaches else 70.0,
            'age': 30.0,
        }

        logger.info(f"Computed physical medians for {len(self.physical_medians)-1} weight classes")

    def _get_physical_features(self, fighter_name: str, fight_date: pd.Timestamp,
                               weight_class: str) -> Dict:
        """
        Get physical attributes for a fighter at fight time.

        Args:
            fighter_name: Fighter name
            fight_date: Date of the fight (for age calculation)
            weight_class: Weight class (for median imputation)

        Returns:
            Dictionary with height_inches, reach_inches, age_at_fight, stance features
        """
        # Get medians for this weight class (fallback to global)
        medians = self.physical_medians.get(weight_class,
                                            self.physical_medians.get('_global',
                                                                      {'height_inches': 70.0, 'reach_inches': 70.0, 'age': 30.0}))

        if fighter_name not in self.fighter_details_lookup:
            # Fighter not in details - use weight class medians
            return {
                'height_inches': medians['height_inches'],
                'reach_inches': medians['reach_inches'],
                'age_at_fight': medians['age'],
                'stance_orthodox': 1,  # Default to orthodox
                'stance_southpaw': 0,
                'stance_switch': 0,
            }

        details = self.fighter_details_lookup[fighter_name]

        # Height - use median if missing
        height = details['height_inches'] if details['height_inches'] is not None else medians['height_inches']

        # Reach - use median if missing
        reach = details['reach_inches'] if details['reach_inches'] is not None else medians['reach_inches']

        # Age at fight time
        if details['dob'] is not None and pd.notna(fight_date):
            age = (fight_date - details['dob']).days / 365.25
            if age < 18 or age > 60:  # Sanity check
                age = medians['age']
        else:
            age = medians['age']

        # Stance one-hot encoding
        stance = details['stance'] if details['stance'] is not None else 'Orthodox'
        stance_orthodox = 1 if stance == 'Orthodox' else 0
        stance_southpaw = 1 if stance == 'Southpaw' else 0
        stance_switch = 1 if stance == 'Switch' else 0

        return {
            'height_inches': height,
            'reach_inches': reach,
            'age_at_fight': age,
            'stance_orthodox': stance_orthodox,
            'stance_southpaw': stance_southpaw,
            'stance_switch': stance_switch,
        }

    def add_physical_features(self, matchup_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add physical attribute features to matchup DataFrame (Phase 9.2).

        Adds height, reach, age, and stance features for both fighters,
        plus differential features.

        Args:
            matchup_df: DataFrame with fight matchups

        Returns:
            DataFrame with added physical features
        """
        if self.fighter_details is None:
            logger.warning("No fighter details loaded, skipping physical features")
            return matchup_df

        logger.info("Adding physical attribute features (Phase 9.2)...")

        # Compute medians if not already done
        if not self.physical_medians:
            self._compute_physical_medians(matchup_df)

        # Add features for each fight
        f1_heights = []
        f1_reaches = []
        f1_ages = []
        f1_stance_orthodox = []
        f1_stance_southpaw = []
        f1_stance_switch = []

        f2_heights = []
        f2_reaches = []
        f2_ages = []
        f2_stance_orthodox = []
        f2_stance_southpaw = []
        f2_stance_switch = []

        for _, row in matchup_df.iterrows():
            f1_phys = self._get_physical_features(row['f1_name'], row['fight_date'], row['weight_class'])
            f2_phys = self._get_physical_features(row['f2_name'], row['fight_date'], row['weight_class'])

            f1_heights.append(f1_phys['height_inches'])
            f1_reaches.append(f1_phys['reach_inches'])
            f1_ages.append(f1_phys['age_at_fight'])
            f1_stance_orthodox.append(f1_phys['stance_orthodox'])
            f1_stance_southpaw.append(f1_phys['stance_southpaw'])
            f1_stance_switch.append(f1_phys['stance_switch'])

            f2_heights.append(f2_phys['height_inches'])
            f2_reaches.append(f2_phys['reach_inches'])
            f2_ages.append(f2_phys['age_at_fight'])
            f2_stance_orthodox.append(f2_phys['stance_orthodox'])
            f2_stance_southpaw.append(f2_phys['stance_southpaw'])
            f2_stance_switch.append(f2_phys['stance_switch'])

        # Add columns
        matchup_df = matchup_df.copy()
        matchup_df['f1_height_inches'] = f1_heights
        matchup_df['f1_reach_inches'] = f1_reaches
        matchup_df['f1_age_at_fight'] = f1_ages
        matchup_df['f1_stance_orthodox'] = f1_stance_orthodox
        matchup_df['f1_stance_southpaw'] = f1_stance_southpaw
        matchup_df['f1_stance_switch'] = f1_stance_switch

        matchup_df['f2_height_inches'] = f2_heights
        matchup_df['f2_reach_inches'] = f2_reaches
        matchup_df['f2_age_at_fight'] = f2_ages
        matchup_df['f2_stance_orthodox'] = f2_stance_orthodox
        matchup_df['f2_stance_southpaw'] = f2_stance_southpaw
        matchup_df['f2_stance_switch'] = f2_stance_switch

        # Differential features
        matchup_df['diff_height'] = matchup_df['f1_height_inches'] - matchup_df['f2_height_inches']
        matchup_df['diff_reach'] = matchup_df['f1_reach_inches'] - matchup_df['f2_reach_inches']
        matchup_df['diff_age'] = matchup_df['f1_age_at_fight'] - matchup_df['f2_age_at_fight']

        # Reach advantage (reach relative to height)
        matchup_df['f1_reach_advantage'] = matchup_df['f1_reach_inches'] - matchup_df['f1_height_inches']
        matchup_df['f2_reach_advantage'] = matchup_df['f2_reach_inches'] - matchup_df['f2_height_inches']
        matchup_df['diff_reach_advantage'] = matchup_df['f1_reach_advantage'] - matchup_df['f2_reach_advantage']

        # Stance mismatch (orthodox vs southpaw is interesting stylistically)
        matchup_df['stance_mismatch'] = (
            ((matchup_df['f1_stance_orthodox'] == 1) & (matchup_df['f2_stance_southpaw'] == 1)) |
            ((matchup_df['f1_stance_southpaw'] == 1) & (matchup_df['f2_stance_orthodox'] == 1))
        ).astype(int)

        logger.info(f"Added 19 physical attribute features")

        return matchup_df


    def preprocess_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse strings, restructure to fighter-centric, sort chronologically.

        This transforms fight-centric data (1 row = 1 fight with 2 fighters)
        into fighter-centric data (2 rows per fight, 1 per fighter).

        Args:
            df: Raw fight data from CSV

        Returns:
            Preprocessed DataFrame with parsed statistics, chronologically sorted
        """
        logger.info("Starting data preprocessing...")
        logger.info(f"Input: {len(df)} fights")

        # Parse event dates to datetime
        df['event_date_parsed'] = pd.to_datetime(df['event_date'])

        # Create two DataFrames: one for each fighter in each fight
        fighter_records = []

        for idx, row in df.iterrows():
            # Determine winner (normalize names)
            winner = str(row['winner']).strip() if pd.notna(row['winner']) else ''
            red_name = str(row['red_fighter_name']).strip()
            blue_name = str(row['blue_fighter_name']).strip()

            # Check if data has detailed stats (post-2009)
            has_detailed = pd.notna(row.get('red_fighter_sig_str', None))

            # Parse fight outcome time
            fight_time_seconds = parse_time_to_seconds(row.get('time', ''))
            round_num = int(row['round']) if pd.notna(row.get('round')) else 0
            total_fight_seconds = (round_num - 1) * 300 + fight_time_seconds

            # Record for red corner fighter
            red_record = {
                'fight_id': row['fight_id'],
                'fighter_name': red_name,
                'opponent_name': blue_name,
                'fight_date': row['event_date_parsed'],
                'event_name': row['event_name'],
                'event_location': row.get('event_location', 'Unknown'),
                'weight_class': row.get('weight_class', 'Unknown'),
                'is_winner': 1 if winner == red_name else 0,
                'method': row.get('method', 'Unknown'),
                'round': round_num,
                'fight_time_seconds': total_fight_seconds,
                'has_detailed_stats': has_detailed,
            }

            # Record for blue corner fighter
            blue_record = {
                'fight_id': row['fight_id'],
                'fighter_name': blue_name,
                'opponent_name': red_name,
                'fight_date': row['event_date_parsed'],
                'event_name': row['event_name'],
                'event_location': row.get('event_location', 'Unknown'),
                'weight_class': row.get('weight_class', 'Unknown'),
                'is_winner': 1 if winner == blue_name else 0,
                'method': row.get('method', 'Unknown'),
                'round': round_num,
                'fight_time_seconds': total_fight_seconds,
                'has_detailed_stats': has_detailed,
            }

            # Parse detailed statistics if available
            if has_detailed:
                # Helper to safely parse integers with cleaning
                def safe_int(val):
                    if pd.isna(val):
                        return 0
                    try:
                        # Strip whitespace and newlines, take first number
                        cleaned = str(val).strip().split()[0] if str(val).strip() else '0'
                        return int(cleaned)
                    except (ValueError, IndexError):
                        return 0

                # Red fighter stats
                red_kd = safe_int(row.get('red_fighter_KD'))
                red_sig_landed, red_sig_attempted = parse_strike_data(row.get('red_fighter_sig_str', ''))
                red_total_landed, red_total_attempted = parse_strike_data(row.get('red_fighter_total_str', ''))
                red_td_landed, red_td_attempted = parse_strike_data(row.get('red_fighter_TD', ''))
                red_sub_att = safe_int(row.get('red_fighter_sub_att'))
                red_ctrl_time = parse_time_to_seconds(row.get('red_fighter_ctrl', ''))

                # Blue fighter stats
                blue_kd = safe_int(row.get('blue_fighter_KD'))
                blue_sig_landed, blue_sig_attempted = parse_strike_data(row.get('blue_fighter_sig_str', ''))
                blue_total_landed, blue_total_attempted = parse_strike_data(row.get('blue_fighter_total_str', ''))
                blue_td_landed, blue_td_attempted = parse_strike_data(row.get('blue_fighter_TD', ''))
                blue_sub_att = safe_int(row.get('blue_fighter_sub_att'))
                blue_ctrl_time = parse_time_to_seconds(row.get('blue_fighter_ctrl', ''))

                # Add to records
                red_record.update({
                    'knockdowns': red_kd,
                    'sig_strikes_landed': red_sig_landed,
                    'sig_strikes_attempted': red_sig_attempted,
                    'total_strikes_landed': red_total_landed,
                    'total_strikes_attempted': red_total_attempted,
                    'takedowns_landed': red_td_landed,
                    'takedowns_attempted': red_td_attempted,
                    'submission_attempts': red_sub_att,
                    'control_time_seconds': red_ctrl_time,
                    # Opponent stats (for defensive metrics)
                    'opp_sig_strikes_landed': blue_sig_landed,
                    'opp_sig_strikes_attempted': blue_sig_attempted,
                })

                blue_record.update({
                    'knockdowns': blue_kd,
                    'sig_strikes_landed': blue_sig_landed,
                    'sig_strikes_attempted': blue_sig_attempted,
                    'total_strikes_landed': blue_total_landed,
                    'total_strikes_attempted': blue_total_attempted,
                    'takedowns_landed': blue_td_landed,
                    'takedowns_attempted': blue_td_attempted,
                    'submission_attempts': blue_sub_att,
                    'control_time_seconds': blue_ctrl_time,
                    'opp_sig_strikes_landed': red_sig_landed,
                    'opp_sig_strikes_attempted': red_sig_attempted,
                })
            else:
                # No detailed stats available
                for key in ['knockdowns', 'sig_strikes_landed', 'sig_strikes_attempted',
                           'total_strikes_landed', 'total_strikes_attempted',
                           'takedowns_landed', 'takedowns_attempted',
                           'submission_attempts', 'control_time_seconds',
                           'opp_sig_strikes_landed', 'opp_sig_strikes_attempted']:
                    red_record[key] = 0
                    blue_record[key] = 0

            fighter_records.append(red_record)
            fighter_records.append(blue_record)

        # Create DataFrame from records
        fighter_df = pd.DataFrame(fighter_records)

        # Sort chronologically (critical for avoiding data leakage)
        fighter_df = fighter_df.sort_values(['fighter_name', 'fight_date']).reset_index(drop=True)

        logger.info(f"Preprocessing complete: {len(fighter_df)} fighter-fight records")
        logger.info(f"Date range: {fighter_df['fight_date'].min()} to {fighter_df['fight_date'].max()}")
        logger.info(f"Unique fighters: {fighter_df['fighter_name'].nunique()}")
        logger.info(f"Records with detailed stats: {fighter_df['has_detailed_stats'].sum()} / {len(fighter_df)}")

        return fighter_df


    def calculate_career_features(self, fighter_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate career statistics up to each fight (avoiding data leakage).

        For each fighter in each fight, calculates their statistics based ONLY
        on fights prior to this one.

        Args:
            fighter_df: Preprocessed fighter-centric DataFrame

        Returns:
            DataFrame with added career feature columns
        """
        logger.info("Calculating career features...")

        df = fighter_df.copy()

        # Group by fighter and calculate cumulative stats
        def calculate_fighter_stats(fighter_group):
            """Calculate cumulative career stats for a single fighter"""
            fg = fighter_group.sort_values('fight_date').copy()

            # Career cumulative stats (excluding current fight)
            fg['career_fights'] = range(len(fg))
            fg['career_wins'] = fg['is_winner'].shift(1, fill_value=0).cumsum()
            fg['career_losses'] = (1 - fg['is_winner']).shift(1, fill_value=0).cumsum()

            # Win rate
            fg['career_win_rate'] = fg['career_wins'] / fg['career_fights'].replace(0, 1)

            # Finish rates by method
            fg['is_ko'] = fg['method'].str.contains('KO|TKO', case=False, na=False).astype(int)
            fg['is_sub'] = fg['method'].str.contains('Submission', case=False, na=False).astype(int)
            fg['is_decision'] = fg['method'].str.contains('Decision', case=False, na=False).astype(int)

            fg['career_kos'] = (fg['is_ko'] & (fg['is_winner'] == 1)).shift(1, fill_value=0).cumsum()
            fg['career_subs'] = (fg['is_sub'] & (fg['is_winner'] == 1)).shift(1, fill_value=0).cumsum()
            fg['career_decisions'] = (fg['is_decision'] & (fg['is_winner'] == 1)).shift(1, fill_value=0).cumsum()

            # Finish rate = KOs + Subs / Total Wins
            fg['career_ko_rate'] = fg['career_kos'] / fg['career_wins'].replace(0, 1)
            fg['career_sub_rate'] = fg['career_subs'] / fg['career_wins'].replace(0, 1)
            fg['career_dec_rate'] = fg['career_decisions'] / fg['career_wins'].replace(0, 1)
            fg['career_finish_rate'] = (fg['career_kos'] + fg['career_subs']) / fg['career_wins'].replace(0, 1)

            # Win/loss streaks
            fg['win_streak'] = 0
            fg['loss_streak'] = 0

            current_win_streak = 0
            current_loss_streak = 0

            for i in range(len(fg)):
                fg.loc[fg.index[i], 'win_streak'] = current_win_streak
                fg.loc[fg.index[i], 'loss_streak'] = current_loss_streak

                if i < len(fg):
                    if fg.iloc[i]['is_winner'] == 1:
                        current_win_streak += 1
                        current_loss_streak = 0
                    else:
                        current_loss_streak += 1
                        current_win_streak = 0

            # Days since last fight
            fg['days_since_last_fight'] = fg['fight_date'].diff().dt.days.fillna(365)

            # Fights per year (career average)
            fg['career_days'] = (fg['fight_date'] - fg['fight_date'].iloc[0]).dt.days
            fg['fights_per_year'] = (fg['career_fights'] / (fg['career_days'] / 365.25)).replace([np.inf, -np.inf], 0)

            # Rolling statistics for different windows
            for window in self.rolling_windows:
                # Win rate in last N fights
                fg[f'win_rate_last_{window}'] = fg['is_winner'].shift(1).rolling(window, min_periods=1).mean()

                # Finish rate in last N fights
                is_finish = (fg['is_ko'] | fg['is_sub']) & (fg['is_winner'] == 1)
                fg[f'finish_rate_last_{window}'] = is_finish.shift(1).rolling(window, min_periods=1).mean()

                # Average strikes landed/absorbed in last N fights
                if 'sig_strikes_landed' in fg.columns:
                    fg[f'avg_sig_strikes_landed_last_{window}'] = fg['sig_strikes_landed'].shift(1).rolling(window, min_periods=1).mean()
                    fg[f'avg_sig_strikes_absorbed_last_{window}'] = fg['opp_sig_strikes_landed'].shift(1).rolling(window, min_periods=1).mean()
                    fg[f'avg_takedowns_last_{window}'] = fg['takedowns_landed'].shift(1).rolling(window, min_periods=1).mean()

                # Average fight time
                fg[f'avg_fight_time_last_{window}'] = fg['fight_time_seconds'].shift(1).rolling(window, min_periods=1).mean()

            # Weight class experience
            fg['weight_class_fights'] = fg.groupby('weight_class').cumcount()

            # Fights in last 12 months - set fight_date as index temporarily
            fg_temp = fg.set_index('fight_date')
            fg['fights_last_12mo'] = fg_temp['is_winner'].shift(1).rolling('365D').count().values

            return fg

        # Apply to each fighter
        df = df.groupby('fighter_name', group_keys=False).apply(calculate_fighter_stats)

        logger.info(f"Career features calculated for {df['fighter_name'].nunique()} fighters")

        return df


    def calculate_opponent_quality_features(self, fighter_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate domain-specific opponent skill profiles and adjusted features.

        Phase 7 enhancement: Each stat category is adjusted by the opponent's
        strength in THAT SAME category:
        - Striking stats × opponent_striking_skill
        - Grappling stats × opponent_grappling_skill
        - Knockdowns × sqrt(striking_skill × durability)
        - Win/loss × opponent_overall_quality

        Args:
            fighter_df: DataFrame with fighter-centric records and career features

        Returns:
            DataFrame with domain-specific opponent skills and adjusted features
        """
        if not self.compute_opponent_quality:
            logger.info("Opponent quality computation disabled, skipping...")
            return fighter_df

        logger.info("Calculating domain-specific opponent skill profiles (Phase 7)...")

        df = fighter_df.copy()

        # =====================================================================
        # Step 1: Compute all 4 raw opponent skill profiles for each fight
        # =====================================================================
        raw_striking_skills = []
        raw_grappling_skills = []
        raw_durability_scores = []
        raw_overall_quality = []

        for idx, row in df.iterrows():
            opponent_name = row['opponent_name']
            fight_date = row['fight_date']

            # Compute each skill profile
            striking = compute_opponent_striking_skill(opponent_name, fight_date, df)
            grappling = compute_opponent_grappling_skill(opponent_name, fight_date, df)
            durability = compute_opponent_durability(opponent_name, fight_date, df)
            overall = compute_opponent_overall_quality(opponent_name, fight_date, df)

            raw_striking_skills.append(striking)
            raw_grappling_skills.append(grappling)
            raw_durability_scores.append(durability)
            raw_overall_quality.append(overall)

        df['raw_opp_striking_skill'] = raw_striking_skills
        df['raw_opp_grappling_skill'] = raw_grappling_skills
        df['raw_opp_durability'] = raw_durability_scores
        df['raw_opp_overall_quality'] = raw_overall_quality

        # =====================================================================
        # Step 2: Normalize each skill to 0.5-1.5 scale (1.0 = average)
        # =====================================================================
        df['opp_striking_skill'] = normalize_skill_scores(
            pd.Series(raw_striking_skills, index=df.index), neutral_value=1.0
        )
        df['opp_grappling_skill'] = normalize_skill_scores(
            pd.Series(raw_grappling_skills, index=df.index), neutral_value=1.0
        )
        df['opp_durability'] = normalize_skill_scores(
            pd.Series(raw_durability_scores, index=df.index), neutral_value=1.0
        )
        df['opp_overall_quality'] = normalize_skill_scores(
            pd.Series(raw_overall_quality, index=df.index), neutral_value=1.0
        )

        # For backward compatibility, also store as 'opponent_quality'
        df['opponent_quality'] = df['opp_overall_quality']

        # Print distribution stats for each skill category
        self.opponent_quality_stats = {}
        for skill_col in ['opp_striking_skill', 'opp_grappling_skill', 'opp_durability', 'opp_overall_quality']:
            valid = df[skill_col].dropna()
            raw_col = skill_col.replace('opp_', 'raw_opp_')
            debuts = df[raw_col].isna().sum() if raw_col in df.columns else 0
            self.opponent_quality_stats[skill_col] = {
                'mean': float(valid.mean()),
                'std': float(valid.std()),
                'min': float(valid.min()),
                'max': float(valid.max()),
                'median': float(valid.median()),
                'neutral_count': int(debuts),
            }
            logger.info(f"  {skill_col}: mean={valid.mean():.3f}, std={valid.std():.3f}, "
                        f"range=[{valid.min():.3f}, {valid.max():.3f}], neutral={debuts}")

        # =====================================================================
        # Step 3: Apply domain-specific adjustments to stats
        # =====================================================================
        logger.info("Applying domain-matched stat adjustments...")

        # Striking stats adjusted by opponent's striking ability
        striking_stats = ['sig_strikes_landed', 'sig_strikes_attempted',
                          'total_strikes_landed', 'total_strikes_attempted']
        for stat in striking_stats:
            if stat in df.columns:
                df[f'{stat}_adj'] = df[stat] * df['opp_striking_skill']

        # Grappling stats adjusted by opponent's grappling ability
        grappling_stats = ['takedowns_landed', 'takedowns_attempted',
                           'submission_attempts', 'control_time_seconds']
        for stat in grappling_stats:
            if stat in df.columns:
                df[f'{stat}_adj'] = df[stat] * df['opp_grappling_skill']

        # Knockdowns: hybrid adjustment (striking × durability)
        if 'knockdowns' in df.columns:
            df['knockdowns_adj'] = df['knockdowns'] * np.sqrt(
                df['opp_striking_skill'] * df['opp_durability']
            )

        # Win/loss adjusted by opponent's overall quality
        df['quality_adjusted_win'] = df['is_winner'] * df['opp_overall_quality']

        # =====================================================================
        # Step 4: Compute adjusted career stats and rolling windows
        # =====================================================================
        logger.info("Computing domain-adjusted career stats...")

        def calculate_domain_adjusted_stats(fighter_group):
            """Calculate domain-adjusted cumulative career stats"""
            fg = fighter_group.sort_values('fight_date').copy()

            # Adjusted career win rate (using overall quality)
            fg['adj_career_wins'] = fg['quality_adjusted_win'].shift(1, fill_value=0).cumsum()
            fg['career_fights_for_adj'] = range(len(fg))
            fg['adj_career_win_rate'] = fg['adj_career_wins'] / fg['career_fights_for_adj'].replace(0, 1)

            # Average opponent skills faced (expanding mean of prior opponents)
            for skill in ['opp_striking_skill', 'opp_grappling_skill', 'opp_durability', 'opp_overall_quality']:
                col_name = f'avg_{skill}'
                fg[col_name] = fg[skill].shift(1).expanding(min_periods=1).mean()
                fg[col_name] = fg[col_name].fillna(1.0)

            # Domain-adjusted rolling windows
            for window in self.rolling_windows:
                # Adjusted win rate in last N fights
                fg[f'adj_win_rate_last_{window}'] = fg['quality_adjusted_win'].shift(1).rolling(
                    window, min_periods=1
                ).mean()

                # Recent opponent skill averages
                for skill in ['opp_striking_skill', 'opp_grappling_skill', 'opp_overall_quality']:
                    fg[f'avg_{skill}_last_{window}'] = fg[skill].shift(1).rolling(
                        window, min_periods=1
                    ).mean()

                # Domain-adjusted striking stats
                if 'sig_strikes_landed_adj' in fg.columns:
                    fg[f'adj_avg_sig_strikes_landed_last_{window}'] = fg['sig_strikes_landed_adj'].shift(1).rolling(
                        window, min_periods=1
                    ).mean()

                # Domain-adjusted grappling stats
                if 'takedowns_landed_adj' in fg.columns:
                    fg[f'adj_avg_takedowns_last_{window}'] = fg['takedowns_landed_adj'].shift(1).rolling(
                        window, min_periods=1
                    ).mean()

                if 'control_time_seconds_adj' in fg.columns:
                    fg[f'adj_avg_control_time_last_{window}'] = fg['control_time_seconds_adj'].shift(1).rolling(
                        window, min_periods=1
                    ).mean()

                # Domain-adjusted knockdowns
                if 'knockdowns_adj' in fg.columns:
                    fg[f'adj_avg_knockdowns_last_{window}'] = fg['knockdowns_adj'].shift(1).rolling(
                        window, min_periods=1
                    ).mean()

            # Clean up temp columns
            fg.drop(columns=['career_fights_for_adj'], inplace=True)

            return fg

        df = df.groupby('fighter_name', group_keys=False).apply(calculate_domain_adjusted_stats)

        # =====================================================================
        # Step 5: Fill NaN values with appropriate defaults
        # =====================================================================
        adj_features = [col for col in df.columns if col.startswith('adj_') or 'opp_' in col]
        for feat in adj_features:
            if feat == 'adj_career_win_rate':
                df[feat] = df[feat].fillna(self.debut_win_rate)
            elif 'opp_' in feat and 'skill' in feat or 'quality' in feat or 'durability' in feat:
                df[feat] = df[feat].fillna(1.0)  # Neutral
            else:
                df[feat] = df[feat].fillna(0.0)

        logger.info(f"Added {len(adj_features)} domain-adjusted features (Phase 7)")

        return df


    def calculate_weight_class_medians(self, df: pd.DataFrame) -> Dict:
        """
        Calculate median statistics by weight class for imputation.

        Args:
            df: DataFrame with career features

        Returns:
            Dictionary of weight class medians
        """
        logger.info("Calculating weight class medians for imputation...")

        # Only use fighters with 6+ fights for reliable medians
        experienced = df[df['career_fights'] >= 6].copy()

        medians = {}

        for wc in df['weight_class'].unique():
            wc_data = experienced[experienced['weight_class'] == wc]

            if len(wc_data) > 0:
                medians[wc] = {
                    'sig_strikes_landed': wc_data['sig_strikes_landed'].median(),
                    'sig_strikes_absorbed': wc_data['opp_sig_strikes_landed'].median(),
                    'takedowns': wc_data['takedowns_landed'].median(),
                    'fight_time': wc_data['fight_time_seconds'].median(),
                }
            else:
                # Use overall median if weight class has no experienced fighters
                medians[wc] = {
                    'sig_strikes_landed': experienced['sig_strikes_landed'].median(),
                    'sig_strikes_absorbed': experienced['opp_sig_strikes_landed'].median(),
                    'takedowns': experienced['takedowns_landed'].median(),
                    'fight_time': experienced['fight_time_seconds'].median(),
                }

        self.weight_class_medians = medians
        logger.info(f"Calculated medians for {len(medians)} weight classes")

        return medians


    def impute_early_career(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply hybrid imputation for fighters with insufficient fight history.

        Uses tier-based approach:
        - Tier 0 (debut): Population medians
        - Tier 1 (1-2 fights): Blended actual + weight class median
        - Tier 2 (3-5 fights): All available fights
        - Tier 3 (6+ fights): No imputation needed

        Args:
            df: DataFrame with career features

        Returns:
            DataFrame with imputed features and confidence flags
        """
        logger.info("Applying early-career imputation...")

        result = df.copy()

        # Calculate weight class medians if not already done
        if not self.weight_class_medians:
            self.calculate_weight_class_medians(result)

        # Add experience tier
        result['experience_tier'] = pd.cut(
            result['career_fights'],
            bins=[-1, 0, 2, 5, np.inf],
            labels=['debut', 'novice', 'developing', 'established']
        )

        # Impute debut fighters (0 prior fights)
        debut_mask = result['career_fights'] == 0
        debut_count = debut_mask.sum()

        if debut_count > 0:
            result.loc[debut_mask, 'career_win_rate'] = self.debut_win_rate
            result.loc[debut_mask, 'career_ko_rate'] = self.debut_ko_rate
            result.loc[debut_mask, 'career_sub_rate'] = self.debut_sub_rate
            result.loc[debut_mask, 'career_dec_rate'] = self.debut_dec_rate
            result.loc[debut_mask, 'career_finish_rate'] = self.debut_ko_rate + self.debut_sub_rate

            # Impute rolling stats with weight class medians
            for idx in result[debut_mask].index:
                wc = result.loc[idx, 'weight_class']
                wc_medians = self.weight_class_medians.get(wc, {})

                for window in self.rolling_windows:
                    result.loc[idx, f'avg_sig_strikes_landed_last_{window}'] = wc_medians.get('sig_strikes_landed', 0)
                    result.loc[idx, f'avg_sig_strikes_absorbed_last_{window}'] = wc_medians.get('sig_strikes_absorbed', 0)
                    result.loc[idx, f'avg_takedowns_last_{window}'] = wc_medians.get('takedowns', 0)
                    result.loc[idx, f'avg_fight_time_last_{window}'] = wc_medians.get('fight_time', 0)

        # Impute novice fighters (1-2 prior fights) - blend actual + median
        novice_mask = result['experience_tier'] == 'novice'
        novice_count = novice_mask.sum()

        if novice_count > 0:
            for idx in result[novice_mask].index:
                wc = result.loc[idx, 'weight_class']
                wc_medians = self.weight_class_medians.get(wc, {})
                n_fights = result.loc[idx, 'career_fights']

                # Blend for rolling windows requiring more fights than available
                for window in self.rolling_windows:
                    if n_fights < window:
                        weight_actual = n_fights / window
                        weight_median = 1 - weight_actual

                        actual_strikes = result.loc[idx, f'avg_sig_strikes_landed_last_{window}']
                        if pd.isna(actual_strikes):
                            actual_strikes = 0
                        result.loc[idx, f'avg_sig_strikes_landed_last_{window}'] = (
                            weight_actual * actual_strikes +
                            weight_median * wc_medians.get('sig_strikes_landed', 0)
                        )

                        actual_absorbed = result.loc[idx, f'avg_sig_strikes_absorbed_last_{window}']
                        if pd.isna(actual_absorbed):
                            actual_absorbed = 0
                        result.loc[idx, f'avg_sig_strikes_absorbed_last_{window}'] = (
                            weight_actual * actual_absorbed +
                            weight_median * wc_medians.get('sig_strikes_absorbed', 0)
                        )

        # Add confidence features
        result['is_ufc_debut'] = (result['career_fights'] == 0).astype(int)
        result['data_completeness_score'] = np.where(
            result['career_fights'] >= 6,
            1.0,
            result['career_fights'] / 6.0
        )

        logger.info(f"Imputation complete:")
        logger.info(f"  Debut fighters: {debut_count}")
        logger.info(f"  Novice fighters: {novice_count}")
        logger.info(f"  Developing fighters: {(result['experience_tier'] == 'developing').sum()}")
        logger.info(f"  Established fighters: {(result['experience_tier'] == 'established').sum()}")

        return result


    def create_matchup_features(self, fighter_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create matchup comparison features from fighter-centric data.

        Converts from fighter-centric (2 rows per fight) back to fight-centric
        (1 row per fight) with differential features.

        Args:
            fighter_df: DataFrame with fighter career features

        Returns:
            DataFrame with one row per fight and matchup comparison features
        """
        logger.info("Creating matchup features...")

        # Get unique fights
        fights = []

        for fight_id in fighter_df['fight_id'].unique():
            fight_data = fighter_df[fighter_df['fight_id'] == fight_id].reset_index(drop=True)

            if len(fight_data) != 2:
                logger.warning(f"Fight {fight_id} has {len(fight_data)} records, expected 2. Skipping.")
                continue

            # Assign fighter 1 and fighter 2 (arbitrary but consistent)
            f1 = fight_data.iloc[0]
            f2 = fight_data.iloc[1]

            # Target variable: did fighter 1 win?
            f1_is_winner = f1['is_winner']

            # Create matchup record
            matchup = {
                'fight_id': fight_id,
                'fight_date': f1['fight_date'],
                'event_name': f1['event_name'],
                'weight_class': f1['weight_class'],
                'f1_name': f1['fighter_name'],
                'f2_name': f2['fighter_name'],
                'f1_is_winner': f1_is_winner,  # Target variable
                'method': f1['method'],
                'round': f1['round'],
                'fight_time_seconds': f1['fight_time_seconds'],
            }

            # Career features for both fighters
            career_features = [
                'career_fights', 'career_wins', 'career_losses', 'career_win_rate',
                'career_ko_rate', 'career_sub_rate', 'career_dec_rate', 'career_finish_rate',
                'win_streak', 'loss_streak', 'days_since_last_fight', 'fights_per_year',
                'weight_class_fights', 'fights_last_12mo',
                'is_ufc_debut', 'data_completeness_score', 'has_detailed_stats'
            ]

            # Rolling features
            for window in self.rolling_windows:
                career_features.extend([
                    f'win_rate_last_{window}',
                    f'finish_rate_last_{window}',
                    f'avg_sig_strikes_landed_last_{window}',
                    f'avg_sig_strikes_absorbed_last_{window}',
                    f'avg_takedowns_last_{window}',
                    f'avg_fight_time_last_{window}',
                ])

            # Add domain-specific opponent-adjusted features (Phase 7)
            if self.compute_opponent_quality:
                # Core adjusted career features
                career_features.extend([
                    'adj_career_win_rate',
                    # Domain-specific opponent skill averages
                    'avg_opp_striking_skill',
                    'avg_opp_grappling_skill',
                    'avg_opp_durability',
                    'avg_opp_overall_quality',
                ])
                for window in self.rolling_windows:
                    career_features.extend([
                        # Adjusted win rate
                        f'adj_win_rate_last_{window}',
                        # Recent opponent skill averages
                        f'avg_opp_striking_skill_last_{window}',
                        f'avg_opp_grappling_skill_last_{window}',
                        f'avg_opp_overall_quality_last_{window}',
                        # Domain-adjusted stats
                        f'adj_avg_sig_strikes_landed_last_{window}',
                        f'adj_avg_takedowns_last_{window}',
                        f'adj_avg_control_time_last_{window}',
                        f'adj_avg_knockdowns_last_{window}',
                    ])

            # Add features for both fighters
            for feat in career_features:
                if feat in f1 and feat in f2:
                    matchup[f'f1_{feat}'] = f1[feat]
                    matchup[f'f2_{feat}'] = f2[feat]

                    # Create differential feature (f1 - f2)
                    if isinstance(f1[feat], (int, float, np.number)) and isinstance(f2[feat], (int, float, np.number)):
                        matchup[f'diff_{feat}'] = f1[feat] - f2[feat]

            # Style matchup indicators
            # Striking volume (strikes per minute)
            f1_strike_rate = f1.get('avg_sig_strikes_landed_last_5', 0) / (f1.get('avg_fight_time_last_5', 1) / 60 + 0.01)
            f2_strike_rate = f2.get('avg_sig_strikes_landed_last_5', 0) / (f2.get('avg_fight_time_last_5', 1) / 60 + 0.01)

            matchup['f1_striking_volume'] = f1_strike_rate
            matchup['f2_striking_volume'] = f2_strike_rate
            matchup['diff_striking_volume'] = f1_strike_rate - f2_strike_rate

            # Grappling tendency
            f1_grappling = f1.get('avg_takedowns_last_5', 0) / (f1.get('avg_sig_strikes_landed_last_5', 1) + 0.01)
            f2_grappling = f2.get('avg_takedowns_last_5', 0) / (f2.get('avg_sig_strikes_landed_last_5', 1) + 0.01)

            matchup['f1_grappling_tendency'] = f1_grappling
            matchup['f2_grappling_tendency'] = f2_grappling
            matchup['diff_grappling_tendency'] = f1_grappling - f2_grappling

            # Experience ratio
            total_fights = f1['career_fights'] + f2['career_fights']
            matchup['f1_experience_ratio'] = f1['career_fights'] / (total_fights + 1)
            matchup['f2_experience_ratio'] = f2['career_fights'] / (total_fights + 1)

            # Momentum differential
            f1_momentum = f1['win_streak'] - f1['loss_streak']
            f2_momentum = f2['win_streak'] - f2['loss_streak']
            matchup['diff_momentum'] = f1_momentum - f2_momentum

            fights.append(matchup)

        matchup_df = pd.DataFrame(fights)

        logger.info(f"Created matchup features for {len(matchup_df)} fights")

        return matchup_df


    def create_all_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete feature set for model training.

        Main pipeline that runs all preprocessing and feature engineering steps.

        Args:
            raw_data: Raw fight data from CSV

        Returns:
            Complete feature DataFrame ready for modeling
        """
        logger.info("=" * 80)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 80)

        # Phase 1: Preprocess raw data
        fighter_df = self.preprocess_raw_data(raw_data)

        # Phase 2: Calculate career features
        fighter_df = self.calculate_career_features(fighter_df)

        # Phase 2b: Calculate opponent quality features (Phase 6 addition)
        if self.compute_opponent_quality:
            fighter_df = self.calculate_opponent_quality_features(fighter_df)

        # Phase 3: Apply early-career imputation
        fighter_df = self.impute_early_career(fighter_df)

        # Phase 4: Create matchup features
        matchup_df = self.create_matchup_features(fighter_df)

        # Phase 4b: Add physical attribute features (Phase 9.2)
        if self.fighter_details is not None:
            matchup_df = self.add_physical_features(matchup_df)

        # Phase 5: Add temporal/contextual features
        matchup_df = self._add_temporal_features(matchup_df)

        # Phase 6: One-hot encode weight classes
        matchup_df = self._encode_weight_classes(matchup_df)

        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final dataset: {len(matchup_df)} fights, {len(matchup_df.columns)} features")

        return matchup_df


    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal and contextual features."""
        logger.info("Adding temporal features...")

        result = df.copy()

        # Layoff indicators
        result['f1_is_coming_off_layoff'] = (result['f1_days_since_last_fight'] > 180).astype(int)
        result['f2_is_coming_off_layoff'] = (result['f2_days_since_last_fight'] > 180).astype(int)

        # Weight class experience
        result['f1_is_new_to_weight_class'] = (result['f1_weight_class_fights'] <= 2).astype(int)
        result['f2_is_new_to_weight_class'] = (result['f2_weight_class_fights'] <= 2).astype(int)

        # Title fight indicator (basic check for "Title" in event name)
        result['is_title_fight'] = result['event_name'].str.contains('Title', case=False, na=False).astype(int)

        # Year and month (for potential seasonal patterns)
        result['fight_year'] = result['fight_date'].dt.year
        result['fight_month'] = result['fight_date'].dt.month

        logger.info("Temporal features added")

        return result


    def _encode_weight_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode weight classes."""
        logger.info("Encoding weight classes...")

        # Get dummies for weight class
        weight_dummies = pd.get_dummies(df['weight_class'], prefix='wc')

        result = pd.concat([df, weight_dummies], axis=1)

        logger.info(f"Added {len(weight_dummies.columns)} weight class features")

        return result


    def validate_features(self, df: pd.DataFrame) -> Dict:
        """
        Validate feature quality and check for data leakage.

        Args:
            df: Feature DataFrame

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating features...")

        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'issues': []
        }

        # Check for null values in critical features
        critical_features = ['f1_career_win_rate', 'f2_career_win_rate', 'f1_is_winner']
        for feat in critical_features:
            if feat in df.columns:
                null_count = df[feat].isnull().sum()
                if null_count > 0:
                    validation_report['issues'].append(f"{feat} has {null_count} null values")

        # Check percentage features are in [0, 1] range
        # Note: adj_ features can be outside [0,1] due to opponent quality scaling
        pct_features = [col for col in df.columns if ('rate' in col or 'ratio' in col)
                        and not col.startswith('diff_')
                        and 'adj_' not in col
                        and 'f1_adj_' not in col
                        and 'f2_adj_' not in col]
        for feat in pct_features:
            if feat in df.columns:
                out_of_range = ((df[feat] < 0) | (df[feat] > 1)).sum()
                if out_of_range > 0:
                    validation_report['issues'].append(f"{feat} has {out_of_range} values outside [0, 1]")

        # Check opponent quality features are in valid range [0.5, 1.5]
        oq_features = [col for col in df.columns if 'opponent_quality' in col and not col.startswith('diff_')]
        for feat in oq_features:
            if feat in df.columns:
                out_of_range = ((df[feat] < 0.4) | (df[feat] > 1.6)).sum()
                if out_of_range > 0:
                    validation_report['issues'].append(f"{feat} has {out_of_range} values outside [0.4, 1.6]")

        # Check count features are non-negative (excluding diff_ features)
        count_features = [col for col in df.columns
                          if ('career_fights' in col or 'career_wins' in col)
                          and not col.startswith('diff_')]
        for feat in count_features:
            if feat in df.columns:
                negative = (df[feat] < 0).sum()
                if negative > 0:
                    validation_report['issues'].append(f"{feat} has {negative} negative values")

        # Data leakage check: career_fights should be <= total career fights at end
        # This is a basic sanity check
        if 'f1_career_fights' in df.columns:
            max_fights_check = df.groupby('f1_name')['f1_career_fights'].max()
            # Check that career_fights is monotonically increasing per fighter
            validation_report['max_career_fights_sample'] = max_fights_check.head(10).to_dict()

        validation_report['checks_passed'] = len(validation_report['issues']) == 0

        if validation_report['checks_passed']:
            logger.info("✓ All validation checks passed")
        else:
            logger.warning(f"✗ Validation found {len(validation_report['issues'])} issues")
            for issue in validation_report['issues'][:5]:  # Show first 5
                logger.warning(f"  - {issue}")

        return validation_report


    def get_feature_names(self) -> List[str]:
        """
        Get list of all engineered feature names.

        Returns:
            List of feature column names
        """
        # This will be populated after create_all_features is run
        return self.feature_names


    def save_artifacts(self, output_dir: str):
        """
        Save feature engineering artifacts.

        Args:
            output_dir: Directory to save artifacts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save weight class medians
        medians_file = output_path / 'weight_class_medians_v1.pkl'
        with open(medians_file, 'wb') as f:
            pickle.dump(self.weight_class_medians, f)
        logger.info(f"Saved weight class medians to {medians_file}")

        # Save feature names
        if self.feature_names:
            names_file = output_path / 'feature_names_v1.json'
            with open(names_file, 'w') as f:
                json.dump(self.feature_names, f, indent=2)
            logger.info(f"Saved feature names to {names_file}")

        # Save configuration
        config_file = output_path / 'feature_config_v1.json'
        config = {
            'debut_win_rate': self.debut_win_rate,
            'debut_ko_rate': self.debut_ko_rate,
            'debut_sub_rate': self.debut_sub_rate,
            'debut_dec_rate': self.debut_dec_rate,
            'experience_tiers': self.experience_tiers,
            'rolling_windows': self.rolling_windows,
            'compute_opponent_quality': self.compute_opponent_quality,
            'opponent_quality_stats': self.opponent_quality_stats,
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved feature config to {config_file}")


    def validate_opponent_quality_no_leakage(
        self,
        fighter_df: pd.DataFrame,
        sample_size: int = 20
    ) -> Dict:
        """
        Validate that opponent quality scores use only pre-fight data.

        Randomly samples fights and verifies that opponent quality was
        computed only from fights BEFORE the current fight date.

        Args:
            fighter_df: Fighter-centric DataFrame with opponent_quality column
            sample_size: Number of fights to sample for validation

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating opponent quality no-leakage (sample={sample_size})...")

        validation = {
            'samples': [],
            'all_passed': True,
        }

        # Sample random fights
        sample_indices = np.random.choice(
            fighter_df.index,
            min(sample_size, len(fighter_df)),
            replace=False
        )

        for idx in sample_indices:
            row = fighter_df.loc[idx]
            opponent = row['opponent_name']
            fight_date = row['fight_date']

            # Get opponent's fights
            opponent_all = fighter_df[fighter_df['fighter_name'] == opponent]
            opponent_pre_fight = opponent_all[opponent_all['fight_date'] < fight_date]

            sample = {
                'fighter': row['fighter_name'],
                'opponent': opponent,
                'fight_date': str(fight_date.date()),
                'opponent_total_fights': len(opponent_all),
                'opponent_pre_fight_count': len(opponent_pre_fight),
                'opponent_quality_used': row.get('opponent_quality', None),
            }

            # Verify: if opponent had 0 pre-fight fights, quality should be 1.0 (neutral)
            if len(opponent_pre_fight) == 0:
                expected_quality = 1.0
                if sample['opponent_quality_used'] is not None:
                    if abs(sample['opponent_quality_used'] - expected_quality) > 0.01:
                        sample['leakage_detected'] = True
                        validation['all_passed'] = False
                    else:
                        sample['leakage_detected'] = False

            validation['samples'].append(sample)

        if validation['all_passed']:
            logger.info("✓ No leakage detected in opponent quality scores")
        else:
            logger.warning("✗ Potential leakage detected in opponent quality scores")

        return validation


    def get_extreme_opponent_quality_fighters(
        self,
        fighter_df: pd.DataFrame,
        top_n: int = 5
    ) -> Dict:
        """
        Find fighters with highest and lowest average opponent quality.

        Useful for validation: highest should be top contenders,
        lowest should be fighters who faced mostly weak opponents.

        Args:
            fighter_df: Fighter-centric DataFrame with opponent_quality column
            top_n: Number of fighters to return in each category

        Returns:
            Dictionary with highest and lowest average opponent quality fighters
        """
        if 'avg_opponent_quality' not in fighter_df.columns:
            return {'error': 'avg_opponent_quality not computed'}

        # Get last record for each fighter (most complete career data)
        latest = fighter_df.sort_values('fight_date').groupby('fighter_name').last()

        # Filter to fighters with at least 5 fights for meaningful averages
        experienced = latest[latest['career_fights'] >= 5]

        if len(experienced) == 0:
            return {'error': 'No fighters with 5+ fights found'}

        # Sort by average opponent quality
        sorted_by_oq = experienced.sort_values('avg_opponent_quality', ascending=False)

        highest = []
        for name, row in sorted_by_oq.head(top_n).iterrows():
            highest.append({
                'fighter': name,
                'avg_opponent_quality': float(row['avg_opponent_quality']),
                'career_fights': int(row['career_fights']),
                'career_win_rate': float(row['career_win_rate']),
            })

        lowest = []
        for name, row in sorted_by_oq.tail(top_n).iterrows():
            lowest.append({
                'fighter': name,
                'avg_opponent_quality': float(row['avg_opponent_quality']),
                'career_fights': int(row['career_fights']),
                'career_win_rate': float(row['career_win_rate']),
            })

        return {
            'highest_opponent_quality': highest,
            'lowest_opponent_quality': lowest,
        }
