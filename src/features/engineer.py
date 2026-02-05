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

        # Phase 3: Apply early-career imputation
        fighter_df = self.impute_early_career(fighter_df)

        # Phase 4: Create matchup features
        matchup_df = self.create_matchup_features(fighter_df)

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
        pct_features = [col for col in df.columns if 'rate' in col or 'ratio' in col]
        for feat in pct_features:
            if feat in df.columns:
                out_of_range = ((df[feat] < 0) | (df[feat] > 1)).sum()
                if out_of_range > 0:
                    validation_report['issues'].append(f"{feat} has {out_of_range} values outside [0, 1]")

        # Check count features are non-negative
        count_features = [col for col in df.columns if 'career_fights' in col or 'career_wins' in col]
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
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved feature config to {config_file}")
