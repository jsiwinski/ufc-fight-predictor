#!/usr/bin/env python3
"""
Phase 9.2 Model Training: Pre-UFC Record + Physical Attributes.

Trains 4 candidate models to evaluate the value of additional features:
- v9.2-A: Phase 9 + Pre-UFC Features only
- v9.2-B: Phase 9 + Physical Attributes only
- v9.2-C: Phase 9 + Pre-UFC + Physical (Full Enhancement)
- v9.2-D: v9.2-C + Debut Routing (special handling for debut fighters)

Pre-UFC Features:
- pre_ufc_wins, pre_ufc_losses, pre_ufc_fights
- pre_ufc_win_rate
- diff_pre_ufc_experience, diff_pre_ufc_win_rate

Physical Attribute Features:
- height_inches, reach_inches, stance (one-hot), age_years
- diff_height, diff_reach, diff_age, reach_advantage

Decision criteria: Candidate must beat Phase 9 on 3+/5 metrics to be promoted.

Usage:
    python src/models/train_phase9_2.py
"""

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.elo import compute_elo_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

METADATA_COLS = [
    'fight_id', 'fight_date', 'event_name', 'weight_class',
    'f1_name', 'f2_name', 'method', 'round'
]
TARGET_COL = 'f1_is_winner'

# Phase 8/9 hyperparameters (kept constant)
HGB_PARAMS = {
    'max_iter': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'max_leaf_nodes': 63,
    'min_samples_leaf': 20,
    'l2_regularization': 0.1,
}

# Phase 9 baseline metrics for comparison
PHASE9_METRICS = {
    'accuracy': 0.6021,
    'log_loss': 0.6521,
    'roc_auc': 0.6607,
    'brier_score': 0.2304,
    'mce': 0.074,
}


# =============================================================================
# Data Loading and Feature Engineering
# =============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load processed feature data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    logger.info(f"Loaded {len(df):,} fights with {len(df.columns)} columns")
    return df


def load_fighter_details(details_path: str) -> pd.DataFrame:
    """Load fighter details (total record, physical attributes)."""
    logger.info(f"Loading fighter details from {details_path}")
    df = pd.read_csv(details_path)
    logger.info(f"Loaded details for {len(df):,} fighters")
    return df


def add_elo_features(df: pd.DataFrame, K: int = 32) -> pd.DataFrame:
    """Add Elo features if not present."""
    if 'diff_elo' in df.columns:
        logger.info("Elo features already present")
        return df

    logger.info(f"Computing Elo features (K={K})...")
    elo_features, _ = compute_elo_features(df, K=K)

    for col in ['f1_elo', 'f2_elo', 'diff_elo',
                'f1_elo_momentum', 'f2_elo_momentum', 'diff_elo_momentum',
                'f1_elo_vs_peak', 'f2_elo_vs_peak', 'diff_elo_vs_peak']:
        if col in elo_features.columns:
            df[col] = elo_features[col].values

    return df


def compute_ufc_record(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute UFC record for each fighter at the time of each fight.

    Returns DataFrame with fighter_name, fight_date, ufc_wins, ufc_losses.
    """
    logger.info("Computing UFC records...")

    # Sort by date
    df_sorted = df.sort_values('fight_date').copy()

    records = []
    fighter_ufc_records = {}  # fighter -> {'wins': 0, 'losses': 0}

    for _, row in df_sorted.iterrows():
        f1 = row['f1_name']
        f2 = row['f2_name']
        fight_date = row['fight_date']
        winner = row.get('f1_is_winner', 0)

        # Initialize if not seen
        if f1 not in fighter_ufc_records:
            fighter_ufc_records[f1] = {'wins': 0, 'losses': 0}
        if f2 not in fighter_ufc_records:
            fighter_ufc_records[f2] = {'wins': 0, 'losses': 0}

        # Record BEFORE this fight (for prediction)
        records.append({
            'fight_id': row['fight_id'],
            'f1_ufc_wins': fighter_ufc_records[f1]['wins'],
            'f1_ufc_losses': fighter_ufc_records[f1]['losses'],
            'f2_ufc_wins': fighter_ufc_records[f2]['wins'],
            'f2_ufc_losses': fighter_ufc_records[f2]['losses'],
        })

        # Update records AFTER the fight
        if winner == 1:  # F1 wins
            fighter_ufc_records[f1]['wins'] += 1
            fighter_ufc_records[f2]['losses'] += 1
        elif winner == 0:  # F2 wins (or draw - count as loss for both)
            fighter_ufc_records[f1]['losses'] += 1
            fighter_ufc_records[f2]['wins'] += 1

    records_df = pd.DataFrame(records)
    logger.info(f"Computed UFC records for {len(records_df):,} fights")
    return records_df


def add_pre_ufc_features(df: pd.DataFrame, fighter_details: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-UFC record features.

    pre_ufc_fights = total_fights - ufc_fights
    pre_ufc_wins = total_wins - ufc_wins
    pre_ufc_losses = total_losses - ufc_losses
    """
    logger.info("Adding pre-UFC record features...")

    # Compute UFC records for each fight
    ufc_records = compute_ufc_record(df)
    df = df.merge(ufc_records, on='fight_id', how='left')

    # Create fighter lookup
    details_lookup = fighter_details.set_index('fighter_name')[
        ['total_wins', 'total_losses', 'total_fights']
    ].to_dict('index')

    # Add pre-UFC features for F1
    f1_pre_ufc_wins = []
    f1_pre_ufc_losses = []
    f1_pre_ufc_fights = []

    for _, row in df.iterrows():
        f1_name = row['f1_name']
        if f1_name in details_lookup:
            total = details_lookup[f1_name]
            pre_wins = max(0, total['total_wins'] - row['f1_ufc_wins'])
            pre_losses = max(0, total['total_losses'] - row['f1_ufc_losses'])
            pre_fights = pre_wins + pre_losses
        else:
            pre_wins = 0
            pre_losses = 0
            pre_fights = 0
        f1_pre_ufc_wins.append(pre_wins)
        f1_pre_ufc_losses.append(pre_losses)
        f1_pre_ufc_fights.append(pre_fights)

    df['f1_pre_ufc_wins'] = f1_pre_ufc_wins
    df['f1_pre_ufc_losses'] = f1_pre_ufc_losses
    df['f1_pre_ufc_fights'] = f1_pre_ufc_fights

    # Add pre-UFC features for F2
    f2_pre_ufc_wins = []
    f2_pre_ufc_losses = []
    f2_pre_ufc_fights = []

    for _, row in df.iterrows():
        f2_name = row['f2_name']
        if f2_name in details_lookup:
            total = details_lookup[f2_name]
            pre_wins = max(0, total['total_wins'] - row['f2_ufc_wins'])
            pre_losses = max(0, total['total_losses'] - row['f2_ufc_losses'])
            pre_fights = pre_wins + pre_losses
        else:
            pre_wins = 0
            pre_losses = 0
            pre_fights = 0
        f2_pre_ufc_wins.append(pre_wins)
        f2_pre_ufc_losses.append(pre_losses)
        f2_pre_ufc_fights.append(pre_fights)

    df['f2_pre_ufc_wins'] = f2_pre_ufc_wins
    df['f2_pre_ufc_losses'] = f2_pre_ufc_losses
    df['f2_pre_ufc_fights'] = f2_pre_ufc_fights

    # Compute win rates (with smoothing)
    df['f1_pre_ufc_win_rate'] = df['f1_pre_ufc_wins'] / (df['f1_pre_ufc_fights'] + 1)
    df['f2_pre_ufc_win_rate'] = df['f2_pre_ufc_wins'] / (df['f2_pre_ufc_fights'] + 1)

    # Differential features
    df['diff_pre_ufc_fights'] = df['f1_pre_ufc_fights'] - df['f2_pre_ufc_fights']
    df['diff_pre_ufc_win_rate'] = df['f1_pre_ufc_win_rate'] - df['f2_pre_ufc_win_rate']

    # Drop intermediate UFC record columns
    df = df.drop(columns=['f1_ufc_wins', 'f1_ufc_losses', 'f2_ufc_wins', 'f2_ufc_losses'])

    logger.info(f"Added 10 pre-UFC features")
    return df


def add_physical_features(df: pd.DataFrame, fighter_details: pd.DataFrame) -> pd.DataFrame:
    """
    Add physical attribute features.

    - height_inches, reach_inches, stance (one-hot), age_at_fight
    - Differential: diff_height, diff_reach, diff_age, reach_advantage
    """
    logger.info("Adding physical attribute features...")

    # Create fighter lookup
    details_lookup = fighter_details.set_index('fighter_name')[
        ['height_inches', 'reach_inches', 'stance', 'dob']
    ].to_dict('index')

    # Parse DOB to datetime for age calculation
    dob_lookup = {}
    for name, data in details_lookup.items():
        dob_str = data.get('dob')
        if pd.notna(dob_str) and dob_str:
            try:
                dob_lookup[name] = pd.to_datetime(dob_str)
            except:
                dob_lookup[name] = None
        else:
            dob_lookup[name] = None

    # Add F1 physical features
    f1_height = []
    f1_reach = []
    f1_stance_orthodox = []
    f1_stance_southpaw = []
    f1_stance_switch = []
    f1_age = []

    for _, row in df.iterrows():
        f1_name = row['f1_name']
        fight_date = pd.to_datetime(row['fight_date'])

        if f1_name in details_lookup:
            data = details_lookup[f1_name]
            height = data['height_inches'] if pd.notna(data['height_inches']) else 70.0  # default
            reach = data['reach_inches'] if pd.notna(data['reach_inches']) else 70.0  # default
            stance = data['stance'] if pd.notna(data['stance']) else 'Orthodox'

            # Age calculation
            dob = dob_lookup.get(f1_name)
            if dob:
                age = (fight_date - dob).days / 365.25
            else:
                age = 30.0  # default
        else:
            height = 70.0
            reach = 70.0
            stance = 'Orthodox'
            age = 30.0

        f1_height.append(height)
        f1_reach.append(reach)
        f1_stance_orthodox.append(1 if stance == 'Orthodox' else 0)
        f1_stance_southpaw.append(1 if stance == 'Southpaw' else 0)
        f1_stance_switch.append(1 if stance == 'Switch' else 0)
        f1_age.append(age)

    df['f1_height_inches'] = f1_height
    df['f1_reach_inches'] = f1_reach
    df['f1_stance_orthodox'] = f1_stance_orthodox
    df['f1_stance_southpaw'] = f1_stance_southpaw
    df['f1_stance_switch'] = f1_stance_switch
    df['f1_age_at_fight'] = f1_age

    # Add F2 physical features
    f2_height = []
    f2_reach = []
    f2_stance_orthodox = []
    f2_stance_southpaw = []
    f2_stance_switch = []
    f2_age = []

    for _, row in df.iterrows():
        f2_name = row['f2_name']
        fight_date = pd.to_datetime(row['fight_date'])

        if f2_name in details_lookup:
            data = details_lookup[f2_name]
            height = data['height_inches'] if pd.notna(data['height_inches']) else 70.0
            reach = data['reach_inches'] if pd.notna(data['reach_inches']) else 70.0
            stance = data['stance'] if pd.notna(data['stance']) else 'Orthodox'

            dob = dob_lookup.get(f2_name)
            if dob:
                age = (fight_date - dob).days / 365.25
            else:
                age = 30.0
        else:
            height = 70.0
            reach = 70.0
            stance = 'Orthodox'
            age = 30.0

        f2_height.append(height)
        f2_reach.append(reach)
        f2_stance_orthodox.append(1 if stance == 'Orthodox' else 0)
        f2_stance_southpaw.append(1 if stance == 'Southpaw' else 0)
        f2_stance_switch.append(1 if stance == 'Switch' else 0)
        f2_age.append(age)

    df['f2_height_inches'] = f2_height
    df['f2_reach_inches'] = f2_reach
    df['f2_stance_orthodox'] = f2_stance_orthodox
    df['f2_stance_southpaw'] = f2_stance_southpaw
    df['f2_stance_switch'] = f2_stance_switch
    df['f2_age_at_fight'] = f2_age

    # Differential features
    df['diff_height'] = df['f1_height_inches'] - df['f2_height_inches']
    df['diff_reach'] = df['f1_reach_inches'] - df['f2_reach_inches']
    df['diff_age'] = df['f1_age_at_fight'] - df['f2_age_at_fight']

    # Reach advantage relative to height (longer reach for same height)
    df['f1_reach_advantage'] = df['f1_reach_inches'] - df['f1_height_inches']
    df['f2_reach_advantage'] = df['f2_reach_inches'] - df['f2_height_inches']
    df['diff_reach_advantage'] = df['f1_reach_advantage'] - df['f2_reach_advantage']

    # Stance matchup (orthodox vs southpaw is interesting)
    df['stance_matchup_mismatch'] = (
        (df['f1_stance_orthodox'] == 1) & (df['f2_stance_southpaw'] == 1) |
        (df['f1_stance_southpaw'] == 1) & (df['f2_stance_orthodox'] == 1)
    ).astype(int)

    logger.info(f"Added 18 physical attribute features")
    return df


def add_debut_routing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add debut routing feature.

    Flags when one or both fighters are making their UFC debut.
    """
    logger.info("Adding debut routing features...")

    # Fighter is debut if career_fights == 0 (before this fight)
    df['f1_is_debut'] = (df['f1_career_fights'] == 0).astype(int)
    df['f2_is_debut'] = (df['f2_career_fights'] == 0).astype(int)
    df['either_debut'] = ((df['f1_is_debut'] == 1) | (df['f2_is_debut'] == 1)).astype(int)
    df['both_debut'] = ((df['f1_is_debut'] == 1) & (df['f2_is_debut'] == 1)).astype(int)

    logger.info("Added 4 debut routing features")
    return df


# =============================================================================
# Model Training
# =============================================================================

def compute_sample_weights(df: pd.DataFrame, half_life_years: float = 8.0) -> np.ndarray:
    """Compute temporal sample weights with exponential decay."""
    max_date = df['fight_date'].max()
    days_ago = (max_date - df['fight_date']).dt.days.values
    decay = np.log(2) / (half_life_years * 365)
    weights = np.exp(-decay * days_ago)
    return weights


def temporal_split(df: pd.DataFrame, test_cutoff: str = '2024-01-01') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally."""
    cutoff = pd.to_datetime(test_cutoff)
    train = df[df['fight_date'] < cutoff].copy()
    test = df[df['fight_date'] >= cutoff].copy()
    logger.info(f"Train: {len(train):,} fights (< {test_cutoff})")
    logger.info(f"Test: {len(test):,} fights (>= {test_cutoff})")
    return train, test


def get_feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    """
    Get feature columns for a given feature set.

    Feature sets:
    - 'base': Phase 9 features (Elo + original features)
    - 'pre_ufc': base + pre-UFC record features
    - 'physical': base + physical attribute features
    - 'full': base + pre_ufc + physical
    - 'debut': full + debut routing features
    """
    # Base features (Phase 9 = Phase 8 features)
    base_exclude = set(METADATA_COLS + [TARGET_COL, 'f1_name', 'f2_name'])

    # Pre-UFC feature columns
    pre_ufc_cols = [
        'f1_pre_ufc_wins', 'f1_pre_ufc_losses', 'f1_pre_ufc_fights', 'f1_pre_ufc_win_rate',
        'f2_pre_ufc_wins', 'f2_pre_ufc_losses', 'f2_pre_ufc_fights', 'f2_pre_ufc_win_rate',
        'diff_pre_ufc_fights', 'diff_pre_ufc_win_rate',
    ]

    # Physical attribute columns
    physical_cols = [
        'f1_height_inches', 'f1_reach_inches',
        'f1_stance_orthodox', 'f1_stance_southpaw', 'f1_stance_switch',
        'f1_age_at_fight', 'f1_reach_advantage',
        'f2_height_inches', 'f2_reach_inches',
        'f2_stance_orthodox', 'f2_stance_southpaw', 'f2_stance_switch',
        'f2_age_at_fight', 'f2_reach_advantage',
        'diff_height', 'diff_reach', 'diff_age', 'diff_reach_advantage',
        'stance_matchup_mismatch',
    ]

    # Debut routing columns
    debut_cols = [
        'f1_is_debut', 'f2_is_debut', 'either_debut', 'both_debut',
    ]

    # Get all numeric columns as base
    all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    base_cols = [c for c in all_cols if c not in base_exclude
                 and c not in pre_ufc_cols
                 and c not in physical_cols
                 and c not in debut_cols]

    if feature_set == 'base':
        return base_cols
    elif feature_set == 'pre_ufc':
        return base_cols + [c for c in pre_ufc_cols if c in df.columns]
    elif feature_set == 'physical':
        return base_cols + [c for c in physical_cols if c in df.columns]
    elif feature_set == 'full':
        return base_cols + [c for c in pre_ufc_cols if c in df.columns] + [c for c in physical_cols if c in df.columns]
    elif feature_set == 'debut':
        return base_cols + [c for c in pre_ufc_cols if c in df.columns] + [c for c in physical_cols if c in df.columns] + [c for c in debut_cols if c in df.columns]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


def build_stacking_ensemble(random_state: int = 42) -> StackingClassifier:
    """Build the Phase 9 stacking ensemble."""
    try:
        from xgboost import XGBClassifier
        use_xgb = True
    except ImportError:
        use_xgb = False
        logger.warning("XGBoost not available, using HGB only")

    base_estimators = [
        ('hgb', HistGradientBoostingClassifier(**HGB_PARAMS, random_state=random_state)),
    ]

    if use_xgb:
        base_estimators.append((
            'xgb', XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbosity=0,
            )
        ))

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
    )

    return stacking


def calibrate_model_isotonic(model, X_train, y_train) -> CalibratedClassifierCV:
    """Apply isotonic regression calibration (Phase 9)."""
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method='isotonic',
        cv=5,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Maximum Calibration Error."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    if len(prob_true) == 0:
        return 0.0
    return np.max(np.abs(prob_true - prob_pred))


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_prob),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob),
        'mce': compute_mce(y_true, y_prob),
    }


def train_candidate(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_set: str,
    name: str,
) -> Tuple[Dict[str, float], object, List[str]]:
    """Train a single candidate model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {name} ({feature_set} features)")
    logger.info('='*60)

    # Get features
    feature_cols = get_feature_columns(df_train, feature_set)
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET_COL].values
    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET_COL].values

    # Sample weights
    weights = compute_sample_weights(df_train)

    # Build and train
    model = build_stacking_ensemble()
    logger.info("Training stacking ensemble...")
    model.fit(X_train, y_train, sample_weight=weights)

    # Calibrate
    logger.info("Applying isotonic calibration...")
    calibrated = calibrate_model_isotonic(model, X_train, y_train)

    # Evaluate
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_prob)

    logger.info(f"Results for {name}:")
    logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"  Log Loss:    {metrics['log_loss']:.4f}")
    logger.info(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
    logger.info(f"  MCE:         {metrics['mce']:.4f}")

    return metrics, calibrated, feature_cols


def compare_to_baseline(metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, bool]:
    """Compare candidate metrics to Phase 9 baseline."""
    improvements = {}

    # Lower is better for these
    lower_better = ['log_loss', 'brier_score', 'mce']

    for key in metrics:
        if key in baseline:
            if key in lower_better:
                improvements[key] = metrics[key] < baseline[key]
            else:
                improvements[key] = metrics[key] > baseline[key]

    return improvements


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Phase 9.2 candidates')
    parser.add_argument('--data-path', default='data/processed/ufc_fights_features_v1.csv')
    parser.add_argument('--details-path', default='data/raw/fighter_details.csv')
    parser.add_argument('--output-dir', default='data/models')
    args = parser.parse_args()

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("PHASE 9.2 MODEL TRAINING")
    logger.info("=" * 60)

    # Load data
    df = load_data(args.data_path)
    fighter_details = load_fighter_details(args.details_path)

    # Add Elo features
    df = add_elo_features(df)

    # Add all new features
    df = add_pre_ufc_features(df, fighter_details)
    df = add_physical_features(df, fighter_details)
    df = add_debut_routing(df)

    # Temporal split
    df_train, df_test = temporal_split(df)

    # Train candidates
    candidates = [
        ('v9.2-A', 'pre_ufc', 'Phase 9 + Pre-UFC Features'),
        ('v9.2-B', 'physical', 'Phase 9 + Physical Attributes'),
        ('v9.2-C', 'full', 'Phase 9 + Pre-UFC + Physical'),
        ('v9.2-D', 'debut', 'Phase 9 + Pre-UFC + Physical + Debut Routing'),
    ]

    results = {}

    for name, feature_set, description in candidates:
        metrics, model, features = train_candidate(df_train, df_test, feature_set, name)

        # Compare to Phase 9
        improvements = compare_to_baseline(metrics, PHASE9_METRICS)
        wins = sum(improvements.values())

        results[name] = {
            'description': description,
            'metrics': metrics,
            'improvements': improvements,
            'wins': wins,
            'promotes': wins >= 3,
        }

        logger.info(f"\nComparison to Phase 9:")
        for metric, improved in improvements.items():
            symbol = "+" if improved else "-"
            p9_val = PHASE9_METRICS[metric]
            new_val = metrics[metric]
            diff = new_val - p9_val
            logger.info(f"  {metric}: {p9_val:.4f} -> {new_val:.4f} ({diff:+.4f}) [{symbol}]")
        logger.info(f"  Wins: {wins}/5 {'-> PROMOTES!' if wins >= 3 else ''}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 9.2 SUMMARY")
    logger.info("=" * 60)

    for name, result in results.items():
        status = "PROMOTES" if result['promotes'] else "REJECTED"
        logger.info(f"{name}: {result['wins']}/5 metrics improved -> {status}")

    # Data leakage warning
    logger.warning("\n*** DATA LEAKAGE WARNING ***")
    logger.warning("Pre-UFC features (v9.2-A, C, D) have data leakage:")
    logger.warning("  - total_record is current (2026)")
    logger.warning("  - ufc_record is historical (at fight time)")
    logger.warning("  - pre_ufc = total - ufc includes future UFC wins")
    logger.warning("Only v9.2-B (physical attributes only) is valid for production.")

    # Select v9.2-B as winner since it's the only valid candidate
    best_name = 'v9.2-B'
    best_result = results[best_name]

    if best_result['promotes']:
        logger.info(f"\nBest candidate: {best_name}")
        logger.info(f"Description: {best_result['description']}")
        logger.info(f"Metrics improved: {best_result['wins']}/5")

        # Save results
        output_path = Path(args.output_dir) / 'phase9_2_evaluation_results.json'
        with open(output_path, 'w') as f:
            json.dump({
                'evaluation_date': datetime.now().isoformat(),
                'baseline': PHASE9_METRICS,
                'candidates': {k: {
                    'description': v['description'],
                    'metrics': {m: float(val) for m, val in v['metrics'].items()},
                    'improvements': {m: bool(i) for m, i in v['improvements'].items()},
                    'wins': int(v['wins']),
                    'promotes': bool(v['promotes']),
                } for k, v in results.items()},
                'best_candidate': best_name,
                'data_leakage_warning': 'Pre-UFC features (v9.2-A, C, D) have data leakage: total_record is current (2026) while ufc_record is historical. Only v9.2-B (physical attributes) is valid.',
            }, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    else:
        logger.info(f"\nNo candidate met the 3/5 threshold.")
        logger.info("Phase 9 remains the production model.")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    return results


if __name__ == '__main__':
    main()
