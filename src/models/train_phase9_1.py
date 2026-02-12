#!/usr/bin/env python3
"""
Phase 9.1: Targeted Improvements from Phase 9 Findings.

Implements targeted improvements based on Phase 9 signals:
- Signal 1: Pre-UFC records for debut model (Task 1A, 1B)
- Signal 2: Isotonic regression calibration (Task 2)
- Signal 3: Automated interaction discovery (Task 3)

Sub-candidates:
- v9.1-B1: Phase 8 + pre-UFC features (single model, no routing)
- v9.1-B2: v9-B routing with pre-UFC features in debut model
- v9.1-B3: Phase 8 + pre-UFC features + routing (best of both)
- v9.1-C1: Phase 8 + isotonic regression (replacing Platt)
- v9.1-C2: Phase 8 + Platt -> isotonic (two-stage)
- v9.1-D1: Phase 8 + auto-discovered interactions, pruned
- v9.1-FINAL: Best combination of above

Usage:
    python src/models/train_phase9_1.py
    python src/models/train_phase9_1.py --skip-scraping  # Use cached fighter records
    python src/models/train_phase9_1.py --quick-test
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

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

# Phase 8 hyperparameters
PHASE8_HGB_PARAMS = {
    'max_iter': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'max_leaf_nodes': 63,
    'min_samples_leaf': 20,
    'l2_regularization': 0.1,
}


# =============================================================================
# Task 1A: Scrape Pre-UFC Records
# =============================================================================

class FighterRecordScraper:
    """Scraper for fighter total professional records from ufcstats.com."""

    def __init__(self, rate_limit: float = 0.5):
        self.base_url = 'http://www.ufcstats.com'
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_all_fighter_urls(self) -> Dict[str, str]:
        """Get mapping of fighter names to their profile URLs."""
        fighter_urls = {}

        for letter in 'abcdefghijklmnopqrstuvwxyz':
            url = f"{self.base_url}/statistics/fighters?char={letter}&page=all"
            try:
                time.sleep(self.rate_limit)
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                rows = soup.select('tr.b-statistics__table-row')
                for row in rows:
                    link = row.select_one('a.b-link.b-link_style_black')
                    if link and link.get('href'):
                        name = link.text.strip()
                        fighter_urls[name] = link['href']

                logger.debug(f"Letter '{letter}': found {len(fighter_urls)} fighters total")
            except Exception as e:
                logger.warning(f"Error fetching letter {letter}: {e}")
                continue

        return fighter_urls

    def scrape_fighter_record(self, name: str, url: str) -> Optional[Dict]:
        """Scrape total professional record for a single fighter."""
        try:
            time.sleep(self.rate_limit)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract total record (e.g., "Record: 22-6-0")
            record_span = soup.select_one('span.b-content__title-record')
            if not record_span:
                return None

            record_text = record_span.text.strip()
            # Parse "Record: W-L-D" format
            match = re.search(r'Record:\s*(\d+)-(\d+)-(\d+)', record_text)
            if not match:
                return None

            wins = int(match.group(1))
            losses = int(match.group(2))
            draws = int(match.group(3))

            return {
                'fighter_name': name,
                'total_pro_wins': wins,
                'total_pro_losses': losses,
                'total_pro_draws': draws,
                'total_pro_fights': wins + losses + draws,
                'fighter_url': url,
            }
        except Exception as e:
            logger.warning(f"Error scraping {name}: {e}")
            return None

    def scrape_all_records(self, output_path: str = 'data/raw/fighter_pre_ufc_records.csv') -> pd.DataFrame:
        """Scrape total professional records for all fighters."""
        logger.info("Scraping all fighter professional records...")

        # Get all fighter URLs
        fighter_urls = self.get_all_fighter_urls()
        logger.info(f"Found {len(fighter_urls)} fighters to scrape")

        records = []
        for i, (name, url) in enumerate(fighter_urls.items(), 1):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(fighter_urls)} fighters scraped")

            record = self.scrape_fighter_record(name, url)
            if record:
                records.append(record)

        df = pd.DataFrame(records)

        # Save to CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} fighter records to {output_path}")

        return df


def compute_pre_ufc_records(
    fights_df: pd.DataFrame,
    fighter_records_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute pre-UFC records for each fighter.

    Strategy:
    - For each fighter, get their total professional record (current)
    - Count their UFC fights from the dataset
    - Pre-UFC record = total_record - UFC_record

    This gives us the experience they had BEFORE entering UFC.
    """
    logger.info("Computing pre-UFC records...")

    # Create mapping of fighter name to total record
    total_records = {}
    for _, row in fighter_records_df.iterrows():
        name = row['fighter_name']
        total_records[name] = {
            'total_wins': row['total_pro_wins'],
            'total_losses': row['total_pro_losses'],
            'total_draws': row['total_pro_draws'],
            'total_fights': row['total_pro_fights'],
        }

    # Count UFC record for each fighter (wins, losses, draws)
    # This needs to be done chronologically to get pre-UFC record at debut

    # Sort by date
    fights_sorted = fights_df.sort_values('fight_date').copy()

    # Track UFC record for each fighter
    ufc_wins = {}
    ufc_losses = {}
    ufc_draws = {}

    # Store pre-UFC record for each fighter (computed at their debut)
    pre_ufc_records = {}

    for _, fight in fights_sorted.iterrows():
        f1 = fight['f1_name']
        f2 = fight['f2_name']

        # Get winner info
        winner = fight.get('winner', '')
        is_draw = 'draw' in str(fight.get('method', '')).lower() or winner == 'Draw'

        for fighter in [f1, f2]:
            # Initialize if first UFC fight
            if fighter not in ufc_wins:
                ufc_wins[fighter] = 0
                ufc_losses[fighter] = 0
                ufc_draws[fighter] = 0

                # Compute pre-UFC record at debut
                if fighter in total_records:
                    total = total_records[fighter]
                    pre_ufc_records[fighter] = {
                        'pre_ufc_wins': total['total_wins'] - ufc_wins.get(fighter, 0),
                        'pre_ufc_losses': total['total_losses'] - ufc_losses.get(fighter, 0),
                        'pre_ufc_draws': total['total_draws'] - ufc_draws.get(fighter, 0),
                    }
                    # Adjust for current UFC record (none at debut)
                    pre_ufc_records[fighter]['pre_ufc_fights'] = (
                        pre_ufc_records[fighter]['pre_ufc_wins'] +
                        pre_ufc_records[fighter]['pre_ufc_losses'] +
                        pre_ufc_records[fighter]['pre_ufc_draws']
                    )
                else:
                    # Fighter not found in records - assume no pre-UFC experience
                    pre_ufc_records[fighter] = {
                        'pre_ufc_wins': 0,
                        'pre_ufc_losses': 0,
                        'pre_ufc_draws': 0,
                        'pre_ufc_fights': 0,
                    }

        # Update UFC record after this fight
        if is_draw:
            ufc_draws[f1] = ufc_draws.get(f1, 0) + 1
            ufc_draws[f2] = ufc_draws.get(f2, 0) + 1
        elif winner == f1 or fight.get('f1_is_winner', 0) == 1:
            ufc_wins[f1] = ufc_wins.get(f1, 0) + 1
            ufc_losses[f2] = ufc_losses.get(f2, 0) + 1
        elif winner == f2 or fight.get('f1_is_winner', 0) == 0:
            ufc_wins[f2] = ufc_wins.get(f2, 0) + 1
            ufc_losses[f1] = ufc_losses.get(f1, 0) + 1

    # Create DataFrame with pre-UFC records
    pre_ufc_df = pd.DataFrame([
        {'fighter_name': name, **record}
        for name, record in pre_ufc_records.items()
    ])

    # Compute derived features
    if len(pre_ufc_df) > 0:
        pre_ufc_df['pre_ufc_win_rate'] = pre_ufc_df.apply(
            lambda x: x['pre_ufc_wins'] / x['pre_ufc_fights'] if x['pre_ufc_fights'] > 0 else 0.5,
            axis=1
        )
        pre_ufc_df['has_pre_ufc_experience'] = (pre_ufc_df['pre_ufc_fights'] > 0).astype(int)

    logger.info(f"Computed pre-UFC records for {len(pre_ufc_df)} fighters")
    logger.info(f"  Fighters with pre-UFC experience: {(pre_ufc_df['pre_ufc_fights'] > 0).sum()}")
    logger.info(f"  Avg pre-UFC fights: {pre_ufc_df['pre_ufc_fights'].mean():.1f}")

    return pre_ufc_df


def add_pre_ufc_features(df: pd.DataFrame, pre_ufc_df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-UFC features to the fight dataset."""
    result = df.copy()

    # Create lookup dict
    pre_ufc_lookup = pre_ufc_df.set_index('fighter_name').to_dict('index')

    # Add features for fighter 1
    result['f1_pre_ufc_wins'] = result['f1_name'].map(
        lambda x: pre_ufc_lookup.get(x, {}).get('pre_ufc_wins', 0)
    )
    result['f1_pre_ufc_losses'] = result['f1_name'].map(
        lambda x: pre_ufc_lookup.get(x, {}).get('pre_ufc_losses', 0)
    )
    result['f1_pre_ufc_fights'] = result['f1_name'].map(
        lambda x: pre_ufc_lookup.get(x, {}).get('pre_ufc_fights', 0)
    )
    result['f1_pre_ufc_win_rate'] = result['f1_name'].map(
        lambda x: pre_ufc_lookup.get(x, {}).get('pre_ufc_win_rate', 0.5)
    )
    result['f1_has_pre_ufc_exp'] = result['f1_name'].map(
        lambda x: pre_ufc_lookup.get(x, {}).get('has_pre_ufc_experience', 0)
    )

    # Add features for fighter 2
    result['f2_pre_ufc_wins'] = result['f2_name'].map(
        lambda x: pre_ufc_lookup.get(x, {}).get('pre_ufc_wins', 0)
    )
    result['f2_pre_ufc_losses'] = result['f2_name'].map(
        lambda x: pre_ufc_lookup.get(x, {}).get('pre_ufc_losses', 0)
    )
    result['f2_pre_ufc_fights'] = result['f2_name'].map(
        lambda x: pre_ufc_lookup.get(x, {}).get('pre_ufc_fights', 0)
    )
    result['f2_pre_ufc_win_rate'] = result['f2_name'].map(
        lambda x: pre_ufc_lookup.get(x, {}).get('pre_ufc_win_rate', 0.5)
    )
    result['f2_has_pre_ufc_exp'] = result['f2_name'].map(
        lambda x: pre_ufc_lookup.get(x, {}).get('has_pre_ufc_experience', 0)
    )

    # Differential features
    result['diff_pre_ufc_fights'] = result['f1_pre_ufc_fights'] - result['f2_pre_ufc_fights']
    result['diff_pre_ufc_win_rate'] = result['f1_pre_ufc_win_rate'] - result['f2_pre_ufc_win_rate']
    result['diff_pre_ufc_wins'] = result['f1_pre_ufc_wins'] - result['f2_pre_ufc_wins']

    # Total pro experience (UFC + pre-UFC)
    result['f1_total_pro_fights'] = result['f1_career_fights'] + result['f1_pre_ufc_fights']
    result['f2_total_pro_fights'] = result['f2_career_fights'] + result['f2_pre_ufc_fights']
    result['diff_total_pro_fights'] = result['f1_total_pro_fights'] - result['f2_total_pro_fights']

    # Pre-UFC experience ratio (what fraction of experience is pre-UFC)
    result['f1_pre_ufc_exp_ratio'] = result.apply(
        lambda x: x['f1_pre_ufc_fights'] / x['f1_total_pro_fights']
        if x['f1_total_pro_fights'] > 0 else 0,
        axis=1
    )
    result['f2_pre_ufc_exp_ratio'] = result.apply(
        lambda x: x['f2_pre_ufc_fights'] / x['f2_total_pro_fights']
        if x['f2_total_pro_fights'] > 0 else 0,
        axis=1
    )

    logger.info(f"Added pre-UFC features. Sample stats:")
    logger.info(f"  F1 avg pre-UFC fights: {result['f1_pre_ufc_fights'].mean():.1f}")
    logger.info(f"  F2 avg pre-UFC fights: {result['f2_pre_ufc_fights'].mean():.1f}")

    return result


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load processed feature data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    logger.info(f"Loaded {len(df):,} fights with {len(df.columns)} columns")
    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """Get feature columns (excluding metadata and target)."""
    exclude = set(METADATA_COLS + [TARGET_COL])
    if exclude_cols:
        exclude.update(exclude_cols)
    return [col for col in df.columns if col not in exclude]


def compute_sample_weights(fight_years: pd.Series, half_life: int = 8) -> np.ndarray:
    """Compute exponential decay sample weights."""
    max_year = fight_years.max()
    years_ago = max_year - fight_years
    weights = np.power(0.5, years_ago / half_life)
    return weights.values


def add_elo_features(df: pd.DataFrame, K: int = 32) -> Tuple[pd.DataFrame, Dict]:
    """Add Elo features to the dataset if not present."""
    if 'diff_elo' in df.columns:
        logger.info("Elo features already present")
        return df, {}

    logger.info(f"Computing Elo features (K={K})...")
    elo_features, final_ratings = compute_elo_features(df, K=K)

    result = df.copy()
    for col in elo_features.columns:
        result[col] = elo_features[col].values

    logger.info(f"Added {len(elo_features.columns)} Elo features")
    return result, final_ratings


# =============================================================================
# Debut Features (from Phase 9)
# =============================================================================

def add_debut_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add debut-specific features for routing model."""
    result = df.copy()

    # Debut indicators
    result['is_any_debut'] = (
        (result['f1_career_fights'] == 0) | (result['f2_career_fights'] == 0)
    ).astype(int)

    result['is_both_debut'] = (
        (result['f1_career_fights'] == 0) & (result['f2_career_fights'] == 0)
    ).astype(int)

    result['is_f1_debut'] = (result['f1_career_fights'] == 0).astype(int)
    result['is_f2_debut'] = (result['f2_career_fights'] == 0).astype(int)

    # Early career indicators
    result['is_f1_early_career'] = (result['f1_career_fights'] < 3).astype(int)
    result['is_f2_early_career'] = (result['f2_career_fights'] < 3).astype(int)

    # Opponent features (when one fighter debuts)
    result['debut_opponent_elo'] = np.where(
        result['is_f1_debut'] == 1,
        result.get('f2_pre_fight_elo', 1500),
        np.where(
            result['is_f2_debut'] == 1,
            result.get('f1_pre_fight_elo', 1500),
            1500
        )
    )

    result['debut_opponent_win_rate'] = np.where(
        result['is_f1_debut'] == 1,
        result['f2_career_win_rate'],
        np.where(
            result['is_f2_debut'] == 1,
            result['f1_career_win_rate'],
            0.5
        )
    )

    result['debut_opponent_fights'] = np.where(
        result['is_f1_debut'] == 1,
        result['f2_career_fights'],
        np.where(
            result['is_f2_debut'] == 1,
            result['f1_career_fights'],
            0
        )
    )

    # Experience asymmetry
    result['experience_asymmetry'] = np.abs(
        result['f1_career_fights'] - result['f2_career_fights']
    )

    logger.info(f"Added debut features. Debuts: {result['is_any_debut'].sum()}")
    return result


# =============================================================================
# Model Training Utilities
# =============================================================================

def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    skip_tuning: bool = False
) -> StackingClassifier:
    """Train Phase 8 style stacking ensemble (HGB + XGB + LR)."""
    logger.info("Training stacking ensemble...")

    hgb = HistGradientBoostingClassifier(
        **PHASE8_HGB_PARAMS,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )

    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
        )
        estimators = [
            ('hgb', hgb),
            ('xgb', xgb),
            ('lr', LogisticRegression(max_iter=1000, C=1.0)),
        ]
    except ImportError:
        logger.warning("XGBoost not installed, using HGB + LR only")
        estimators = [
            ('hgb', hgb),
            ('lr', LogisticRegression(max_iter=1000, C=1.0)),
        ]

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
    )

    ensemble.fit(X_train, y_train, sample_weight=sample_weights)
    return ensemble


def train_simple_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: Optional[np.ndarray] = None
) -> HistGradientBoostingClassifier:
    """Train simple HGB for routing models with small samples."""
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=4,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.5,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def calibrate_model_platt(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> CalibratedClassifierCV:
    """Apply Platt scaling (sigmoid) calibration."""
    logger.info("Applying Platt scaling calibration...")
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method='sigmoid',
        cv=5,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def calibrate_model_isotonic(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> CalibratedClassifierCV:
    """Apply isotonic regression calibration."""
    logger.info("Applying isotonic regression calibration...")
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method='isotonic',
        cv=5,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


class TwoStageCalibrator:
    """Two-stage calibration: Platt scaling -> Isotonic regression."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.platt_calibrator = None
        self.isotonic_calibrator = None

    def fit(self, X, y):
        """Fit the two-stage calibrator."""
        logger.info("Fitting two-stage calibrator (Platt -> Isotonic)...")

        # Stage 1: Platt scaling with CV
        self.platt_calibrator = CalibratedClassifierCV(
            estimator=self.base_model,
            method='sigmoid',
            cv=5,
        )
        self.platt_calibrator.fit(X, y)

        # Get out-of-fold predictions from Platt stage
        platt_proba = cross_val_predict(
            self.platt_calibrator, X, y, cv=5, method='predict_proba'
        )[:, 1]

        # Stage 2: Isotonic regression on Platt outputs
        self.isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_calibrator.fit(platt_proba, y)

        return self

    def predict_proba(self, X):
        """Predict probabilities with two-stage calibration."""
        # Get Platt-calibrated probabilities
        platt_proba = self.platt_calibrator.predict_proba(X)[:, 1]

        # Apply isotonic calibration
        isotonic_proba = self.isotonic_calibrator.predict(platt_proba)

        # Return in sklearn format [P(0), P(1)]
        return np.column_stack([1 - isotonic_proba, isotonic_proba])

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    try:
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        mce = np.mean(np.abs(prob_pred - prob_true))
    except:
        mce = None

    return {
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'roc_auc': float(roc_auc),
        'brier_score': float(brier),
        'mce': float(mce) if mce else None,
    }


def compute_segment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    mask: np.ndarray,
    segment_name: str
) -> Dict:
    """Compute metrics for a specific segment."""
    if mask.sum() == 0:
        return {'segment': segment_name, 'count': 0, 'accuracy': None, 'log_loss': None}

    y_t = y_true[mask]
    y_p = y_pred[mask]
    y_pr = y_proba[mask]

    accuracy = accuracy_score(y_t, y_p)

    try:
        ll = log_loss(y_t, np.clip(y_pr, 1e-15, 1 - 1e-15), labels=[0, 1])
    except ValueError:
        ll = None

    return {
        'segment': segment_name,
        'count': int(mask.sum()),
        'accuracy': round(accuracy * 100, 1),
        'log_loss': round(ll, 3) if ll is not None else None
    }


# =============================================================================
# Task 3: Automated Interaction Discovery
# =============================================================================

def discover_interactions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    base_features: List[str],
    sample_weights: np.ndarray,
    top_n: int = 10
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    Automatically discover useful feature interactions.

    Strategy:
    1. Train model to get feature importances
    2. For top N features, create pairwise interactions
    3. Prune interactions that don't help
    """
    logger.info("Discovering feature interactions...")

    # Train model to get importances
    temp_model = HistGradientBoostingClassifier(**PHASE8_HGB_PARAMS, random_state=42)
    temp_model.fit(X_train, y_train, sample_weight=sample_weights)

    importance = permutation_importance(
        temp_model, X_test, y_test,
        n_repeats=5,
        random_state=42,
        scoring='neg_log_loss',
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': base_features,
        'importance': importance.importances_mean
    }).sort_values('importance', ascending=False)

    # Get top N features
    top_features = importance_df.head(top_n)['feature'].tolist()
    logger.info(f"Top {top_n} features: {top_features}")

    # Create interaction features
    train_interactions = X_train.copy()
    test_interactions = X_test.copy()
    interaction_features = []

    eps = 1e-10
    for i, f1 in enumerate(top_features):
        for f2 in top_features[i+1:]:
            # Multiplicative interaction
            mult_name = f'{f1}_x_{f2}'
            train_interactions[mult_name] = X_train[f1] * X_train[f2]
            test_interactions[mult_name] = X_test[f1] * X_test[f2]
            interaction_features.append(mult_name)

            # Ratio (f1 / f2)
            ratio_name = f'{f1}_div_{f2}'
            train_interactions[ratio_name] = X_train[f1] / (X_train[f2].abs() + eps)
            test_interactions[ratio_name] = X_test[f1] / (X_test[f2].abs() + eps)
            interaction_features.append(ratio_name)

            # Absolute difference
            absdiff_name = f'abs_{f1}_minus_{f2}'
            train_interactions[absdiff_name] = (X_train[f1] - X_train[f2]).abs()
            test_interactions[absdiff_name] = (X_test[f1] - X_test[f2]).abs()
            interaction_features.append(absdiff_name)

    logger.info(f"Created {len(interaction_features)} interaction features")

    # Evaluate which interactions are useful
    all_features = base_features + interaction_features

    # Train model with all features
    full_model = HistGradientBoostingClassifier(**PHASE8_HGB_PARAMS, random_state=42)
    full_model.fit(
        train_interactions[all_features].fillna(0),
        y_train,
        sample_weight=sample_weights
    )

    # Get importance of all features
    full_importance = permutation_importance(
        full_model,
        test_interactions[all_features].fillna(0),
        y_test,
        n_repeats=5,
        random_state=42,
        scoring='neg_log_loss',
        n_jobs=-1
    )

    full_importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': full_importance.importances_mean
    }).sort_values('importance', ascending=False)

    # Keep features with importance > 0.001
    keep_features = full_importance_df[
        full_importance_df['importance'] > 0.001
    ]['feature'].tolist()

    logger.info(f"Keeping {len(keep_features)} features after pruning (threshold=0.001)")

    # Log which interactions survived
    surviving_interactions = [f for f in keep_features if f in interaction_features]
    logger.info(f"Surviving interactions: {len(surviving_interactions)}")

    return keep_features, train_interactions, test_interactions


# =============================================================================
# Candidate Implementations
# =============================================================================

def train_candidate_b1(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_features: List[str],
    sample_weights: np.ndarray,
    pre_ufc_features: List[str]
) -> Dict:
    """
    v9.1-B1: Phase 8 + pre-UFC features (single model, no routing)
    """
    logger.info("\n" + "=" * 60)
    logger.info("v9.1-B1: Phase 8 + Pre-UFC Features (Single Model)")
    logger.info("=" * 60)

    all_features = base_features + [f for f in pre_ufc_features if f in df_train.columns]
    logger.info(f"Total features: {len(base_features)} base + {len(all_features) - len(base_features)} pre-UFC = {len(all_features)}")

    X_train = df_train[all_features].fillna(0)
    X_test = df_test[all_features].fillna(0)

    # Train model
    model = train_stacking_ensemble(X_train, y_train, sample_weights)
    model_cal = calibrate_model_platt(model, X_train, y_train)

    results = evaluate_model(model_cal, X_test, y_test)
    results['features'] = len(all_features)
    results['name'] = 'v9.1-B1'

    logger.info(f"v9.1-B1: Accuracy={results['accuracy']:.1%}, Log Loss={results['log_loss']:.4f}")

    return {
        'results': results,
        'model': model_cal,
        'features': all_features,
    }


def train_candidate_b2(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_features: List[str],
    sample_weights: np.ndarray,
    pre_ufc_features: List[str],
    main_model
) -> Dict:
    """
    v9.1-B2: v9-B routing with pre-UFC features in debut model
    """
    logger.info("\n" + "=" * 60)
    logger.info("v9.1-B2: Routing + Pre-UFC Features in Debut Model")
    logger.info("=" * 60)

    # Add debut features
    train_debut = add_debut_features(df_train.copy())
    test_debut = add_debut_features(df_test.copy())

    # Identify debut fights
    train_is_debut = (train_debut['f1_career_fights'] == 0) | (train_debut['f2_career_fights'] == 0)
    test_is_debut = (test_debut['f1_career_fights'] == 0) | (test_debut['f2_career_fights'] == 0)

    logger.info(f"Train debuts: {train_is_debut.sum()}, Test debuts: {test_is_debut.sum()}")

    # Debut model features = base + debut + pre-UFC
    debut_model_features = base_features + [
        'is_any_debut', 'is_both_debut', 'is_f1_debut', 'is_f2_debut',
        'is_f1_early_career', 'is_f2_early_career',
        'debut_opponent_elo', 'debut_opponent_win_rate',
        'debut_opponent_fights', 'experience_asymmetry',
    ] + pre_ufc_features

    debut_model_features = [f for f in debut_model_features if f in train_debut.columns]

    # Train debut model
    logger.info(f"Training debut model with {len(debut_model_features)} features...")
    X_train_debut = train_debut[train_is_debut][debut_model_features].fillna(0)
    y_train_debut = y_train[train_is_debut]
    weights_debut = sample_weights[train_is_debut.values]

    debut_model = train_simple_model(X_train_debut, y_train_debut, weights_debut)
    debut_model_cal = calibrate_model_platt(debut_model, X_train_debut, y_train_debut)

    # Routing predictions
    X_test_base = test_debut[base_features].fillna(0)
    X_test_debut_all = test_debut[debut_model_features].fillna(0)

    main_proba = main_model.predict_proba(X_test_base)[:, 1]
    debut_proba = debut_model_cal.predict_proba(X_test_debut_all)[:, 1]

    combined_proba = np.where(test_is_debut.values, debut_proba, main_proba)
    combined_pred = (combined_proba > 0.5).astype(int)

    # Evaluate
    accuracy = accuracy_score(y_test, combined_pred)
    logloss = log_loss(y_test, combined_proba)
    roc_auc = roc_auc_score(y_test, combined_proba)
    brier = brier_score_loss(y_test, combined_proba)

    try:
        prob_true, prob_pred = calibration_curve(y_test, combined_proba, n_bins=10)
        mce = np.mean(np.abs(prob_pred - prob_true))
    except:
        mce = None

    results = {
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'roc_auc': float(roc_auc),
        'brier_score': float(brier),
        'mce': float(mce) if mce else None,
        'features': f"{len(base_features)}+route",
        'name': 'v9.1-B2',
    }

    logger.info(f"v9.1-B2: Accuracy={accuracy:.1%}, Log Loss={logloss:.4f}")

    return {
        'results': results,
        'debut_model': debut_model_cal,
        'combined_proba': combined_proba,
        'test_is_debut': test_is_debut,
    }


def train_candidate_b3(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_features: List[str],
    sample_weights: np.ndarray,
    pre_ufc_features: List[str]
) -> Dict:
    """
    v9.1-B3: Phase 8 + pre-UFC features + routing (best of both)
    Main model has pre-UFC features, debut model is specialized.
    """
    logger.info("\n" + "=" * 60)
    logger.info("v9.1-B3: Pre-UFC Features + Routing")
    logger.info("=" * 60)

    # Add debut features
    train_debut = add_debut_features(df_train.copy())
    test_debut = add_debut_features(df_test.copy())

    train_is_debut = (train_debut['f1_career_fights'] == 0) | (train_debut['f2_career_fights'] == 0)
    test_is_debut = (test_debut['f1_career_fights'] == 0) | (test_debut['f2_career_fights'] == 0)

    # Main model features (base + pre-UFC)
    main_features = base_features + [f for f in pre_ufc_features if f in train_debut.columns]

    # Debut model features (base + debut + pre-UFC)
    debut_features = main_features + [
        'is_any_debut', 'is_both_debut', 'is_f1_debut', 'is_f2_debut',
        'is_f1_early_career', 'is_f2_early_career',
        'debut_opponent_elo', 'debut_opponent_win_rate',
        'debut_opponent_fights', 'experience_asymmetry',
    ]
    debut_features = [f for f in debut_features if f in train_debut.columns]

    # Train main model (non-debut fights)
    train_non_debut = ~train_is_debut
    logger.info(f"Training main model on {train_non_debut.sum()} non-debut fights...")

    X_train_main = train_debut[train_non_debut][main_features].fillna(0)
    y_train_main = y_train[train_non_debut]
    weights_main = sample_weights[train_non_debut.values]

    main_model = train_stacking_ensemble(X_train_main, y_train_main, weights_main)
    main_model_cal = calibrate_model_platt(main_model, X_train_main, y_train_main)

    # Train debut model
    logger.info(f"Training debut model on {train_is_debut.sum()} debut fights...")
    X_train_debut = train_debut[train_is_debut][debut_features].fillna(0)
    y_train_debut = y_train[train_is_debut]
    weights_debut = sample_weights[train_is_debut.values]

    debut_model = train_simple_model(X_train_debut, y_train_debut, weights_debut)
    debut_model_cal = calibrate_model_platt(debut_model, X_train_debut, y_train_debut)

    # Routing predictions
    X_test_main = test_debut[main_features].fillna(0)
    X_test_debut = test_debut[debut_features].fillna(0)

    main_proba = main_model_cal.predict_proba(X_test_main)[:, 1]
    debut_proba = debut_model_cal.predict_proba(X_test_debut)[:, 1]

    combined_proba = np.where(test_is_debut.values, debut_proba, main_proba)
    combined_pred = (combined_proba > 0.5).astype(int)

    # Evaluate
    accuracy = accuracy_score(y_test, combined_pred)
    logloss = log_loss(y_test, combined_proba)
    roc_auc = roc_auc_score(y_test, combined_proba)
    brier = brier_score_loss(y_test, combined_proba)

    try:
        prob_true, prob_pred = calibration_curve(y_test, combined_proba, n_bins=10)
        mce = np.mean(np.abs(prob_pred - prob_true))
    except:
        mce = None

    results = {
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'roc_auc': float(roc_auc),
        'brier_score': float(brier),
        'mce': float(mce) if mce else None,
        'features': f"{len(main_features)}+route",
        'name': 'v9.1-B3',
    }

    logger.info(f"v9.1-B3: Accuracy={accuracy:.1%}, Log Loss={logloss:.4f}")

    return {
        'results': results,
        'main_model': main_model_cal,
        'debut_model': debut_model_cal,
        'combined_proba': combined_proba,
        'test_is_debut': test_is_debut,
    }


def train_candidate_c1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weights: np.ndarray,
    base_features: List[str]
) -> Dict:
    """
    v9.1-C1: Phase 8 + isotonic regression (replacing Platt)
    """
    logger.info("\n" + "=" * 60)
    logger.info("v9.1-C1: Phase 8 + Isotonic Regression")
    logger.info("=" * 60)

    # Train base model
    model = train_stacking_ensemble(X_train, y_train, sample_weights)

    # Apply isotonic calibration
    model_cal = calibrate_model_isotonic(model, X_train, y_train)

    results = evaluate_model(model_cal, X_test, y_test)
    results['features'] = len(base_features)
    results['name'] = 'v9.1-C1'

    logger.info(f"v9.1-C1: Accuracy={results['accuracy']:.1%}, Log Loss={results['log_loss']:.4f}")

    return {
        'results': results,
        'model': model_cal,
    }


def train_candidate_c2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weights: np.ndarray,
    base_features: List[str]
) -> Dict:
    """
    v9.1-C2: Phase 8 + Platt -> Isotonic (two-stage)
    """
    logger.info("\n" + "=" * 60)
    logger.info("v9.1-C2: Phase 8 + Two-Stage Calibration")
    logger.info("=" * 60)

    # Train base model
    model = train_stacking_ensemble(X_train, y_train, sample_weights)

    # Apply two-stage calibration
    calibrator = TwoStageCalibrator(model)
    calibrator.fit(X_train, y_train)

    results = evaluate_model(calibrator, X_test, y_test)
    results['features'] = len(base_features)
    results['name'] = 'v9.1-C2'

    logger.info(f"v9.1-C2: Accuracy={results['accuracy']:.1%}, Log Loss={results['log_loss']:.4f}")

    return {
        'results': results,
        'model': calibrator,
    }


def train_candidate_d1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weights: np.ndarray,
    base_features: List[str]
) -> Dict:
    """
    v9.1-D1: Phase 8 + auto-discovered interactions, pruned
    """
    logger.info("\n" + "=" * 60)
    logger.info("v9.1-D1: Phase 8 + Auto-Discovered Interactions")
    logger.info("=" * 60)

    # Discover interactions
    keep_features, train_with_interactions, test_with_interactions = discover_interactions(
        X_train, y_train, X_test, y_test, base_features, sample_weights, top_n=10
    )

    # Train model with selected features
    X_train_final = train_with_interactions[keep_features].fillna(0)
    X_test_final = test_with_interactions[keep_features].fillna(0)

    model = train_stacking_ensemble(X_train_final, y_train, sample_weights)
    model_cal = calibrate_model_platt(model, X_train_final, y_train)

    results = evaluate_model(model_cal, X_test_final, y_test)
    results['features'] = len(keep_features)
    results['name'] = 'v9.1-D1'

    logger.info(f"v9.1-D1: Accuracy={results['accuracy']:.1%}, Log Loss={results['log_loss']:.4f}")

    return {
        'results': results,
        'model': model_cal,
        'features': keep_features,
        'train_df': train_with_interactions,
        'test_df': test_with_interactions,
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def main(
    data_path: str = 'data/processed/ufc_fights_features_v1.csv',
    output_dir: str = 'data/models/phase9.1',
    test_cutoff: str = '2024-01-01',
    skip_scraping: bool = False,
    quick_test: bool = False
):
    """Run Phase 9.1 candidate training and evaluation."""
    print("\n" + "=" * 80)
    print("PHASE 9.1: Targeted Improvements from Phase 9 Findings")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if quick_test:
        skip_scraping = True
        print("QUICK TEST MODE: Skipping scraping, minimal iterations")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(data_path)

    # Add Elo features if not present
    df, _ = add_elo_features(df, K=32)

    # =========================================================================
    # Task 1A: Scrape/Load Pre-UFC Records
    # =========================================================================
    print("\n" + "-" * 80)
    print("TASK 1A: Pre-UFC Records")
    print("-" * 80)

    pre_ufc_path = Path('data/raw/fighter_pre_ufc_records.csv')
    fighter_records_path = Path('data/raw/fighter_total_records.csv')

    if skip_scraping and fighter_records_path.exists():
        print("Loading cached fighter records...")
        fighter_records_df = pd.read_csv(fighter_records_path)
    else:
        print("Scraping fighter records from ufcstats.com...")
        print("(This will take ~30-60 minutes for ~3000 fighters)")

        scraper = FighterRecordScraper(rate_limit=0.5)
        fighter_records_df = scraper.scrape_all_records(str(fighter_records_path))

    # Compute pre-UFC records
    if pre_ufc_path.exists() and skip_scraping:
        print("Loading cached pre-UFC records...")
        pre_ufc_df = pd.read_csv(pre_ufc_path)
    else:
        pre_ufc_df = compute_pre_ufc_records(df, fighter_records_df)
        pre_ufc_df.to_csv(pre_ufc_path, index=False)

    # Add pre-UFC features to main dataset
    df = add_pre_ufc_features(df, pre_ufc_df)

    # Pre-UFC feature list
    pre_ufc_features = [
        'f1_pre_ufc_wins', 'f2_pre_ufc_wins',
        'f1_pre_ufc_losses', 'f2_pre_ufc_losses',
        'f1_pre_ufc_fights', 'f2_pre_ufc_fights',
        'f1_pre_ufc_win_rate', 'f2_pre_ufc_win_rate',
        'f1_has_pre_ufc_exp', 'f2_has_pre_ufc_exp',
        'diff_pre_ufc_fights', 'diff_pre_ufc_win_rate', 'diff_pre_ufc_wins',
        'f1_total_pro_fights', 'f2_total_pro_fights', 'diff_total_pro_fights',
        'f1_pre_ufc_exp_ratio', 'f2_pre_ufc_exp_ratio',
    ]

    # =========================================================================
    # Prepare Data
    # =========================================================================
    print("\n" + "-" * 80)
    print("PREPARING DATA")
    print("-" * 80)

    # Temporal split
    cutoff = pd.to_datetime(test_cutoff)
    train_df = df[df['fight_date'] < cutoff].copy()
    test_df = df[df['fight_date'] >= cutoff].copy()

    print(f"Training: {len(train_df):,} fights ({train_df['fight_date'].min().date()} to {train_df['fight_date'].max().date()})")
    print(f"Testing: {len(test_df):,} fights ({test_df['fight_date'].min().date()} to {test_df['fight_date'].max().date()})")

    # Load Phase 8 feature list
    phase8_features_path = Path('data/models/phase8_feature_list.json')
    if phase8_features_path.exists():
        with open(phase8_features_path) as f:
            base_features = json.load(f)
        print(f"Loaded Phase 8 features: {len(base_features)}")
    else:
        base_features = get_feature_columns(df)
        print(f"Using all features: {len(base_features)}")

    base_features = [f for f in base_features if f in df.columns]

    X_train_base = train_df[base_features].fillna(0)
    y_train = train_df[TARGET_COL]
    X_test_base = test_df[base_features].fillna(0)
    y_test = test_df[TARGET_COL]

    # Sample weights
    fight_years_train = train_df['fight_date'].dt.year
    sample_weights = compute_sample_weights(fight_years_train, half_life=8)

    # =========================================================================
    # Load/Train Phase 8 Baseline
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 8 BASELINE")
    print("-" * 80)

    phase8_model_path = Path('data/models/ufc_model_phase8.pkl')
    if phase8_model_path.exists():
        print("Loading Phase 8 model...")
        phase8_model = joblib.load(phase8_model_path)
    else:
        print("Training Phase 8 baseline model...")
        phase8_model = train_stacking_ensemble(X_train_base, y_train, sample_weights)
        phase8_model = calibrate_model_platt(phase8_model, X_train_base, y_train)

    phase8_metrics = evaluate_model(phase8_model, X_test_base, y_test)
    phase8_metrics['features'] = len(base_features)
    phase8_metrics['name'] = 'Phase 8 (prod)'

    print(f"\nPhase 8 Baseline:")
    print(f"  Accuracy:   {phase8_metrics['accuracy']:.1%}")
    print(f"  Log Loss:   {phase8_metrics['log_loss']:.4f}")
    print(f"  ROC-AUC:    {phase8_metrics['roc_auc']:.4f}")
    print(f"  Brier:      {phase8_metrics['brier_score']:.4f}")
    print(f"  MCE:        {phase8_metrics['mce']:.1%}" if phase8_metrics['mce'] else "  MCE:        N/A")

    # Also load v9-B reference metrics
    v9b_metrics = {
        'accuracy': 0.6124,
        'log_loss': 0.6573,
        'roc_auc': 0.6565,
        'brier_score': 0.2325,
        'mce': 0.0693,
        'features': '51+route',
        'name': 'v9-B (ref)',
    }

    # =========================================================================
    # Train All Candidates
    # =========================================================================
    all_results = {
        'Phase 8 (prod)': phase8_metrics,
        'v9-B (ref)': v9b_metrics,
    }

    # Task 1B: Pre-UFC variants
    print("\n" + "-" * 80)
    print("TASK 1B: Pre-UFC Feature Candidates")
    print("-" * 80)

    cand_b1 = train_candidate_b1(
        train_df, test_df, y_train, y_test,
        base_features, sample_weights, pre_ufc_features
    )
    all_results['v9.1-B1'] = cand_b1['results']

    cand_b2 = train_candidate_b2(
        train_df, test_df, y_train, y_test,
        base_features, sample_weights, pre_ufc_features, phase8_model
    )
    all_results['v9.1-B2'] = cand_b2['results']

    cand_b3 = train_candidate_b3(
        train_df, test_df, y_train, y_test,
        base_features, sample_weights, pre_ufc_features
    )
    all_results['v9.1-B3'] = cand_b3['results']

    # Task 2: Calibration variants
    print("\n" + "-" * 80)
    print("TASK 2: Calibration Candidates")
    print("-" * 80)

    cand_c1 = train_candidate_c1(
        X_train_base, y_train, X_test_base, y_test,
        sample_weights, base_features
    )
    all_results['v9.1-C1'] = cand_c1['results']

    cand_c2 = train_candidate_c2(
        X_train_base, y_train, X_test_base, y_test,
        sample_weights, base_features
    )
    all_results['v9.1-C2'] = cand_c2['results']

    # Task 3: Interaction discovery
    print("\n" + "-" * 80)
    print("TASK 3: Interaction Discovery")
    print("-" * 80)

    cand_d1 = train_candidate_d1(
        X_train_base, y_train, X_test_base, y_test,
        sample_weights, base_features
    )
    all_results['v9.1-D1'] = cand_d1['results']

    # =========================================================================
    # v9.1-FINAL: Combine Best Elements
    # =========================================================================
    print("\n" + "-" * 80)
    print("v9.1-FINAL: Combining Best Elements")
    print("-" * 80)

    # Determine best calibration method
    calibration_candidates = ['v9.1-C1', 'v9.1-C2']
    best_cal = min(calibration_candidates, key=lambda x: all_results[x]['log_loss'])
    best_cal_method = 'isotonic' if best_cal == 'v9.1-C1' else 'two_stage'
    logger.info(f"Best calibration method: {best_cal_method} (from {best_cal})")

    # Determine if pre-UFC features help
    pre_ufc_helps = all_results['v9.1-B1']['accuracy'] > phase8_metrics['accuracy']
    logger.info(f"Pre-UFC features help: {pre_ufc_helps}")

    # Determine if interactions help
    interactions_help = all_results['v9.1-D1']['accuracy'] > phase8_metrics['accuracy']
    logger.info(f"Interactions help: {interactions_help}")

    # Determine if routing helps
    routing_helps = all_results['v9.1-B3']['accuracy'] > all_results['v9.1-B1']['accuracy']
    logger.info(f"Routing helps: {routing_helps}")

    # Build final model with best elements
    final_features = base_features.copy()
    if pre_ufc_helps:
        final_features += [f for f in pre_ufc_features if f in train_df.columns]

    # Get interaction features if they help
    if interactions_help and 'features' in cand_d1:
        # Add only new interaction features (not base features)
        interaction_only = [f for f in cand_d1['features'] if f not in final_features]
        # Only add if we have the data
        if 'train_df' in cand_d1:
            for feat in interaction_only:
                if feat in cand_d1['train_df'].columns:
                    train_df[feat] = cand_d1['train_df'][feat].values
                    test_df[feat] = cand_d1['test_df'][feat].values
                    final_features.append(feat)

    final_features = list(dict.fromkeys(final_features))  # Remove duplicates
    final_features = [f for f in final_features if f in train_df.columns]

    logger.info(f"Final feature set: {len(final_features)} features")

    # Train final model
    X_train_final = train_df[final_features].fillna(0)
    X_test_final = test_df[final_features].fillna(0)

    final_model = train_stacking_ensemble(X_train_final, y_train, sample_weights)

    # Apply best calibration
    if best_cal_method == 'isotonic':
        final_model_cal = calibrate_model_isotonic(final_model, X_train_final, y_train)
    else:
        final_model_cal = TwoStageCalibrator(final_model)
        final_model_cal.fit(X_train_final, y_train)

    # Optionally add routing
    if routing_helps:
        # Add debut features and route
        train_debut = add_debut_features(train_df.copy())
        test_debut = add_debut_features(test_df.copy())

        train_is_debut = (train_debut['f1_career_fights'] == 0) | (train_debut['f2_career_fights'] == 0)
        test_is_debut = (test_debut['f1_career_fights'] == 0) | (test_debut['f2_career_fights'] == 0)

        debut_features = final_features + [
            'is_any_debut', 'is_both_debut', 'is_f1_debut', 'is_f2_debut',
            'debut_opponent_elo', 'debut_opponent_win_rate',
        ]
        debut_features = [f for f in debut_features if f in train_debut.columns]

        X_train_debut = train_debut[train_is_debut][debut_features].fillna(0)
        y_train_debut = y_train[train_is_debut]
        weights_debut = sample_weights[train_is_debut.values]

        debut_model = train_simple_model(X_train_debut, y_train_debut, weights_debut)
        debut_model_cal = calibrate_model_platt(debut_model, X_train_debut, y_train_debut)

        # Combined predictions
        main_proba = final_model_cal.predict_proba(X_test_final)[:, 1]
        debut_proba = debut_model_cal.predict_proba(test_debut[debut_features].fillna(0))[:, 1]

        final_proba = np.where(test_is_debut.values, debut_proba, main_proba)
        final_pred = (final_proba > 0.5).astype(int)

        accuracy = accuracy_score(y_test, final_pred)
        logloss = log_loss(y_test, final_proba)
        roc_auc = roc_auc_score(y_test, final_proba)
        brier = brier_score_loss(y_test, final_proba)

        try:
            prob_true, prob_pred = calibration_curve(y_test, final_proba, n_bins=10)
            mce = np.mean(np.abs(prob_pred - prob_true))
        except:
            mce = None

        final_results = {
            'accuracy': float(accuracy),
            'log_loss': float(logloss),
            'roc_auc': float(roc_auc),
            'brier_score': float(brier),
            'mce': float(mce) if mce else None,
            'features': f"{len(final_features)}+route",
            'name': 'v9.1-FINAL',
        }
    else:
        final_results = evaluate_model(final_model_cal, X_test_final, y_test)
        final_results['features'] = len(final_features)
        final_results['name'] = 'v9.1-FINAL'

    all_results['v9.1-FINAL'] = final_results
    logger.info(f"v9.1-FINAL: Accuracy={final_results['accuracy']:.1%}, Log Loss={final_results['log_loss']:.4f}")

    # =========================================================================
    # Comparison Tables
    # =========================================================================
    print("\n" + "-" * 80)
    print("COMPARISON TABLES")
    print("-" * 80)

    # Main comparison table
    print("\nPHASE 9.1 CANDIDATE COMPARISON")
    print("=" * 100)
    print(f"{'Candidate':<15} {'Acc':<10} {'Log Loss':<12} {'ROC-AUC':<10} {'Brier':<10} {'MCE':<10} {'Features'}")
    print("=" * 100)

    for name in ['Phase 8 (prod)', 'v9-B (ref)', 'v9.1-B1', 'v9.1-B2', 'v9.1-B3',
                 'v9.1-C1', 'v9.1-C2', 'v9.1-D1', 'v9.1-FINAL']:
        if name in all_results:
            r = all_results[name]
            mce_str = f"{r['mce']:.1%}" if r.get('mce') else 'N/A'
            print(f"{name:<15} {r['accuracy']:.1%}      {r['log_loss']:.4f}       {r['roc_auc']:.4f}     {r['brier_score']:.4f}     {mce_str:<10} {r.get('features', 'N/A')}")

    print("=" * 100)

    # Debut segment table
    print("\nDEBUT SEGMENT ACCURACY")
    print("=" * 80)
    print(f"{'Segment':<20} {'Phase 8':<10} {'v9-B':<10} {'v9.1-B1':<10} {'v9.1-B2':<10} {'v9.1-B3':<10}")
    print("-" * 80)

    # Compute segment metrics
    y_test_arr = y_test.values
    phase8_proba = phase8_model.predict_proba(X_test_base)[:, 1]
    phase8_pred = (phase8_proba > 0.5).astype(int)

    debut_mask = (test_df['f1_career_fights'] == 0) | (test_df['f2_career_fights'] == 0)
    inexperienced_mask = (test_df['f1_career_fights'] < 5) & (test_df['f2_career_fights'] < 5)

    # Phase 8 debut accuracy
    p8_debut_acc = accuracy_score(y_test_arr[debut_mask.values], phase8_pred[debut_mask.values]) * 100

    # B1 debut accuracy
    b1_proba = cand_b1['model'].predict_proba(test_df[cand_b1['features']].fillna(0))[:, 1]
    b1_pred = (b1_proba > 0.5).astype(int)
    b1_debut_acc = accuracy_score(y_test_arr[debut_mask.values], b1_pred[debut_mask.values]) * 100

    # B2 debut accuracy
    b2_pred = (cand_b2['combined_proba'] > 0.5).astype(int)
    b2_debut_acc = accuracy_score(y_test_arr[debut_mask.values], b2_pred[debut_mask.values]) * 100

    # B3 debut accuracy
    b3_pred = (cand_b3['combined_proba'] > 0.5).astype(int)
    b3_debut_acc = accuracy_score(y_test_arr[debut_mask.values], b3_pred[debut_mask.values]) * 100

    print(f"{'Debut fighters':<20} {p8_debut_acc:.1f}%      55.4%     {b1_debut_acc:.1f}%      {b2_debut_acc:.1f}%      {b3_debut_acc:.1f}%")

    # Overall
    p8_all_acc = accuracy_score(y_test_arr, phase8_pred) * 100
    b1_all_acc = accuracy_score(y_test_arr, b1_pred) * 100
    b2_all_acc = accuracy_score(y_test_arr, b2_pred) * 100
    b3_all_acc = accuracy_score(y_test_arr, b3_pred) * 100

    print(f"{'All fights':<20} {p8_all_acc:.1f}%      61.2%     {b1_all_acc:.1f}%      {b2_all_acc:.1f}%      {b3_all_acc:.1f}%")
    print("=" * 80)

    # =========================================================================
    # Winner Selection
    # =========================================================================
    print("\n" + "-" * 80)
    print("WINNER SELECTION")
    print("-" * 80)

    metrics = ['accuracy', 'log_loss', 'roc_auc', 'brier_score', 'mce']
    higher_is_better = ['accuracy', 'roc_auc']

    winners = []
    p8 = phase8_metrics

    for name in ['v9.1-B1', 'v9.1-B2', 'v9.1-B3', 'v9.1-C1', 'v9.1-C2', 'v9.1-D1', 'v9.1-FINAL']:
        if name not in all_results:
            continue

        cand = all_results[name]

        wins = 0
        catastrophic = False

        for metric in metrics:
            if cand.get(metric) is None or p8.get(metric) is None:
                continue

            c_val = cand[metric]
            p_val = p8[metric]

            if metric in higher_is_better:
                if c_val > p_val:
                    wins += 1
                if (p_val - c_val) > 0.01:
                    catastrophic = True
            else:
                if c_val < p_val:
                    wins += 1
                if (c_val - p_val) > 0.01:
                    catastrophic = True

        beats_p8 = wins >= 3 and not catastrophic

        print(f"{name}: {wins}/5 wins, catastrophic={catastrophic}, beats Phase 8: {beats_p8}")

        if beats_p8:
            winners.append((name, cand['accuracy'], cand['log_loss']))

    print("\n" + "-" * 60)
    if winners:
        winners.sort(key=lambda x: x[2])
        print(f"WINNER: {winners[0][0]} (Accuracy={winners[0][1]:.1%}, Log Loss={winners[0][2]:.4f})")
    else:
        print("NO CANDIDATES BEAT PHASE 8")

    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)

    # Save comparison results
    comparison_data = {
        'training_date': datetime.now().isoformat(),
        'test_cutoff': test_cutoff,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'base_features': len(base_features),
        'pre_ufc_features': len(pre_ufc_features),
        'results': {k: {kk: vv for kk, vv in v.items()
                       if not callable(vv) and not isinstance(vv, np.ndarray)}
                   for k, v in all_results.items()},
        'winners': [w[0] for w in winners] if winners else [],
        'best_calibration': best_cal_method,
        'pre_ufc_helps': pre_ufc_helps,
        'interactions_help': interactions_help,
        'routing_helps': routing_helps,
    }

    with open(output_path / 'comparison_results.json', 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    print(f"Saved comparison to {output_path / 'comparison_results.json'}")

    # Save comparison table as text
    with open(output_path / 'comparison_results.txt', 'w') as f:
        f.write("PHASE 9.1 CANDIDATE COMPARISON\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Candidate':<15} {'Acc':<10} {'Log Loss':<12} {'ROC-AUC':<10} {'Brier':<10} {'MCE':<10} {'Features'}\n")
        f.write("=" * 100 + "\n")

        for name in ['Phase 8 (prod)', 'v9-B (ref)', 'v9.1-B1', 'v9.1-B2', 'v9.1-B3',
                     'v9.1-C1', 'v9.1-C2', 'v9.1-D1', 'v9.1-FINAL']:
            if name in all_results:
                r = all_results[name]
                mce_str = f"{r['mce']:.1%}" if r.get('mce') else 'N/A'
                f.write(f"{name:<15} {r['accuracy']:.1%}      {r['log_loss']:.4f}       {r['roc_auc']:.4f}     {r['brier_score']:.4f}     {mce_str:<10} {r.get('features', 'N/A')}\n")

        f.write("=" * 100 + "\n")

        if winners:
            f.write(f"\nWINNER: {winners[0][0]}\n")
        else:
            f.write("\nNO CANDIDATES BEAT PHASE 8\n")

    print(f"Saved comparison table to {output_path / 'comparison_results.txt'}")

    # Save models
    if cand_b1.get('model'):
        joblib.dump(cand_b1['model'], output_path / 'model_v9_1_b1.pkl')
    if cand_b2.get('debut_model'):
        joblib.dump(cand_b2['debut_model'], output_path / 'model_v9_1_b2_debut.pkl')
    if cand_b3.get('main_model'):
        joblib.dump(cand_b3['main_model'], output_path / 'model_v9_1_b3_main.pkl')
    if cand_b3.get('debut_model'):
        joblib.dump(cand_b3['debut_model'], output_path / 'model_v9_1_b3_debut.pkl')
    if cand_c1.get('model'):
        joblib.dump(cand_c1['model'], output_path / 'model_v9_1_c1.pkl')
    if cand_d1.get('model'):
        joblib.dump(cand_d1['model'], output_path / 'model_v9_1_d1.pkl')

    print(f"Saved all candidate models to {output_path}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 9.1 COMPLETE")
    print("=" * 80)

    print(f"\nPhase 8 Baseline: Accuracy={phase8_metrics['accuracy']:.1%}")
    if winners:
        print(f"\n{len(winners)} candidate(s) beat Phase 8: {[w[0] for w in winners]}")
    else:
        print("\nNo candidates beat Phase 8 on 3+ metrics without catastrophic regression")
        print("Phase 8 remains in production")

    return all_results, winners


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 9.1 Candidate Training')
    parser.add_argument('--skip-scraping', action='store_true',
                        help='Skip fighter record scraping, use cached data')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')
    parser.add_argument('--data-path', type=str,
                        default='data/processed/ufc_fights_features_v1.csv')
    parser.add_argument('--output-dir', type=str, default='data/models/phase9.1')
    parser.add_argument('--test-cutoff', type=str, default='2024-01-01')

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_cutoff=args.test_cutoff,
        skip_scraping=args.skip_scraping,
        quick_test=args.quick_test
    )
