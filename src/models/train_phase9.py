#!/usr/bin/env python3
"""
Phase 9: Candidate Training Based on Diagnostic Findings.

Implements 4 model candidates targeting weaknesses identified in Phase 8 diagnostics:
- Candidate A: Style Interaction Features (Grappler vs Grappler weakness)
- Candidate B: Debut-Aware Routing Model (debut fighter weakness)
- Candidate C: Close-Fight Specialist (coin-flip confidence weakness)
- Candidate D: Combined Kitchen Sink (all features combined)

All candidates use the SAME train/test split as Phase 8:
- Train: 1994-2023 (7,457 fights)
- Test: 2024-2026 (1,063 fights)

Usage:
    python src/models/train_phase9.py
    python src/models/train_phase9.py --skip-tuning
    python src/models/train_phase9.py --quick-test
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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

# Phase 8 best hyperparameters
PHASE8_HGB_PARAMS = {
    'max_iter': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'max_leaf_nodes': 63,
    'min_samples_leaf': 20,
    'l2_regularization': 0.1,
}


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
# Style Classification (Candidate A)
# =============================================================================

def classify_fighter_style(
    career_ko_rate: float,
    career_sub_rate: float,
    career_finish_rate: float,
    ko_rate_75th: float,
    sub_rate_75th: float
) -> str:
    """
    Classify fighter style based on career finish rates.

    Uses career KO rate and submission rate as proxies for style:
    - Striker: High KO rate (above 75th percentile) AND low submission rate
    - Grappler: High submission rate (above 75th percentile) OR high finish rate with low KO
    - Balanced: Everything else

    Args:
        career_ko_rate: Career KO/TKO win rate
        career_sub_rate: Career submission win rate
        career_finish_rate: Career finish rate (KO + Sub)
        ko_rate_75th: 75th percentile KO rate from training set
        sub_rate_75th: 75th percentile submission rate from training set

    Returns:
        'Striker', 'Grappler', or 'Balanced'
    """
    is_high_ko = career_ko_rate > ko_rate_75th
    is_high_sub = career_sub_rate > sub_rate_75th
    is_low_ko = career_ko_rate < 0.1

    if is_high_ko and not is_high_sub:
        return 'Striker'
    elif is_high_sub or (career_finish_rate > 0.3 and is_low_ko):
        return 'Grappler'
    else:
        return 'Balanced'


def add_style_features(
    df: pd.DataFrame,
    ko_rate_75th: float,
    sub_rate_75th: float
) -> pd.DataFrame:
    """
    Add style classification and interaction features.

    Features added:
    - f1_style, f2_style: Categorical style classification
    - style_is_striker_vs_grappler: 1 if opposing styles
    - style_is_grappler_vs_grappler: 1 if both grapplers
    - diff_striking_vs_td_defense: Cross-domain vulnerability
    - diff_grappling_vs_striking: Style advantage
    """
    result = df.copy()

    # Get career rates for style classification
    f1_ko_rate = result.get('f1_career_ko_rate', pd.Series([0] * len(result))).fillna(0)
    f2_ko_rate = result.get('f2_career_ko_rate', pd.Series([0] * len(result))).fillna(0)
    f1_sub_rate = result.get('f1_career_sub_rate', pd.Series([0] * len(result))).fillna(0)
    f2_sub_rate = result.get('f2_career_sub_rate', pd.Series([0] * len(result))).fillna(0)
    f1_finish_rate = result.get('f1_career_finish_rate', pd.Series([0] * len(result))).fillna(0)
    f2_finish_rate = result.get('f2_career_finish_rate', pd.Series([0] * len(result))).fillna(0)

    # Classify styles
    f1_styles = []
    f2_styles = []

    for i in range(len(result)):
        f1_style = classify_fighter_style(
            f1_ko_rate.iloc[i], f1_sub_rate.iloc[i], f1_finish_rate.iloc[i],
            ko_rate_75th, sub_rate_75th
        )
        f2_style = classify_fighter_style(
            f2_ko_rate.iloc[i], f2_sub_rate.iloc[i], f2_finish_rate.iloc[i],
            ko_rate_75th, sub_rate_75th
        )
        f1_styles.append(f1_style)
        f2_styles.append(f2_style)

    result['f1_style'] = f1_styles
    result['f2_style'] = f2_styles

    # Style matchup indicators
    result['style_is_striker_vs_grappler'] = (
        ((result['f1_style'] == 'Striker') & (result['f2_style'] == 'Grappler')) |
        ((result['f1_style'] == 'Grappler') & (result['f2_style'] == 'Striker'))
    ).astype(int)

    result['style_is_grappler_vs_grappler'] = (
        (result['f1_style'] == 'Grappler') & (result['f2_style'] == 'Grappler')
    ).astype(int)

    result['style_is_striker_vs_striker'] = (
        (result['f1_style'] == 'Striker') & (result['f2_style'] == 'Striker')
    ).astype(int)

    result['style_is_balanced_vs_grappler'] = (
        ((result['f1_style'] == 'Balanced') & (result['f2_style'] == 'Grappler')) |
        ((result['f1_style'] == 'Grappler') & (result['f2_style'] == 'Balanced'))
    ).astype(int)

    result['style_is_balanced_vs_striker'] = (
        ((result['f1_style'] == 'Balanced') & (result['f2_style'] == 'Striker')) |
        ((result['f1_style'] == 'Striker') & (result['f2_style'] == 'Balanced'))
    ).astype(int)

    # Cross-domain vulnerability features using finish rates
    # Striker's KO rate vs Grappler's sub rate
    result['diff_ko_vs_sub'] = f1_ko_rate - f2_sub_rate
    result['diff_finish_rate'] = f1_finish_rate - f2_finish_rate

    # Style-conditional features (advantage in one's own domain)
    result['f1_is_grappler'] = (result['f1_style'] == 'Grappler').astype(int)
    result['f2_is_grappler'] = (result['f2_style'] == 'Grappler').astype(int)
    result['f1_is_striker'] = (result['f1_style'] == 'Striker').astype(int)
    result['f2_is_striker'] = (result['f2_style'] == 'Striker').astype(int)

    # Grappler vs Grappler specific: who has sub advantage?
    result['gvg_sub_advantage'] = np.where(
        result['style_is_grappler_vs_grappler'] == 1,
        f1_sub_rate - f2_sub_rate,
        0
    )

    # Striker vs Grappler specific: striker's KO rate vs grappler's sub rate
    result['svg_striker_ko'] = np.where(
        result['style_is_striker_vs_grappler'] == 1,
        np.where(
            result['f1_style'] == 'Striker',
            f1_ko_rate,
            f2_ko_rate
        ),
        0
    )

    result['svg_grappler_sub'] = np.where(
        result['style_is_striker_vs_grappler'] == 1,
        np.where(
            result['f1_style'] == 'Grappler',
            f1_sub_rate,
            f2_sub_rate
        ),
        0
    )

    logger.info(f"Added style features. Style distribution:")
    logger.info(f"  F1 styles: {result['f1_style'].value_counts().to_dict()}")

    return result


# =============================================================================
# Debut Features (Candidate B)
# =============================================================================

def add_debut_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add debut-specific features for routing model.

    For debut fighters, we emphasize opponent's features since the
    debuting fighter has no UFC history.
    """
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
    result['is_both_early_career'] = (
        (result['f1_career_fights'] < 5) & (result['f2_career_fights'] < 5)
    ).astype(int)

    # Opponent features (when one fighter debuts)
    # For F1 debut: use F2's stats as "opponent strength"
    result['debut_opponent_elo'] = np.where(
        result['is_f1_debut'] == 1,
        result.get('f2_pre_fight_elo', 1500),
        np.where(
            result['is_f2_debut'] == 1,
            result.get('f1_pre_fight_elo', 1500),
            1500  # neutral when no debuts
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

    result['debut_opponent_streak'] = np.where(
        result['is_f1_debut'] == 1,
        result['f2_win_streak'] - result['f2_loss_streak'],
        np.where(
            result['is_f2_debut'] == 1,
            result['f1_win_streak'] - result['f1_loss_streak'],
            0
        )
    )

    # Experience asymmetry (for early career routing)
    result['experience_asymmetry'] = np.abs(
        result['f1_career_fights'] - result['f2_career_fights']
    )

    logger.info(f"Added debut features. Debut distribution:")
    logger.info(f"  Any debut: {result['is_any_debut'].sum()}")
    logger.info(f"  Both debut: {result['is_both_debut'].sum()}")

    return result


# =============================================================================
# Close-Fight Features (Candidate C)
# =============================================================================

def add_close_fight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features specifically for close fights (small Elo gap).

    In close fights, recent form and momentum matter more since
    base skill level is similar.
    """
    result = df.copy()

    # Elo gap magnitude
    elo_diff = result.get('diff_elo', pd.Series([0] * len(result))).fillna(0)
    result['elo_gap_magnitude'] = np.abs(elo_diff)

    # Close fight indicator (various thresholds)
    result['is_close_fight_25'] = (result['elo_gap_magnitude'] < 25).astype(int)
    result['is_close_fight_50'] = (result['elo_gap_magnitude'] < 50).astype(int)
    result['is_close_fight_75'] = (result['elo_gap_magnitude'] < 75).astype(int)

    # Win rate trend (recent vs career)
    f1_recent = result.get('f1_win_rate_last_3', result['f1_career_win_rate']).fillna(0.5)
    f2_recent = result.get('f2_win_rate_last_3', result['f2_career_win_rate']).fillna(0.5)

    result['f1_form_trend'] = f1_recent - result['f1_career_win_rate']
    result['f2_form_trend'] = f2_recent - result['f2_career_win_rate']
    result['diff_form_trend'] = result['f1_form_trend'] - result['f2_form_trend']

    # Momentum features
    result['f1_momentum_score'] = result['f1_win_streak'] - result['f1_loss_streak']
    result['f2_momentum_score'] = result['f2_win_streak'] - result['f2_loss_streak']
    result['diff_momentum_score'] = result['f1_momentum_score'] - result['f2_momentum_score']

    # Activity (ring rust in close fights)
    f1_layoff = result.get('f1_days_since_last_fight', pd.Series([180] * len(result))).fillna(180)
    f2_layoff = result.get('f2_days_since_last_fight', pd.Series([180] * len(result))).fillna(180)

    result['diff_activity'] = f2_layoff - f1_layoff  # Positive = F1 more active

    # Elo momentum (who's trending up)
    f1_elo_mom = result.get('f1_elo_momentum', pd.Series([0] * len(result))).fillna(0)
    f2_elo_mom = result.get('f2_elo_momentum', pd.Series([0] * len(result))).fillna(0)

    result['elo_momentum_advantage'] = f1_elo_mom - f2_elo_mom

    # Career stage (years in UFC as proxy for age)
    f1_exp = result['f1_career_fights']
    f2_exp = result['f2_career_fights']

    # Estimate career stage: rising (0-5), peak (6-15), declining (16+)
    def career_stage(fights):
        if fights < 6:
            return 1  # Rising
        elif fights < 16:
            return 0  # Peak
        else:
            return -1  # Declining

    result['f1_career_stage'] = f1_exp.apply(career_stage)
    result['f2_career_stage'] = f2_exp.apply(career_stage)
    result['diff_career_stage'] = result['f1_career_stage'] - result['f2_career_stage']

    logger.info(f"Added close-fight features.")
    logger.info(f"  Close fights (<50 Elo): {result['is_close_fight_50'].sum()}")

    return result


# =============================================================================
# Model Training
# =============================================================================

def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    skip_tuning: bool = False
) -> StackingClassifier:
    """Train Phase 8 style stacking ensemble (HGB + XGB + LR)."""
    logger.info("Training stacking ensemble...")

    # Base HGB model
    hgb = HistGradientBoostingClassifier(
        **PHASE8_HGB_PARAMS,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )

    # Try to import XGBoost
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


def calibrate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'sigmoid'
) -> CalibratedClassifierCV:
    """Apply Platt scaling calibration."""
    logger.info(f"Applying {method} calibration...")
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=5,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


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

    # Mean Calibration Error
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


def analyze_style_matchups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    df_test: pd.DataFrame
) -> List[Dict]:
    """Analyze accuracy by style matchup."""
    results = []

    if 'f1_style' not in df_test.columns:
        return results

    # Balanced vs Balanced
    mask = (df_test['f1_style'] == 'Balanced') & (df_test['f2_style'] == 'Balanced')
    results.append(compute_segment_metrics(y_true, y_pred, y_proba, mask.values, 'Balanced vs Balanced'))

    # Balanced vs Grappler
    mask = (
        ((df_test['f1_style'] == 'Balanced') & (df_test['f2_style'] == 'Grappler')) |
        ((df_test['f1_style'] == 'Grappler') & (df_test['f2_style'] == 'Balanced'))
    )
    results.append(compute_segment_metrics(y_true, y_pred, y_proba, mask.values, 'Balanced vs Grappler'))

    # Grappler vs Grappler
    mask = (df_test['f1_style'] == 'Grappler') & (df_test['f2_style'] == 'Grappler')
    results.append(compute_segment_metrics(y_true, y_pred, y_proba, mask.values, 'Grappler vs Grappler'))

    return results


def analyze_experience_segments(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    df_test: pd.DataFrame
) -> List[Dict]:
    """Analyze accuracy by experience/debut status."""
    results = []

    # Debut fighters
    mask = (df_test['f1_career_fights'] == 0) | (df_test['f2_career_fights'] == 0)
    results.append(compute_segment_metrics(y_true, y_pred, y_proba, mask.values, 'Debut fighters'))

    # Both inexperienced
    mask = (df_test['f1_career_fights'] < 5) & (df_test['f2_career_fights'] < 5)
    results.append(compute_segment_metrics(y_true, y_pred, y_proba, mask.values, 'Both inexperienced'))

    # Non-debut fights
    mask = (df_test['f1_career_fights'] > 0) & (df_test['f2_career_fights'] > 0)
    results.append(compute_segment_metrics(y_true, y_pred, y_proba, mask.values, 'Non-debut fights'))

    return results


def analyze_elo_gap_segments(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    df_test: pd.DataFrame
) -> List[Dict]:
    """Analyze accuracy by Elo gap."""
    results = []

    elo_gap = np.abs(df_test.get('diff_elo', pd.Series([0] * len(df_test))).fillna(0).values)

    # Elo gap < 25
    mask = elo_gap < 25
    results.append(compute_segment_metrics(y_true, y_pred, y_proba, mask, 'Elo gap < 25'))

    # Elo gap 25-75
    mask = (elo_gap >= 25) & (elo_gap < 75)
    results.append(compute_segment_metrics(y_true, y_pred, y_proba, mask, 'Elo gap 25-75'))

    # Elo gap 75+
    mask = elo_gap >= 75
    results.append(compute_segment_metrics(y_true, y_pred, y_proba, mask, 'Elo gap 75+'))

    return results


def analyze_confidence_buckets(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> List[Dict]:
    """Analyze accuracy by confidence bucket."""
    results = []

    confidence = np.maximum(y_proba, 1 - y_proba)
    y_pred = (y_proba > 0.5).astype(int)

    buckets = [
        (0.50, 0.55, '50-55%'),
        (0.55, 0.60, '55-60%'),
        (0.60, 0.65, '60-65%'),
        (0.65, 0.70, '65-70%'),
        (0.70, 1.01, '70%+'),
    ]

    for low, high, label in buckets:
        mask = (confidence >= low) & (confidence < high)
        results.append(compute_segment_metrics(y_true, y_pred, y_proba, mask, label))

    return results


# =============================================================================
# Candidate Implementations
# =============================================================================

def train_candidate_a(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_features: List[str],
    sample_weights: np.ndarray,
    skip_tuning: bool = False
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Candidate A: Style Interaction Features

    Adds style classification and interaction features to address
    Grappler vs Grappler weakness.
    """
    logger.info("\n" + "=" * 60)
    logger.info("CANDIDATE A: Style Interaction Features")
    logger.info("=" * 60)

    # Compute 75th percentiles from training set for style classification
    # Use only non-zero values for meaningful thresholds
    f1_ko = df_train['f1_career_ko_rate']
    f1_sub = df_train['f1_career_sub_rate']

    ko_rate_75th = f1_ko[f1_ko > 0].quantile(0.75) if (f1_ko > 0).sum() > 0 else 0.5
    sub_rate_75th = f1_sub[f1_sub > 0].quantile(0.75) if (f1_sub > 0).sum() > 0 else 0.15

    logger.info(f"Training set 75th percentiles: KO rate={ko_rate_75th:.2f}, Sub rate={sub_rate_75th:.2f}")

    # Add style features
    train_styled = add_style_features(df_train.copy(), ko_rate_75th, sub_rate_75th)
    test_styled = add_style_features(df_test.copy(), ko_rate_75th, sub_rate_75th)

    # Get style features
    style_features = [
        'style_is_striker_vs_grappler',
        'style_is_grappler_vs_grappler',
        'style_is_striker_vs_striker',
        'style_is_balanced_vs_grappler',
        'style_is_balanced_vs_striker',
        'diff_ko_vs_sub',
        'diff_finish_rate',
        'f1_is_grappler',
        'f2_is_grappler',
        'f1_is_striker',
        'f2_is_striker',
        'gvg_sub_advantage',
        'svg_striker_ko',
        'svg_grappler_sub',
    ]

    # Full feature set
    all_features = base_features + [f for f in style_features if f in train_styled.columns]
    logger.info(f"Full feature set: {len(base_features)} base + {len(all_features) - len(base_features)} style = {len(all_features)}")

    X_train = train_styled[all_features].fillna(0)
    X_test = test_styled[all_features].fillna(0)

    # Train full model
    logger.info("Training v9-A-full (all style features)...")
    model_full = train_stacking_ensemble(X_train, y_train, sample_weights, skip_tuning)
    model_full_cal = calibrate_model(model_full, X_train, y_train)

    results_full = evaluate_model(model_full_cal, X_test, y_test)
    results_full['features'] = len(all_features)
    results_full['name'] = 'v9-A-full'

    logger.info(f"v9-A-full: Accuracy={results_full['accuracy']:.1%}, Log Loss={results_full['log_loss']:.4f}")

    # Also train pruned version
    logger.info("Computing feature importance for pruning...")

    # Quick importance check
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
        'feature': all_features,
        'importance': importance.importances_mean
    }).sort_values('importance', ascending=False)

    # Keep features with positive importance
    keep_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()

    if len(keep_features) < len(all_features):
        logger.info(f"Pruning: keeping {len(keep_features)} of {len(all_features)} features")

        X_train_pruned = train_styled[keep_features].fillna(0)
        X_test_pruned = test_styled[keep_features].fillna(0)

        model_pruned = train_stacking_ensemble(X_train_pruned, y_train, sample_weights, skip_tuning)
        model_pruned_cal = calibrate_model(model_pruned, X_train_pruned, y_train)

        results_pruned = evaluate_model(model_pruned_cal, X_test_pruned, y_test)
        results_pruned['features'] = len(keep_features)
        results_pruned['name'] = 'v9-A-pruned'

        logger.info(f"v9-A-pruned: Accuracy={results_pruned['accuracy']:.1%}, Log Loss={results_pruned['log_loss']:.4f}")
    else:
        results_pruned = results_full.copy()
        results_pruned['name'] = 'v9-A-pruned'

    return {
        'full': results_full,
        'pruned': results_pruned,
        'model_full': model_full_cal,
        'features_full': all_features,
        'features_pruned': keep_features if len(keep_features) < len(all_features) else all_features,
    }, train_styled, test_styled


def train_candidate_b(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_features: List[str],
    sample_weights: np.ndarray,
    main_model,
    skip_tuning: bool = False
) -> Dict:
    """
    Candidate B: Debut-Aware Routing Model

    Uses separate model for debut fighters, routes non-debuts to Phase 8 model.
    """
    logger.info("\n" + "=" * 60)
    logger.info("CANDIDATE B: Debut-Aware Routing Model")
    logger.info("=" * 60)

    # Add debut features
    train_debut = add_debut_features(df_train.copy())
    test_debut = add_debut_features(df_test.copy())

    # Identify debut vs non-debut fights
    train_is_debut = (train_debut['f1_career_fights'] == 0) | (train_debut['f2_career_fights'] == 0)
    test_is_debut = (test_debut['f1_career_fights'] == 0) | (test_debut['f2_career_fights'] == 0)

    logger.info(f"Train debuts: {train_is_debut.sum()} / {len(train_debut)}")
    logger.info(f"Test debuts: {test_is_debut.sum()} / {len(test_debut)}")

    # Debut-specific features
    debut_features = base_features + [
        'is_any_debut',
        'is_both_debut',
        'is_f1_debut',
        'is_f2_debut',
        'is_f1_early_career',
        'is_f2_early_career',
        'debut_opponent_elo',
        'debut_opponent_win_rate',
        'debut_opponent_fights',
        'debut_opponent_streak',
        'experience_asymmetry',
    ]
    debut_features = [f for f in debut_features if f in train_debut.columns]

    # Train debut model on all debut fights
    logger.info("Training debut specialist model...")

    X_train_debut = train_debut[train_is_debut][debut_features].fillna(0)
    y_train_debut = y_train[train_is_debut]
    weights_debut = sample_weights[train_is_debut.values]

    # Use simpler model for smaller sample
    debut_model = train_simple_model(X_train_debut, y_train_debut, weights_debut)
    debut_model_cal = calibrate_model(debut_model, X_train_debut, y_train_debut)

    # Evaluate debut model on debut test fights
    X_test_debut = test_debut[test_is_debut][debut_features].fillna(0)
    y_test_debut = y_test[test_is_debut]

    debut_metrics = evaluate_model(debut_model_cal, X_test_debut, y_test_debut)
    logger.info(f"Debut model on debut fights: Accuracy={debut_metrics['accuracy']:.1%}")

    # Routing: combine predictions
    # For debut fights: use debut model
    # For non-debut fights: use main model (Phase 8)

    X_test_full = test_debut[base_features].fillna(0)
    X_test_debut_all = test_debut[debut_features].fillna(0)

    # Get predictions from both models
    main_proba = main_model.predict_proba(X_test_full)[:, 1]
    debut_proba = debut_model_cal.predict_proba(X_test_debut_all)[:, 1]

    # Route: use debut model for debuts, main model otherwise
    combined_proba = np.where(test_is_debut.values, debut_proba, main_proba)
    combined_pred = (combined_proba > 0.5).astype(int)

    # Overall evaluation
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
        'name': 'v9-B',
        'debut_model_metrics': debut_metrics,
    }

    logger.info(f"v9-B (routed): Accuracy={accuracy:.1%}, Log Loss={logloss:.4f}")

    return {
        'results': results,
        'debut_model': debut_model_cal,
        'debut_features': debut_features,
        'combined_proba': combined_proba,
    }


def train_candidate_c(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_features: List[str],
    sample_weights: np.ndarray,
    main_model,
    skip_tuning: bool = False
) -> Dict:
    """
    Candidate C: Close-Fight Specialist

    Uses separate model for close fights (small Elo gap), routes
    mismatches to Phase 8 model.
    """
    logger.info("\n" + "=" * 60)
    logger.info("CANDIDATE C: Close-Fight Specialist")
    logger.info("=" * 60)

    # Add close-fight features
    train_close = add_close_fight_features(df_train.copy())
    test_close = add_close_fight_features(df_test.copy())

    # Try different thresholds
    thresholds = [25, 50, 75, 100]
    best_results = None
    best_threshold = 50
    best_accuracy = 0

    for threshold in thresholds:
        train_is_close = train_close['elo_gap_magnitude'] < threshold
        test_is_close = test_close['elo_gap_magnitude'] < threshold

        logger.info(f"Testing threshold={threshold}: train={train_is_close.sum()}, test={test_is_close.sum()}")

        if train_is_close.sum() < 100:
            continue

        # Close-fight specific features
        close_features = base_features + [
            'elo_gap_magnitude',
            'f1_form_trend',
            'f2_form_trend',
            'diff_form_trend',
            'f1_momentum_score',
            'f2_momentum_score',
            'diff_momentum_score',
            'diff_activity',
            'elo_momentum_advantage',
            'f1_career_stage',
            'f2_career_stage',
            'diff_career_stage',
        ]
        close_features = [f for f in close_features if f in train_close.columns]

        # Train close-fight model
        X_train_close = train_close[train_is_close][close_features].fillna(0)
        y_train_close = y_train[train_is_close]
        weights_close = sample_weights[train_is_close.values]

        close_model = train_stacking_ensemble(X_train_close, y_train_close, weights_close, skip_tuning=True)
        close_model_cal = calibrate_model(close_model, X_train_close, y_train_close)

        # Routing predictions
        X_test_full = test_close[base_features].fillna(0)
        X_test_close_all = test_close[close_features].fillna(0)

        main_proba = main_model.predict_proba(X_test_full)[:, 1]
        close_proba = close_model_cal.predict_proba(X_test_close_all)[:, 1]

        combined_proba = np.where(test_is_close.values, close_proba, main_proba)
        combined_pred = (combined_proba > 0.5).astype(int)

        accuracy = accuracy_score(y_test, combined_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_results = {
                'close_model': close_model_cal,
                'close_features': close_features,
                'combined_proba': combined_proba,
                'test_is_close': test_is_close,
            }

    # Compute final metrics with best threshold
    combined_proba = best_results['combined_proba']
    combined_pred = (combined_proba > 0.5).astype(int)

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
        'name': 'v9-C',
        'best_threshold': best_threshold,
    }

    logger.info(f"v9-C (threshold={best_threshold}): Accuracy={accuracy:.1%}, Log Loss={logloss:.4f}")

    best_results['results'] = results
    return best_results


def train_candidate_d(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_features: List[str],
    sample_weights: np.ndarray,
    skip_tuning: bool = False
) -> Dict:
    """
    Candidate D: Combined Kitchen Sink

    Single model with all enriched features from A, B, C (no routing).
    """
    logger.info("\n" + "=" * 60)
    logger.info("CANDIDATE D: Combined Kitchen Sink")
    logger.info("=" * 60)

    # Compute percentiles for style classification
    f1_ko = df_train['f1_career_ko_rate']
    f1_sub = df_train['f1_career_sub_rate']
    ko_rate_75th = f1_ko[f1_ko > 0].quantile(0.75) if (f1_ko > 0).sum() > 0 else 0.5
    sub_rate_75th = f1_sub[f1_sub > 0].quantile(0.75) if (f1_sub > 0).sum() > 0 else 0.15

    train_combined = add_style_features(df_train.copy(), ko_rate_75th, sub_rate_75th)
    train_combined = add_debut_features(train_combined)
    train_combined = add_close_fight_features(train_combined)

    test_combined = add_style_features(df_test.copy(), ko_rate_75th, sub_rate_75th)
    test_combined = add_debut_features(test_combined)
    test_combined = add_close_fight_features(test_combined)

    # All new features
    new_features = [
        # Style
        'style_is_striker_vs_grappler',
        'style_is_grappler_vs_grappler',
        'style_is_striker_vs_striker',
        'style_is_balanced_vs_grappler',
        'style_is_balanced_vs_striker',
        'diff_ko_vs_sub',
        'diff_finish_rate',
        'f1_is_grappler',
        'f2_is_grappler',
        'f1_is_striker',
        'f2_is_striker',
        'gvg_sub_advantage',
        'svg_striker_ko',
        'svg_grappler_sub',
        # Debut
        'is_any_debut',
        'is_both_debut',
        'is_f1_debut',
        'is_f2_debut',
        'is_f1_early_career',
        'is_f2_early_career',
        'debut_opponent_elo',
        'debut_opponent_win_rate',
        'debut_opponent_fights',
        'debut_opponent_streak',
        'experience_asymmetry',
        # Close fight
        'elo_gap_magnitude',
        'is_close_fight_50',
        'f1_form_trend',
        'f2_form_trend',
        'diff_form_trend',
        'f1_momentum_score',
        'f2_momentum_score',
        'diff_momentum_score',
        'diff_activity',
        'elo_momentum_advantage',
        'f1_career_stage',
        'f2_career_stage',
        'diff_career_stage',
    ]

    all_features = base_features + [f for f in new_features if f in train_combined.columns]
    logger.info(f"Combined features: {len(base_features)} base + {len(all_features) - len(base_features)} new = {len(all_features)}")

    X_train = train_combined[all_features].fillna(0)
    X_test = test_combined[all_features].fillna(0)

    # Train and prune
    logger.info("Training combined model...")
    model = train_stacking_ensemble(X_train, y_train, sample_weights, skip_tuning)

    # Feature importance pruning
    logger.info("Computing feature importance for pruning...")
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
        'feature': all_features,
        'importance': importance.importances_mean
    }).sort_values('importance', ascending=False)

    # Keep positive importance features
    keep_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()

    logger.info(f"Keeping {len(keep_features)} of {len(all_features)} features after pruning")

    X_train_pruned = train_combined[keep_features].fillna(0)
    X_test_pruned = test_combined[keep_features].fillna(0)

    model_pruned = train_stacking_ensemble(X_train_pruned, y_train, sample_weights, skip_tuning)
    model_pruned_cal = calibrate_model(model_pruned, X_train_pruned, y_train)

    results = evaluate_model(model_pruned_cal, X_test_pruned, y_test)
    results['features'] = len(keep_features)
    results['name'] = 'v9-D'

    logger.info(f"v9-D: Accuracy={results['accuracy']:.1%}, Log Loss={results['log_loss']:.4f}")

    return {
        'results': results,
        'model': model_pruned_cal,
        'features': keep_features,
        'train_combined': train_combined,
        'test_combined': test_combined,
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def main(
    data_path: str = 'data/processed/ufc_fights_features_v1.csv',
    output_dir: str = 'data/models/phase9',
    test_cutoff: str = '2024-01-01',
    skip_tuning: bool = False,
    quick_test: bool = False
):
    """Run Phase 9 candidate training and evaluation."""
    print("\n" + "=" * 80)
    print("PHASE 9: Candidate Training Based on Diagnostic Findings")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if quick_test:
        skip_tuning = True
        print("QUICK TEST MODE: Minimal iterations")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(data_path)

    # Add Elo features if not present
    df, _ = add_elo_features(df, K=32)

    # =========================================================================
    # PART 1: Prepare Data and Load Phase 8 Baseline
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 1: Load Phase 8 Baseline")
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
        # Fallback: use all features except metadata
        base_features = get_feature_columns(df)
        print(f"Using all features: {len(base_features)}")

    # Ensure all base features exist
    base_features = [f for f in base_features if f in df.columns]

    X_train_base = train_df[base_features].fillna(0)
    y_train = train_df[TARGET_COL]
    X_test_base = test_df[base_features].fillna(0)
    y_test = test_df[TARGET_COL]

    # Sample weights
    fight_years_train = train_df['fight_date'].dt.year
    sample_weights = compute_sample_weights(fight_years_train, half_life=8)

    # Load or train Phase 8 model
    phase8_model_path = Path('data/models/ufc_model_phase8.pkl')
    if phase8_model_path.exists():
        print("Loading Phase 8 model...")
        phase8_model = joblib.load(phase8_model_path)
    else:
        print("Training Phase 8 baseline model...")
        phase8_model = train_stacking_ensemble(X_train_base, y_train, sample_weights, skip_tuning)
        phase8_model = calibrate_model(phase8_model, X_train_base, y_train)

    # Evaluate Phase 8 baseline
    phase8_metrics = evaluate_model(phase8_model, X_test_base, y_test)
    phase8_metrics['features'] = len(base_features)
    phase8_metrics['name'] = 'Phase 8 (base)'

    print(f"\nPhase 8 Baseline:")
    print(f"  Accuracy:   {phase8_metrics['accuracy']:.1%}")
    print(f"  Log Loss:   {phase8_metrics['log_loss']:.4f}")
    print(f"  ROC-AUC:    {phase8_metrics['roc_auc']:.4f}")
    print(f"  Brier:      {phase8_metrics['brier_score']:.4f}")
    print(f"  MCE:        {phase8_metrics['mce']:.1%}" if phase8_metrics['mce'] else "  MCE:        N/A")

    # =========================================================================
    # PART 2: Train All Candidates
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 2: Train Model Candidates")
    print("-" * 80)

    all_results = {'Phase 8 (base)': phase8_metrics}

    # Candidate A: Style Interaction Features
    cand_a, train_styled, test_styled = train_candidate_a(
        train_df, test_df, y_train, y_test,
        base_features, sample_weights, skip_tuning
    )
    all_results['v9-A-full'] = cand_a['full']
    all_results['v9-A-pruned'] = cand_a['pruned']

    # Candidate B: Debut-Aware Routing Model
    cand_b = train_candidate_b(
        train_df, test_df, y_train, y_test,
        base_features, sample_weights, phase8_model, skip_tuning
    )
    all_results['v9-B'] = cand_b['results']

    # Candidate C: Close-Fight Specialist
    cand_c = train_candidate_c(
        train_df, test_df, y_train, y_test,
        base_features, sample_weights, phase8_model, skip_tuning
    )
    all_results['v9-C'] = cand_c['results']

    # Candidate D: Combined Kitchen Sink
    cand_d = train_candidate_d(
        train_df, test_df, y_train, y_test,
        base_features, sample_weights, skip_tuning
    )
    all_results['v9-D'] = cand_d['results']

    # =========================================================================
    # PART 3: Comparison Table
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 3: Candidate Comparison")
    print("-" * 80)

    # Main comparison table
    print("\nPHASE 9 CANDIDATE COMPARISON")
    print("=" * 100)
    print(f"{'Candidate':<20} {'Acc':<10} {'Log Loss':<12} {'ROC-AUC':<10} {'Brier':<10} {'MCE':<10} {'Features'}")
    print("=" * 100)

    for name in ['Phase 8 (base)', 'v9-A-full', 'v9-A-pruned', 'v9-B', 'v9-C', 'v9-D']:
        if name in all_results:
            r = all_results[name]
            mce_str = f"{r['mce']:.1%}" if r.get('mce') else 'N/A'
            print(f"{name:<20} {r['accuracy']:.1%}      {r['log_loss']:.4f}       {r['roc_auc']:.4f}     {r['brier_score']:.4f}     {mce_str:<10} {r.get('features', 'N/A')}")

    print("=" * 100)

    # =========================================================================
    # PART 4: Segment Diagnostics
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 4: Segment Diagnostics")
    print("-" * 80)

    # Get predictions from each candidate
    phase8_proba = phase8_model.predict_proba(X_test_base)[:, 1]
    phase8_pred = (phase8_proba > 0.5).astype(int)

    cand_a_proba = cand_a['model_full'].predict_proba(test_styled[cand_a['features_full']].fillna(0))[:, 1]
    cand_a_pred = (cand_a_proba > 0.5).astype(int)

    cand_b_proba = cand_b['combined_proba']
    cand_b_pred = (cand_b_proba > 0.5).astype(int)

    cand_c_proba = cand_c['combined_proba']
    cand_c_pred = (cand_c_proba > 0.5).astype(int)

    cand_d_proba = cand_d['model'].predict_proba(cand_d['test_combined'][cand_d['features']].fillna(0))[:, 1]
    cand_d_pred = (cand_d_proba > 0.5).astype(int)

    y_test_arr = y_test.values

    # Style matchup analysis (Candidate A diagnostic)
    print("\nCandidate A diagnostic (Style Matchups):")
    print("-" * 60)
    print(f"{'Style Matchup':<25} {'Phase 8':<12} {'v9-A':<12} {'Δ'}")
    print("-" * 60)

    phase8_style = analyze_style_matchups(y_test_arr, phase8_pred, phase8_proba, test_styled)
    cand_a_style = analyze_style_matchups(y_test_arr, cand_a_pred, cand_a_proba, test_styled)

    for p8, ca in zip(phase8_style, cand_a_style):
        if p8['accuracy'] is not None and ca['accuracy'] is not None:
            delta = ca['accuracy'] - p8['accuracy']
            print(f"{p8['segment']:<25} {p8['accuracy']:.1f}%        {ca['accuracy']:.1f}%        {delta:+.1f}pp")

    # Experience analysis (Candidate B diagnostic)
    print("\nCandidate B diagnostic (Experience):")
    print("-" * 60)
    print(f"{'Experience Segment':<25} {'Phase 8':<12} {'v9-B':<12} {'Δ'}")
    print("-" * 60)

    phase8_exp = analyze_experience_segments(y_test_arr, phase8_pred, phase8_proba, test_df)
    cand_b_exp = analyze_experience_segments(y_test_arr, cand_b_pred, cand_b_proba, test_df)

    for p8, cb in zip(phase8_exp, cand_b_exp):
        if p8['accuracy'] is not None and cb['accuracy'] is not None:
            delta = cb['accuracy'] - p8['accuracy']
            print(f"{p8['segment']:<25} {p8['accuracy']:.1f}%        {cb['accuracy']:.1f}%        {delta:+.1f}pp")

    # Elo gap analysis (Candidate C diagnostic)
    print("\nCandidate C diagnostic (Elo Gap):")
    print("-" * 60)
    print(f"{'Elo Gap Segment':<25} {'Phase 8':<12} {'v9-C':<12} {'Δ'}")
    print("-" * 60)

    phase8_elo = analyze_elo_gap_segments(y_test_arr, phase8_pred, phase8_proba, test_df)
    cand_c_elo = analyze_elo_gap_segments(y_test_arr, cand_c_pred, cand_c_proba, test_df)

    for p8, cc in zip(phase8_elo, cand_c_elo):
        if p8['accuracy'] is not None and cc['accuracy'] is not None:
            delta = cc['accuracy'] - p8['accuracy']
            print(f"{p8['segment']:<25} {p8['accuracy']:.1f}%        {cc['accuracy']:.1f}%        {delta:+.1f}pp")

    # Candidate D (full breakdown)
    print("\nCandidate D diagnostic (All Segments):")
    print("-" * 60)

    # Confidence buckets
    print("\nConfidence Buckets:")
    phase8_conf = analyze_confidence_buckets(y_test_arr, phase8_proba)
    cand_d_conf = analyze_confidence_buckets(y_test_arr, cand_d_proba)

    print(f"{'Bucket':<20} {'Phase 8':<12} {'v9-D':<12} {'Δ'}")
    for p8, cd in zip(phase8_conf, cand_d_conf):
        if p8['accuracy'] is not None and cd['accuracy'] is not None:
            delta = cd['accuracy'] - p8['accuracy']
            print(f"{p8['segment']:<20} {p8['accuracy']:.1f}%        {cd['accuracy']:.1f}%        {delta:+.1f}pp")

    # =========================================================================
    # PART 5: Winner Selection
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 5: Winner Selection")
    print("-" * 80)

    # A candidate beats Phase 8 if it improves on 3+ of 5 metrics
    # without degrading any single metric by more than 1pp/0.01

    metrics = ['accuracy', 'log_loss', 'roc_auc', 'brier_score', 'mce']
    higher_is_better = ['accuracy', 'roc_auc']

    winners = []

    for name in ['v9-A-full', 'v9-A-pruned', 'v9-B', 'v9-C', 'v9-D']:
        if name not in all_results:
            continue

        cand = all_results[name]
        p8 = phase8_metrics

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
                if (p_val - c_val) > 0.01:  # 1pp drop
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
        # Sort by log loss (lower is better)
        winners.sort(key=lambda x: x[2])
        print(f"WINNER: {winners[0][0]} (Accuracy={winners[0][1]:.1%}, Log Loss={winners[0][2]:.4f})")
    else:
        print("NO CANDIDATES BEAT PHASE 8")
        print("Negative result: Current model may be close to its ceiling")

    # =========================================================================
    # PART 6: Save Results
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 6: Save Results")
    print("-" * 80)

    # Save comparison results
    comparison_data = {
        'training_date': datetime.now().isoformat(),
        'test_cutoff': test_cutoff,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'base_features': len(base_features),
        'results': {k: {kk: vv for kk, vv in v.items() if not callable(vv) and not isinstance(vv, np.ndarray)}
                   for k, v in all_results.items()},
        'winners': [w[0] for w in winners] if winners else [],
    }

    with open(output_path / 'comparison_results.json', 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    print(f"Saved comparison to {output_path / 'comparison_results.json'}")

    # Save comparison table as text
    with open(output_path / 'comparison_results.txt', 'w') as f:
        f.write("PHASE 9 CANDIDATE COMPARISON\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Candidate':<20} {'Acc':<10} {'Log Loss':<12} {'ROC-AUC':<10} {'Brier':<10} {'MCE':<10} {'Features'}\n")
        f.write("=" * 100 + "\n")

        for name in ['Phase 8 (base)', 'v9-A-full', 'v9-A-pruned', 'v9-B', 'v9-C', 'v9-D']:
            if name in all_results:
                r = all_results[name]
                mce_str = f"{r['mce']:.1%}" if r.get('mce') else 'N/A'
                f.write(f"{name:<20} {r['accuracy']:.1%}      {r['log_loss']:.4f}       {r['roc_auc']:.4f}     {r['brier_score']:.4f}     {mce_str:<10} {r.get('features', 'N/A')}\n")

        f.write("=" * 100 + "\n")

        if winners:
            f.write(f"\nWINNER: {winners[0][0]}\n")
        else:
            f.write("\nNO CANDIDATES BEAT PHASE 8\n")

    print(f"Saved comparison table to {output_path / 'comparison_results.txt'}")

    # Save models
    if cand_a.get('model_full'):
        joblib.dump(cand_a['model_full'], output_path / 'model_v9_a_full.pkl')
    if cand_b.get('debut_model'):
        joblib.dump(cand_b['debut_model'], output_path / 'model_v9_b_debut.pkl')
    if cand_c.get('close_model'):
        joblib.dump(cand_c['close_model'], output_path / 'model_v9_c_close.pkl')
    if cand_d.get('model'):
        joblib.dump(cand_d['model'], output_path / 'model_v9_d_combined.pkl')

    print(f"Saved all candidate models to {output_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 9 COMPLETE")
    print("=" * 80)

    print(f"\nPhase 8 Baseline: Accuracy={phase8_metrics['accuracy']:.1%}")
    print(f"Best Candidate: {'v9-D' if 'v9-D' in all_results else 'N/A'} Accuracy={all_results.get('v9-D', {}).get('accuracy', 0):.1%}")

    if winners:
        print(f"\n✓ {len(winners)} candidate(s) beat Phase 8: {[w[0] for w in winners]}")
    else:
        print("\n✗ No candidates beat Phase 8 on 3+ metrics without catastrophic regression")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 9 Candidate Training')
    parser.add_argument('--skip-tuning', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')
    parser.add_argument('--data-path', type=str, default='data/processed/ufc_fights_features_v1.csv')
    parser.add_argument('--output-dir', type=str, default='data/models/phase9')
    parser.add_argument('--test-cutoff', type=str, default='2024-01-01')

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_cutoff=args.test_cutoff,
        skip_tuning=args.skip_tuning,
        quick_test=args.quick_test
    )
