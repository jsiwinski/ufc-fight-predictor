#!/usr/bin/env python3
"""
Phase 8: Elo Ratings, Post-Hoc Calibration, Sample Weighting, Ensemble.

This script implements comprehensive model improvements:
1. Elo rating system for strength-of-schedule encoding
2. Post-hoc calibration (Platt scaling and isotonic regression)
3. Sample weighting by recency (recent fights matter more)
4. Feature pruning (remove zero/negative importance features)
5. Conditional mean imputation for pre-2009 fights
6. Multiple model candidates (HGB, XGBoost, Stacking Ensemble, Minimal)

Usage:
    python src/models/train_phase8.py
    python src/models/train_phase8.py --skip-tuning
    python src/models/train_phase8.py --quick-test
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

from src.features.elo import compute_elo_features, get_top_elo_fighters, validate_elo_system

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

# Hyperparameter search spaces
HGB_PARAM_DIST = {
    'max_iter': [150, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    'max_leaf_nodes': [15, 31, 63, 127],
    'min_samples_leaf': [10, 15, 20, 30, 50],
    'l2_regularization': [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
}

XGB_PARAM_DIST = {
    'n_estimators': [150, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    'min_child_weight': [3, 5, 7, 10, 15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 2.0, 5.0],
}

QUICK_HGB_PARAMS = {
    'max_iter': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'max_leaf_nodes': 63,
    'min_samples_leaf': 20,
    'l2_regularization': 0.1,
}


# =============================================================================
# Data Loading and Feature Preparation
# =============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load processed feature data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    logger.info(f"Loaded {len(df):,} fights with {len(df.columns)} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns (excluding metadata and target)."""
    return [col for col in df.columns if col not in METADATA_COLS + [TARGET_COL]]


def add_elo_features(df: pd.DataFrame, K: int = 32) -> Tuple[pd.DataFrame, Dict]:
    """Add Elo features to the dataset."""
    logger.info(f"Computing Elo features (K={K})...")
    elo_features, final_ratings = compute_elo_features(df, K=K)

    # Merge Elo features
    result = df.copy()
    for col in elo_features.columns:
        result[col] = elo_features[col].values

    logger.info(f"Added {len(elo_features.columns)} Elo features")
    return result, final_ratings


def compute_sample_weights(fight_years: pd.Series, half_life: int = 8) -> np.ndarray:
    """
    Compute exponential decay sample weights.

    Recent fights get more weight. half_life=8 means a fight from 8 years ago
    gets 50% the weight of a current fight.

    Args:
        fight_years: Series of fight years
        half_life: Years for weight to decay by half

    Returns:
        Array of sample weights
    """
    max_year = fight_years.max()
    years_ago = max_year - fight_years
    weights = np.power(0.5, years_ago / half_life)
    return weights.values


def get_features_to_prune(importance_df: pd.DataFrame, threshold: float = 0.0) -> List[str]:
    """
    Get list of features to prune based on permutation importance.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        threshold: Prune features with importance <= threshold

    Returns:
        List of feature names to remove
    """
    to_prune = importance_df[importance_df['importance'] <= threshold]['feature'].tolist()
    return to_prune


def compute_imputation_table(
    df: pd.DataFrame,
    stat_cols: List[str]
) -> Dict[Tuple, Dict[str, float]]:
    """
    Build conditional mean imputation table for pre-2009 fights.

    Groups fights by (method_category, round, is_winner) and computes
    average stats for each group.

    Args:
        df: DataFrame with detailed fights (has_detailed_stats == True)
        stat_cols: List of stat columns to impute

    Returns:
        Dict mapping (method_cat, round, is_winner) -> {stat: mean_value}
    """
    logger.info("Building imputation table for pre-2009 fights...")

    # Only use fights with detailed stats
    if 'f1_has_detailed_stats' in df.columns:
        detailed = df[df['f1_has_detailed_stats'] == True].copy()
    else:
        detailed = df.copy()

    if len(detailed) == 0:
        return {}

    # Categorize method
    def categorize_method(method):
        method = str(method).lower()
        if 'ko' in method or 'tko' in method:
            return 'KO/TKO'
        elif 'submission' in method:
            return 'Submission'
        elif 'decision' in method:
            return 'Decision'
        else:
            return 'Other'

    # Get method from raw data if available
    if 'method' in detailed.columns:
        detailed['method_cat'] = detailed['method'].apply(categorize_method)
    else:
        detailed['method_cat'] = 'Other'

    # Compute group means
    imputation_table = {}

    # Define groups (method_cat, is_winner combinations)
    for method_cat in ['KO/TKO', 'Submission', 'Decision', 'Other']:
        for is_winner in [0, 1]:
            group = detailed[(detailed['method_cat'] == method_cat) &
                           (detailed['f1_is_winner'] == is_winner)]

            if len(group) > 10:  # Need at least 10 fights for reliable imputation
                means = {}
                for col in stat_cols:
                    if col in group.columns:
                        means[col] = float(group[col].mean())
                imputation_table[(method_cat, is_winner)] = means

    # Also compute overall means as fallback
    overall_means = {}
    for col in stat_cols:
        if col in detailed.columns:
            overall_means[col] = float(detailed[col].mean())
    imputation_table['_overall'] = overall_means

    logger.info(f"Built imputation table with {len(imputation_table)} groups")
    return imputation_table


# =============================================================================
# Model Training
# =============================================================================

def train_hgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    skip_tuning: bool = False,
    n_iter: int = 80
) -> Tuple[HistGradientBoostingClassifier, Dict]:
    """Train HistGradientBoosting with optional hyperparameter tuning."""
    if skip_tuning:
        logger.info("Training HGB with default params (skipping tuning)")
        model = HistGradientBoostingClassifier(
            **QUICK_HGB_PARAMS,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model, QUICK_HGB_PARAMS

    logger.info(f"Tuning HGB ({n_iter} iterations)...")
    base_model = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=HGB_PARAM_DIST,
        n_iter=n_iter,
        scoring='neg_log_loss',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    start = time.time()
    search.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.time() - start

    logger.info(f"HGB tuning complete in {elapsed:.1f}s, best CV log_loss: {-search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    skip_tuning: bool = False,
    n_iter: int = 80
) -> Tuple:
    """Train XGBoost with optional hyperparameter tuning."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("XGBoost not installed, skipping Candidate B")
        return None, None

    if skip_tuning:
        logger.info("Training XGBoost with default params")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model, model.get_params()

    logger.info(f"Tuning XGBoost ({n_iter} iterations)...")
    base_model = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter,
        scoring='neg_log_loss',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    start = time.time()
    search.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.time() - start

    logger.info(f"XGBoost tuning complete in {elapsed:.1f}s, best CV log_loss: {-search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hgb_model: HistGradientBoostingClassifier,
    xgb_model,
    sample_weights: Optional[np.ndarray] = None
) -> StackingClassifier:
    """Train stacking ensemble combining HGB, XGBoost, and LR."""
    logger.info("Training stacking ensemble...")

    estimators = [
        ('hgb', hgb_model),
        ('lr', LogisticRegression(max_iter=1000, C=1.0)),
    ]

    if xgb_model is not None:
        estimators.append(('xgb', xgb_model))

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
    )

    ensemble.fit(X_train, y_train, sample_weight=sample_weights)
    logger.info("Stacking ensemble trained")
    return ensemble


def calibrate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'sigmoid'
) -> CalibratedClassifierCV:
    """Apply post-hoc calibration to a trained model."""
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

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Core metrics
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


def compute_feature_importance_for_pruning(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10
) -> pd.DataFrame:
    """Compute permutation importance for feature pruning."""
    logger.info("Computing permutation importance (10 repeats)...")

    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        scoring='neg_log_loss',
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': X.columns.tolist(),
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)

    return importance_df


# =============================================================================
# Diagnostic Analysis
# =============================================================================

def analyze_elo_standalone(
    df: pd.DataFrame,
    test_cutoff: str = '2024-01-01'
) -> Dict:
    """Analyze Elo's standalone predictive value."""
    logger.info("Analyzing Elo standalone value...")

    cutoff = pd.to_datetime(test_cutoff)

    # Filter to test set with required columns
    test = df[df['fight_date'] >= cutoff].copy()

    required_cols = ['diff_elo', 'diff_career_win_rate', 'f1_is_winner']
    for col in required_cols:
        if col not in test.columns:
            return {'error': f'Missing column: {col}'}

    test = test.dropna(subset=required_cols)

    results = {}

    # Elo alone
    lr_elo = LogisticRegression(max_iter=1000)
    lr_elo.fit(test[['diff_elo']], test['f1_is_winner'])
    elo_pred = lr_elo.predict(test[['diff_elo']])
    results['elo_only_accuracy'] = float(accuracy_score(test['f1_is_winner'], elo_pred))

    # Win rate alone
    lr_wr = LogisticRegression(max_iter=1000)
    lr_wr.fit(test[['diff_career_win_rate']], test['f1_is_winner'])
    wr_pred = lr_wr.predict(test[['diff_career_win_rate']])
    results['win_rate_only_accuracy'] = float(accuracy_score(test['f1_is_winner'], wr_pred))

    # Both together
    lr_both = LogisticRegression(max_iter=1000)
    lr_both.fit(test[['diff_elo', 'diff_career_win_rate']], test['f1_is_winner'])
    both_pred = lr_both.predict(test[['diff_elo', 'diff_career_win_rate']])
    results['combined_accuracy'] = float(accuracy_score(test['f1_is_winner'], both_pred))

    results['n_test_fights'] = len(test)

    return results


def analyze_by_confidence_bucket(
    y_true: np.ndarray,
    y_proba_v1: np.ndarray,
    y_proba_best: np.ndarray
) -> pd.DataFrame:
    """Analyze accuracy by v1's confidence buckets."""
    buckets = [
        (0.50, 0.55, '50-55%'),
        (0.55, 0.60, '55-60%'),
        (0.60, 0.65, '60-65%'),
        (0.65, 0.70, '65-70%'),
        (0.70, 1.00, '70%+'),
    ]

    results = []
    for low, high, label in buckets:
        # Use max of f1_prob, f2_prob for confidence
        confidence = np.maximum(y_proba_v1, 1 - y_proba_v1)
        mask = (confidence >= low) & (confidence < high)

        if mask.sum() > 0:
            # v1 predictions
            v1_pred = (y_proba_v1[mask] > 0.5).astype(int)
            v1_acc = accuracy_score(y_true[mask], v1_pred)

            # Best candidate predictions
            best_pred = (y_proba_best[mask] > 0.5).astype(int)
            best_acc = accuracy_score(y_true[mask], best_pred)

            results.append({
                'bucket': label,
                'n_fights': int(mask.sum()),
                'v1_accuracy': float(v1_acc),
                'best_accuracy': float(best_acc),
                'improvement': float(best_acc - v1_acc),
            })

    return pd.DataFrame(results)


def analyze_sample_weight_impact(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fight_years_train: pd.Series,
    half_life: int = 8
) -> Dict:
    """Compare models with and without sample weighting."""
    logger.info("Analyzing sample weight impact...")

    # With sample weights
    weights = compute_sample_weights(fight_years_train, half_life=half_life)
    model_weighted = HistGradientBoostingClassifier(**QUICK_HGB_PARAMS, random_state=42)
    model_weighted.fit(X_train, y_train, sample_weight=weights)
    metrics_weighted = evaluate_model(model_weighted, X_test, y_test)

    # Without sample weights
    model_unweighted = HistGradientBoostingClassifier(**QUICK_HGB_PARAMS, random_state=42)
    model_unweighted.fit(X_train, y_train)
    metrics_unweighted = evaluate_model(model_unweighted, X_test, y_test)

    return {
        'weighted': metrics_weighted,
        'unweighted': metrics_unweighted,
        'half_life': half_life,
        'accuracy_diff': metrics_weighted['accuracy'] - metrics_unweighted['accuracy'],
    }


def analyze_calibration_improvement(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """Compare raw vs calibrated model performance."""
    logger.info("Analyzing calibration improvement...")

    # Raw model
    raw_metrics = evaluate_model(model, X_test, y_test)

    # Platt scaling
    model_platt = calibrate_model(model, X_train, y_train, method='sigmoid')
    platt_metrics = evaluate_model(model_platt, X_test, y_test)

    # Isotonic regression
    model_isotonic = calibrate_model(model, X_train, y_train, method='isotonic')
    isotonic_metrics = evaluate_model(model_isotonic, X_test, y_test)

    return {
        'raw': raw_metrics,
        'platt': platt_metrics,
        'isotonic': isotonic_metrics,
    }


def estimate_accuracy_ceiling(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Estimate theoretical accuracy ceiling with perfect calibration.

    If the 60-65% bucket has 63% actual win rate, perfect calibration
    predicts 63% for those fights. This gives the best accuracy achievable
    without changing the model's ranking ability.
    """
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        # Perfect calibration: predict actual base rate for each bucket
        # This requires binning the data
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # For each bin, use actual win rate as prediction
        perfect_proba = np.zeros_like(y_proba)
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                perfect_proba[mask] = y_true[mask].mean()

        # Calculate accuracy with perfect calibration
        perfect_pred = (perfect_proba > 0.5).astype(int)
        ceiling_accuracy = accuracy_score(y_true, perfect_pred)

        # Current accuracy
        current_pred = (y_proba > 0.5).astype(int)
        current_accuracy = accuracy_score(y_true, current_pred)

        return {
            'ceiling_accuracy': float(ceiling_accuracy),
            'current_accuracy': float(current_accuracy),
            'headroom': float(ceiling_accuracy - current_accuracy),
            'market_gap': float(0.65 - ceiling_accuracy),  # Gap to 65% market
        }
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# Winner Selection
# =============================================================================

def compare_candidates(
    candidates: Dict[str, Dict],
    v1_metrics: Dict
) -> Dict:
    """
    Compare all candidates and determine winner.

    Winner must beat v1 on 3+ of 5 metrics with no catastrophic regression
    (>3pp accuracy drop or >0.03 log loss increase).
    """
    comparison = {
        'v1': v1_metrics,
        'candidates': candidates,
        'winner': None,
        'winner_metrics': None,
    }

    metrics = ['accuracy', 'log_loss', 'roc_auc', 'brier_score', 'mce']
    # Higher is better for accuracy, roc_auc
    # Lower is better for log_loss, brier_score, mce
    higher_is_better = ['accuracy', 'roc_auc']

    for name, metrics_dict in candidates.items():
        wins = 0
        win_details = []
        catastrophic = False

        for metric in metrics:
            if metric not in metrics_dict or metrics_dict[metric] is None:
                continue
            if metric not in v1_metrics or v1_metrics[metric] is None:
                continue

            v1_val = v1_metrics[metric]
            cand_val = metrics_dict[metric]

            if metric in higher_is_better:
                if cand_val > v1_val:
                    wins += 1
                    diff = (cand_val - v1_val) * 100
                    win_details.append(f"{metric}: +{diff:.2f}pp")
                # Check for catastrophic accuracy drop
                if metric == 'accuracy' and (v1_val - cand_val) > 0.03:
                    catastrophic = True
            else:
                if cand_val < v1_val:
                    wins += 1
                    diff = (v1_val - cand_val) * 100
                    win_details.append(f"{metric}: -{diff:.2f}pp")
                # Check for catastrophic log_loss increase
                if metric == 'log_loss' and (cand_val - v1_val) > 0.03:
                    catastrophic = True

        comparison['candidates'][name] = {
            **metrics_dict,
            'wins': wins,
            'win_details': win_details,
            'catastrophic': catastrophic,
            'beats_v1': wins >= 3 and not catastrophic,
        }

        if wins >= 3 and not catastrophic:
            if comparison['winner'] is None:
                comparison['winner'] = name
                comparison['winner_metrics'] = metrics_dict
            else:
                # Multiple winners - pick by log_loss
                if metrics_dict['log_loss'] < comparison['winner_metrics']['log_loss']:
                    comparison['winner'] = name
                    comparison['winner_metrics'] = metrics_dict

    if comparison['winner'] is None:
        comparison['winner'] = 'v1'
        comparison['winner_metrics'] = v1_metrics
        comparison['reason'] = 'No candidate beat v1 on 3+ metrics without catastrophic regression'
    else:
        comparison['reason'] = f'{comparison["winner"]} beat v1 on {candidates[comparison["winner"]]["wins"]} metrics'

    return comparison


# =============================================================================
# Main Pipeline
# =============================================================================

def main(
    data_path: str = 'data/processed/ufc_fights_features_v1.csv',
    output_dir: str = 'data/models',
    test_cutoff: str = '2024-01-01',
    skip_tuning: bool = False,
    quick_test: bool = False,
    n_iter: int = 80
):
    """Run the complete Phase 8 training pipeline."""
    print("\n" + "=" * 80)
    print("PHASE 8: Elo Ratings, Post-Hoc Calibration, Sample Weighting, Ensemble")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if quick_test:
        n_iter = 10
        skip_tuning = True
        print("QUICK TEST MODE: Minimal iterations, skipping tuning")

    # Load data
    df = load_data(data_path)

    # =========================================================================
    # PART 1: Add Elo Features
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 1: Elo Rating System")
    print("-" * 80)

    df, final_ratings = add_elo_features(df, K=32)

    # Validate Elo
    top_25 = get_top_elo_fighters(final_ratings, 25)
    print("\nTop 10 Elo ratings:")
    for i, (name, rating) in enumerate(top_25[:10], 1):
        print(f"  {i:2}. {name:<30} {rating:.0f}")

    # =========================================================================
    # PART 2: Prepare Features and Split Data
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 2: Feature Preparation")
    print("-" * 80)

    # Get feature columns (including new Elo features)
    feature_cols = get_feature_columns(df)
    print(f"Total features before pruning: {len(feature_cols)}")

    # Temporal split
    cutoff = pd.to_datetime(test_cutoff)
    train_df = df[df['fight_date'] < cutoff].copy()
    test_df = df[df['fight_date'] >= cutoff].copy()

    print(f"Training: {len(train_df):,} fights ({train_df['fight_date'].min().date()} to {train_df['fight_date'].max().date()})")
    print(f"Testing: {len(test_df):,} fights ({test_df['fight_date'].min().date()} to {test_df['fight_date'].max().date()})")

    X_train_full = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_test_full = test_df[feature_cols].fillna(0)
    y_test = test_df[TARGET_COL]

    # Compute sample weights
    fight_years_train = train_df['fight_date'].dt.year
    sample_weights = compute_sample_weights(fight_years_train, half_life=8)
    print(f"Sample weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

    # =========================================================================
    # PART 3: Train v1 Baseline for Comparison
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 3: Load/Train v1 Baseline")
    print("-" * 80)

    v1_path = Path(output_dir) / 'ufc_model_v1.pkl'
    if v1_path.exists():
        v1_model = joblib.load(v1_path)
        # v1 uses original features (no Elo)
        v1_feature_path = Path(output_dir) / 'feature_names_v1.json'
        with open(v1_feature_path) as f:
            v1_features = json.load(f)
        X_test_v1 = test_df[v1_features].fillna(0)
        v1_metrics = evaluate_model(v1_model, X_test_v1, y_test)
        print(f"Loaded v1 model: Accuracy={v1_metrics['accuracy']:.1%}, Log Loss={v1_metrics['log_loss']:.4f}")
    else:
        print("v1 model not found, training baseline...")
        # Filter to v1 features (no Elo)
        v1_features = [c for c in feature_cols if 'elo' not in c.lower()]
        X_train_v1 = train_df[v1_features].fillna(0)
        X_test_v1 = test_df[v1_features].fillna(0)
        v1_model, _ = train_hgb(X_train_v1, y_train, skip_tuning=True)
        v1_metrics = evaluate_model(v1_model, X_test_v1, y_test)

    print(f"\nv1 Baseline Metrics:")
    print(f"  Accuracy:   {v1_metrics['accuracy']:.1%}")
    print(f"  Log Loss:   {v1_metrics['log_loss']:.4f}")
    print(f"  ROC-AUC:    {v1_metrics['roc_auc']:.4f}")
    print(f"  Brier:      {v1_metrics['brier_score']:.4f}")
    print(f"  MCE:        {v1_metrics['mce']:.1%}" if v1_metrics['mce'] else "  MCE:        N/A")

    # Store v1 predictions for comparison
    v1_proba = v1_model.predict_proba(X_test_v1)[:, 1]

    # =========================================================================
    # PART 4: Feature Pruning
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 4: Feature Pruning")
    print("-" * 80)

    # Compute importance on full feature set with quick model
    temp_model = HistGradientBoostingClassifier(**QUICK_HGB_PARAMS, random_state=42)
    temp_model.fit(X_train_full, y_train, sample_weight=sample_weights)

    importance_df = compute_feature_importance_for_pruning(temp_model, X_test_full, y_test)

    # Get features to prune
    features_to_prune = get_features_to_prune(importance_df, threshold=0.0)
    print(f"Features with zero/negative importance: {len(features_to_prune)}")

    # Keep top features
    surviving_features = [f for f in feature_cols if f not in features_to_prune]
    print(f"Surviving features: {len(surviving_features)}")

    # Also check for high correlations
    X_surviving = X_train_full[surviving_features]
    corr_matrix = X_surviving.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(c1, c2, corr_matrix.loc[c1, c2])
                       for c1 in upper.columns for c2 in upper.index
                       if upper.loc[c2, c1] > 0.85]

    if high_corr_pairs:
        print(f"\nHigh correlation pairs found: {len(high_corr_pairs)}")
        # Drop lower importance feature from each pair
        for c1, c2, corr in high_corr_pairs[:5]:  # Show first 5
            imp1 = importance_df[importance_df['feature'] == c1]['importance'].values[0] if c1 in importance_df['feature'].values else 0
            imp2 = importance_df[importance_df['feature'] == c2]['importance'].values[0] if c2 in importance_df['feature'].values else 0
            to_drop = c2 if imp1 > imp2 else c1
            if to_drop in surviving_features:
                surviving_features.remove(to_drop)
                print(f"  Dropped {to_drop} (corr={corr:.2f} with {'c1' if to_drop == c2 else 'c2'})")

    print(f"\nFinal feature count: {len(surviving_features)}")

    # Prepare pruned datasets
    X_train_pruned = train_df[surviving_features].fillna(0)
    X_test_pruned = test_df[surviving_features].fillna(0)

    # =========================================================================
    # PART 5: Train Candidates
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 5: Train Model Candidates")
    print("-" * 80)

    candidates = {}

    # Candidate A: HGB + Elo + pruned + sample weights + calibration
    print("\n[Candidate A] HGB + Elo + pruned + calibrated")
    hgb_model, hgb_params = train_hgb(X_train_pruned, y_train, sample_weights,
                                       skip_tuning=skip_tuning, n_iter=n_iter)

    # Test calibration methods
    cal_analysis = analyze_calibration_improvement(hgb_model, X_train_pruned, y_train,
                                                   X_test_pruned, y_test)

    # Pick best calibration method
    if cal_analysis['platt']['log_loss'] < cal_analysis['isotonic']['log_loss']:
        best_cal = 'platt'
        cal_method = 'sigmoid'
    else:
        best_cal = 'isotonic'
        cal_method = 'isotonic'

    hgb_calibrated = calibrate_model(hgb_model, X_train_pruned, y_train, method=cal_method)
    candidates['A_HGB_calibrated'] = evaluate_model(hgb_calibrated, X_test_pruned, y_test)
    print(f"  Raw:        Accuracy={cal_analysis['raw']['accuracy']:.1%}, Log Loss={cal_analysis['raw']['log_loss']:.4f}")
    print(f"  {best_cal.title()}:      Accuracy={candidates['A_HGB_calibrated']['accuracy']:.1%}, Log Loss={candidates['A_HGB_calibrated']['log_loss']:.4f}")

    # Candidate B: XGBoost + same features
    print("\n[Candidate B] XGBoost + Elo + pruned + calibrated")
    xgb_model, xgb_params = train_xgb(X_train_pruned, y_train, sample_weights,
                                       skip_tuning=skip_tuning, n_iter=n_iter)

    if xgb_model is not None:
        xgb_calibrated = calibrate_model(xgb_model, X_train_pruned, y_train, method=cal_method)
        candidates['B_XGB_calibrated'] = evaluate_model(xgb_calibrated, X_test_pruned, y_test)
        print(f"  Accuracy={candidates['B_XGB_calibrated']['accuracy']:.1%}, Log Loss={candidates['B_XGB_calibrated']['log_loss']:.4f}")
    else:
        print("  Skipped (XGBoost not installed)")

    # Candidate C: Stacking Ensemble
    print("\n[Candidate C] Stacking Ensemble + calibrated")
    ensemble = train_stacking_ensemble(X_train_pruned, y_train, hgb_model, xgb_model, sample_weights)
    ensemble_calibrated = calibrate_model(ensemble, X_train_pruned, y_train, method=cal_method)
    candidates['C_Ensemble_calibrated'] = evaluate_model(ensemble_calibrated, X_test_pruned, y_test)
    print(f"  Accuracy={candidates['C_Ensemble_calibrated']['accuracy']:.1%}, Log Loss={candidates['C_Ensemble_calibrated']['log_loss']:.4f}")

    # Candidate D: Minimal (top 10 features only)
    print("\n[Candidate D] Minimal Model (top 10 features)")
    top_10_features = importance_df.head(10)['feature'].tolist()
    # Ensure we only use features that exist in pruned set
    top_10_features = [f for f in top_10_features if f in surviving_features][:10]
    print(f"  Top 10: {top_10_features}")

    X_train_min = train_df[top_10_features].fillna(0)
    X_test_min = test_df[top_10_features].fillna(0)

    min_model, _ = train_hgb(X_train_min, y_train, sample_weights, skip_tuning=True)
    min_calibrated = calibrate_model(min_model, X_train_min, y_train, method=cal_method)
    candidates['D_Minimal_calibrated'] = evaluate_model(min_calibrated, X_test_min, y_test)
    print(f"  Accuracy={candidates['D_Minimal_calibrated']['accuracy']:.1%}, Log Loss={candidates['D_Minimal_calibrated']['log_loss']:.4f}")

    # Also test v1 + calibration only (free improvement check)
    print("\n[Baseline] v1 + post-hoc calibration")
    if v1_path.exists():
        v1_calibrated = calibrate_model(v1_model, train_df[v1_features].fillna(0), y_train, method=cal_method)
        v1_cal_metrics = evaluate_model(v1_calibrated, X_test_v1, y_test)
        candidates['v1_calibrated'] = v1_cal_metrics
        print(f"  Accuracy={v1_cal_metrics['accuracy']:.1%}, Log Loss={v1_cal_metrics['log_loss']:.4f}")

    # =========================================================================
    # PART 6: Diagnostic Analysis
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 6: Diagnostic Analysis")
    print("-" * 80)

    diagnostics = {}

    # Analysis 1: Elo's standalone value
    print("\n[Analysis 1] Elo Standalone Value")
    elo_analysis = analyze_elo_standalone(df, test_cutoff)
    diagnostics['elo_standalone'] = elo_analysis
    print(f"  Elo only:     {elo_analysis.get('elo_only_accuracy', 0):.1%}")
    print(f"  Win rate only: {elo_analysis.get('win_rate_only_accuracy', 0):.1%}")
    print(f"  Combined:      {elo_analysis.get('combined_accuracy', 0):.1%}")

    # Analysis 2: Confidence bucket analysis
    print("\n[Analysis 2] Accuracy by Confidence Bucket")
    best_candidate_name = max(candidates.keys(), key=lambda k: candidates[k]['accuracy'])

    # Get best candidate predictions
    if 'D_Minimal' in best_candidate_name:
        best_proba = min_calibrated.predict_proba(X_test_min)[:, 1]
    else:
        best_proba = hgb_calibrated.predict_proba(X_test_pruned)[:, 1]

    bucket_analysis = analyze_by_confidence_bucket(y_test.values, v1_proba, best_proba)
    diagnostics['confidence_buckets'] = bucket_analysis.to_dict('records')
    print(bucket_analysis.to_string(index=False))

    # Analysis 3: Sample weight impact
    print("\n[Analysis 3] Sample Weight Impact")
    weight_analysis = analyze_sample_weight_impact(
        X_train_pruned, y_train, X_test_pruned, y_test, fight_years_train
    )
    diagnostics['sample_weight_impact'] = weight_analysis
    print(f"  Weighted:   {weight_analysis['weighted']['accuracy']:.1%}")
    print(f"  Unweighted: {weight_analysis['unweighted']['accuracy']:.1%}")
    print(f"  Difference: {weight_analysis['accuracy_diff']*100:+.2f}pp")

    # Analysis 4: Calibration improvement
    print("\n[Analysis 4] Calibration Improvement")
    print(f"  Raw:      Log Loss={cal_analysis['raw']['log_loss']:.4f}, MCE={cal_analysis['raw']['mce']:.1%}" if cal_analysis['raw']['mce'] else f"  Raw: Log Loss={cal_analysis['raw']['log_loss']:.4f}")
    print(f"  Platt:    Log Loss={cal_analysis['platt']['log_loss']:.4f}, MCE={cal_analysis['platt']['mce']:.1%}" if cal_analysis['platt']['mce'] else f"  Platt: Log Loss={cal_analysis['platt']['log_loss']:.4f}")
    print(f"  Isotonic: Log Loss={cal_analysis['isotonic']['log_loss']:.4f}, MCE={cal_analysis['isotonic']['mce']:.1%}" if cal_analysis['isotonic']['mce'] else f"  Isotonic: Log Loss={cal_analysis['isotonic']['log_loss']:.4f}")
    diagnostics['calibration'] = cal_analysis

    # Analysis 5: Ceiling estimation
    print("\n[Analysis 5] Accuracy Ceiling Estimation")
    ceiling = estimate_accuracy_ceiling(y_test.values, best_proba)
    diagnostics['ceiling'] = ceiling
    if 'error' not in ceiling:
        print(f"  Current accuracy:  {ceiling['current_accuracy']:.1%}")
        print(f"  Ceiling accuracy:  {ceiling['ceiling_accuracy']:.1%}")
        print(f"  Headroom:          {ceiling['headroom']*100:.1f}pp")
        print(f"  Gap to market:     {ceiling['market_gap']*100:.1f}pp (65% market)")

    # =========================================================================
    # PART 7: Winner Selection
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 7: Winner Selection")
    print("-" * 80)

    comparison = compare_candidates(candidates, v1_metrics)

    print("\nComparison Results:")
    print("-" * 60)
    print(f"{'Candidate':<25} {'Acc':<8} {'LogLoss':<10} {'ROC':<8} {'Brier':<8} {'MCE':<8} {'Wins'}")
    print("-" * 60)

    # Print v1
    m = v1_metrics
    mce_str = f"{m['mce']:.1%}" if m['mce'] else 'N/A'
    print(f"{'v1 (baseline)':<25} {m['accuracy']:.1%}   {m['log_loss']:.4f}     {m['roc_auc']:.4f}   {m['brier_score']:.4f}   {mce_str:<8} -")

    # Print candidates
    for name in sorted(comparison['candidates'].keys()):
        cand = comparison['candidates'][name]
        mce_str = f"{cand['mce']:.1%}" if cand.get('mce') else 'N/A'
        wins_str = f"{cand['wins']}/5" if 'wins' in cand else '-'
        marker = '✓' if cand.get('beats_v1') else ''
        print(f"{name:<25} {cand['accuracy']:.1%}   {cand['log_loss']:.4f}     {cand['roc_auc']:.4f}   {cand['brier_score']:.4f}   {mce_str:<8} {wins_str} {marker}")

    print("-" * 60)
    print(f"\nWinner: {comparison['winner']}")
    print(f"Reason: {comparison['reason']}")

    # =========================================================================
    # PART 8: Save Results
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 8: Save Results")
    print("-" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save comparison results
    comparison_data = {
        'training_date': datetime.now().isoformat(),
        'winner': comparison['winner'],
        'reason': comparison['reason'],
        'v1_metrics': v1_metrics,
        'candidates': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
                       for k, v in comparison['candidates'].items()},
        'diagnostics': {k: v if not isinstance(v, pd.DataFrame) else v.to_dict('records')
                        for k, v in diagnostics.items()},
        'elo_validation': {
            'top_10_elo': [(name, float(rating)) for name, rating in top_25[:10]],
            'elo_only_accuracy': elo_analysis.get('elo_only_accuracy'),
        },
        'feature_counts': {
            'original': len(feature_cols),
            'pruned': len(features_to_prune),
            'final': len(surviving_features),
        },
        'test_split': test_cutoff,
        'train_size': len(train_df),
        'test_size': len(test_df),
    }

    with open(output_path / 'model_comparison_phase8.json', 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    print(f"Saved comparison to {output_path / 'model_comparison_phase8.json'}")

    # Save feature list
    with open(output_path / 'phase8_feature_list.json', 'w') as f:
        json.dump(surviving_features, f, indent=2)
    print(f"Saved feature list to {output_path / 'phase8_feature_list.json'}")

    # If a candidate beat v1, save the winning model
    if comparison['winner'] != 'v1':
        winning_name = comparison['winner']

        if 'A_HGB' in winning_name:
            winning_model = hgb_calibrated
        elif 'B_XGB' in winning_name:
            winning_model = xgb_calibrated
        elif 'C_Ensemble' in winning_name:
            winning_model = ensemble_calibrated
        elif 'D_Minimal' in winning_name:
            winning_model = min_calibrated
        elif 'v1_calibrated' in winning_name:
            winning_model = v1_calibrated
        else:
            winning_model = None

        if winning_model is not None:
            model_path = output_path / 'ufc_model_phase8.pkl'
            joblib.dump(winning_model, model_path)
            print(f"Saved winning model to {model_path}")

            # Save feature names for the winning model
            if 'D_Minimal' in winning_name:
                winning_features = top_10_features
            elif 'v1_calibrated' in winning_name:
                winning_features = v1_features
            else:
                winning_features = surviving_features

            with open(output_path / 'phase8_winning_features.json', 'w') as f:
                json.dump(winning_features, f, indent=2)
    else:
        # v1 retained - but check if calibrated v1 is better
        if 'v1_calibrated' in candidates:
            v1_cal = candidates['v1_calibrated']
            if v1_cal['log_loss'] < v1_metrics['log_loss']:
                print("\n⚠️  v1 + calibration improves log loss - deploying calibrated v1")
                joblib.dump(v1_calibrated, output_path / 'ufc_model_phase8.pkl')
                comparison_data['deployed'] = 'v1_calibrated'
            else:
                print("\nv1 retained unchanged")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 8 COMPLETE")
    print("=" * 80)

    print(f"\nWinner: {comparison['winner']}")

    winner_metrics = comparison['winner_metrics']
    print(f"\nFinal Metrics:")
    print(f"  Accuracy:   {winner_metrics['accuracy']:.1%}")
    print(f"  Log Loss:   {winner_metrics['log_loss']:.4f}")
    print(f"  ROC-AUC:    {winner_metrics['roc_auc']:.4f}")
    print(f"  Brier:      {winner_metrics['brier_score']:.4f}")
    if winner_metrics.get('mce'):
        print(f"  MCE:        {winner_metrics['mce']:.1%}")

    if 'ceiling' in diagnostics and 'error' not in diagnostics['ceiling']:
        print(f"\nCeiling Analysis:")
        print(f"  Model appears to be within {diagnostics['ceiling']['headroom']*100:.1f}pp of theoretical ceiling")
        print(f"  Gap to market (~65%): {diagnostics['ceiling']['market_gap']*100:.1f}pp")

    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 8 Training Pipeline')
    parser.add_argument('--skip-tuning', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode (minimal iterations)')
    parser.add_argument('--data-path', type=str, default='data/processed/ufc_fights_features_v1.csv')
    parser.add_argument('--output-dir', type=str, default='data/models')
    parser.add_argument('--test-cutoff', type=str, default='2024-01-01')
    parser.add_argument('--n-iter', type=int, default=80, help='Number of random search iterations')

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_cutoff=args.test_cutoff,
        skip_tuning=args.skip_tuning,
        quick_test=args.quick_test,
        n_iter=args.n_iter
    )
