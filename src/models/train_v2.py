"""
Model training module for UFC fight prediction - v2 (Opponent-Quality Adjusted).

This module trains three model variants using opponent-quality-adjusted features
and compares them against the v1 baseline.

Variants:
    A: Original + Adjusted features together
    B: Adjusted features only (replace originals)
    C: Original + Opponent Quality features only (no adjusted stats)

Usage:
    python src/models/train_v2.py
    python src/models/train_v2.py --skip-tuning
    python src/models/train_v2.py --quick  # Fast training with fewer iterations
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
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings during training
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# Metadata columns (not used for training)
METADATA_COLS = [
    'fight_id', 'fight_date', 'event_name', 'weight_class',
    'f1_name', 'f2_name', 'method', 'round'
]

# Target column
TARGET_COL = 'f1_is_winner'

# Hyperparameter search space
PARAM_DIST = {
    'max_iter': [150, 200, 300, 400],
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.03, 0.05, 0.08, 0.1],
    'max_leaf_nodes': [31, 63, 127],
    'min_samples_leaf': [10, 15, 20, 30],
    'l2_regularization': [0.05, 0.1, 0.2, 0.5],
}

# Quick training params (for --skip-tuning or --quick)
QUICK_PARAMS = {
    'max_iter': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'max_leaf_nodes': 63,
    'min_samples_leaf': 20,
    'l2_regularization': 0.1,
}

# Original features (from v1 model)
ORIGINAL_FEATURE_PATTERNS = [
    'f1_career_', 'f2_career_', 'diff_career_',
    'f1_win_streak', 'f2_win_streak', 'diff_win_streak',
    'f1_loss_streak', 'f2_loss_streak', 'diff_loss_streak',
    'f1_days_since', 'f2_days_since', 'diff_days_since',
    'f1_fights_per_year', 'f2_fights_per_year', 'diff_fights_per_year',
    'f1_weight_class_fights', 'f2_weight_class_fights', 'diff_weight_class_fights',
    'f1_fights_last_12mo', 'f2_fights_last_12mo', 'diff_fights_last_12mo',
    'f1_is_ufc_debut', 'f2_is_ufc_debut', 'diff_is_ufc_debut',
    'f1_data_completeness', 'f2_data_completeness', 'diff_data_completeness',
    'f1_has_detailed_stats', 'f2_has_detailed_stats',
    'f1_win_rate_last_', 'f2_win_rate_last_', 'diff_win_rate_last_',
    'f1_finish_rate_last_', 'f2_finish_rate_last_', 'diff_finish_rate_last_',
    'f1_avg_sig_strikes_landed_last_', 'f2_avg_sig_strikes_landed_last_',
    'diff_avg_sig_strikes_landed_last_',
    'f1_avg_sig_strikes_absorbed_last_', 'f2_avg_sig_strikes_absorbed_last_',
    'diff_avg_sig_strikes_absorbed_last_',
    'f1_avg_takedowns_last_', 'f2_avg_takedowns_last_', 'diff_avg_takedowns_last_',
    'f1_avg_fight_time_last_', 'f2_avg_fight_time_last_', 'diff_avg_fight_time_last_',
    'f1_striking_volume', 'f2_striking_volume', 'diff_striking_volume',
    'f1_grappling_tendency', 'f2_grappling_tendency', 'diff_grappling_tendency',
    'f1_experience_ratio', 'f2_experience_ratio',
    'diff_momentum',
    'f1_is_coming_off_layoff', 'f2_is_coming_off_layoff',
    'f1_is_new_to_weight_class', 'f2_is_new_to_weight_class',
    'is_title_fight', 'fight_year', 'fight_month',
    'wc_',
]

# Adjusted feature patterns (new in v2)
ADJUSTED_FEATURE_PATTERNS = [
    'f1_adj_', 'f2_adj_', 'diff_adj_',
    'f1_avg_opponent_quality', 'f2_avg_opponent_quality', 'diff_avg_opponent_quality',
]

# Opponent quality only patterns (for variant C)
OPPONENT_QUALITY_PATTERNS = [
    'opponent_quality',
]


# ============================================================================
# Feature Selection Helpers
# ============================================================================

def get_original_features(df: pd.DataFrame) -> List[str]:
    """Get list of original (v1) feature columns."""
    features = []
    for col in df.columns:
        if col in METADATA_COLS or col == TARGET_COL:
            continue
        # Check if it matches any adjusted pattern (exclude these)
        is_adjusted = any(p in col for p in ['adj_', 'opponent_quality'])
        if not is_adjusted:
            features.append(col)
    return features


def get_adjusted_features(df: pd.DataFrame) -> List[str]:
    """Get list of opponent-quality-adjusted feature columns."""
    features = []
    for col in df.columns:
        if col in METADATA_COLS or col == TARGET_COL:
            continue
        if 'adj_' in col or 'opponent_quality' in col:
            features.append(col)
    return features


def get_opponent_quality_only_features(df: pd.DataFrame) -> List[str]:
    """Get list of opponent quality features (not adjusted stats)."""
    features = []
    for col in df.columns:
        if col in METADATA_COLS or col == TARGET_COL:
            continue
        if 'opponent_quality' in col and 'adj_' not in col:
            features.append(col)
    return features


def get_variant_features(df: pd.DataFrame, variant: str) -> List[str]:
    """
    Get feature columns for a specific variant.

    Variants:
        A: Original + Adjusted features together
        B: Adjusted features only (replace originals with adjusted versions)
        C: Original + Opponent Quality features only
    """
    original = get_original_features(df)
    adjusted = get_adjusted_features(df)
    oq_only = get_opponent_quality_only_features(df)

    if variant == 'A':
        # All features
        return original + adjusted
    elif variant == 'B':
        # Replace original career/rolling with adjusted versions
        # Keep: metadata, weight class dummies, temporal, non-adjusted features
        # Replace: career_win_rate -> adj_career_win_rate, etc.
        adjusted_only = []
        for col in df.columns:
            if col in METADATA_COLS or col == TARGET_COL:
                continue
            # Include all adjusted features
            if 'adj_' in col or 'opponent_quality' in col:
                adjusted_only.append(col)
            # Include non-replaceable original features
            elif not any(x in col for x in [
                'career_win_rate', 'win_rate_last_',
                'avg_sig_strikes_landed_last_', 'avg_takedowns_last_',
                'avg_control_time_last_'
            ]):
                # Keep features that don't have adjusted versions
                adjusted_only.append(col)
        return adjusted_only
    elif variant == 'C':
        # Original + opponent quality only (no adjusted stats)
        return original + oq_only
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load processed feature data from CSV."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    logger.info(f"Loaded {len(df):,} fights with {len(df.columns)} columns")
    return df


def temporal_split(
    df: pd.DataFrame,
    cutoff_date: str = '2024-01-01'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally (train on past, test on future)."""
    cutoff = pd.to_datetime(cutoff_date)
    train_df = df[df['fight_date'] < cutoff].copy()
    test_df = df[df['fight_date'] >= cutoff].copy()

    logger.info(f"Temporal split at {cutoff_date}:")
    logger.info(f"  Training: {len(train_df):,} fights")
    logger.info(f"  Testing: {len(test_df):,} fights")

    return train_df, test_df


# ============================================================================
# Model Training
# ============================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    skip_tuning: bool = False,
    n_iter: int = 50,
    cv_folds: int = 5
) -> Tuple[HistGradientBoostingClassifier, Dict]:
    """Train a HistGradientBoostingClassifier with optional hyperparameter tuning."""
    if skip_tuning:
        logger.info("Using quick training params (no tuning)")
        model = HistGradientBoostingClassifier(
            **QUICK_PARAMS,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
        )
        model.fit(X_train, y_train)
        return model, QUICK_PARAMS

    logger.info(f"Hyperparameter tuning ({n_iter} iterations, {cv_folds}-fold CV)")

    base_model = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring='neg_log_loss',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start_time

    logger.info(f"Tuning completed in {elapsed:.1f}s")
    logger.info(f"Best CV log loss: {-search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(
    model: HistGradientBoostingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    mce = np.mean(np.abs(prob_pred - prob_true))

    return {
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'roc_auc': float(roc_auc),
        'brier_score': float(brier),
        'mce': float(mce),
    }


def compute_feature_importance(
    model: HistGradientBoostingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_samples: int = 1000
) -> pd.DataFrame:
    """Compute permutation feature importance."""
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
    else:
        X_sample = X
        y_sample = y

    result = permutation_importance(
        model, X_sample, y_sample,
        n_repeats=10,
        random_state=42,
        scoring='accuracy'
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)

    return importance_df


# ============================================================================
# Comparison
# ============================================================================

def compare_models(results: Dict[str, Dict]) -> Tuple[str, Dict]:
    """
    Compare model variants and select winner.

    Winner must beat v1 on at least 3 of 5 metrics with no catastrophic regression.
    Priority: Log Loss > Brier Score > Accuracy > ROC-AUC > MCE
    """
    v1 = results.get('v1')
    if v1 is None:
        # v1 results not provided, just pick best v2 variant
        variants = [k for k in results.keys() if k != 'v1']
        best = min(variants, key=lambda x: results[x]['log_loss'])
        return best, {'winner': best, 'reason': 'No v1 baseline provided'}

    comparison = {'v1': v1}
    variant_wins = {}

    for variant in ['v2-A', 'v2-B', 'v2-C']:
        if variant not in results:
            continue

        v2 = results[variant]
        wins = 0
        details = []

        # Compare each metric
        # Accuracy (higher is better)
        if v2['accuracy'] > v1['accuracy']:
            wins += 1
            details.append(f"Accuracy: +{(v2['accuracy']-v1['accuracy'])*100:.2f}pp")
        elif v2['accuracy'] < v1['accuracy'] - 0.03:
            details.append(f"Accuracy: REGRESSION {(v2['accuracy']-v1['accuracy'])*100:.2f}pp")

        # Log Loss (lower is better)
        if v2['log_loss'] < v1['log_loss']:
            wins += 1
            details.append(f"Log Loss: {v2['log_loss']-v1['log_loss']:.4f}")
        elif v2['log_loss'] > v1['log_loss'] + 0.02:
            details.append(f"Log Loss: REGRESSION +{v2['log_loss']-v1['log_loss']:.4f}")

        # ROC-AUC (higher is better)
        if v2['roc_auc'] > v1['roc_auc']:
            wins += 1
            details.append(f"ROC-AUC: +{v2['roc_auc']-v1['roc_auc']:.4f}")

        # Brier Score (lower is better)
        if v2['brier_score'] < v1['brier_score']:
            wins += 1
            details.append(f"Brier: {v2['brier_score']-v1['brier_score']:.4f}")

        # MCE (lower is better)
        if v2['mce'] < v1['mce']:
            wins += 1
            details.append(f"MCE: {(v2['mce']-v1['mce'])*100:.2f}pp")

        variant_wins[variant] = {
            'wins': wins,
            'details': details,
            'beats_v1': wins >= 3
        }
        comparison[variant] = v2

    # Select winner
    candidates = [v for v, info in variant_wins.items() if info['beats_v1']]

    if len(candidates) == 0:
        # No variant beats v1
        return 'v1', {
            'winner': 'v1',
            'reason': 'No v2 variant beat v1 on 3+ metrics',
            'variant_analysis': variant_wins,
        }

    # Among candidates, pick the one with best log loss (primary metric)
    best = min(candidates, key=lambda x: results[x]['log_loss'])

    return best, {
        'winner': best,
        'reason': f'{best} beats v1 on {variant_wins[best]["wins"]}/5 metrics',
        'variant_analysis': variant_wins,
    }


# ============================================================================
# Main Pipeline
# ============================================================================

def main(
    data_path: str = 'data/processed/ufc_fights_features_v2.csv',
    output_dir: str = 'data/models',
    test_cutoff: str = '2024-01-01',
    skip_tuning: bool = False,
    n_iter: int = 50,
    quick: bool = False,
):
    """Run the v2 training pipeline with all variants."""
    print("=" * 70)
    print("UFC FIGHT PREDICTION MODEL v2 TRAINING")
    print("Opponent-Quality-Adjusted Features")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if quick:
        skip_tuning = True
        n_iter = 10

    # Load data
    df = load_data(data_path)

    # Split
    train_df, test_df = temporal_split(df, test_cutoff)

    # Load v1 results for comparison
    v1_results_path = Path(output_dir) / 'evaluation_results_v1.json'
    if v1_results_path.exists():
        with open(v1_results_path, 'r') as f:
            v1_data = json.load(f)
            v1_metrics = v1_data['test_metrics']
            v1_metrics['mce'] = v1_data.get('mean_calibration_error', 0.137)
        print(f"\nv1 baseline loaded: accuracy={v1_metrics['accuracy']:.1%}, log_loss={v1_metrics['log_loss']:.4f}")
    else:
        print("\nWARNING: v1 results not found, will train v1 from scratch")
        v1_metrics = None

    # Results storage
    all_results = {}
    all_models = {}
    all_importance = {}

    # If no v1 metrics, train v1 first
    if v1_metrics is None:
        print("\n" + "=" * 70)
        print("TRAINING v1 (BASELINE)")
        print("=" * 70)
        v1_features = get_original_features(df)
        print(f"Features: {len(v1_features)}")

        X_train = train_df[v1_features]
        y_train = train_df[TARGET_COL]
        X_test = test_df[v1_features]
        y_test = test_df[TARGET_COL]

        model, params = train_model(X_train, y_train, skip_tuning, n_iter)
        metrics = evaluate_model(model, X_test, y_test)

        all_results['v1'] = metrics
        all_models['v1'] = model
        v1_metrics = metrics
        print(f"v1: accuracy={metrics['accuracy']:.1%}, log_loss={metrics['log_loss']:.4f}")
    else:
        all_results['v1'] = v1_metrics

    # Train v2 variants
    variants = {
        'v2-A': 'Original + Adjusted',
        'v2-B': 'Adjusted Only',
        'v2-C': 'Original + OQ Only',
    }

    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    for variant_id, variant_name in variants.items():
        print("\n" + "=" * 70)
        print(f"TRAINING {variant_id}: {variant_name}")
        print("=" * 70)

        variant_letter = variant_id.split('-')[1]
        feature_cols = get_variant_features(df, variant_letter)
        print(f"Features: {len(feature_cols)}")

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        model, params = train_model(X_train, y_train, skip_tuning, n_iter)
        metrics = evaluate_model(model, X_test, y_test)

        all_results[variant_id] = metrics
        all_models[variant_id] = model

        # Compute feature importance for winner candidates
        importance = compute_feature_importance(model, X_test, y_test, feature_cols)
        all_importance[variant_id] = importance

        print(f"{variant_id}: accuracy={metrics['accuracy']:.1%}, log_loss={metrics['log_loss']:.4f}")

    # Compare and select winner
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    winner, comparison = compare_models(all_results)

    print(f"\n{'Metric':<15} {'v1':<12} {'v2-A':<12} {'v2-B':<12} {'v2-C':<12}")
    print("-" * 63)

    for metric in ['accuracy', 'log_loss', 'roc_auc', 'brier_score', 'mce']:
        row = f"{metric:<15}"
        for model_id in ['v1', 'v2-A', 'v2-B', 'v2-C']:
            if model_id in all_results:
                val = all_results[model_id].get(metric, 0)
                if metric == 'accuracy' or metric == 'mce':
                    row += f"{val*100:.1f}%{'':<7}"
                else:
                    row += f"{val:.4f}{'':<6}"
            else:
                row += f"{'N/A':<12}"
        print(row)

    print(f"\nWINNER: {winner}")
    print(f"Reason: {comparison['reason']}")

    if 'variant_analysis' in comparison:
        print("\nVariant Analysis:")
        for v, info in comparison['variant_analysis'].items():
            status = "✓ BEATS v1" if info['beats_v1'] else "✗ LOSES to v1"
            print(f"  {v}: {info['wins']}/5 wins - {status}")
            for detail in info['details'][:3]:
                print(f"    {detail}")

    # Save artifacts
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save comparison results
    comparison_file = output_path / 'v1_vs_v2_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump({
            'training_date': datetime.now().isoformat(),
            'winner': winner,
            'comparison': comparison,
            'all_metrics': all_results,
            'test_split': test_cutoff,
            'train_size': len(train_df),
            'test_size': len(test_df),
        }, f, indent=2)
    print(f"Saved comparison to {comparison_file}")

    # If a v2 variant won, save it as the new production model
    if winner.startswith('v2'):
        model = all_models[winner]
        variant_letter = winner.split('-')[1]
        feature_cols = get_variant_features(df, variant_letter)

        # Save model
        model_file = output_path / 'ufc_model_v2.pkl'
        joblib.dump(model, model_file)
        print(f"Saved winning model to {model_file}")

        # Save feature names
        features_file = output_path / 'feature_names_v2.json'
        with open(features_file, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        print(f"Saved feature names to {features_file}")

        # Save feature importance
        importance = all_importance[winner]
        importance_file = output_path / 'feature_importance_v2.csv'
        importance.to_csv(importance_file, index=False)
        print(f"Saved feature importance to {importance_file}")

        # Save evaluation results
        eval_file = output_path / 'evaluation_results_v2.json'
        with open(eval_file, 'w') as f:
            json.dump({
                'training_date': datetime.now().isoformat(),
                'model_type': 'HistGradientBoostingClassifier',
                'variant': winner,
                'test_metrics': all_results[winner],
                'improvement_over_v1': {
                    'accuracy': all_results[winner]['accuracy'] - v1_metrics['accuracy'],
                    'log_loss': all_results[winner]['log_loss'] - v1_metrics['log_loss'],
                    'roc_auc': all_results[winner]['roc_auc'] - v1_metrics['roc_auc'],
                    'brier_score': all_results[winner]['brier_score'] - v1_metrics['brier_score'],
                },
                'top_features': [
                    {'feature': row['feature'], 'importance': float(row['importance'])}
                    for _, row in importance.head(10).iterrows()
                ]
            }, f, indent=2)
        print(f"Saved evaluation results to {eval_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Winner: {winner}")

    if winner.startswith('v2'):
        v2_metrics = all_results[winner]
        print(f"  Accuracy: {v2_metrics['accuracy']:.1%} (v1: {v1_metrics['accuracy']:.1%})")
        print(f"  Log Loss: {v2_metrics['log_loss']:.4f} (v1: {v1_metrics['log_loss']:.4f})")
        print(f"  ROC-AUC: {v2_metrics['roc_auc']:.4f} (v1: {v1_metrics['roc_auc']:.4f})")
        print(f"  Brier: {v2_metrics['brier_score']:.4f} (v1: {v1_metrics['brier_score']:.4f})")
    else:
        print(f"  v1 retained (no v2 variant improved enough)")

    return winner, all_results, all_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train UFC fight prediction model v2 variants'
    )
    parser.add_argument(
        '--skip-tuning',
        action='store_true',
        help='Skip hyperparameter tuning (use defaults)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training with fewer iterations'
    )
    parser.add_argument(
        '--test-cutoff',
        type=str,
        default='2024-01-01',
        help='Date for temporal train/test split'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/ufc_fights_features_v2.csv',
        help='Path to v2 feature data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models',
        help='Directory to save models and artifacts'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=50,
        help='Number of random search iterations'
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_cutoff=args.test_cutoff,
        skip_tuning=args.skip_tuning,
        n_iter=args.n_iter,
        quick=args.quick,
    )
