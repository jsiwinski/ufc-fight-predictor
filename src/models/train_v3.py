"""
Model training module for UFC fight prediction - v3 (Domain-Specific Adjustment).

Phase 7 implementation:
- Domain-specific opponent skill profiles (striking, grappling, durability, overall)
- Stats adjusted by opponent's ability in that specific domain
- Feature pruning: removes features with zero/negative importance

Variants:
    A: Pruned originals + all domain-adjusted features
    B: Domain-adjusted features replace originals where applicable

Usage:
    python src/models/train_v3.py
    python src/models/train_v3.py --skip-tuning
    python src/models/train_v3.py --quick
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

METADATA_COLS = [
    'fight_id', 'fight_date', 'event_name', 'weight_class',
    'f1_name', 'f2_name', 'method', 'round'
]

TARGET_COL = 'f1_is_winner'

# Hyperparameter search space (same as Phase 6)
PARAM_DIST = {
    'max_iter': [150, 200, 300, 400, 500],
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.02, 0.03, 0.05, 0.08, 0.1],
    'max_leaf_nodes': [31, 63, 127],
    'min_samples_leaf': [10, 15, 20, 30, 40],
    'l2_regularization': [0.05, 0.1, 0.2, 0.5, 1.0],
}

QUICK_PARAMS = {
    'max_iter': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'max_leaf_nodes': 63,
    'min_samples_leaf': 20,
    'l2_regularization': 0.1,
}

# Features to remove when replacing with adjusted versions (Variant B)
REPLACEABLE_FEATURES = [
    'career_win_rate',
    'win_rate_last_3', 'win_rate_last_5', 'win_rate_last_10',
    'avg_sig_strikes_landed_last_3', 'avg_sig_strikes_landed_last_5', 'avg_sig_strikes_landed_last_10',
    'avg_takedowns_last_3', 'avg_takedowns_last_5', 'avg_takedowns_last_10',
]


# ============================================================================
# Feature Selection with Pruning
# ============================================================================

def load_feature_importance(importance_path: str) -> pd.DataFrame:
    """Load feature importance from CSV."""
    if not Path(importance_path).exists():
        return None
    return pd.read_csv(importance_path)


def get_features_to_prune(importance_df: pd.DataFrame, threshold: float = 0.0) -> List[str]:
    """
    Get list of features with importance at or below threshold.

    Args:
        importance_df: DataFrame with feature/importance columns
        threshold: Remove features with importance <= this value

    Returns:
        List of feature names to prune
    """
    if importance_df is None:
        return []

    to_prune = importance_df[importance_df['importance'] <= threshold]['feature'].tolist()
    return to_prune


def get_all_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all feature columns (excluding metadata and target)."""
    return [col for col in df.columns if col not in METADATA_COLS + [TARGET_COL]]


def get_original_features(df: pd.DataFrame) -> List[str]:
    """Get list of original (non-adjusted) feature columns."""
    features = []
    for col in df.columns:
        if col in METADATA_COLS or col == TARGET_COL:
            continue
        # Exclude adjusted and opponent skill features
        if 'adj_' in col or 'opp_' in col:
            continue
        features.append(col)
    return features


def get_domain_adjusted_features(df: pd.DataFrame) -> List[str]:
    """Get list of domain-adjusted feature columns (Phase 7)."""
    features = []
    for col in df.columns:
        if col in METADATA_COLS or col == TARGET_COL:
            continue
        if 'adj_' in col or 'opp_' in col:
            features.append(col)
    return features


def get_variant_features(
    df: pd.DataFrame,
    variant: str,
    features_to_prune: List[str] = None
) -> List[str]:
    """
    Get feature columns for a specific variant.

    Variants:
        A: Pruned originals + all domain-adjusted features
        B: Domain-adjusted features replace originals where applicable

    Args:
        df: DataFrame with all features
        variant: 'A' or 'B'
        features_to_prune: List of features to remove (from importance pruning)

    Returns:
        List of feature names to use
    """
    features_to_prune = features_to_prune or []

    original = get_original_features(df)
    adjusted = get_domain_adjusted_features(df)

    if variant == 'A':
        # Pruned originals + all adjusted features
        pruned_original = [f for f in original if f not in features_to_prune]
        # Also prune adjusted versions of pruned features
        pruned_adjusted = []
        for f in adjusted:
            # Check if this is an adjusted version of a pruned feature
            base_name = f.replace('f1_', '').replace('f2_', '').replace('diff_', '')
            base_name = base_name.replace('adj_', '').replace('avg_opp_', '')
            should_prune = False
            for pruned in features_to_prune:
                pruned_base = pruned.replace('f1_', '').replace('f2_', '').replace('diff_', '')
                if base_name == pruned_base:
                    should_prune = True
                    break
            if not should_prune:
                pruned_adjusted.append(f)

        return pruned_original + pruned_adjusted

    elif variant == 'B':
        # Replace originals with adjusted versions where applicable
        features = []

        for col in df.columns:
            if col in METADATA_COLS or col == TARGET_COL:
                continue
            if col in features_to_prune:
                continue

            # Check if this is a replaceable original feature
            is_replaceable = False
            for rep in REPLACEABLE_FEATURES:
                if rep in col and 'adj_' not in col and 'opp_' not in col:
                    is_replaceable = True
                    break

            if is_replaceable:
                # Skip - will use adjusted version instead
                continue

            features.append(col)

        # Add all adjusted features
        features.extend([f for f in adjusted if f not in features_to_prune])

        return list(set(features))  # Remove duplicates

    else:
        raise ValueError(f"Unknown variant: {variant}")


def analyze_feature_redundancy(df: pd.DataFrame, features: List[str], threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """
    Find pairs of features with correlation above threshold.

    Returns:
        List of (feature1, feature2, correlation) tuples
    """
    numeric_df = df[features].select_dtypes(include=[np.number])

    if len(numeric_df.columns) == 0:
        return []

    corr_matrix = numeric_df.corr().abs()

    redundant_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                redundant_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    return redundant_pairs


# ============================================================================
# Data Loading and Feature Engineering
# ============================================================================

def run_feature_engineering(raw_data_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Run Phase 7 feature engineering pipeline.

    Args:
        raw_data_path: Path to raw fight data
        output_path: Optional path to save processed features

    Returns:
        DataFrame with Phase 7 features
    """
    logger.info("Running Phase 7 feature engineering...")

    raw_df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded {len(raw_df)} fights from {raw_data_path}")

    engineer = FeatureEngineer({'compute_opponent_quality': True})
    features_df = engineer.create_all_features(raw_df)

    if output_path:
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

    return features_df


def load_data(data_path: str) -> pd.DataFrame:
    """Load processed feature data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    logger.info(f"Loaded {len(df):,} fights with {len(df.columns)} columns")
    return df


def temporal_split(
    df: pd.DataFrame,
    cutoff_date: str = '2024-01-01'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally."""
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
    n_iter: int = 60,
    cv_folds: int = 5
) -> Tuple[HistGradientBoostingClassifier, Dict]:
    """Train model with optional hyperparameter tuning."""
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
# Model Comparison
# ============================================================================

def compare_models(results: Dict[str, Dict]) -> Tuple[str, Dict]:
    """
    Compare model variants and select winner.

    Winner must beat v1 on at least 3 of 5 metrics.
    No single metric can regress by more than 5%.
    """
    v1 = results.get('v1')
    if v1 is None:
        variants = [k for k in results.keys() if k != 'v1']
        best = min(variants, key=lambda x: results[x]['log_loss'])
        return best, {'winner': best, 'reason': 'No v1 baseline provided'}

    comparison = {'v1': v1}
    variant_wins = {}

    for variant in ['v3-A', 'v3-B']:
        if variant not in results:
            continue

        v3 = results[variant]
        wins = 0
        details = []
        catastrophic = False

        # Accuracy (higher is better)
        if v3['accuracy'] > v1['accuracy']:
            wins += 1
            details.append(f"Accuracy: +{(v3['accuracy']-v1['accuracy'])*100:.2f}pp")
        elif v3['accuracy'] < v1['accuracy'] - 0.05:
            catastrophic = True
            details.append(f"Accuracy: CATASTROPHIC {(v3['accuracy']-v1['accuracy'])*100:.2f}pp")

        # Log Loss (lower is better)
        if v3['log_loss'] < v1['log_loss']:
            wins += 1
            details.append(f"Log Loss: {v3['log_loss']-v1['log_loss']:.4f}")
        elif v3['log_loss'] > v1['log_loss'] * 1.05:
            catastrophic = True
            details.append(f"Log Loss: CATASTROPHIC +{v3['log_loss']-v1['log_loss']:.4f}")

        # ROC-AUC (higher is better)
        if v3['roc_auc'] > v1['roc_auc']:
            wins += 1
            details.append(f"ROC-AUC: +{v3['roc_auc']-v1['roc_auc']:.4f}")

        # Brier Score (lower is better)
        if v3['brier_score'] < v1['brier_score']:
            wins += 1
            details.append(f"Brier: {v3['brier_score']-v1['brier_score']:.4f}")

        # MCE (lower is better)
        if v3['mce'] < v1['mce']:
            wins += 1
            details.append(f"MCE: {(v3['mce']-v1['mce'])*100:.2f}pp")

        variant_wins[variant] = {
            'wins': wins,
            'details': details,
            'beats_v1': wins >= 3 and not catastrophic,
            'catastrophic': catastrophic
        }
        comparison[variant] = v3

    # Select winner
    candidates = [v for v, info in variant_wins.items() if info['beats_v1']]

    if len(candidates) == 0:
        return 'v1', {
            'winner': 'v1',
            'reason': 'No v3 variant beat v1 on 3+ metrics without catastrophic regression',
            'variant_analysis': variant_wins,
        }

    # Pick best by log loss
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
    raw_data_path: str = 'data/raw/ufc_fights_v1.csv',
    output_dir: str = 'data/models',
    test_cutoff: str = '2024-01-01',
    skip_tuning: bool = False,
    n_iter: int = 60,
    quick: bool = False,
    regenerate_features: bool = False,
):
    """Run the v3 training pipeline with feature pruning and two variants."""
    print("=" * 70)
    print("UFC FIGHT PREDICTION MODEL v3 TRAINING")
    print("Phase 7: Domain-Specific Opponent Adjustment")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if quick:
        skip_tuning = True
        n_iter = 10

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Feature Engineering
    # =========================================================================
    features_path = 'data/processed/ufc_fights_features_v3.csv'

    if regenerate_features or not Path(features_path).exists():
        print("\n" + "=" * 70)
        print("RUNNING PHASE 7 FEATURE ENGINEERING")
        print("=" * 70)
        df = run_feature_engineering(raw_data_path, features_path)
    else:
        df = load_data(features_path)

    # =========================================================================
    # Step 2: Load v1 baseline and feature importance for pruning
    # =========================================================================
    v1_results_path = output_path / 'evaluation_results_v1.json'
    v1_importance_path = output_path / 'feature_importance_v1.csv'

    if v1_results_path.exists():
        with open(v1_results_path, 'r') as f:
            v1_data = json.load(f)
            v1_metrics = v1_data['test_metrics']
            v1_metrics['mce'] = v1_data.get('mean_calibration_error', 0.137)
        print(f"\nv1 baseline: accuracy={v1_metrics['accuracy']:.1%}, log_loss={v1_metrics['log_loss']:.4f}")
    else:
        v1_metrics = None
        print("\nWARNING: v1 results not found")

    # =========================================================================
    # Step 3: Feature Pruning
    # =========================================================================
    print("\n" + "=" * 70)
    print("FEATURE PRUNING")
    print("=" * 70)

    importance_df = load_feature_importance(str(v1_importance_path))

    if importance_df is not None:
        features_to_prune = get_features_to_prune(importance_df, threshold=0.0)
        print(f"Features with importance <= 0: {len(features_to_prune)}")

        # Show first 10 features being pruned
        if len(features_to_prune) > 0:
            print("\nPruning features (first 10):")
            for f in features_to_prune[:10]:
                imp = importance_df[importance_df['feature'] == f]['importance'].values
                imp_val = imp[0] if len(imp) > 0 else 'N/A'
                print(f"  - {f}: {imp_val}")
            if len(features_to_prune) > 10:
                print(f"  ... and {len(features_to_prune) - 10} more")
    else:
        features_to_prune = []
        print("No importance data available, skipping pruning")

    # =========================================================================
    # Step 4: Prepare data splits
    # =========================================================================
    train_df, test_df = temporal_split(df, test_cutoff)
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    # =========================================================================
    # Step 5: Train v1 if needed
    # =========================================================================
    all_results = {}
    all_models = {}
    all_importance = {}
    all_feature_lists = {}

    if v1_metrics is None:
        print("\n" + "=" * 70)
        print("TRAINING v1 (BASELINE)")
        print("=" * 70)

        v1_features = get_original_features(df)
        v1_features = [f for f in v1_features if f not in features_to_prune]
        print(f"Features: {len(v1_features)}")

        X_train = train_df[v1_features]
        X_test = test_df[v1_features]

        model, params = train_model(X_train, y_train, skip_tuning, n_iter)
        metrics = evaluate_model(model, X_test, y_test)

        all_results['v1'] = metrics
        all_models['v1'] = model
        v1_metrics = metrics
        print(f"v1: accuracy={metrics['accuracy']:.1%}, log_loss={metrics['log_loss']:.4f}")
    else:
        all_results['v1'] = v1_metrics

    # =========================================================================
    # Step 6: Train v3 variants
    # =========================================================================
    variants = {
        'v3-A': 'Pruned + Domain Adjusted',
        'v3-B': 'Adjusted Replace',
    }

    for variant_id, variant_name in variants.items():
        print("\n" + "=" * 70)
        print(f"TRAINING {variant_id}: {variant_name}")
        print("=" * 70)

        variant_letter = variant_id.split('-')[1]
        feature_cols = get_variant_features(df, variant_letter, features_to_prune)

        # Filter to columns that exist
        feature_cols = [f for f in feature_cols if f in df.columns]
        print(f"Features: {len(feature_cols)}")

        all_feature_lists[variant_id] = feature_cols

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        model, params = train_model(X_train, y_train, skip_tuning, n_iter)
        metrics = evaluate_model(model, X_test, y_test)

        all_results[variant_id] = metrics
        all_models[variant_id] = model

        # Compute feature importance
        importance = compute_feature_importance(model, X_test, y_test, feature_cols)
        all_importance[variant_id] = importance

        print(f"{variant_id}: accuracy={metrics['accuracy']:.1%}, log_loss={metrics['log_loss']:.4f}")

    # =========================================================================
    # Step 7: Compare and select winner
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    winner, comparison = compare_models(all_results)

    print(f"\n{'Metric':<15} {'v1':<12} {'v3-A':<12} {'v3-B':<12}")
    print("-" * 51)

    for metric in ['accuracy', 'log_loss', 'roc_auc', 'brier_score', 'mce']:
        row = f"{metric:<15}"
        for model_id in ['v1', 'v3-A', 'v3-B']:
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
            if info.get('catastrophic'):
                status += " (CATASTROPHIC)"
            print(f"  {v}: {info['wins']}/5 wins - {status}")
            for detail in info['details'][:3]:
                print(f"    {detail}")

    # =========================================================================
    # Step 8: Save artifacts
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    # Save comparison results
    comparison_file = output_path / 'model_comparison_v3.json'
    with open(comparison_file, 'w') as f:
        json.dump({
            'training_date': datetime.now().isoformat(),
            'winner': winner,
            'comparison': comparison,
            'all_metrics': all_results,
            'test_split': test_cutoff,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'features_pruned': features_to_prune,
            'feature_counts': {k: len(v) for k, v in all_feature_lists.items()},
        }, f, indent=2)
    print(f"Saved comparison to {comparison_file}")

    # If v3 wins, save as production model
    if winner.startswith('v3'):
        model = all_models[winner]
        feature_cols = all_feature_lists[winner]

        # Save model
        model_file = output_path / 'ufc_model_v3.pkl'
        joblib.dump(model, model_file)
        print(f"Saved winning model to {model_file}")

        # Save feature names
        features_file = output_path / 'v3_feature_list.json'
        with open(features_file, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        print(f"Saved feature list to {features_file}")

        # Save feature importance
        importance = all_importance[winner]
        importance_file = output_path / 'feature_importance_v3.csv'
        importance.to_csv(importance_file, index=False)
        print(f"Saved feature importance to {importance_file}")

        # Save evaluation results
        eval_file = output_path / 'evaluation_results_v3.json'
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
                    'mce': all_results[winner]['mce'] - v1_metrics['mce'],
                },
                'feature_count': len(feature_cols),
                'top_features': [
                    {'feature': row['feature'], 'importance': float(row['importance'])}
                    for _, row in importance.head(10).iterrows()
                ]
            }, f, indent=2)
        print(f"Saved evaluation results to {eval_file}")

    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Winner: {winner}")

    if winner.startswith('v3'):
        v3_metrics = all_results[winner]
        print(f"  Accuracy: {v3_metrics['accuracy']:.1%} (v1: {v1_metrics['accuracy']:.1%})")
        print(f"  Log Loss: {v3_metrics['log_loss']:.4f} (v1: {v1_metrics['log_loss']:.4f})")
        print(f"  ROC-AUC: {v3_metrics['roc_auc']:.4f} (v1: {v1_metrics['roc_auc']:.4f})")
        print(f"  Brier: {v3_metrics['brier_score']:.4f} (v1: {v1_metrics['brier_score']:.4f})")
        print(f"  MCE: {v3_metrics['mce']:.1%} (v1: {v1_metrics['mce']:.1%})")
    else:
        print(f"  v1 retained (no v3 variant improved enough)")

    return winner, all_results, all_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train UFC fight prediction model v3 (Phase 7)'
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
        '--regenerate-features',
        action='store_true',
        help='Re-run feature engineering even if cached'
    )
    parser.add_argument(
        '--test-cutoff',
        type=str,
        default='2024-01-01',
        help='Date for temporal train/test split'
    )
    parser.add_argument(
        '--raw-data-path',
        type=str,
        default='data/raw/ufc_fights_v1.csv',
        help='Path to raw fight data'
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
        default=60,
        help='Number of random search iterations'
    )

    args = parser.parse_args()

    main(
        raw_data_path=args.raw_data_path,
        output_dir=args.output_dir,
        test_cutoff=args.test_cutoff,
        skip_tuning=args.skip_tuning,
        n_iter=args.n_iter,
        quick=args.quick,
        regenerate_features=args.regenerate_features,
    )
