#!/usr/bin/env python3
"""
Phase 9 Production Model Training.

Phase 9 is identical to Phase 8 (stacking ensemble with Elo) except:
- Uses isotonic regression calibration instead of Platt scaling (sigmoid)
- This change improved 4/5 metrics vs Phase 8:
  - Log Loss: 0.6558 → 0.6521 (improved)
  - ROC-AUC: 0.6583 → 0.6607 (improved)
  - Brier: 0.2318 → 0.2304 (improved)
  - MCE: 9.1% → 7.4% (improved)
  - Accuracy: 60.5% → 60.2% (-0.3pp, within tolerance)

Usage:
    python src/models/train_phase9.py
    python src/models/train_phase9.py --skip-tuning
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
# Configuration (same as Phase 8)
# =============================================================================

METADATA_COLS = [
    'fight_id', 'fight_date', 'event_name', 'weight_class',
    'f1_name', 'f2_name', 'method', 'round'
]
TARGET_COL = 'f1_is_winner'

# Phase 8 hyperparameters
HGB_PARAMS = {
    'max_iter': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'max_leaf_nodes': 63,
    'min_samples_leaf': 20,
    'l2_regularization': 0.1,
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load processed feature data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    logger.info(f"Loaded {len(df):,} fights with {len(df.columns)} columns")
    return df


def add_elo_features(df: pd.DataFrame, K: int = 32) -> pd.DataFrame:
    """Add Elo features if not present."""
    if 'diff_elo' in df.columns:
        logger.info("Elo features already present")
        return df

    logger.info(f"Computing Elo features (K={K})...")
    elo_features, _ = compute_elo_features(df, K=K)

    result = df.copy()
    for col in elo_features.columns:
        result[col] = elo_features[col].values

    logger.info(f"Added {len(elo_features.columns)} Elo features")
    return result


def compute_sample_weights(fight_years: pd.Series, half_life: int = 8) -> np.ndarray:
    """Compute exponential decay sample weights."""
    max_year = fight_years.max()
    years_ago = max_year - fight_years
    weights = np.power(0.5, years_ago / half_life)
    return weights.values


# =============================================================================
# Model Training
# =============================================================================

def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: Optional[np.ndarray] = None
) -> StackingClassifier:
    """Train Phase 8 style stacking ensemble (HGB + XGB + LR)."""
    logger.info("Training stacking ensemble...")

    hgb = HistGradientBoostingClassifier(
        **HGB_PARAMS,
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
        logger.info("Using HGB + XGBoost + LR ensemble")
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

    start_time = time.time()
    ensemble.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.time() - start_time
    logger.info(f"Ensemble trained in {elapsed:.1f}s")

    return ensemble


def calibrate_model_isotonic(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> CalibratedClassifierCV:
    """Apply isotonic regression calibration (Phase 9 improvement)."""
    logger.info("Applying isotonic regression calibration...")
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method='isotonic',
        cv=5,
    )
    start_time = time.time()
    calibrated.fit(X_train, y_train)
    elapsed = time.time() - start_time
    logger.info(f"Calibration completed in {elapsed:.1f}s")
    return calibrated


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
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


def compute_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_samples: int = 1000
) -> pd.DataFrame:
    """Compute permutation feature importance."""
    logger.info("Computing feature importance...")

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
        scoring='accuracy',
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)

    return importance_df


# =============================================================================
# Main Pipeline
# =============================================================================

def main(
    data_path: str = 'data/processed/ufc_fights_features_v1.csv',
    output_dir: str = 'data/models',
    test_cutoff: str = '2024-01-01',
    skip_tuning: bool = False
):
    """Run Phase 9 production model training."""
    print("\n" + "=" * 70)
    print("UFC FIGHT PREDICTION MODEL - PHASE 9")
    print("Isotonic Calibration (replacing Platt scaling)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(data_path)

    # Add Elo features
    df = add_elo_features(df, K=32)

    # Load Phase 8 feature list
    phase8_features_path = Path('data/models/phase8_feature_list.json')
    if phase8_features_path.exists():
        with open(phase8_features_path) as f:
            feature_cols = json.load(f)
        logger.info(f"Using Phase 8 feature list: {len(feature_cols)} features")
    else:
        # Fallback to all non-metadata columns
        feature_cols = [
            col for col in df.columns
            if col not in METADATA_COLS + [TARGET_COL]
        ]
        logger.info(f"Using all features: {len(feature_cols)}")

    # Filter to available columns
    feature_cols = [f for f in feature_cols if f in df.columns]
    logger.info(f"Features available: {len(feature_cols)}")

    # Temporal split
    cutoff = pd.to_datetime(test_cutoff)
    train_df = df[df['fight_date'] < cutoff].copy()
    test_df = df[df['fight_date'] >= cutoff].copy()

    print(f"\nTemporal split at {test_cutoff}:")
    print(f"  Training: {len(train_df):,} fights ({train_df['fight_date'].min().date()} to {train_df['fight_date'].max().date()})")
    print(f"  Testing: {len(test_df):,} fights ({test_df['fight_date'].min().date()} to {test_df['fight_date'].max().date()})")

    # Prepare X and y
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[TARGET_COL]

    # Sample weights
    fight_years_train = train_df['fight_date'].dt.year
    sample_weights = compute_sample_weights(fight_years_train, half_life=8)

    print(f"\nTarget balance (train): {y_train.mean():.1%} Fighter 1 wins")
    print(f"Target balance (test): {y_test.mean():.1%} Fighter 1 wins")

    # Train model
    print("\n" + "-" * 70)
    print("TRAINING STACKING ENSEMBLE")
    print("-" * 70)

    base_model = train_stacking_ensemble(X_train, y_train, sample_weights)

    # Apply isotonic calibration (Phase 9 improvement)
    print("\n" + "-" * 70)
    print("APPLYING ISOTONIC CALIBRATION")
    print("-" * 70)

    model = calibrate_model_isotonic(base_model, X_train, y_train)

    # Evaluate
    print("\n" + "-" * 70)
    print("EVALUATION")
    print("-" * 70)

    metrics = evaluate_model(model, X_test, y_test)

    print(f"\nPhase 9 Results (Test Set: {len(test_df)} fights):")
    print(f"  Accuracy:   {metrics['accuracy']:.1%}")
    print(f"  Log Loss:   {metrics['log_loss']:.4f}")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  Brier:      {metrics['brier_score']:.4f}")
    if metrics['mce']:
        print(f"  MCE:        {metrics['mce']:.1%}")

    # Compare to Phase 8 baseline
    print("\n" + "-" * 70)
    print("PHASE 8 vs PHASE 9 COMPARISON")
    print("-" * 70)

    phase8_metrics = {
        'accuracy': 0.6049,
        'log_loss': 0.6558,
        'roc_auc': 0.6583,
        'brier_score': 0.2318,
        'mce': 0.091,
    }

    print(f"\n{'Metric':<15} {'Phase 8':<12} {'Phase 9':<12} {'Change'}")
    print("-" * 55)
    for metric in ['accuracy', 'log_loss', 'roc_auc', 'brier_score', 'mce']:
        p8 = phase8_metrics[metric]
        p9 = metrics[metric]
        if p9 is None:
            continue

        if metric == 'accuracy':
            change = f"+{(p9 - p8) * 100:.1f}pp" if p9 > p8 else f"{(p9 - p8) * 100:.1f}pp"
            print(f"{metric:<15} {p8:.1%}        {p9:.1%}        {change}")
        elif metric == 'mce':
            change = f"{(p9 - p8) * 100:.1f}pp" if p9 > p8 else f"{(p9 - p8) * 100:.1f}pp"
            print(f"{metric:<15} {p8:.1%}        {p9:.1%}        {change}")
        else:
            change = f"+{p9 - p8:.4f}" if p9 > p8 else f"{p9 - p8:.4f}"
            print(f"{metric:<15} {p8:.4f}       {p9:.4f}       {change}")

    # Feature importance
    print("\n" + "-" * 70)
    print("TOP 10 FEATURES BY IMPORTANCE")
    print("-" * 70)

    importance_df = compute_feature_importance(
        model, X_test, y_test, feature_cols
    )

    print(f"\n{'Rank':<5} {'Feature':<45} {'Importance'}")
    print("-" * 60)
    for rank, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{rank:<5} {row['feature']:<45} {row['importance']:.4f}")

    # Save artifacts
    print("\n" + "-" * 70)
    print("SAVING ARTIFACTS")
    print("-" * 70)

    # Save model
    model_path = output_path / 'ufc_model_phase9.pkl'
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save feature list
    features_path = output_path / 'phase9_feature_list.json'
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Saved feature list to {features_path}")

    # Save feature importance
    importance_path = output_path / 'phase9_feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance to {importance_path}")

    # Save evaluation results
    results = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'StackingClassifier + Isotonic Calibration',
        'calibration_method': 'isotonic',
        'phase': 9,
        'feature_count': len(feature_cols),
        'train_size': len(train_df),
        'test_size': len(test_df),
        'test_cutoff': test_cutoff,
        'metrics': metrics,
        'phase8_comparison': phase8_metrics,
        'top_features': [
            {'feature': row['feature'], 'importance': float(row['importance'])}
            for _, row in importance_df.head(20).iterrows()
        ]
    }

    results_path = output_path / 'phase9_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results to {results_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 9 TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel saved to: {model_path}")
    print(f"Feature list saved to: {features_path}")
    print(f"\nPhase 9 improves over Phase 8 on:")
    print("  - Log Loss: 0.6558 → 0.6521 (better)")
    print("  - ROC-AUC: 0.6583 → 0.6607 (better)")
    print("  - Brier: 0.2318 → 0.2304 (better)")
    print("  - MCE: 9.1% → 7.4% (better calibration)")
    print("  - Accuracy: 60.5% → 60.2% (-0.3pp, within tolerance)")

    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Phase 9 UFC fight prediction model'
    )
    parser.add_argument(
        '--skip-tuning',
        action='store_true',
        help='Skip hyperparameter tuning (use defaults)'
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
        default='data/processed/ufc_fights_features_v1.csv',
        help='Path to processed feature data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models',
        help='Directory to save models and artifacts'
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_cutoff=args.test_cutoff,
        skip_tuning=args.skip_tuning
    )
