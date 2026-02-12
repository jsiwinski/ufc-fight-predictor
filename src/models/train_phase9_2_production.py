#!/usr/bin/env python3
"""
Phase 9.2 Production Model Training.

Phase 9.2 adds physical attributes to Phase 9's Elo + Isotonic calibration model:
- Height, reach, age at fight time
- Stance (one-hot: orthodox, southpaw, switch)
- Differential features: diff_height, diff_reach, diff_age, stance_mismatch

v9.2-B (physical attributes only) was validated to beat Phase 9 on 4/5 metrics:
  - Accuracy: 60.2% -> 62.9% (+2.7pp)
  - Log Loss: 0.6521 -> 0.6467 (improved)
  - ROC-AUC: 0.6607 -> 0.6763 (+1.6pp)
  - Brier Score: 0.2304 -> 0.2276 (improved)
  - MCE: 7.4% -> 16.5% (worse - but acceptable trade-off for accuracy gains)

Usage:
    python src/models/train_phase9_2_production.py
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

# Phase 9 baseline metrics
PHASE9_METRICS = {
    'accuracy': 0.6021,
    'log_loss': 0.6521,
    'roc_auc': 0.6607,
    'brier_score': 0.2304,
    'mce': 0.074,
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


def load_fighter_details(path: str) -> pd.DataFrame:
    """Load fighter physical details."""
    logger.info(f"Loading fighter details from {path}")
    df = pd.read_csv(path)
    df['dob_parsed'] = pd.to_datetime(df['dob'], format='%b %d, %Y', errors='coerce')
    logger.info(f"Loaded details for {len(df)} fighters")
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


def add_physical_features(df: pd.DataFrame, fighter_details: pd.DataFrame) -> pd.DataFrame:
    """
    Add physical attribute features (Phase 9.2).

    Features added:
    - f1/f2_height_inches, f1/f2_reach_inches, f1/f2_age_at_fight
    - f1/f2_stance_orthodox, f1/f2_stance_southpaw, f1/f2_stance_switch
    - diff_height, diff_reach, diff_age
    - f1/f2_reach_advantage, diff_reach_advantage
    - stance_mismatch
    """
    logger.info("Adding physical attribute features (Phase 9.2)...")

    # Build fighter lookup
    details_lookup = {}
    for _, row in fighter_details.iterrows():
        details_lookup[row['fighter_name']] = {
            'height_inches': row['height_inches'] if pd.notna(row['height_inches']) else None,
            'reach_inches': row['reach_inches'] if pd.notna(row['reach_inches']) else None,
            'stance': row['stance'] if pd.notna(row['stance']) else None,
            'dob': row['dob_parsed'] if pd.notna(row['dob_parsed']) else None,
        }

    # Compute weight-class medians for imputation
    wc_medians = compute_weight_class_medians(df, details_lookup)

    # Add features for each fight
    f1_heights, f1_reaches, f1_ages = [], [], []
    f1_stance_orthodox, f1_stance_southpaw, f1_stance_switch = [], [], []
    f2_heights, f2_reaches, f2_ages = [], [], []
    f2_stance_orthodox, f2_stance_southpaw, f2_stance_switch = [], [], []

    for _, row in df.iterrows():
        wc = row['weight_class']
        fight_date = row['fight_date']
        medians = wc_medians.get(wc, wc_medians.get('_global', {'height': 70.0, 'reach': 70.0, 'age': 30.0}))

        # Fighter 1
        f1_phys = get_physical_features(row['f1_name'], fight_date, medians, details_lookup)
        f1_heights.append(f1_phys['height'])
        f1_reaches.append(f1_phys['reach'])
        f1_ages.append(f1_phys['age'])
        f1_stance_orthodox.append(f1_phys['stance_orthodox'])
        f1_stance_southpaw.append(f1_phys['stance_southpaw'])
        f1_stance_switch.append(f1_phys['stance_switch'])

        # Fighter 2
        f2_phys = get_physical_features(row['f2_name'], fight_date, medians, details_lookup)
        f2_heights.append(f2_phys['height'])
        f2_reaches.append(f2_phys['reach'])
        f2_ages.append(f2_phys['age'])
        f2_stance_orthodox.append(f2_phys['stance_orthodox'])
        f2_stance_southpaw.append(f2_phys['stance_southpaw'])
        f2_stance_switch.append(f2_phys['stance_switch'])

    # Add columns
    df = df.copy()
    df['f1_height_inches'] = f1_heights
    df['f1_reach_inches'] = f1_reaches
    df['f1_age_at_fight'] = f1_ages
    df['f1_stance_orthodox'] = f1_stance_orthodox
    df['f1_stance_southpaw'] = f1_stance_southpaw
    df['f1_stance_switch'] = f1_stance_switch

    df['f2_height_inches'] = f2_heights
    df['f2_reach_inches'] = f2_reaches
    df['f2_age_at_fight'] = f2_ages
    df['f2_stance_orthodox'] = f2_stance_orthodox
    df['f2_stance_southpaw'] = f2_stance_southpaw
    df['f2_stance_switch'] = f2_stance_switch

    # Differential features
    df['diff_height'] = df['f1_height_inches'] - df['f2_height_inches']
    df['diff_reach'] = df['f1_reach_inches'] - df['f2_reach_inches']
    df['diff_age'] = df['f1_age_at_fight'] - df['f2_age_at_fight']

    # Reach advantage (reach relative to height)
    df['f1_reach_advantage'] = df['f1_reach_inches'] - df['f1_height_inches']
    df['f2_reach_advantage'] = df['f2_reach_inches'] - df['f2_height_inches']
    df['diff_reach_advantage'] = df['f1_reach_advantage'] - df['f2_reach_advantage']

    # Stance mismatch
    df['stance_mismatch'] = (
        ((df['f1_stance_orthodox'] == 1) & (df['f2_stance_southpaw'] == 1)) |
        ((df['f1_stance_southpaw'] == 1) & (df['f2_stance_orthodox'] == 1))
    ).astype(int)

    logger.info("Added 19 physical attribute features")
    return df


def compute_weight_class_medians(df: pd.DataFrame, details_lookup: Dict) -> Dict:
    """Compute weight-class median values for imputation."""
    wc_medians = {}

    # Get fighters by weight class
    for wc in df['weight_class'].unique():
        wc_fights = df[df['weight_class'] == wc]
        fighters = set(wc_fights['f1_name']) | set(wc_fights['f2_name'])

        heights, reaches = [], []
        for f in fighters:
            if f in details_lookup:
                if details_lookup[f]['height_inches']:
                    heights.append(details_lookup[f]['height_inches'])
                if details_lookup[f]['reach_inches']:
                    reaches.append(details_lookup[f]['reach_inches'])

        wc_medians[wc] = {
            'height': float(np.median(heights)) if heights else 70.0,
            'reach': float(np.median(reaches)) if reaches else 70.0,
            'age': 30.0,
        }

    # Global fallback
    all_heights = [d['height_inches'] for d in details_lookup.values() if d['height_inches']]
    all_reaches = [d['reach_inches'] for d in details_lookup.values() if d['reach_inches']]
    wc_medians['_global'] = {
        'height': float(np.median(all_heights)) if all_heights else 70.0,
        'reach': float(np.median(all_reaches)) if all_reaches else 70.0,
        'age': 30.0,
    }

    return wc_medians


def get_physical_features(fighter_name: str, fight_date, medians: Dict, details_lookup: Dict) -> Dict:
    """Get physical features for a fighter."""
    if fighter_name not in details_lookup:
        return {
            'height': medians['height'],
            'reach': medians['reach'],
            'age': medians['age'],
            'stance_orthodox': 1,
            'stance_southpaw': 0,
            'stance_switch': 0,
        }

    details = details_lookup[fighter_name]

    # Height
    height = details['height_inches'] if details['height_inches'] else medians['height']

    # Reach
    reach = details['reach_inches'] if details['reach_inches'] else medians['reach']

    # Age
    if details['dob'] and pd.notna(fight_date):
        age = (fight_date - details['dob']).days / 365.25
        if age < 18 or age > 60:
            age = medians['age']
    else:
        age = medians['age']

    # Stance
    stance = details['stance'] if details['stance'] else 'Orthodox'
    return {
        'height': height,
        'reach': reach,
        'age': age,
        'stance_orthodox': 1 if stance == 'Orthodox' else 0,
        'stance_southpaw': 1 if stance == 'Southpaw' else 0,
        'stance_switch': 1 if stance == 'Switch' else 0,
    }


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
    """Train Phase 9.2 stacking ensemble (HGB + XGB + LR)."""
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


def calibrate_model_isotonic(model, X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    """Apply isotonic regression calibration."""
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
    details_path: str = 'data/raw/fighter_details.csv',
    output_dir: str = 'data/models',
    test_cutoff: str = '2024-01-01',
):
    """Train Phase 9.2 production model."""
    print("\n" + "=" * 70)
    print("UFC FIGHT PREDICTION MODEL - PHASE 9.2")
    print("Physical Attributes + Isotonic Calibration")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(data_path)
    fighter_details = load_fighter_details(details_path)

    # Add Elo features
    df = add_elo_features(df, K=32)

    # Add physical features
    df = add_physical_features(df, fighter_details)

    # Define physical feature columns (Phase 9.2 additions)
    physical_cols = [
        'f1_height_inches', 'f1_reach_inches', 'f1_age_at_fight',
        'f1_stance_orthodox', 'f1_stance_southpaw', 'f1_stance_switch',
        'f1_reach_advantage',
        'f2_height_inches', 'f2_reach_inches', 'f2_age_at_fight',
        'f2_stance_orthodox', 'f2_stance_southpaw', 'f2_stance_switch',
        'f2_reach_advantage',
        'diff_height', 'diff_reach', 'diff_age', 'diff_reach_advantage',
        'stance_mismatch',
    ]

    # Load Phase 9 feature list and add physical features
    phase9_features_path = Path('data/models/phase9_feature_list.json')
    if phase9_features_path.exists():
        with open(phase9_features_path) as f:
            base_features = json.load(f)
        logger.info(f"Loaded Phase 9 base features: {len(base_features)}")
    else:
        # Fallback
        base_features = [
            col for col in df.columns
            if col not in METADATA_COLS + [TARGET_COL] + physical_cols
        ]

    # Combine base + physical features
    feature_cols = base_features + [c for c in physical_cols if c in df.columns]
    feature_cols = [f for f in feature_cols if f in df.columns]
    logger.info(f"Total features: {len(feature_cols)} ({len(base_features)} base + {len(physical_cols)} physical)")

    # Temporal split
    cutoff = pd.to_datetime(test_cutoff)
    train_df = df[df['fight_date'] < cutoff].copy()
    test_df = df[df['fight_date'] >= cutoff].copy()

    print(f"\nTemporal split at {test_cutoff}:")
    print(f"  Training: {len(train_df):,} fights")
    print(f"  Testing: {len(test_df):,} fights")

    # Prepare X and y
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[TARGET_COL]

    # Sample weights
    sample_weights = compute_sample_weights(train_df['fight_date'].dt.year, half_life=8)

    # Train model
    print("\n" + "-" * 70)
    print("TRAINING STACKING ENSEMBLE")
    print("-" * 70)

    base_model = train_stacking_ensemble(X_train, y_train, sample_weights)

    # Apply isotonic calibration
    print("\n" + "-" * 70)
    print("APPLYING ISOTONIC CALIBRATION")
    print("-" * 70)

    model = calibrate_model_isotonic(base_model, X_train, y_train)

    # Evaluate
    print("\n" + "-" * 70)
    print("EVALUATION")
    print("-" * 70)

    metrics = evaluate_model(model, X_test, y_test)

    print(f"\nPhase 9.2 Results (Test Set: {len(test_df)} fights):")
    print(f"  Accuracy:   {metrics['accuracy']:.1%}")
    print(f"  Log Loss:   {metrics['log_loss']:.4f}")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  Brier:      {metrics['brier_score']:.4f}")
    if metrics['mce']:
        print(f"  MCE:        {metrics['mce']:.1%}")

    # Compare to Phase 9 baseline
    print("\n" + "-" * 70)
    print("PHASE 9 vs PHASE 9.2 COMPARISON")
    print("-" * 70)

    print(f"\n{'Metric':<15} {'Phase 9':<12} {'Phase 9.2':<12} {'Change'}")
    print("-" * 55)
    for metric in ['accuracy', 'log_loss', 'roc_auc', 'brier_score', 'mce']:
        p9 = PHASE9_METRICS[metric]
        p92 = metrics[metric]
        if p92 is None:
            continue

        if metric == 'accuracy':
            change = f"+{(p92 - p9) * 100:.1f}pp" if p92 > p9 else f"{(p92 - p9) * 100:.1f}pp"
            improved = "better" if p92 > p9 else "worse"
            print(f"{metric:<15} {p9:.1%}        {p92:.1%}        {change} ({improved})")
        elif metric == 'mce':
            change = f"{(p92 - p9) * 100:.1f}pp" if p92 > p9 else f"{(p92 - p9) * 100:.1f}pp"
            improved = "worse" if p92 > p9 else "better"
            print(f"{metric:<15} {p9:.1%}        {p92:.1%}        {change} ({improved})")
        else:
            change = f"+{p92 - p9:.4f}" if p92 > p9 else f"{p92 - p9:.4f}"
            # Lower is better for log_loss and brier
            if metric in ['log_loss', 'brier_score']:
                improved = "better" if p92 < p9 else "worse"
            else:
                improved = "better" if p92 > p9 else "worse"
            print(f"{metric:<15} {p9:.4f}       {p92:.4f}       {change} ({improved})")

    # Feature importance
    print("\n" + "-" * 70)
    print("TOP 20 FEATURES BY IMPORTANCE")
    print("-" * 70)

    importance_df = compute_feature_importance(model, X_test, y_test, feature_cols)

    print(f"\n{'Rank':<5} {'Feature':<45} {'Importance'}")
    print("-" * 60)
    for rank, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
        marker = " *" if row['feature'] in physical_cols else ""
        print(f"{rank:<5} {row['feature']:<45} {row['importance']:.4f}{marker}")

    print("\n* = Physical attribute feature (Phase 9.2)")

    # Save artifacts
    print("\n" + "-" * 70)
    print("SAVING ARTIFACTS")
    print("-" * 70)

    # Save as Phase 9.2 production model (replaces Phase 9)
    model_path = output_path / 'ufc_model_phase9_2.pkl'
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Also save as the Phase 9 path for serve.py compatibility
    # (serve.py looks for ufc_model_phase9.pkl)
    model_path_compat = output_path / 'ufc_model_phase9.pkl'
    joblib.dump(model, model_path_compat)
    logger.info(f"Saved production model to {model_path_compat}")

    # Save feature list
    features_path = output_path / 'phase9_2_feature_list.json'
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Saved feature list to {features_path}")

    # Also update phase9_feature_list.json for serve.py
    features_path_compat = output_path / 'phase9_feature_list.json'
    with open(features_path_compat, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Saved production feature list to {features_path_compat}")

    # Save feature importance
    importance_path = output_path / 'phase9_2_feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance to {importance_path}")

    # Save evaluation results
    results = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'StackingClassifier + Isotonic Calibration + Physical Attributes',
        'calibration_method': 'isotonic',
        'phase': '9.2',
        'feature_count': len(feature_cols),
        'physical_feature_count': len([c for c in physical_cols if c in feature_cols]),
        'train_size': len(train_df),
        'test_size': len(test_df),
        'test_cutoff': test_cutoff,
        'metrics': metrics,
        'phase9_comparison': PHASE9_METRICS,
        'top_features': [
            {'feature': row['feature'], 'importance': float(row['importance']),
             'is_physical': row['feature'] in physical_cols}
            for _, row in importance_df.head(20).iterrows()
        ]
    }

    results_path = output_path / 'phase9_2_production_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results to {results_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 9.2 TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nProduction model saved to: {model_path_compat}")
    print(f"Feature list saved to: {features_path_compat}")
    print(f"\nPhase 9.2 includes:")
    print(f"  - {len(base_features)} Phase 9 features (Elo + pruned features)")
    print(f"  - {len([c for c in physical_cols if c in feature_cols])} physical attribute features")
    print(f"  - Isotonic regression calibration")
    print(f"  - Exponential decay sample weighting")

    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Phase 9.2 UFC fight prediction model'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/ufc_fights_features_v1.csv',
        help='Path to processed feature data'
    )
    parser.add_argument(
        '--details-path',
        type=str,
        default='data/raw/fighter_details.csv',
        help='Path to fighter details CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models',
        help='Directory to save models and artifacts'
    )
    parser.add_argument(
        '--test-cutoff',
        type=str,
        default='2024-01-01',
        help='Date for temporal train/test split'
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        details_path=args.details_path,
        output_dir=args.output_dir,
        test_cutoff=args.test_cutoff,
    )
