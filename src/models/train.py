"""
Model training module for UFC fight prediction.

This module handles:
- Training machine learning models
- Hyperparameter tuning
- Model validation and evaluation
- Model persistence

Usage:
    python src/models/train.py
    python src/models/train.py --skip-tuning
    python src/models/train.py --test-cutoff 2024-01-01
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
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

# Default hyperparameter search space
DEFAULT_PARAM_DIST = {
    'max_iter': [100, 200, 300, 500],
    'max_depth': [4, 5, 6, 8, None],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_leaf_nodes': [31, 63, 127, None],
    'min_samples_leaf': [10, 20, 50],
    'l2_regularization': [0.0, 0.1, 1.0],
}

# Quick training params (when --skip-tuning)
QUICK_PARAMS = {
    'max_iter': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'max_leaf_nodes': 63,
    'min_samples_leaf': 20,
    'l2_regularization': 0.1,
}

# Target metrics
TARGETS = {
    'accuracy': (0.62, 0.65),  # 62-65%
    'log_loss': 0.60,  # <0.60
    'roc_auc': (0.68, 0.72),  # 0.68-0.72
    'brier_score': 0.25,  # <0.25
    'market_baseline': 0.65,  # ~65%
}


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load processed feature data from CSV."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    logger.info(f"Loaded {len(df):,} fights with {len(df.columns)} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (excluding metadata and target)."""
    feature_cols = [
        col for col in df.columns
        if col not in METADATA_COLS + [TARGET_COL]
    ]
    return feature_cols


def temporal_split(
    df: pd.DataFrame,
    cutoff_date: str = '2024-01-01'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (train on past, test on future).

    Args:
        df: Full dataset with fight_date column
        cutoff_date: Date string for train/test split

    Returns:
        Tuple of (train_df, test_df)
    """
    cutoff = pd.to_datetime(cutoff_date)
    train_df = df[df['fight_date'] < cutoff].copy()
    test_df = df[df['fight_date'] >= cutoff].copy()

    logger.info(f"Temporal split at {cutoff_date}:")
    logger.info(f"  Training: {len(train_df):,} fights "
                f"({train_df['fight_date'].min().date()} to {train_df['fight_date'].max().date()})")
    logger.info(f"  Testing: {len(test_df):,} fights "
                f"({test_df['fight_date'].min().date()} to {test_df['fight_date'].max().date()})")

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
    """
    Train a HistGradientBoostingClassifier with optional hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training target
        skip_tuning: If True, use default params without tuning
        n_iter: Number of random search iterations
        cv_folds: Number of cross-validation folds

    Returns:
        Tuple of (trained model, best parameters dict)
    """
    if skip_tuning:
        logger.info("Skipping hyperparameter tuning (using defaults)")
        model = HistGradientBoostingClassifier(
            **QUICK_PARAMS,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
        )
        model.fit(X_train, y_train)
        return model, QUICK_PARAMS

    # Hyperparameter tuning
    logger.info(f"Starting hyperparameter tuning ({n_iter} iterations, {cv_folds}-fold CV)")

    base_model = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=DEFAULT_PARAM_DIST,
        n_iter=n_iter,
        scoring='neg_log_loss',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start_time

    logger.info(f"Tuning completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Best CV log loss: {-search.best_score_:.4f}")
    logger.info(f"Best parameters: {search.best_params_}")

    return search.best_estimator_, search.best_params_


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(
    model: HistGradientBoostingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary with all evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Core metrics
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    mean_calibration_error = np.mean(np.abs(prob_pred - prob_true))

    metrics = {
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'roc_auc': float(roc_auc),
        'brier_score': float(brier),
        'confusion_matrix': cm.tolist(),
        'mean_calibration_error': float(mean_calibration_error),
        'predictions': {
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'y_true': y_test.tolist()
        }
    }

    return metrics


def print_evaluation_report(metrics: Dict):
    """Print formatted evaluation report."""
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)

    accuracy = metrics['accuracy']
    logloss = metrics['log_loss']
    roc_auc = metrics['roc_auc']
    brier = metrics['brier_score']

    print(f"\n{'Metric':<20} {'Value':<15} {'Target':<15} {'Status'}")
    print("-"*65)

    # Accuracy
    acc_status = '✓ BEAT' if accuracy >= 0.62 else '✗ MISS'
    print(f"{'Accuracy':<20} {accuracy:.1%}{'':<10} {'62-65%':<15} {acc_status}")

    # Log Loss
    ll_status = '✓ BEAT' if logloss < 0.60 else '✗ MISS'
    print(f"{'Log Loss':<20} {logloss:.4f}{'':<10} {'<0.60':<15} {ll_status}")

    # ROC-AUC
    auc_status = '✓ BEAT' if roc_auc >= 0.68 else '✗ MISS'
    print(f"{'ROC-AUC':<20} {roc_auc:.4f}{'':<10} {'0.68-0.72':<15} {auc_status}")

    # Brier Score
    brier_status = '✓ BEAT' if brier < 0.25 else '✗ MISS'
    print(f"{'Brier Score':<20} {brier:.4f}{'':<10} {'<0.25':<15} {brier_status}")

    # Market baseline
    print(f"\n{'Market Baseline':<20} ~65%")
    if accuracy > 0.65:
        print(f"Model BEATS market baseline by {(accuracy - 0.65)*100:.1f}pp")
    else:
        print(f"Model is {(0.65 - accuracy)*100:.1f}pp below market baseline")

    # Confusion matrix
    cm = metrics['confusion_matrix']
    print("\n" + "-"*65)
    print("CONFUSION MATRIX")
    print("-"*65)
    print(f"                    Predicted F2 Win    Predicted F1 Win")
    print(f"Actual F2 Win       {cm[0][0]:<20} {cm[0][1]}")
    print(f"Actual F1 Win       {cm[1][0]:<20} {cm[1][1]}")


# ============================================================================
# Feature Importance
# ============================================================================

def compute_feature_importance(
    model: HistGradientBoostingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_samples: int = 1000
) -> pd.DataFrame:
    """
    Compute permutation feature importance.

    Args:
        model: Trained model
        X: Feature data
        y: Target data
        feature_names: List of feature names
        n_samples: Number of samples to use (for speed)

    Returns:
        DataFrame with feature importance scores
    """
    logger.info("Computing feature importance (permutation method)...")

    # Sample data for faster computation
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
# Model Persistence
# ============================================================================

def save_artifacts(
    model: HistGradientBoostingClassifier,
    metrics: Dict,
    best_params: Dict,
    feature_importance: pd.DataFrame,
    feature_names: List[str],
    train_info: Dict,
    output_dir: str = 'data/models'
):
    """
    Save all training artifacts.

    Args:
        model: Trained model
        metrics: Evaluation metrics
        best_params: Best hyperparameters
        feature_importance: Feature importance DataFrame
        feature_names: List of feature names
        train_info: Training metadata (dates, sizes, etc.)
        output_dir: Directory to save artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_path / 'ufc_model_v1.pkl'
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save feature importance
    importance_path = output_path / 'feature_importance_v1.csv'
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance to {importance_path}")

    # Save feature names
    names_path = output_path / 'feature_names_v1.json'
    with open(names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Saved feature names to {names_path}")

    # Save evaluation results
    evaluation_results = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'HistGradientBoostingClassifier',
        'dataset': train_info,
        'best_params': {k: (v if v is not None else 'None') for k, v in best_params.items()},
        'test_metrics': {
            'accuracy': metrics['accuracy'],
            'log_loss': metrics['log_loss'],
            'roc_auc': metrics['roc_auc'],
            'brier_score': metrics['brier_score'],
        },
        'mean_calibration_error': metrics['mean_calibration_error'],
        'target_comparison': {
            'accuracy_target': '62-65%',
            'accuracy_met': bool(metrics['accuracy'] >= 0.62),
            'log_loss_target': '<0.60',
            'log_loss_met': bool(metrics['log_loss'] < 0.60),
            'roc_auc_target': '0.68-0.72',
            'roc_auc_met': bool(metrics['roc_auc'] >= 0.68),
            'brier_target': '<0.25',
            'brier_met': bool(metrics['brier_score'] < 0.25),
            'market_baseline': 0.65,
            'beat_market': bool(metrics['accuracy'] > 0.65)
        },
        'top_features': [
            {'feature': row['feature'], 'importance': float(row['importance'])}
            for _, row in feature_importance.head(10).iterrows()
        ]
    }

    results_path = output_path / 'evaluation_results_v1.json'
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"Saved evaluation results to {results_path}")


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main(
    data_path: str = 'data/processed/ufc_fights_features_v1.csv',
    output_dir: str = 'data/models',
    test_cutoff: str = '2024-01-01',
    skip_tuning: bool = False,
    n_iter: int = 50
):
    """
    Run the complete training pipeline.

    Args:
        data_path: Path to processed feature data
        output_dir: Directory to save models and artifacts
        test_cutoff: Date string for temporal train/test split
        skip_tuning: If True, skip hyperparameter tuning
        n_iter: Number of random search iterations
    """
    print("="*70)
    print("UFC FIGHT PREDICTION MODEL TRAINING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: HistGradientBoostingClassifier (sklearn)")
    print(f"Tuning: {'Disabled' if skip_tuning else f'RandomizedSearchCV ({n_iter} iterations)'}")

    # Load data
    df = load_data(data_path)
    feature_cols = get_feature_columns(df)
    print(f"\nFeatures: {len(feature_cols)}")

    # Temporal split
    train_df, test_df = temporal_split(df, test_cutoff)

    # Prepare X and y
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    print(f"\nTarget balance (train): {y_train.mean():.1%} Fighter 1 wins")
    print(f"Target balance (test): {y_test.mean():.1%} Fighter 1 wins")
    print(f"Missing values (train): {X_train.isnull().sum().sum():,}")
    print(f"Missing values (test): {X_test.isnull().sum().sum():,}")

    # Train model
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)

    model, best_params = train_model(
        X_train, y_train,
        skip_tuning=skip_tuning,
        n_iter=n_iter
    )

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print_evaluation_report(metrics)

    # Feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (Top 10)")
    print("="*70)

    importance_df = compute_feature_importance(
        model, X_test, y_test, feature_cols
    )

    print(f"\n{'Rank':<5} {'Feature':<45} {'Importance'}")
    print("-"*60)
    for rank, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{rank:<5} {row['feature']:<45} {row['importance']:.4f}")

    zero_importance = (importance_df['importance'] <= 0).sum()
    print(f"\nFeatures with zero/negative importance: {zero_importance}")

    # Save artifacts
    print("\n" + "="*70)
    print("SAVING ARTIFACTS")
    print("="*70)

    train_info = {
        'total_fights': int(len(df)),
        'train_fights': int(len(train_df)),
        'test_fights': int(len(test_df)),
        'train_date_range': [
            str(train_df['fight_date'].min().date()),
            str(train_df['fight_date'].max().date())
        ],
        'test_date_range': [
            str(test_df['fight_date'].min().date()),
            str(test_df['fight_date'].max().date())
        ],
        'feature_count': int(len(feature_cols)),
        'target_balance_train': float(y_train.mean()),
        'target_balance_test': float(y_test.mean())
    }

    save_artifacts(
        model, metrics, best_params,
        importance_df, feature_cols, train_info,
        output_dir
    )

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Accuracy: {metrics['accuracy']:.1%} (target: 62-65%)")
    print(f"  Log Loss: {metrics['log_loss']:.4f} (target: <0.60)")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f} (target: 0.68-0.72)")
    print(f"  Brier Score: {metrics['brier_score']:.4f} (target: <0.25)")

    print(f"\nArtifacts saved to: {output_dir}/")
    print(f"  - ufc_model_v1.pkl")
    print(f"  - evaluation_results_v1.json")
    print(f"  - feature_importance_v1.csv")
    print(f"  - feature_names_v1.json")

    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train UFC fight prediction model'
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
        help='Date for temporal train/test split (YYYY-MM-DD)'
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
        n_iter=args.n_iter
    )
