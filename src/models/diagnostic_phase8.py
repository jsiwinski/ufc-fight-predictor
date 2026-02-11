#!/usr/bin/env python3
"""
Phase 8 Model Diagnostic: Performance by Segment Analysis.

Analyzes where the Phase 8 stacking ensemble model performs best and worst
by slicing predictions across multiple dimensions.

Usage:
    python src/models/diagnostic_phase8.py
    python src/models/diagnostic_phase8.py --output json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

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

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'ufc_fights_features_v1.csv'
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ufc_fights_v1.csv'
MODEL_PATH = PROJECT_ROOT / 'data' / 'models' / 'ufc_model_phase8.pkl'
FEATURE_PATH = PROJECT_ROOT / 'data' / 'models' / 'phase8_winning_features.json'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'models' / 'phase8_diagnostic_results.json'


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed and raw data."""
    logger.info("Loading data...")

    # Load processed features
    df = pd.read_csv(DATA_PATH)
    df['fight_date'] = pd.to_datetime(df['fight_date'])

    # Load raw data for additional info (method, round, etc.)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    raw_df['fight_date'] = pd.to_datetime(raw_df['event_date'], format='mixed')

    logger.info(f"Loaded {len(df):,} fights")
    return df, raw_df


def prepare_predictions(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    model,
    feature_names: List[str],
    test_cutoff: str = '2024-01-01'
) -> pd.DataFrame:
    """
    Prepare DataFrame with model predictions and all metadata for slicing.
    """
    logger.info("Computing Elo features...")
    elo_features, _ = compute_elo_features(df, K=32)

    # Merge Elo features
    for col in elo_features.columns:
        df[col] = elo_features[col].values

    # Ensure all feature columns exist
    available_features = [f for f in feature_names if f in df.columns]
    logger.info(f"Using {len(available_features)}/{len(feature_names)} features")

    # Get features
    X = df[available_features].fillna(0)

    # Generate predictions
    logger.info("Generating predictions...")
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    # Create analysis DataFrame
    analysis_df = df[['fight_date', 'event_name', 'weight_class',
                      'f1_name', 'f2_name', 'f1_is_winner']].copy()

    # Add predictions
    analysis_df['y_true'] = df['f1_is_winner'].values
    analysis_df['y_proba'] = y_proba
    analysis_df['y_pred'] = y_pred
    analysis_df['correct'] = (analysis_df['y_pred'] == analysis_df['y_true']).astype(int)
    analysis_df['confidence'] = np.maximum(y_proba, 1 - y_proba)

    # Add features for slicing
    analysis_df['diff_elo'] = df['diff_elo'].values if 'diff_elo' in df.columns else 0
    analysis_df['f1_elo'] = df['f1_pre_fight_elo'].values if 'f1_pre_fight_elo' in df.columns else 1500
    analysis_df['f2_elo'] = df['f2_pre_fight_elo'].values if 'f2_pre_fight_elo' in df.columns else 1500

    # Career stats
    analysis_df['f1_career_fights'] = df['f1_career_fights'].values if 'f1_career_fights' in df.columns else 0
    analysis_df['f2_career_fights'] = df['f2_career_fights'].values if 'f2_career_fights' in df.columns else 0
    analysis_df['f1_career_win_rate'] = df['f1_career_win_rate'].values if 'f1_career_win_rate' in df.columns else 0.5
    analysis_df['f2_career_win_rate'] = df['f2_career_win_rate'].values if 'f2_career_win_rate' in df.columns else 0.5

    # Win/loss streaks (column names vary by dataset version)
    f1_streak_col = 'f1_win_streak' if 'f1_win_streak' in df.columns else 'f1_current_win_streak'
    f2_streak_col = 'f2_win_streak' if 'f2_win_streak' in df.columns else 'f2_current_win_streak'
    f1_loss_col = 'f1_loss_streak' if 'f1_loss_streak' in df.columns else 'f1_current_loss_streak'
    f2_loss_col = 'f2_loss_streak' if 'f2_loss_streak' in df.columns else 'f2_current_loss_streak'

    analysis_df['f1_current_win_streak'] = df[f1_streak_col].values if f1_streak_col in df.columns else 0
    analysis_df['f2_current_win_streak'] = df[f2_streak_col].values if f2_streak_col in df.columns else 0
    analysis_df['f1_current_loss_streak'] = df[f1_loss_col].values if f1_loss_col in df.columns else 0
    analysis_df['f2_current_loss_streak'] = df[f2_loss_col].values if f2_loss_col in df.columns else 0

    # Striking/grappling stats for style classification
    for col in ['f1_avg_sig_strikes_landed', 'f2_avg_sig_strikes_landed',
                'f1_avg_takedowns_landed', 'f2_avg_takedowns_landed',
                'f1_career_sub_rate', 'f2_career_sub_rate']:
        if col in df.columns:
            analysis_df[col] = df[col].values
        else:
            analysis_df[col] = 0

    # Method and round are already in processed data - use directly
    analysis_df['method'] = df['method'].fillna('Unknown').values
    analysis_df['round'] = df['round'].fillna(3).values

    # Add fight year
    analysis_df['fight_year'] = analysis_df['fight_date'].dt.year

    # Add train/test split indicator
    cutoff = pd.to_datetime(test_cutoff)
    analysis_df['is_test'] = (analysis_df['fight_date'] >= cutoff).astype(int)

    logger.info(f"Analysis DataFrame ready: {len(analysis_df):,} fights")
    return analysis_df


def compute_segment_metrics(df: pd.DataFrame, segment_name: str = "Overall") -> Dict:
    """Compute accuracy, log loss, and count for a segment."""
    if len(df) == 0:
        return {'segment': segment_name, 'count': 0, 'accuracy': None, 'log_loss': None}

    accuracy = accuracy_score(df['y_true'], df['y_pred'])

    # Compute log loss safely
    y_proba = df['y_proba'].values
    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)

    # Handle case where segment has only one class
    try:
        ll = log_loss(df['y_true'], y_proba, labels=[0, 1])
    except ValueError:
        ll = None

    return {
        'segment': segment_name,
        'count': len(df),
        'accuracy': round(accuracy * 100, 1),
        'log_loss': round(ll, 3) if ll is not None else None
    }


def analyze_confidence_buckets(df: pd.DataFrame) -> List[Dict]:
    """Segment 1: Split fights by model confidence."""
    buckets = [
        (0.50, 0.55, '50-55% (coin flips)'),
        (0.55, 0.60, '55-60% (slight lean)'),
        (0.60, 0.65, '60-65% (moderate)'),
        (0.65, 0.70, '65-70% (confident)'),
        (0.70, 0.75, '70-75% (strong pick)'),
        (0.75, 1.01, '75%+ (heavy favorite)')
    ]

    results = []
    for low, high, label in buckets:
        mask = (df['confidence'] >= low) & (df['confidence'] < high)
        results.append(compute_segment_metrics(df[mask], label))

    results.append(compute_segment_metrics(df, 'Overall'))
    return results


def analyze_win_streaks(df: pd.DataFrame) -> List[Dict]:
    """Segment 2: Split by fighter win streaks."""
    results = []

    # Both on win streaks (2+)
    both_streaking = (df['f1_current_win_streak'] >= 2) & (df['f2_current_win_streak'] >= 2)
    results.append(compute_segment_metrics(df[both_streaking], 'Both streaking (2+)'))

    # One streaking (2+), one not
    one_streak = ((df['f1_current_win_streak'] >= 2) & (df['f2_current_win_streak'] < 2)) | \
                 ((df['f2_current_win_streak'] >= 2) & (df['f1_current_win_streak'] < 2))
    results.append(compute_segment_metrics(df[one_streak], 'One streaking'))

    # Both on losing streaks
    both_losing = (df['f1_current_loss_streak'] >= 2) & (df['f2_current_loss_streak'] >= 2)
    results.append(compute_segment_metrics(df[both_losing], 'Both losing'))

    # Long streak (5+) vs non-streaking
    long_vs_short = ((df['f1_current_win_streak'] >= 5) & (df['f2_current_win_streak'] < 2)) | \
                    ((df['f2_current_win_streak'] >= 5) & (df['f1_current_win_streak'] < 2))
    results.append(compute_segment_metrics(df[long_vs_short], 'Streak 5+ vs <2'))

    results.append(compute_segment_metrics(df, 'Overall'))
    return results


def analyze_experience_asymmetry(df: pd.DataFrame) -> List[Dict]:
    """Segment 3: Split by fighter experience gap."""
    results = []

    # Experience classifications
    f1_exp = df['f1_career_fights']
    f2_exp = df['f2_career_fights']
    exp_gap = abs(f1_exp - f2_exp)

    # Both experienced (10+ each)
    both_exp = (f1_exp >= 10) & (f2_exp >= 10)
    results.append(compute_segment_metrics(df[both_exp], 'Both experienced (10+)'))

    # Both inexperienced (<5 each)
    both_inexp = (f1_exp < 5) & (f2_exp < 5)
    results.append(compute_segment_metrics(df[both_inexp], 'Both inexperienced (<5)'))

    # Large experience gap (>8 fights difference)
    large_gap = exp_gap > 8
    results.append(compute_segment_metrics(df[large_gap], 'Large gap (>8 fights)'))

    # Debut vs veteran
    debut_vs_vet = ((f1_exp == 0) & (f2_exp >= 5)) | ((f2_exp == 0) & (f1_exp >= 5))
    results.append(compute_segment_metrics(df[debut_vs_vet], 'Debut vs veteran'))

    # Any debut
    any_debut = (f1_exp == 0) | (f2_exp == 0)
    results.append(compute_segment_metrics(df[any_debut], 'Any debut'))

    # No debuts
    no_debuts = (f1_exp > 0) & (f2_exp > 0)
    results.append(compute_segment_metrics(df[no_debuts], 'No debuts'))

    results.append(compute_segment_metrics(df, 'Overall'))
    return results


def analyze_weight_class(df: pd.DataFrame) -> List[Dict]:
    """Segment 4: Accuracy by division."""
    results = []

    divisions = [
        'Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight',
        'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight',
        "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
        "Women's Featherweight"
    ]

    for div in divisions:
        mask = df['weight_class'].str.contains(div, case=False, na=False)
        if mask.sum() >= 10:  # Only report if sufficient sample
            results.append(compute_segment_metrics(df[mask], div))

    results.append(compute_segment_metrics(df, 'Overall'))
    return results


def analyze_fight_method(df: pd.DataFrame) -> List[Dict]:
    """Segment 5: Accuracy by fight method (post-hoc)."""
    results = []

    method_str = df['method'].str.upper()

    # KO/TKO (matches both "KO/TKO" and "KO" and "TKO")
    ko = method_str.str.contains('KO|TKO', na=False, regex=True)
    results.append(compute_segment_metrics(df[ko], 'KO/TKO'))

    # Submission (matches "SUB" and "SUBMISSION")
    sub = method_str.str.contains('SUB', na=False)
    results.append(compute_segment_metrics(df[sub], 'Submission'))

    # Unanimous decision (matches "U-DEC" and "UNANIMOUS")
    ud = method_str.str.contains('U-DEC|UNANIMOUS', na=False, regex=True)
    results.append(compute_segment_metrics(df[ud], 'Unanimous Decision'))

    # Split decision (matches "S-DEC" and "SPLIT")
    split = method_str.str.contains('S-DEC|SPLIT', na=False, regex=True)
    results.append(compute_segment_metrics(df[split], 'Split Decision'))

    # Majority decision (matches "M-DEC" and "MAJORITY")
    majority = method_str.str.contains('M-DEC|MAJORITY', na=False, regex=True)
    results.append(compute_segment_metrics(df[majority], 'Majority Decision'))

    # All decisions
    decision = method_str.str.contains('DEC|DECISION', na=False, regex=True)
    results.append(compute_segment_metrics(df[decision], 'All Decisions'))

    results.append(compute_segment_metrics(df, 'Overall'))
    return results


def analyze_round_of_finish(df: pd.DataFrame) -> List[Dict]:
    """Segment 6: Split by method type (round data not available in this dataset)."""
    results = []

    method_str = df['method'].str.upper()

    # Finishes (KO/TKO or SUB)
    is_finish = method_str.str.contains('KO|TKO|SUB', na=False, regex=True)
    results.append(compute_segment_metrics(df[is_finish], 'Finishes (KO/SUB)'))

    # Decisions
    is_decision = method_str.str.contains('DEC|DECISION', na=False, regex=True)
    results.append(compute_segment_metrics(df[is_decision], 'Decisions'))

    # Early finishes are harder to measure without round data
    # Use method as proxy

    results.append(compute_segment_metrics(df, 'Overall'))
    return results


def analyze_elo_gap(df: pd.DataFrame) -> List[Dict]:
    """Segment 7: Split by Elo differential."""
    results = []

    elo_gap = abs(df['diff_elo'])

    buckets = [
        (0, 25, 'Elo gap <25 (very close)'),
        (25, 75, 'Elo gap 25-75 (slight)'),
        (75, 150, 'Elo gap 75-150 (clear)'),
        (150, 250, 'Elo gap 150-250 (big)'),
        (250, 9999, 'Elo gap 250+ (severe)')
    ]

    for low, high, label in buckets:
        mask = (elo_gap >= low) & (elo_gap < high)
        results.append(compute_segment_metrics(df[mask], label))

    results.append(compute_segment_metrics(df, 'Overall'))
    return results


def analyze_era(df: pd.DataFrame) -> List[Dict]:
    """Segment 8: Split by time period."""
    results = []

    year = df['fight_year']

    eras = [
        (1994, 2005, '1994-2005 (early UFC)'),
        (2006, 2010, '2006-2010 (stats begin)'),
        (2011, 2015, '2011-2015 (modern era)'),
        (2016, 2020, '2016-2020 (competitive)'),
        (2021, 2023, '2021-2023 (train tail)'),
        (2024, 2030, '2024-2026 (test set)')
    ]

    for start, end, label in eras:
        mask = (year >= start) & (year <= end)
        if mask.sum() >= 10:
            results.append(compute_segment_metrics(df[mask], label))

    results.append(compute_segment_metrics(df, 'Overall'))
    return results


def classify_fighter_style(
    strikes: float,
    takedowns: float,
    sub_rate: float,
    strike_median: float = 20,
    td_median: float = 1.0,
    sub_median: float = 0.1
) -> str:
    """Classify fighter as Striker, Grappler, or Balanced."""
    is_striker = strikes > strike_median and takedowns < td_median
    is_grappler = takedowns > td_median or sub_rate > sub_median

    if is_striker and not is_grappler:
        return 'Striker'
    elif is_grappler and not is_striker:
        return 'Grappler'
    else:
        return 'Balanced'


def analyze_style_matchups(df: pd.DataFrame) -> Tuple[List[Dict], Dict]:
    """Segment 10: Accuracy by style matchup."""
    # Compute medians for thresholds
    strike_med = df['f1_avg_sig_strikes_landed'].median()
    td_med = df['f1_avg_takedowns_landed'].median()
    sub_med = df['f1_career_sub_rate'].median()

    # Classify each fighter
    df = df.copy()
    df['f1_style'] = df.apply(lambda r: classify_fighter_style(
        r['f1_avg_sig_strikes_landed'], r['f1_avg_takedowns_landed'],
        r['f1_career_sub_rate'], strike_med, td_med, sub_med
    ), axis=1)
    df['f2_style'] = df.apply(lambda r: classify_fighter_style(
        r['f2_avg_sig_strikes_landed'], r['f2_avg_takedowns_landed'],
        r['f2_career_sub_rate'], strike_med, td_med, sub_med
    ), axis=1)

    # Compute matchup type
    def get_matchup(row):
        styles = sorted([row['f1_style'], row['f2_style']])
        return f"{styles[0]} vs {styles[1]}"

    df['matchup_type'] = df.apply(get_matchup, axis=1)

    # Results by matchup
    results = []
    for matchup in df['matchup_type'].unique():
        mask = df['matchup_type'] == matchup
        if mask.sum() >= 30:  # Need reasonable sample
            results.append(compute_segment_metrics(df[mask], matchup))

    results.append(compute_segment_metrics(df, 'Overall'))

    # Also compute the matrix
    styles = ['Striker', 'Grappler', 'Balanced']
    matrix = {}
    for s1 in styles:
        for s2 in styles:
            if s1 <= s2:  # Only upper triangle
                mask = ((df['f1_style'] == s1) & (df['f2_style'] == s2)) | \
                       ((df['f1_style'] == s2) & (df['f2_style'] == s1))
                if mask.sum() >= 20:
                    acc = accuracy_score(df.loc[mask, 'y_true'], df.loc[mask, 'y_pred'])
                    matrix[f"{s1} vs {s2}"] = round(acc * 100, 1)

    return results, matrix


def run_all_analyses(df: pd.DataFrame, split_name: str = "All") -> Dict:
    """Run all segment analyses on a DataFrame."""
    logger.info(f"Running analyses on {split_name} ({len(df):,} fights)...")

    results = {
        'split': split_name,
        'total_fights': len(df),
        'overall_accuracy': round(accuracy_score(df['y_true'], df['y_pred']) * 100, 1),
        'segments': {}
    }

    # Run each analysis
    results['segments']['confidence_buckets'] = analyze_confidence_buckets(df)
    results['segments']['win_streaks'] = analyze_win_streaks(df)
    results['segments']['experience'] = analyze_experience_asymmetry(df)
    results['segments']['weight_class'] = analyze_weight_class(df)
    results['segments']['fight_method'] = analyze_fight_method(df)
    results['segments']['round_of_finish'] = analyze_round_of_finish(df)
    results['segments']['elo_gap'] = analyze_elo_gap(df)
    results['segments']['era'] = analyze_era(df)

    style_results, style_matrix = analyze_style_matchups(df)
    results['segments']['style_matchups'] = style_results
    results['style_matrix'] = style_matrix

    return results


def identify_strengths_weaknesses(test_results: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Identify model's strongest and weakest segments."""
    overall_acc = test_results['overall_accuracy']

    all_segments = []

    for segment_type, segments in test_results['segments'].items():
        for seg in segments:
            if seg['count'] and seg['count'] >= 30 and seg['accuracy'] is not None:
                if seg['segment'] != 'Overall':
                    all_segments.append({
                        'category': segment_type,
                        'segment': seg['segment'],
                        'accuracy': seg['accuracy'],
                        'count': seg['count'],
                        'diff_from_overall': round(seg['accuracy'] - overall_acc, 1)
                    })

    # Sort by accuracy
    all_segments.sort(key=lambda x: x['accuracy'], reverse=True)

    # Top 8 strengths (accuracy > overall)
    strengths = [s for s in all_segments if s['diff_from_overall'] > 0][:8]

    # Bottom 8 weaknesses (accuracy < overall or near coin flip)
    weaknesses = [s for s in all_segments if s['accuracy'] <= 55][:8]
    if len(weaknesses) < 5:
        # Add more from lowest performing
        weaknesses = all_segments[-8:]

    return strengths, weaknesses


def print_results(results: Dict):
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print(f"SEGMENT ANALYSIS: {results['split']} (n={results['total_fights']:,})")
    print("=" * 70)
    print(f"Overall Accuracy: {results['overall_accuracy']}%")

    for segment_type, segments in results['segments'].items():
        print(f"\n{segment_type.upper().replace('_', ' ')}")
        print("-" * 50)
        print(f"{'Segment':<35} {'Count':>8} {'Accuracy':>10} {'Log Loss':>10}")
        print("-" * 50)
        for seg in segments:
            count_str = f"{seg['count']:,}" if seg['count'] else "N/A"
            acc_str = f"{seg['accuracy']}%" if seg['accuracy'] else "N/A"
            ll_str = f"{seg['log_loss']:.3f}" if seg['log_loss'] else "N/A"
            print(f"{seg['segment']:<35} {count_str:>8} {acc_str:>10} {ll_str:>10}")


def main():
    """Run the full diagnostic analysis."""
    parser = argparse.ArgumentParser(description='Phase 8 Model Diagnostic')
    parser.add_argument('--output', type=str, default='console',
                        choices=['console', 'json'], help='Output format')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PHASE 8 MODEL DIAGNOSTIC")
    print("=" * 70)

    # Load data and model
    df, raw_df = load_data()

    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    logger.info(f"Loading features from {FEATURE_PATH}")
    with open(FEATURE_PATH) as f:
        feature_names = json.load(f)

    # Prepare predictions DataFrame
    analysis_df = prepare_predictions(df, raw_df, model, feature_names)

    # Split into train and test
    train_df = analysis_df[analysis_df['is_test'] == 0]
    test_df = analysis_df[analysis_df['is_test'] == 1]

    print(f"\nTrain set: {len(train_df):,} fights (1994-2023)")
    print(f"Test set:  {len(test_df):,} fights (2024-2026)")

    # Run analyses
    train_results = run_all_analyses(train_df, "Train (1994-2023)")
    test_results = run_all_analyses(test_df, "Test (2024-2026)")

    # Identify strengths and weaknesses
    strengths, weaknesses = identify_strengths_weaknesses(test_results)

    # Compile full results
    full_results = {
        'analysis_date': datetime.now().isoformat(),
        'model': 'Phase 8 Stacking Ensemble',
        'train': train_results,
        'test': test_results,
        'strengths': strengths,
        'weaknesses': weaknesses
    }

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_PATH}")

    if args.output == 'console':
        # Print test results (primary focus)
        print_results(test_results)

        # Print strengths and weaknesses
        print("\n" + "=" * 70)
        print("MODEL STRENGTHS (accuracy above overall)")
        print("=" * 70)
        for i, s in enumerate(strengths, 1):
            print(f"{i}. {s['segment']}: {s['accuracy']}% ({s['count']} fights, +{s['diff_from_overall']}pp)")

        print("\n" + "=" * 70)
        print("MODEL WEAKNESSES (accuracy at or below coin flip)")
        print("=" * 70)
        for i, w in enumerate(weaknesses, 1):
            diff = w['diff_from_overall']
            sign = '+' if diff >= 0 else ''
            print(f"{i}. {w['segment']}: {w['accuracy']}% ({w['count']} fights, {sign}{diff}pp)")

        # Style matrix
        if 'style_matrix' in test_results:
            print("\n" + "=" * 70)
            print("STYLE MATCHUP MATRIX (Test Set Accuracy)")
            print("=" * 70)
            for matchup, acc in sorted(test_results['style_matrix'].items()):
                print(f"  {matchup:<25} {acc}%")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_PATH}")

    return full_results


if __name__ == '__main__':
    main()
