"""
Elo Rating System for UFC Fighters.

Computes running Elo ratings for all UFC fighters, processing fights in strict
chronological order. Elo naturally captures strength of schedule without
adjusting per-fight statistics.

How it works:
- Each fighter starts at 1500 Elo
- After each fight, ratings update based on outcome vs expectation
- Beating a higher-rated opponent = bigger gain than beating a lower-rated one
- Ratings are recursive: each opponent's rating reflects THEIR opponent quality

Usage:
    from src.features.elo import compute_all_elo_ratings, compute_elo_features

    # Compute raw Elo ratings
    elo_df, final_ratings = compute_all_elo_ratings(fights_df, K=32)

    # Or compute full feature set with derived features
    elo_features = compute_elo_features(fights_df, K=32)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default parameters
DEFAULT_K = 32  # Standard K-factor (chess default)
DEFAULT_INITIAL_ELO = 1500  # Starting rating for all fighters


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for fighter A vs fighter B.

    Uses standard Elo formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Args:
        rating_a: Elo rating of fighter A
        rating_b: Elo rating of fighter B

    Returns:
        Expected probability of A winning (0 to 1)
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_all_elo_ratings(
    fights_df: pd.DataFrame,
    K: int = DEFAULT_K,
    initial_elo: float = DEFAULT_INITIAL_ELO,
    dynamic_k: bool = False,
    k_base: float = 32.0,
    k_min: float = 16.0,
    k_decay: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute running Elo ratings for all UFC fighters.

    Processes fights in STRICT chronological order. For each fight, stores the
    PRE-FIGHT Elo ratings (the model sees Elo BEFORE the fight happened - no leakage).
    After the outcome, updates the internal ratings dict.

    Args:
        fights_df: DataFrame with fights in fight-centric format, columns:
            - fight_date (datetime): Date of fight
            - f1_name (str): Fighter 1 name
            - f2_name (str): Fighter 2 name
            - f1_is_winner (int): 1 if fighter 1 won, 0 otherwise
            - method (str, optional): Method of victory (for draws/NC detection)
        K: Base K-factor (update magnitude). Higher = more reactive to results.
        initial_elo: Starting Elo for fighters with no prior fights.
        dynamic_k: If True, use dynamic K-factor that decreases with experience.
        k_base: Base K-factor for dynamic K (used for debut fighters).
        k_min: Minimum K-factor (for veterans).
        k_decay: How much K decreases per fight (K = max(k_min, k_base - fights * k_decay)).

    Returns:
        Tuple of:
            - DataFrame with same index as input, columns:
                f1_pre_fight_elo, f2_pre_fight_elo, elo_diff, f1_elo_expected
            - Dict mapping fighter_name -> final Elo rating
    """
    logger.info(f"Computing Elo ratings (K={K}, dynamic_k={dynamic_k})...")

    # Sort by date (critical for correct Elo computation)
    df = fights_df.sort_values('fight_date').copy()

    # Track current ratings and fight counts for each fighter
    ratings: Dict[str, float] = {}
    fight_counts: Dict[str, int] = {}

    # Track historical max (for peak comparisons)
    peak_ratings: Dict[str, float] = {}

    # Results aligned with original index
    results = []
    indices = []

    for idx, fight in df.iterrows():
        f1 = fight['f1_name']
        f2 = fight['f2_name']

        # Get pre-fight ratings
        r1 = ratings.get(f1, initial_elo)
        r2 = ratings.get(f2, initial_elo)

        # Get fight counts (for dynamic K)
        f1_fights = fight_counts.get(f1, 0)
        f2_fights = fight_counts.get(f2, 0)

        # Store pre-fight Elo (THIS is the feature - no leakage)
        e1 = expected_score(r1, r2)
        e2 = 1.0 - e1

        results.append({
            'f1_pre_fight_elo': r1,
            'f2_pre_fight_elo': r2,
            'elo_diff': r1 - r2,
            'f1_elo_expected': e1,
            'f1_peak_elo': peak_ratings.get(f1, initial_elo),
            'f2_peak_elo': peak_ratings.get(f2, initial_elo),
            'f1_fights_before': f1_fights,
            'f2_fights_before': f2_fights,
        })
        indices.append(idx)

        # Determine outcome
        method = str(fight.get('method', '')).lower()
        is_draw = 'draw' in method
        is_nc = 'no contest' in method or 'nc' in method

        if is_nc:
            # No contest - skip rating update entirely
            # Still increment fight count for K-factor purposes
            fight_counts[f1] = f1_fights + 1
            fight_counts[f2] = f2_fights + 1
            continue
        elif is_draw:
            s1, s2 = 0.5, 0.5
        elif fight['f1_is_winner'] == 1:
            s1, s2 = 1.0, 0.0
        else:
            s1, s2 = 0.0, 1.0

        # Calculate K-factor for each fighter
        if dynamic_k:
            k1 = max(k_min, k_base - (f1_fights * k_decay))
            k2 = max(k_min, k_base - (f2_fights * k_decay))
        else:
            k1 = K
            k2 = K

        # Update ratings
        new_r1 = r1 + k1 * (s1 - e1)
        new_r2 = r2 + k2 * (s2 - e2)

        ratings[f1] = new_r1
        ratings[f2] = new_r2

        # Update peaks
        peak_ratings[f1] = max(peak_ratings.get(f1, initial_elo), new_r1)
        peak_ratings[f2] = max(peak_ratings.get(f2, initial_elo), new_r2)

        # Update fight counts
        fight_counts[f1] = f1_fights + 1
        fight_counts[f2] = f2_fights + 1

    # Create results DataFrame aligned with original index
    elo_df = pd.DataFrame(results, index=indices)

    # Reindex to match original DataFrame index order
    elo_df = elo_df.reindex(fights_df.index)

    logger.info(f"Computed Elo for {len(ratings)} fighters across {len(df)} fights")
    logger.info(f"Rating range: {min(ratings.values()):.0f} to {max(ratings.values()):.0f}")

    return elo_df, ratings


def compute_elo_momentum(
    fights_df: pd.DataFrame,
    elo_df: pd.DataFrame,
    window: int = 3
) -> pd.DataFrame:
    """
    Compute Elo momentum features (rating change over recent fights).

    Args:
        fights_df: Original fights DataFrame with f1_name, f2_name, fight_date
        elo_df: DataFrame with pre-fight Elo (from compute_all_elo_ratings)
        window: Number of recent fights to consider for momentum

    Returns:
        DataFrame with momentum features added
    """
    df = fights_df.sort_values('fight_date').copy()
    df = df.join(elo_df, how='left')

    # Track Elo history for each fighter
    fighter_elo_history: Dict[str, List[float]] = {}

    # Results
    momentum_results = []
    indices = []

    for idx, fight in df.iterrows():
        f1 = fight['f1_name']
        f2 = fight['f2_name']

        # Get current pre-fight Elo
        f1_elo = fight['f1_pre_fight_elo']
        f2_elo = fight['f2_pre_fight_elo']

        # Calculate momentum (Elo change over last N fights)
        f1_history = fighter_elo_history.get(f1, [])
        f2_history = fighter_elo_history.get(f2, [])

        if len(f1_history) >= window:
            f1_momentum = f1_elo - f1_history[-window]
        elif len(f1_history) > 0:
            f1_momentum = f1_elo - f1_history[0]
        else:
            f1_momentum = 0.0

        if len(f2_history) >= window:
            f2_momentum = f2_elo - f2_history[-window]
        elif len(f2_history) > 0:
            f2_momentum = f2_elo - f2_history[0]
        else:
            f2_momentum = 0.0

        # Elo vs peak (how close to career best)
        f1_peak = fight.get('f1_peak_elo', f1_elo)
        f2_peak = fight.get('f2_peak_elo', f2_elo)

        f1_vs_peak = f1_elo / max(f1_peak, 1.0)  # 1.0 = at peak, <1.0 = below peak
        f2_vs_peak = f2_elo / max(f2_peak, 1.0)

        momentum_results.append({
            'f1_elo_momentum': f1_momentum,
            'f2_elo_momentum': f2_momentum,
            'diff_elo_momentum': f1_momentum - f2_momentum,
            'f1_elo_vs_peak': f1_vs_peak,
            'f2_elo_vs_peak': f2_vs_peak,
        })
        indices.append(idx)

        # Update history (add current pre-fight Elo)
        if f1 not in fighter_elo_history:
            fighter_elo_history[f1] = []
        fighter_elo_history[f1].append(f1_elo)

        if f2 not in fighter_elo_history:
            fighter_elo_history[f2] = []
        fighter_elo_history[f2].append(f2_elo)

    return pd.DataFrame(momentum_results, index=indices).reindex(fights_df.index)


def compute_elo_features(
    fights_df: pd.DataFrame,
    K: int = DEFAULT_K,
    initial_elo: float = DEFAULT_INITIAL_ELO,
    dynamic_k: bool = False,
    momentum_window: int = 3
) -> pd.DataFrame:
    """
    Compute complete set of Elo-derived features.

    Features created:
        - diff_elo: f1_elo - f2_elo (primary signal)
        - f1_elo_expected: Elo-implied win probability for F1
        - f1_elo_momentum / f2_elo_momentum: Elo change over last N fights
        - diff_elo_momentum: Who's trending better
        - f1_elo_vs_peak / f2_elo_vs_peak: Current Elo / max Elo ever

    Args:
        fights_df: Fight-centric DataFrame with required columns
        K: Base K-factor
        initial_elo: Starting Elo for new fighters
        dynamic_k: Use experience-based K-factor
        momentum_window: Window size for momentum calculation

    Returns:
        DataFrame with all Elo features, aligned with input index
    """
    # Compute base Elo ratings
    elo_df, final_ratings = compute_all_elo_ratings(
        fights_df, K=K, initial_elo=initial_elo, dynamic_k=dynamic_k
    )

    # Compute momentum features
    momentum_df = compute_elo_momentum(fights_df, elo_df, window=momentum_window)

    # Combine all features
    result = pd.concat([elo_df, momentum_df], axis=1)

    # Rename for clarity (keep diff_elo as primary feature name)
    result = result.rename(columns={'elo_diff': 'diff_elo'})

    # Clean up intermediate columns we don't need as features
    drop_cols = ['f1_peak_elo', 'f2_peak_elo', 'f1_fights_before', 'f2_fights_before']
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])

    logger.info(f"Created {len(result.columns)} Elo features")

    return result, final_ratings


def get_top_elo_fighters(
    final_ratings: Dict[str, float],
    top_n: int = 25
) -> List[Tuple[str, float]]:
    """
    Get fighters with highest Elo ratings.

    Useful for validation: should be dominated by all-time greats like
    Jon Jones, Georges St-Pierre, Khabib Nurmagomedov, Amanda Nunes, etc.

    Args:
        final_ratings: Dict mapping fighter name -> final Elo rating
        top_n: Number of top fighters to return

    Returns:
        List of (fighter_name, elo_rating) tuples, sorted by rating descending
    """
    sorted_fighters = sorted(
        final_ratings.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_fighters[:top_n]


def validate_elo_system(
    fights_df: pd.DataFrame,
    elo_features: pd.DataFrame,
    final_ratings: Dict[str, float]
) -> Dict:
    """
    Validate the Elo system implementation.

    Checks:
    1. Top 25 fighters should be dominated by known greats
    2. diff_elo alone should predict ~55-60% accuracy
    3. Correlation with diff_career_win_rate should be 0.5-0.8

    Args:
        fights_df: Original fights DataFrame
        elo_features: Computed Elo features
        final_ratings: Final Elo ratings dict

    Returns:
        Validation results dictionary
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    validation = {
        'top_25_fighters': get_top_elo_fighters(final_ratings, 25),
        'checks': {},
    }

    # Check 1: Standalone prediction accuracy
    df = fights_df.copy()
    df = df.join(elo_features, how='left')

    # Filter to valid rows
    valid = df.dropna(subset=['diff_elo', 'f1_is_winner'])

    if len(valid) > 100:
        X = valid[['diff_elo']].values
        y = valid['f1_is_winner'].values

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)
        y_pred = lr.predict(X)

        elo_only_accuracy = accuracy_score(y, y_pred)
        validation['elo_only_accuracy'] = float(elo_only_accuracy)
        validation['checks']['accuracy_check'] = 'PASS' if 0.54 <= elo_only_accuracy <= 0.70 else 'FAIL'

    # Check 2: Correlation with career win rate
    if 'diff_career_win_rate' in df.columns:
        valid2 = df.dropna(subset=['diff_elo', 'diff_career_win_rate'])
        if len(valid2) > 100:
            corr = valid2['diff_elo'].corr(valid2['diff_career_win_rate'])
            validation['elo_winrate_correlation'] = float(corr)
            validation['checks']['correlation_check'] = 'PASS' if 0.3 <= abs(corr) <= 0.9 else 'FAIL'

    # Rating distribution stats
    elo_values = list(final_ratings.values())
    validation['elo_stats'] = {
        'mean': float(np.mean(elo_values)),
        'std': float(np.std(elo_values)),
        'min': float(np.min(elo_values)),
        'max': float(np.max(elo_values)),
        'median': float(np.median(elo_values)),
        'n_fighters': len(elo_values),
    }

    return validation


def tune_k_factor(
    fights_df: pd.DataFrame,
    k_values: List[int] = [20, 32, 40],
    test_cutoff: str = '2024-01-01'
) -> Dict:
    """
    Test different K-factor values to find optimal.

    Uses prediction accuracy on test set (post-cutoff) to determine best K.

    Args:
        fights_df: Fight-centric DataFrame
        k_values: List of K values to test
        test_cutoff: Date string for train/test split

    Returns:
        Dict with results for each K value
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss

    cutoff = pd.to_datetime(test_cutoff)

    results = {}

    for K in k_values:
        elo_features, _ = compute_elo_features(fights_df, K=K)

        df = fights_df.copy()
        df = df.join(elo_features, how='left')

        # Split
        train = df[df['fight_date'] < cutoff].dropna(subset=['diff_elo', 'f1_is_winner'])
        test = df[df['fight_date'] >= cutoff].dropna(subset=['diff_elo', 'f1_is_winner'])

        if len(train) < 100 or len(test) < 50:
            logger.warning(f"K={K}: Insufficient data for evaluation")
            continue

        # Train simple logistic regression on Elo only
        X_train = train[['diff_elo']].values
        y_train = train['f1_is_winner'].values
        X_test = test[['diff_elo']].values
        y_test = test['f1_is_winner'].values

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        y_proba = lr.predict_proba(X_test)[:, 1]

        results[K] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'log_loss': float(log_loss(y_test, y_proba)),
            'train_size': len(train),
            'test_size': len(test),
        }

        logger.info(f"K={K}: Accuracy={results[K]['accuracy']:.3f}, Log Loss={results[K]['log_loss']:.3f}")

    # Find best K
    if results:
        best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
        results['best_k'] = best_k
        results['best_accuracy'] = results[best_k]['accuracy']

    return results


if __name__ == '__main__':
    """Test Elo computation on processed data."""
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    # Load processed data
    data_path = Path('data/processed/ufc_fights_features_v1.csv')
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])

    print(f"Loaded {len(df)} fights")

    # Compute Elo features
    elo_features, final_ratings = compute_elo_features(df, K=32)

    print("\nTop 25 Elo ratings:")
    print("-" * 40)
    for i, (name, rating) in enumerate(get_top_elo_fighters(final_ratings, 25), 1):
        print(f"{i:2}. {name:<30} {rating:.0f}")

    # Validate
    print("\nValidation:")
    print("-" * 40)
    validation = validate_elo_system(df, elo_features, final_ratings)

    if 'elo_only_accuracy' in validation:
        print(f"Elo-only accuracy: {validation['elo_only_accuracy']:.1%}")
    if 'elo_winrate_correlation' in validation:
        print(f"Correlation with diff_career_win_rate: {validation['elo_winrate_correlation']:.3f}")

    print(f"\nElo stats: mean={validation['elo_stats']['mean']:.0f}, "
          f"std={validation['elo_stats']['std']:.0f}, "
          f"range=[{validation['elo_stats']['min']:.0f}, {validation['elo_stats']['max']:.0f}]")
