"""
Run feature engineering pipeline on UFC fight data v1.

This script processes the raw fight data and creates engineered features
for machine learning model training.
"""

import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from src.features.engineer import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("UFC FIGHT PREDICTOR - FEATURE ENGINEERING v1")
    logger.info("=" * 80)

    # Load raw data
    logger.info("Loading raw data...")
    raw_data_path = 'data/raw/ufc_fights_v1.csv'
    df_raw = pd.read_csv(raw_data_path)
    logger.info(f"Loaded {len(df_raw):,} fights from {raw_data_path}")

    # Initialize feature engineer
    logger.info("Initializing feature engineer...")
    engineer = FeatureEngineer()

    # Run feature engineering pipeline
    logger.info("Running feature engineering pipeline...")
    try:
        df_features = engineer.create_all_features(df_raw)

        logger.info("=" * 80)
        logger.info("Feature engineering completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Final dataset shape: {df_features.shape}")
        logger.info(f"  Fights: {len(df_features):,}")
        logger.info(f"  Features: {len(df_features.columns)}")

        # Validate features
        logger.info("\nRunning validation checks...")
        validation_report = engineer.validate_features(df_features)

        logger.info("\nValidation Summary:")
        logger.info(f"  Total rows: {validation_report['total_rows']:,}")
        logger.info(f"  Total columns: {validation_report['total_columns']}")
        logger.info(f"  Checks passed: {validation_report['checks_passed']}")

        if not validation_report['checks_passed']:
            logger.warning(f"  Issues found: {len(validation_report['issues'])}")
            for issue in validation_report['issues']:
                logger.warning(f"    - {issue}")

        # Save processed data
        logger.info("\nSaving processed data...")
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / 'ufc_fights_features_v1.csv'
        df_features.to_csv(output_file, index=False)
        logger.info(f"✓ Saved features to {output_file}")

        file_size = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"  File size: {file_size:.2f} MB")

        # Save feature engineering artifacts
        logger.info("\nSaving feature engineering artifacts...")
        engineer.save_artifacts('data/processed')

        # Generate feature report
        logger.info("\nGenerating feature report...")
        report_file = output_dir / 'feature_engineering_report_v1.md'

        with open(report_file, 'w') as f:
            f.write("# Feature Engineering Report v1\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Input data:** {len(df_raw):,} fights\n")
            f.write(f"- **Output data:** {len(df_features):,} fights\n")
            f.write(f"- **Total features:** {len(df_features.columns)}\n")
            f.write(f"- **Processing time:** {datetime.now() - start_time}\n\n")

            f.write("## Feature Categories\n\n")

            # Count features by category
            feature_categories = {
                'Fighter 1 Career': len([c for c in df_features.columns if c.startswith('f1_career')]),
                'Fighter 2 Career': len([c for c in df_features.columns if c.startswith('f2_career')]),
                'Fighter 1 Rolling': len([c for c in df_features.columns if c.startswith('f1_') and 'last_' in c]),
                'Fighter 2 Rolling': len([c for c in df_features.columns if c.startswith('f2_') and 'last_' in c]),
                'Differential': len([c for c in df_features.columns if c.startswith('diff_')]),
                'Style Indicators': len([c for c in df_features.columns if 'striking_volume' in c or 'grappling_tendency' in c]),
                'Temporal': len([c for c in df_features.columns if 'layoff' in c or 'fight_year' in c or 'fight_month' in c]),
                'Weight Class (one-hot)': len([c for c in df_features.columns if c.startswith('wc_')]),
                'Metadata': len([c for c in df_features.columns if c in ['fight_id', 'fight_date', 'event_name', 'f1_name', 'f2_name']]),
                'Target': 1,  # f1_is_winner
            }

            for category, count in feature_categories.items():
                f.write(f"- **{category}:** {count} features\n")

            f.write(f"\n## Data Quality\n\n")
            f.write(f"- **Missing values:** {df_features.isnull().sum().sum():,} total\n")
            f.write(f"- **Validation checks passed:** {'✓ Yes' if validation_report['checks_passed'] else '✗ No'}\n")

            if not validation_report['checks_passed']:
                f.write(f"\n### Validation Issues\n\n")
                for issue in validation_report['issues']:
                    f.write(f"- {issue}\n")

            f.write(f"\n## Sample Statistics\n\n")

            # Key feature distributions
            key_features = ['f1_career_win_rate', 'f2_career_win_rate', 'diff_career_win_rate',
                           'f1_career_fights', 'f2_career_fights', 'f1_is_winner']

            f.write("| Feature | Mean | Median | Std | Min | Max |\n")
            f.write("|---------|------|--------|-----|-----|-----|\n")

            for feat in key_features:
                if feat in df_features.columns:
                    stats = df_features[feat].describe()
                    f.write(f"| {feat} | {stats['mean']:.3f} | {stats['50%']:.3f} | ")
                    f.write(f"{stats['std']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |\n")

            f.write(f"\n## All Features ({len(df_features.columns)})\n\n")
            for i, col in enumerate(df_features.columns, 1):
                f.write(f"{i}. `{col}`\n")

        logger.info(f"✓ Saved report to {report_file}")

        # Print summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("KEY STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Target variable (f1_is_winner) distribution:")
        logger.info(f"  Fighter 1 wins: {df_features['f1_is_winner'].sum():,} ({df_features['f1_is_winner'].mean()*100:.1f}%)")
        logger.info(f"  Fighter 2 wins: {(1-df_features['f1_is_winner']).sum():,} ({(1-df_features['f1_is_winner']).mean()*100:.1f}%)")

        logger.info(f"\nDebut fighters:")
        logger.info(f"  Fighter 1 debuts: {df_features['f1_is_ufc_debut'].sum():,}")
        logger.info(f"  Fighter 2 debuts: {df_features['f2_is_ufc_debut'].sum():,}")

        logger.info(f"\nFights with detailed stats:")
        logger.info(f"  Fighter 1: {df_features['f1_has_detailed_stats'].sum():,}/{len(df_features)}")
        logger.info(f"  Fighter 2: {df_features['f2_has_detailed_stats'].sum():,}/{len(df_features)}")

        logger.info("\n" + "=" * 80)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 80)

        elapsed = datetime.now() - start_time
        logger.info(f"Total time: {elapsed}")

        logger.info("\nOutput files:")
        logger.info(f"  1. {output_file}")
        logger.info(f"  2. {output_dir / 'weight_class_medians_v1.pkl'}")
        logger.info(f"  3. {output_dir / 'feature_config_v1.json'}")
        logger.info(f"  4. {report_file}")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
