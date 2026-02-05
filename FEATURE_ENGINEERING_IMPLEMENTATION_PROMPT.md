# Feature Engineering Implementation Prompt
## UFC Fight Predictor - Phase 2

**Status:** Awaiting Leadership Approval
**Created:** February 4, 2026
**Data Source:** `data/raw/ufc_fights_v1.csv` (8,520 fights, 760 events, 1994-2026)

---

## Objective

Implement a complete feature engineering pipeline in `src/features/engineer.py` that transforms raw UFC fight statistics into predictive features suitable for machine learning models. The pipeline must handle:
- Historical fighter performance aggregation
- Early-career fighter imputation strategies
- Matchup-based comparative features
- Temporal/momentum indicators
- Missing data from older fights (pre-2009)

---

## Data Landscape

### Current Dataset
- **Total Fights:** 8,520 across 760 events
- **Date Range:** March 11, 1994 to January 31, 2026 (31.9 years)
- **Columns:** 58 (event info, fight outcomes, strike stats, grappling stats)
- **Data Quality:** 79.6% complete
  - 6,114 fights (2009-2026) have full detailed statistics
  - 2,406 fights (1994-2008) missing detailed strike/grappling breakdowns
  - All fights have basic info: fighters, winner, method, round, time

### Key Data Characteristics
1. **String Formats Requiring Parsing:**
   - Strike data: "25 of 50" → (landed: 25, attempted: 50)
   - Control time: "5:32" → 332 seconds
   - Fight time: "2:15" → 135 seconds in round

2. **Missing Data Patterns:**
   - Pre-2009 fights: Missing all `*_sig_str_*`, `*_TD`, `*_sub_att`, `*_rev`, `*_ctrl` columns
   - Some nicknames are missing (not critical for prediction)

3. **Data Asymmetry:**
   - Red vs Blue corner assignment is arbitrary (not predictive)
   - Need to normalize features to be fighter-agnostic

---

## Implementation Plan

### Phase 1: Data Preprocessing & Cleaning

**Task 1.1: Parse String-Based Statistics**
- Create utility functions to parse "X of Y" format into (landed, attempted) tuples
- Parse time strings (MM:SS) into total seconds
- Handle edge cases: empty strings, "N/A", malformed data

**Task 1.2: Create Base Fighter-Fight Records**
- Restructure data from fight-centric (1 row = 1 fight) to fighter-centric (1 row = 1 fighter's performance in 1 fight)
- This creates 17,040 records (8,520 fights × 2 fighters)
- Include columns:
  - `fighter_name`, `opponent_name`, `fight_date`, `is_winner` (target)
  - `strikes_landed`, `strikes_attempted`, `takedowns_landed`, etc.
  - All parsed statistics from raw data

**Task 1.3: Sort Data Chronologically**
- Sort all records by `fight_date` ascending
- This enables calculation of "career statistics at time of fight"
- Critical for avoiding data leakage (can't use future fights to predict past fights)

**Task 1.4: Handle Missing Data**
- For pre-2009 fights missing detailed stats:
  - Flag with `has_detailed_stats = False`
  - Use only available features (outcome, method, round, time)
  - Consider creating separate feature set for "basic stats only" fights
- For missing nicknames/locations: fill with "Unknown"

---

### Phase 2: Fighter Career Features (Rolling Statistics)

**Objective:** For each fighter in each fight, calculate their career statistics *up to but not including* that fight.

**Task 2.1: Basic Career Metrics**
For each fighter at time of fight, calculate:
- `career_fights` - Total fights prior to this one
- `career_wins` - Total wins prior to this one
- `career_losses` - Total losses prior to this one
- `career_win_rate` - Wins / Total Fights
- `career_ko_rate` - (KO/TKO wins) / Total Wins
- `career_sub_rate` - (Submission wins) / Total Wins
- `career_dec_rate` - (Decision wins) / Total Wins

**Task 2.2: Recent Form Metrics (Rolling Windows)**
Calculate performance over last N fights (N = 3, 5, 10):
- `win_rate_last_N` - Win rate in last N fights
- `finish_rate_last_N` - % of wins by finish (not decision)
- `avg_strikes_landed_last_N` - Average significant strikes landed
- `avg_strikes_absorbed_last_N` - Average significant strikes absorbed
- `avg_takedowns_last_N` - Average takedowns per fight
- `avg_fight_time_last_N` - Average time to finish fights

**Task 2.3: Momentum Indicators**
- `win_streak` - Current consecutive wins (0 if coming off loss)
- `loss_streak` - Current consecutive losses (0 if coming off win)
- `days_since_last_fight` - Days between this fight and previous fight
- `fights_per_year` - Fight frequency over career

**Task 2.4: Opponent Strength Adjustments**
- `avg_opponent_win_rate` - Average win rate of past opponents (at time they fought)
- `strength_of_schedule` - Weighted average of opponent quality

---

### Phase 3: Early-Career Fighter Imputation Strategy

**Problem:** Fighters with 0-3 prior fights have insufficient data for rolling statistics.

**Solution: Hybrid Imputation Approach**

**Task 3.1: Define Experience Tiers**
```
Tier 0 (Debut):        0 prior fights
Tier 1 (Novice):       1-2 prior fights
Tier 2 (Developing):   3-5 prior fights
Tier 3 (Established):  6+ prior fights
```

**Task 3.2: Imputation Rules by Tier**

**For Tier 0 (Debut fighters):**
- Career metrics: Set to population median of all UFC debuting fighters
  - Historical data shows UFC debut win rate ≈ 52% (favors established fighters brought in)
  - `career_win_rate = 0.52`
  - `career_ko_rate = 0.25` (median for debuting fighters)
  - `career_sub_rate = 0.15`
  - `career_dec_rate = 0.60`
- Rolling metrics: Set to weight-class-specific medians
  - Example: `avg_strikes_landed_last_3` = median for weight class
- Add flag: `is_ufc_debut = True`

**For Tier 1 (1-2 fights):**
- Use actual career stats for available metrics
- For rolling windows requiring 3+ fights:
  - Blend actual performance with weight-class median
  - Weight: (actual_fights / required_fights) × actual + (1 - weight) × median
  - Example: Fighter with 2 fights calculating last_3 average:
    - `avg_strikes_last_3 = (2/3) × actual_avg + (1/3) × weight_class_median`

**For Tier 2 (3-5 fights):**
- Use actual career stats for all metrics
- For rolling windows > available fights (e.g., last_10 with only 4 fights):
  - Use all available fights (no imputation)
  - Add feature: `fights_available_for_last_N` (helps model understand uncertainty)

**For Tier 3 (6+ fights):**
- Use actual career stats for all metrics (no imputation needed)

**Task 3.3: Add Confidence Features**
For all fighters, add:
- `career_fight_count` - Number of prior fights (helps model weight reliability)
- `data_completeness_score` - % of features that are real vs imputed
- `has_detailed_stats` - Boolean for pre/post 2009

---

### Phase 4: Matchup Comparison Features

**Objective:** Create features comparing Fighter A vs Fighter B in a matchup.

**Task 4.1: Direct Differential Features**
For each fighter metric, create differential:
- `win_rate_diff` = Fighter_A_win_rate - Fighter_B_win_rate
- `strikes_landed_diff` = avg_strikes_A - avg_strikes_B
- `takedown_diff` = avg_TD_A - avg_TD_B
- Apply to all career and rolling metrics

**Task 4.2: Experience Advantage**
- `experience_diff` = career_fights_A - career_fights_B
- `experience_ratio` = career_fights_A / (career_fights_A + career_fights_B)

**Task 4.3: Style Matchup Indicators**
Create fighter style profiles:
- `striking_volume` = avg_strikes_landed per minute
- `grappling_tendency` = takedown_attempts / total_strikes
- `finish_danger` = (ko_rate + sub_rate) / 2
- `defensive_rating` = 1 - (strikes_absorbed / opponent_strikes_attempted)

Then create matchup features:
- `striker_vs_grappler` = sign(striking_volume_diff) × abs(grappling_tendency_diff)
- `aggression_matchup` = both fighters' striking_volume (high-high = war, low-low = tactical)

**Task 4.4: Momentum Differential**
- `momentum_diff` = (win_streak_A - loss_streak_A) - (win_streak_B - loss_streak_B)
- `recent_form_diff` = win_rate_last_3_A - win_rate_last_3_B

**Task 4.5: Head-to-Head History** (if exists)
- `previous_fights` - Number of times they've fought
- `head_to_head_record` - Fighter A's record vs Fighter B
- If no history: set to 0

---

### Phase 5: Temporal & Contextual Features

**Task 5.1: Career Trajectory Features**
- `career_trajectory` - Linear regression slope of win rate over last 10 fights
  - Positive = improving, Negative = declining
- `peak_performance_gap` - Fights since best rolling_3 win rate
  - Example: If best 3-fight stretch was 5 fights ago, value = 5

**Task 5.2: Activity Level**
- `layoff_length` = Days since last fight (already in Phase 2)
- `is_coming_off_layoff` = Boolean (>180 days since last fight)
- `fights_in_last_12_months` - Fight frequency indicator

**Task 5.3: Weight Class Context**
- `weight_class_experience` - Number of fights in this weight class
- `is_new_to_weight_class` - Boolean (first 3 fights in division)
- One-hot encode weight class for model

**Task 5.4: Event Context** (optional, lower priority)
- `is_title_fight` - Boolean (extract from event name if "Title" or champion names)
- `is_main_event` - Boolean (could use fight order if available, otherwise skip)

---

### Phase 6: Feature Engineering Pipeline Implementation

**Task 6.1: Implement FeatureEngineer Class Methods**

Update `src/features/engineer.py` with:

```python
class FeatureEngineer:
    def __init__(self, config: Dict):
        # Set imputation parameters
        self.debut_win_rate = 0.52
        self.weight_class_medians = {}  # Calculate from data
        self.experience_tiers = {
            'debut': 0, 'novice': 2, 'developing': 5, 'established': 6
        }

    def preprocess_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse strings, restructure to fighter-centric, sort chronologically"""
        pass

    def calculate_career_features(self, fighter_history: pd.DataFrame) -> pd.DataFrame:
        """Calculate career stats up to each fight (avoid data leakage)"""
        pass

    def impute_early_career(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply hybrid imputation for fighters with <6 fights"""
        pass

    def create_matchup_features(self, fight_df: pd.DataFrame) -> pd.DataFrame:
        """Create differential and comparative features for matchups"""
        pass

    def create_all_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline: run all steps in sequence"""
        pass
```

**Task 6.2: Feature Naming Convention**
- Career features: `f1_career_*`, `f2_career_*` (fighter 1, fighter 2)
- Rolling features: `f1_last3_*`, `f1_last5_*`, etc.
- Matchup features: `matchup_*` or `diff_*`
- Flags: `is_*`, `has_*`

**Task 6.3: Handle Data Leakage Prevention**
Critical: When calculating fighter features, ONLY use data from fights before the current fight.
- Implementation: Group by fighter, sort by date, use `.expanding()` or custom cumulative functions
- Validation: Assert that feature values don't change retroactively when adding new data

---

### Phase 7: Validation & Quality Assurance

**Task 7.1: Feature Validation Checks**
Create validation tests:
- No null values in critical features (win_rate, career_fights)
- All percentage features in [0, 1] range
- No negative values for counts (career_wins, strikes_landed, etc.)
- Differential features properly symmetric (diff_AB = -diff_BA)
- Early career imputation applied correctly (verify with sample fighters)

**Task 7.2: Data Leakage Testing**
- Take a known fight, verify features only use prior fights
- Check that adding future fights doesn't change past fight features
- Temporal split validation: Train on 1994-2020, test on 2021-2026
  - Features calculated identically regardless of future data

**Task 7.3: Create Feature Documentation**
Generate automated report:
- List of all features (expected: 100-150 features)
- Feature type (numeric, boolean, categorical)
- Missing value percentage per feature
- Distribution statistics (mean, median, std, min, max)
- Correlation matrix for top features

**Task 7.4: Sample Output Inspection**
Manually review 10 sample fights:
- Verify feature values make intuitive sense
- Check imputation for debut fighters
- Confirm matchup differentials calculated correctly

---

### Phase 8: Output & Storage

**Task 8.1: Save Processed Features**
Save engineered features to:
- `data/processed/ufc_fights_features_v1.csv`
- Include all features + target variable (`is_winner` or `winner_encoded`)
- Include metadata columns: `fight_id`, `fighter_name`, `fight_date`

**Task 8.2: Save Feature Engineering Artifacts**
- `data/processed/weight_class_medians_v1.pkl` - Imputation reference values
- `data/processed/feature_names_v1.json` - List of all feature columns
- `data/processed/feature_engineering_report_v1.md` - Validation report

**Task 8.3: Update Config**
Add to `config.yaml`:
```yaml
feature_engineering:
  version: "v1"
  input_file: "data/raw/ufc_fights_v1.csv"
  output_file: "data/processed/ufc_fights_features_v1.csv"

  imputation:
    debut_win_rate: 0.52
    debut_ko_rate: 0.25
    debut_sub_rate: 0.15
    experience_tiers: [0, 2, 5, 6]

  rolling_windows: [3, 5, 10]

  features:
    career_metrics: true
    rolling_metrics: true
    matchup_differentials: true
    style_indicators: true
    temporal_features: true
```

---

## Expected Outcomes

### Deliverables
1. **Fully Implemented `src/features/engineer.py`**
   - All methods functional (no NotImplementedError)
   - Comprehensive docstrings
   - Type hints for all functions

2. **Processed Feature Dataset**
   - `data/processed/ufc_fights_features_v1.csv`
   - 8,520 fights → 17,040 fighter-fight records (or 8,520 matchup records if kept fight-centric)
   - 100-150 engineered features
   - No data leakage

3. **Validation Report**
   - Feature distributions
   - Imputation statistics
   - Data quality metrics
   - Sample predictions (sanity check)

4. **Documentation**
   - Feature dictionary (all 100-150 features explained)
   - Imputation methodology documentation
   - Usage examples for engineer.py

### Success Criteria
- ✓ All 8,520 fights successfully processed
- ✓ Zero data leakage (validated via temporal split)
- ✓ Early-career fighters properly imputed (no nulls in critical features)
- ✓ Feature distributions are reasonable (no extreme outliers from bugs)
- ✓ Code is modular and reusable for new data
- ✓ Can generate features for upcoming fights (prediction mode)

---

## Timeline Estimate
- Phase 1 (Preprocessing): 1-2 hours
- Phase 2 (Career Features): 2-3 hours
- Phase 3 (Imputation): 1-2 hours
- Phase 4 (Matchup Features): 1-2 hours
- Phase 5 (Temporal Features): 1 hour
- Phase 6 (Pipeline Integration): 1-2 hours
- Phase 7 (Validation): 2-3 hours
- Phase 8 (Documentation): 1 hour

**Total: 10-16 hours of implementation work**

---

## Risks & Mitigations

### Risk 1: Data Leakage
**Mitigation:** Strict chronological sorting, use expanding windows, automated validation tests

### Risk 2: Imputation Bias
**Mitigation:** Include `is_imputed` flags so model can learn reliability, use weight-class-specific medians

### Risk 3: Feature Explosion (too many features)
**Mitigation:** Start with core features (~50), add advanced features iteratively, use feature selection later

### Risk 4: Computation Time
**Mitigation:** Vectorize operations with pandas/numpy, cache intermediate results, profile code

---

## Open Questions for Leadership Review

1. **Imputation Strategy:** Approve the hybrid tier-based approach? Alternative suggestions?

2. **Feature Scope:** Start with all features (~100-150) or prioritize subset (~50 core features)?

3. **Data Structure:** Keep fight-centric (8,520 rows, both fighters as columns) or convert to fighter-centric (17,040 rows)?

4. **Pre-2009 Fights:** Include with limited features, or exclude entirely for cleaner dataset?

5. **Target Variable:** Predict binary win/loss, or multi-class (win by KO/Sub/Decision)?

6. **Weight Class Handling:** One-hot encode (creates many features) or treat as ordinal?

---

## Approval
- [ ] Approach approved as-is
- [ ] Approved with modifications (see comments)
- [ ] Requires revision

**Reviewer:** _______________
**Date:** _______________
**Comments:**

---

**Next Step After Approval:** Implement Phase 1-8 in `src/features/engineer.py` and generate processed feature dataset.
