# UFC Fight Predictor — Data Dictionary

**Last Updated:** 2026-02-05
**Dataset Version:** v1
**Total Features:** 145 (8 metadata + 137 predictive features)

---

## Raw Data Sources

| Source | Description | Update Frequency | Format | Location in Repo |
|--------|-------------|------------------|--------|------------------|
| ufcstats.com/events/completed | Historical UFC fight results with detailed statistics | Weekly (new events) | HTML → CSV | [data/raw/ufc_fights_v1.csv](data/raw/ufc_fights_v1.csv) |
| ufcstats.com/fighters | Fighter biographical data and career averages | Monthly | HTML → CSV | [data/raw/ufc_fighters_raw.csv](data/raw/ufc_fighters_raw.csv) |
| **Processed Dataset** | Feature-engineered data ready for ML | On-demand | CSV | [data/processed/ufc_fights_features_v1.csv](data/processed/ufc_fights_features_v1.csv) |

### Raw Data Statistics

**ufc_fights_v1.csv:**
- **Rows:** 8,520 fights
- **Columns:** 58
- **Date Range:** March 11, 1994 to January 31, 2026 (31.9 years)
- **Events:** 760 UFC events
- **Size:** 7.06 MB
- **Completeness:** 79.6% (detailed stats only available from 2009+)

**ufc_fighters_raw.csv:**
- **Rows:** ~3,000 fighters
- **Columns:** 14 (biographical + career stats)
- **Size:** ~500 KB

**ufc_fights_features_v1.csv:**
- **Rows:** 8,520 fights
- **Columns:** 145 features
- **Target Variable:** `f1_is_winner` (binary: 0 or 1)
- **Class Balance:** 48.9% Fighter 1 wins / 51.1% Fighter 2 wins
- **Missing Values:** 33,719 total (due to early-career fighters and pre-2009 data)

---

## Target Variable

### f1_is_winner
- **Name:** `f1_is_winner`
- **Type:** Binary (0 or 1)
- **Encoding:** 1 = Fighter 1 wins, 0 = Fighter 2 wins
- **Source:** Raw `winner` column compared to fighter names
- **Transformation:** Matched winner name to Fighter 1/Fighter 2, encoded as binary
- **Distribution:** 48.9% (4,164 fights) Fighter 1 wins, 51.1% (4,356 fights) Fighter 2 wins
- **Why:** Primary prediction target. Model outputs probability that Fighter 1 wins the fight.
- **Note:** Fighter 1 vs Fighter 2 assignment is arbitrary (not predictive). Model should be symmetric.

---

## Feature Dictionary

### Category 1: Metadata (5 features) — Not used in model training

#### fight_id
- **Name:** `fight_id`
- **Source:** Fight URL on ufcstats.com (e.g., `30ad2050273d016a`)
- **Transformation:** Extracted from URL path
- **Type:** String
- **Range/Scale:** 16-character hex string
- **Used In:** Data joining, prediction tracking
- **Why:** Unique identifier for each fight

#### fight_date
- **Name:** `fight_date`
- **Source:** Raw `event_date` column
- **Transformation:** Parsed "Month DD, YYYY" → datetime → "YYYY-MM-DD"
- **Type:** Date string
- **Range/Scale:** 1994-03-11 to 2026-01-31
- **Used In:** Temporal ordering, train/test split
- **Why:** Critical for chronological feature engineering and preventing data leakage

#### event_name
- **Name:** `event_name`
- **Source:** Raw `event_name` column
- **Transformation:** Direct copy
- **Type:** String
- **Example:** "UFC 308: Topuria vs Holloway"
- **Used In:** Grouping predictions by event
- **Why:** Human-readable event identification

#### weight_class
- **Name:** `weight_class`
- **Source:** Raw `weight_class` column
- **Transformation:** Direct copy (also one-hot encoded separately)
- **Type:** Categorical
- **Range/Scale:** 15 classes (Flyweight, Bantamweight, ..., Open Weight)
- **Used In:** Weight class analysis (stored but one-hot version used in model)
- **Why:** Weight class context (encoded version is predictive feature)

#### f1_name / f2_name
- **Name:** `f1_name`, `f2_name`
- **Source:** Raw `fighter1_name`, `fighter2_name` columns
- **Transformation:** Direct copy
- **Type:** String
- **Example:** "Conor McGregor", "Khabib Nurmagomedov"
- **Used In:** Display, prediction interpretation
- **Why:** Fighter identification for human readability

#### method
- **Name:** `method`
- **Source:** Raw `method` column
- **Transformation:** Direct copy
- **Type:** Categorical
- **Range/Scale:** KO/TKO, Submission, Decision (Unanimous/Split/Majority), Draw, No Contest
- **Used In:** Analysis (not used in model — would leak outcome)
- **Why:** Historical record of how fight ended

#### round
- **Name:** `round`
- **Source:** Raw `round` column
- **Transformation:** Parsed from string to integer
- **Type:** Numeric (integer)
- **Range/Scale:** 1-5 (3 rounds standard, 5 for title fights)
- **Used In:** Analysis (not used in model — would leak outcome)
- **Why:** Historical record of when fight ended

#### fight_time_seconds
- **Name:** `fight_time_seconds`
- **Source:** Raw `time` column (MM:SS format)
- **Transformation:** Parsed "3:45" → 225 seconds
- **Type:** Numeric (integer)
- **Range/Scale:** 0-900 seconds (0-15 minutes for 3-round fight)
- **Used In:** Average fight time rolling windows
- **Why:** Indicates fight pace and finish tendency

---

### Category 2: Career Statistics (16 features)

These features capture each fighter's historical performance up to (but not including) the current fight.

#### f1_career_fights / f2_career_fights
- **Name:** `f1_career_fights`, `f2_career_fights`
- **Source:** Aggregated from fight history
- **Transformation:** `COUNT(previous fights)` for each fighter
- **Type:** Numeric (integer)
- **Range/Scale:** 0-45 (UFC debut to seasoned veteran)
- **Used In:** Model training
- **Why:** Experience level — more fights = more data, but also older/slower fighter

#### diff_career_fights
- **Name:** `diff_career_fights`
- **Source:** `f1_career_fights - f2_career_fights`
- **Transformation:** Subtraction
- **Type:** Numeric (integer)
- **Range/Scale:** -45 to +45
- **Used In:** Model training
- **Why:** Experience differential — positive = Fighter 1 more experienced

#### f1_career_wins / f2_career_wins
- **Name:** `f1_career_wins`, `f2_career_wins`
- **Source:** Aggregated from fight history
- **Transformation:** `COUNT(previous wins)` for each fighter
- **Type:** Numeric (integer)
- **Range/Scale:** 0-30+
- **Used In:** Model training
- **Why:** Raw win count (less useful than win rate, but captures volume)

#### diff_career_wins
- **Name:** `diff_career_wins`
- **Source:** `f1_career_wins - f2_career_wins`
- **Transformation:** Subtraction
- **Type:** Numeric (integer)
- **Range/Scale:** -30 to +30
- **Used In:** Model training
- **Why:** Win count differential

#### f1_career_losses / f2_career_losses
- **Name:** `f1_career_losses`, `f2_career_losses`
- **Source:** Aggregated from fight history
- **Transformation:** `COUNT(previous losses)` for each fighter
- **Type:** Numeric (integer)
- **Range/Scale:** 0-20+
- **Used In:** Model training
- **Why:** Loss count (inverse of wins, captures durability/matchmaking)

#### diff_career_losses
- **Name:** `diff_career_losses`
- **Source:** `f1_career_losses - f2_career_losses`
- **Transformation:** Subtraction
- **Type:** Numeric (integer)
- **Range/Scale:** -20 to +20
- **Used In:** Model training
- **Why:** Loss count differential — positive = Fighter 1 lost more (bad)

#### f1_career_win_rate / f2_career_win_rate
- **Name:** `f1_career_win_rate`, `f2_career_win_rate`
- **Source:** `career_wins / career_fights`
- **Transformation:** Division (0 if no prior fights)
- **Type:** Numeric (float)
- **Range/Scale:** 0.0-1.0 (0% to 100%)
- **Used In:** Model training (PRIMARY CAREER FEATURE)
- **Why:** **Most important career metric** — win percentage captures overall skill level

#### diff_career_win_rate
- **Name:** `diff_career_win_rate`
- **Source:** `f1_career_win_rate - f2_career_win_rate`
- **Transformation:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -1.0 to +1.0 (-100% to +100%)
- **Used In:** Model training (HIGH IMPORTANCE)
- **Why:** **Key predictive feature** — positive = Fighter 1 better historical performer

#### f1_career_ko_rate / f2_career_ko_rate
- **Name:** `f1_career_ko_rate`, `f2_career_ko_rate`
- **Source:** `COUNT(KO/TKO wins) / career_wins`
- **Transformation:** Division (0 if no prior wins)
- **Type:** Numeric (float)
- **Range/Scale:** 0.0-1.0 (0% to 100% of wins by KO)
- **Used In:** Model training
- **Why:** Knockout power — high KO rate = dangerous striker

#### diff_career_ko_rate
- **Name:** `diff_career_ko_rate`
- **Source:** `f1_career_ko_rate - f2_career_ko_rate`
- **Transformation:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -1.0 to +1.0
- **Used In:** Model training
- **Why:** KO power differential — predicts striking-based finishes

#### f1_career_sub_rate / f2_career_sub_rate
- **Name:** `f1_career_sub_rate`, `f2_career_sub_rate`
- **Source:** `COUNT(Submission wins) / career_wins`
- **Transformation:** Division (0 if no prior wins)
- **Type:** Numeric (float)
- **Range/Scale:** 0.0-1.0 (0% to 100% of wins by submission)
- **Used In:** Model training
- **Why:** Submission skill — high sub rate = grappling threat

#### diff_career_sub_rate
- **Name:** `diff_career_sub_rate`
- **Source:** `f1_career_sub_rate - f2_career_sub_rate`
- **Transformation:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -1.0 to +1.0
- **Used In:** Model training
- **Why:** Grappling skill differential

#### f1_career_dec_rate / f2_career_dec_rate
- **Name:** `f1_career_dec_rate`, `f2_career_dec_rate`
- **Source:** `COUNT(Decision wins) / career_wins`
- **Transformation:** Division (0 if no prior wins)
- **Type:** Numeric (float)
- **Range/Scale:** 0.0-1.0 (0% to 100% of wins by decision)
- **Used In:** Model training
- **Why:** Decision tendency — high rate = volume fighter, not finisher

#### diff_career_dec_rate
- **Name:** `diff_career_dec_rate`
- **Source:** `f1_career_dec_rate - f2_career_dec_rate`
- **Transformation:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -1.0 to +1.0
- **Used In:** Model training
- **Why:** Finish ability differential — negative = Fighter 1 more likely to finish

#### f1_career_finish_rate / f2_career_finish_rate
- **Name:** `f1_career_finish_rate`, `f2_career_finish_rate`
- **Source:** `(career_ko_rate + career_sub_rate)` (1 - decision rate)
- **Transformation:** Sum of KO + Sub rates
- **Type:** Numeric (float)
- **Range/Scale:** 0.0-1.0 (0% to 100% of wins by finish)
- **Used In:** Model training
- **Why:** Overall finishing ability (KO or submission)

#### diff_career_finish_rate
- **Name:** `diff_career_finish_rate`
- **Source:** `f1_career_finish_rate - f2_career_finish_rate`
- **Transformation:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -1.0 to +1.0
- **Used In:** Model training
- **Why:** Finish ability differential — predicts stoppage likelihood

---

### Category 3: Momentum & Recent Form (40 features)

Rolling window statistics over last 3, 5, and 10 fights. Captures recent performance trends.

#### f1_win_rate_last_N / f2_win_rate_last_N (N = 3, 5, 10)
- **Name:** `f1_win_rate_last_3`, `f1_win_rate_last_5`, `f1_win_rate_last_10` (and f2 equivalents)
- **Source:** `MEAN(is_winner)` over last N fights
- **Transformation:** Rolling window mean with `.shift(1).rolling(N).mean()`
- **Type:** Numeric (float)
- **Range/Scale:** 0.0-1.0 (0% to 100% win rate in last N fights)
- **Used In:** Model training (HIGH IMPORTANCE)
- **Why:** **Recent form is highly predictive** — fighter on 5-fight win streak vs 5-fight losing streak

#### diff_win_rate_last_N (N = 3, 5, 10)
- **Name:** `diff_win_rate_last_3`, `diff_win_rate_last_5`, `diff_win_rate_last_10`
- **Source:** `f1_win_rate_last_N - f2_win_rate_last_N`
- **Transformation:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -1.0 to +1.0
- **Used In:** Model training (HIGH IMPORTANCE)
- **Why:** **Momentum differential — key predictor** — positive = Fighter 1 on hot streak

#### f1_finish_rate_last_N / f2_finish_rate_last_N (N = 3, 5, 10)
- **Name:** `f1_finish_rate_last_3`, etc. (6 features total)
- **Source:** `MEAN(method in ['KO/TKO', 'Submission'])` over last N wins
- **Transformation:** Rolling window mean of finish outcomes
- **Type:** Numeric (float)
- **Range/Scale:** 0.0-1.0 (0% to 100% of recent wins by finish)
- **Used In:** Model training
- **Why:** Recent finishing ability — hot finishing streak = dangerous

#### diff_finish_rate_last_N (N = 3, 5, 10)
- **Name:** `diff_finish_rate_last_3`, etc. (3 features total)
- **Source:** Subtraction of f1 - f2 finish rates
- **Transformation:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -1.0 to +1.0
- **Used In:** Model training
- **Why:** Recent finishing differential

#### f1_avg_sig_strikes_landed_last_N / f2_avg_sig_strikes_landed_last_N (N = 3, 5, 10)
- **Name:** `f1_avg_sig_strikes_landed_last_3`, etc. (6 features total)
- **Source:** Raw `sig_strikes_landed` from detailed stats
- **Transformation:** Rolling window mean of strikes landed per fight
- **Type:** Numeric (float)
- **Range/Scale:** 0-150+ (average significant strikes per fight)
- **Used In:** Model training
- **Why:** Recent striking output — high volume = aggressive striker
- **Note:** 0 for pre-2009 fights (detailed stats unavailable)

#### diff_avg_sig_strikes_landed_last_N (N = 3, 5, 10)
- **Name:** `diff_avg_sig_strikes_landed_last_3`, etc. (3 features total)
- **Source:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -150 to +150
- **Used In:** Model training
- **Why:** Recent striking volume differential

#### f1_avg_sig_strikes_absorbed_last_N / f2_avg_sig_strikes_absorbed_last_N (N = 3, 5, 10)
- **Name:** `f1_avg_sig_strikes_absorbed_last_3`, etc. (6 features total)
- **Source:** Opponent's `sig_strikes_landed` (fighter's absorbed)
- **Transformation:** Rolling window mean of opponent strikes landed
- **Type:** Numeric (float)
- **Range/Scale:** 0-150+ (average strikes absorbed per fight)
- **Used In:** Model training
- **Why:** Recent defensive ability — low absorbed = good defense
- **Note:** 0 for pre-2009 fights

#### diff_avg_sig_strikes_absorbed_last_N (N = 3, 5, 10)
- **Name:** `diff_avg_sig_strikes_absorbed_last_3`, etc. (3 features total)
- **Source:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -150 to +150
- **Used In:** Model training
- **Why:** Recent defensive ability differential — negative = Fighter 1 better defense

#### f1_avg_takedowns_last_N / f2_avg_takedowns_last_N (N = 3, 5, 10)
- **Name:** `f1_avg_takedowns_last_3`, etc. (6 features total)
- **Source:** Raw `takedowns_landed` from detailed stats
- **Transformation:** Rolling window mean of takedowns per fight
- **Type:** Numeric (float)
- **Range/Scale:** 0-10+ (average takedowns per fight)
- **Used In:** Model training
- **Why:** Recent grappling activity — high takedowns = wrestler
- **Note:** 0 for pre-2009 fights

#### diff_avg_takedowns_last_N (N = 3, 5, 10)
- **Name:** `diff_avg_takedowns_last_3`, etc. (3 features total)
- **Source:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -10 to +10
- **Used In:** Model training
- **Why:** Recent grappling differential

#### f1_avg_fight_time_last_N / f2_avg_fight_time_last_N (N = 3, 5, 10)
- **Name:** `f1_avg_fight_time_last_3`, etc. (6 features total)
- **Source:** `fight_time_seconds` from previous fights
- **Transformation:** Rolling window mean of fight durations
- **Type:** Numeric (float)
- **Range/Scale:** 0-900 seconds (0-15 minutes)
- **Used In:** Model training
- **Why:** Recent fight length — short = fast finisher, long = decision fighter

#### diff_avg_fight_time_last_N (N = 3, 5, 10)
- **Name:** `diff_avg_fight_time_last_3`, etc. (3 features total)
- **Source:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -900 to +900 seconds
- **Used In:** Model training
- **Why:** Fight length differential

---

### Category 4: Streaks & Momentum Indicators (8 features)

#### f1_win_streak / f2_win_streak
- **Name:** `f1_win_streak`, `f2_win_streak`
- **Source:** Consecutive wins before current fight
- **Transformation:** Custom streak calculation (resets to 0 on loss)
- **Type:** Numeric (integer)
- **Range/Scale:** 0-15+ (longest active win streak)
- **Used In:** Model training
- **Why:** Win streak = confidence, momentum, matchmaking quality

#### diff_win_streak
- **Name:** `diff_win_streak`
- **Source:** `f1_win_streak - f2_win_streak`
- **Transformation:** Subtraction
- **Type:** Numeric (integer)
- **Range/Scale:** -15 to +15
- **Used In:** Model training
- **Why:** Momentum differential — fighter on 5-win streak vs 0-win streak

#### f1_loss_streak / f2_loss_streak
- **Name:** `f1_loss_streak`, `f2_loss_streak`
- **Source:** Consecutive losses before current fight
- **Transformation:** Custom streak calculation (resets to 0 on win)
- **Type:** Numeric (integer)
- **Range/Scale:** 0-8+ (longest active losing streak)
- **Used In:** Model training
- **Why:** Loss streak = decline, desperation, poor matchmaking

#### diff_loss_streak
- **Name:** `diff_loss_streak`
- **Source:** `f1_loss_streak - f2_loss_streak`
- **Transformation:** Subtraction
- **Type:** Numeric (integer)
- **Range/Scale:** -8 to +8
- **Used In:** Model training
- **Why:** Losing streak differential — positive = Fighter 1 struggling more

#### f1_days_since_last_fight / f2_days_since_last_fight
- **Name:** `f1_days_since_last_fight`, `f2_days_since_last_fight`
- **Source:** `current_fight_date - last_fight_date` in days
- **Transformation:** Date subtraction
- **Type:** Numeric (integer)
- **Range/Scale:** 30-1000+ days (1 month to 3+ years)
- **Used In:** Model training
- **Why:** Layoff time — long layoff = rust, injury recovery, or inactivity

#### diff_days_since_last_fight
- **Name:** `diff_days_since_last_fight`
- **Source:** `f1_days_since_last_fight - f2_days_since_last_fight`
- **Transformation:** Subtraction
- **Type:** Numeric (integer)
- **Range/Scale:** -1000 to +1000 days
- **Used In:** Model training
- **Why:** Layoff differential — positive = Fighter 1 more inactive (potential rust)

---

### Category 5: Activity & Experience (12 features)

#### f1_fights_per_year / f2_fights_per_year
- **Name:** `f1_fights_per_year`, `f2_fights_per_year`
- **Source:** `career_fights / years_since_debut`
- **Transformation:** Division
- **Type:** Numeric (float)
- **Range/Scale:** 0.5-4.0 (fights per year)
- **Used In:** Model training
- **Why:** Activity level — 3+ fights/year = active, <1 fight/year = inactive/injured

#### diff_fights_per_year
- **Name:** `diff_fights_per_year`
- **Source:** `f1_fights_per_year - f2_fights_per_year`
- **Transformation:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -3.0 to +3.0
- **Used In:** Model training
- **Why:** Activity differential — positive = Fighter 1 more active (better conditioning)

#### f1_weight_class_fights / f2_weight_class_fights
- **Name:** `f1_weight_class_fights`, `f2_weight_class_fights`
- **Source:** `COUNT(previous fights in current weight class)`
- **Transformation:** Aggregation with weight class filter
- **Type:** Numeric (integer)
- **Range/Scale:** 0-40 (fights in this specific weight class)
- **Used In:** Model training
- **Why:** Weight class experience — moving up/down weight class = adaptation period

#### diff_weight_class_fights
- **Name:** `diff_weight_class_fights`
- **Source:** Subtraction
- **Type:** Numeric (integer)
- **Range/Scale:** -40 to +40
- **Used In:** Model training
- **Why:** Weight class experience differential

#### f1_fights_last_12mo / f2_fights_last_12mo
- **Name:** `f1_fights_last_12mo`, `f2_fights_last_12mo`
- **Source:** `COUNT(fights in last 365 days)`
- **Transformation:** Rolling 365-day window count
- **Type:** Numeric (integer)
- **Range/Scale:** 0-5 (fights in last year)
- **Used In:** Model training
- **Why:** Recent activity — 0 fights in last year = long layoff

#### diff_fights_last_12mo
- **Name:** `diff_fights_last_12mo`
- **Source:** Subtraction
- **Type:** Numeric (integer)
- **Range/Scale:** -5 to +5
- **Used In:** Model training
- **Why:** Recent activity differential

#### f1_is_ufc_debut / f2_is_ufc_debut
- **Name:** `f1_is_ufc_debut`, `f2_is_ufc_debut`
- **Source:** `career_fights == 0`
- **Transformation:** Boolean flag (0 or 1)
- **Type:** Binary (0 or 1)
- **Range/Scale:** 0 (veteran) or 1 (debut)
- **Used In:** Model training
- **Why:** UFC debut = unknown quantity, high variance

#### diff_is_ufc_debut
- **Name:** `diff_is_ufc_debut`
- **Source:** Subtraction
- **Type:** Numeric (integer)
- **Range/Scale:** -1, 0, +1
- **Used In:** Model training
- **Why:** Debut differential — 1 = only Fighter 1 is debut

#### f1_data_completeness_score / f2_data_completeness_score
- **Name:** `f1_data_completeness_score`, `f2_data_completeness_score`
- **Source:** Percentage of non-null detailed stats in fight history
- **Transformation:** `COUNT(non-null detailed stats) / career_fights`
- **Type:** Numeric (float)
- **Range/Scale:** 0.0-1.0 (0% to 100% data completeness)
- **Used In:** Model training (DATA QUALITY INDICATOR)
- **Why:** Pre-2009 fighters have low scores (missing detailed stats) — helps model calibrate confidence

#### diff_data_completeness_score
- **Name:** `diff_data_completeness_score`
- **Source:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -1.0 to +1.0
- **Used In:** Model training
- **Why:** Data quality differential — negative = Fighter 1 has less data (older career)

#### f1_has_detailed_stats / f2_has_detailed_stats
- **Name:** `f1_has_detailed_stats`, `f2_has_detailed_stats`
- **Source:** Flag for whether CURRENT fight has detailed strike/grappling stats
- **Transformation:** Boolean flag (0 or 1)
- **Type:** Binary (0 or 1)
- **Range/Scale:** 0 (pre-2009, no detailed stats) or 1 (2009+, detailed stats)
- **Used In:** Model training
- **Why:** Indicates data availability for rolling window calculations

---

### Category 6: Style Indicators (6 features)

#### f1_striking_volume / f2_striking_volume
- **Name:** `f1_striking_volume`, `f2_striking_volume`
- **Source:** `AVG(sig_strikes_landed) over all fights with detailed stats`
- **Transformation:** Mean aggregation
- **Type:** Numeric (float)
- **Range/Scale:** 0-100+ (average strikes per fight)
- **Used In:** Model training
- **Why:** Striker vs counter-striker identification — high volume = pressure fighter

#### diff_striking_volume
- **Name:** `diff_striking_volume`
- **Source:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -100 to +100
- **Used In:** Model training
- **Why:** Striking style differential — positive = Fighter 1 more aggressive striker

#### f1_grappling_tendency / f2_grappling_tendency
- **Name:** `f1_grappling_tendency`, `f2_grappling_tendency`
- **Source:** `AVG(takedowns_landed) over all fights with detailed stats`
- **Transformation:** Mean aggregation
- **Type:** Numeric (float)
- **Range/Scale:** 0-8+ (average takedowns per fight)
- **Used In:** Model training
- **Why:** Wrestler vs striker identification — high takedowns = wrestler

#### diff_grappling_tendency
- **Name:** `diff_grappling_tendency`
- **Source:** Subtraction
- **Type:** Numeric (float)
- **Range/Scale:** -8 to +8
- **Used In:** Model training
- **Why:** Grappling style differential — positive = Fighter 1 more wrestling-focused

#### f1_experience_ratio / f2_experience_ratio
- **Name:** `f1_experience_ratio`, `f2_experience_ratio`
- **Source:** `career_fights / MAX(all fighters' career_fights)`
- **Transformation:** Normalization (0-1 scale)
- **Type:** Numeric (float)
- **Range/Scale:** 0.0-1.0 (0% to 100% of max experience)
- **Used In:** Model training
- **Why:** Relative experience — 0.2 = rookie, 1.0 = most experienced fighter in dataset

---

### Category 7: Matchup Differentials (37 features)

**Note:** All differential features computed as `f1_feature - f2_feature`. Positive values favor Fighter 1.

See Category 2-6 above for individual `diff_*` features. Summary:
- **Career Differentials (8):** Career fights, wins, losses, win rate, KO/sub/dec/finish rates
- **Rolling Window Differentials (24):** Win rate, finish rate, strikes landed/absorbed, takedowns, fight time (across 3/5/10 windows)
- **Momentum Differentials (3):** Win streak, loss streak, days since last fight
- **Activity Differentials (3):** Fights per year, weight class fights, fights last 12 months
- **Style Differentials (3):** Striking volume, grappling tendency, momentum (derived)

#### diff_momentum
- **Name:** `diff_momentum`
- **Source:** `(f1_win_rate_last_5 - f1_loss_streak/10) - (f2_win_rate_last_5 - f2_loss_streak/10)`
- **Transformation:** Composite feature combining recent win rate and loss streak
- **Type:** Numeric (float)
- **Range/Scale:** -2.0 to +2.0
- **Used In:** Model training
- **Why:** **Combined momentum indicator** — captures hot/cold streak in single feature

---

### Category 8: Temporal Features (4 features)

#### fight_year
- **Name:** `fight_year`
- **Source:** Extracted from `fight_date`
- **Transformation:** `datetime.year`
- **Type:** Numeric (integer)
- **Range/Scale:** 1994-2026
- **Used In:** Model training
- **Why:** Captures era effects (rule changes, meta shifts, athlete evolution)

#### fight_month
- **Name:** `fight_month`
- **Source:** Extracted from `fight_date`
- **Transformation:** `datetime.month`
- **Type:** Numeric (integer)
- **Range/Scale:** 1-12 (January to December)
- **Used In:** Model training
- **Why:** Seasonal effects (training camps, holidays, activity cycles)

#### f1_is_coming_off_layoff / f2_is_coming_off_layoff
- **Name:** `f1_is_coming_off_layoff`, `f2_is_coming_off_layoff`
- **Source:** `days_since_last_fight > 365`
- **Transformation:** Boolean flag (0 or 1)
- **Type:** Binary (0 or 1)
- **Range/Scale:** 0 (active) or 1 (layoff)
- **Used In:** Model training
- **Why:** Rust factor — 1+ year layoff = potential ring rust

#### f1_is_new_to_weight_class / f2_is_new_to_weight_class
- **Name:** `f1_is_new_to_weight_class`, `f2_is_new_to_weight_class`
- **Source:** `weight_class_fights < 2`
- **Transformation:** Boolean flag (0 or 1)
- **Type:** Binary (0 or 1)
- **Range/Scale:** 0 (experienced) or 1 (new)
- **Used In:** Model training
- **Why:** Weight class adjustment — moving up/down = adaptation period

---

### Category 9: Weight Class One-Hot Encoding (15 features)

All weight classes encoded as binary 0/1 dummy variables.

#### wc_Bantamweight ... wc_Women's Strawweight
- **Name:** `wc_Bantamweight`, `wc_Catch Weight`, `wc_Featherweight`, `wc_Flyweight`, `wc_Heavyweight`, `wc_Light Heavyweight`, `wc_Lightweight`, `wc_Middleweight`, `wc_Open Weight`, `wc_Super Heavyweight`, `wc_Welterweight`, `wc_Women's Bantamweight`, `wc_Women's Featherweight`, `wc_Women's Flyweight`, `wc_Women's Strawweight`
- **Source:** Raw `weight_class` column
- **Transformation:** One-hot encoding (pandas `get_dummies()`)
- **Type:** Binary (0 or 1)
- **Range/Scale:** Exactly one feature = 1 per fight, rest = 0
- **Used In:** Model training
- **Why:** Weight class effects (heavyweight KO power, flyweight speed, etc.)
- **Note:** Alternative to label encoding (preserves categorical nature)

#### is_title_fight
- **Name:** `is_title_fight`
- **Source:** Detected from `weight_class` containing "Title" or from bout type
- **Transformation:** Boolean flag (0 or 1)
- **Type:** Binary (0 or 1)
- **Range/Scale:** 0 (regular bout) or 1 (title fight)
- **Used In:** Model training
- **Why:** Title fights = 5 rounds (vs 3), higher stakes, better fighters

---

## Data Quality Notes

### Missing Value Patterns

1. **Pre-2009 Detailed Stats (2,406 fights):**
   - Missing: All `*_sig_str_*`, `*_TD`, `*_sub_att`, `*_rev`, `*_ctrl` columns
   - Available: Basic fight info (winner, method, round, time)
   - Handling: Filled with zeros, flagged with `has_detailed_stats = False`

2. **Early-Career Fighters (UFC debuts):**
   - Missing: All career and rolling window features (0 previous fights)
   - Handling: Filled with zeros or NaN (model learns debut pattern)

3. **Differential Features Outside [0,1]:**
   - Features like `diff_career_win_rate` can be negative (Fighter 1 worse)
   - Validation warnings are expected but not errors
   - Range: Typically -1.0 to +1.0 for rate differentials

### Special Handling Required

- **Dates:** Parse "Month DD, YYYY" format before date operations
- **Strikes:** Parse "25 of 50" → (landed: 25, attempted: 50) before calculations
- **Control Time:** Parse "3:45" → 225 seconds before averaging
- **Fighter Names:** Handle non-ASCII characters (e.g., "José Aldo")
- **Nulls:** Check for both empty strings `""` and `NaN`

---

## Feature Importance (Expected)

**Based on similar MMA prediction research:**

### Tier 1: High Importance (Top 10 Features)
1. `diff_career_win_rate` — Historical skill differential
2. `diff_win_rate_last_5` — Recent momentum
3. `diff_career_fights` — Experience differential
4. `diff_win_streak` — Current form
5. `diff_striking_volume` — Style matchup
6. `diff_avg_sig_strikes_landed_last_5` — Recent striking output
7. `diff_days_since_last_fight` — Inactivity
8. `diff_grappling_tendency` — Wrestler vs striker
9. `f1_is_ufc_debut` / `f2_is_ufc_debut` — Unknown quantity
10. `is_title_fight` — Fight context

### Tier 2: Medium Importance
- Weight class indicators
- Career finish rates (KO/sub)
- Recent strikes absorbed (defense)
- Takedown differentials
- Activity levels (fights per year)

### Tier 3: Low Importance
- Fight year/month (weak temporal effects)
- Individual (non-differential) features (model struggles with comparisons)
- Data completeness scores (quality indicators, not predictive)

### Feature Selection Candidates for Removal
- Remove individual features, keep only differentials
- Remove highly correlated features (career_wins ≈ career_win_rate * career_fights)
- Remove low-variance features (all fighters similar)

---

## Feature Engineering Artifacts

### Saved Files
- **[data/processed/weight_class_medians_v1.pkl](data/processed/weight_class_medians_v1.pkl)** — Median values per weight class for imputation
- **[data/processed/feature_config_v1.json](data/processed/feature_config_v1.json)** — Feature names, types, and metadata for model compatibility

### Regeneration
To regenerate features after scraper changes:
```bash
python run_feature_engineering.py
```

This will:
1. Load [data/raw/ufc_fights_v1.csv](data/raw/ufc_fights_v1.csv)
2. Parse strings, restructure data, compute features
3. Save [data/processed/ufc_fights_features_v1.csv](data/processed/ufc_fights_features_v1.csv)
4. Generate [data/processed/feature_engineering_report_v1.md](data/processed/feature_engineering_report_v1.md)

---

## Usage Examples

### Load Processed Data
```python
import pandas as pd

# Load feature-engineered data
df = pd.read_csv('data/processed/ufc_fights_features_v1.csv')
print(df.shape)  # (8520, 145)

# Check target distribution
print(df['f1_is_winner'].value_counts())
# 0    4356 (51.1%)
# 1    4164 (48.9%)
```

### Split Features and Target
```python
# Metadata columns (exclude from training)
metadata_cols = ['fight_id', 'fight_date', 'event_name', 'weight_class', 'f1_name', 'f2_name', 'method', 'round', 'fight_time_seconds']

# Target variable
y = df['f1_is_winner']

# Feature columns (everything except metadata and target)
feature_cols = [col for col in df.columns if col not in metadata_cols + ['f1_is_winner']]
X = df[feature_cols]

print(f"Features: {len(feature_cols)}")  # 136 predictive features
```

### Temporal Train/Test Split
```python
# Train on pre-2024 fights, test on 2024+
df['fight_date'] = pd.to_datetime(df['fight_date'])
train = df[df['fight_date'] < '2024-01-01']
test = df[df['fight_date'] >= '2024-01-01']

print(f"Train: {len(train)} fights")  # ~8,000
print(f"Test: {len(test)} fights")    # ~500
```

### Check for Missing Values
```python
# Missing value summary
missing = X.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# Features with most missing values
# (typically rolling window features for early-career fighters)
```

---

## References

- **Feature Engineering Implementation:** [src/features/engineer.py](src/features/engineer.py)
- **Raw Data Schema:** [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)
- **Feature Engineering Report:** [data/processed/feature_engineering_report_v1.md](data/processed/feature_engineering_report_v1.md)
- **Feature Engineering Plan:** [FEATURE_ENGINEERING_IMPLEMENTATION_PROMPT.md](FEATURE_ENGINEERING_IMPLEMENTATION_PROMPT.md)
