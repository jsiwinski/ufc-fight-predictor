# CLAUDE.md — Session Starter Guide

**Purpose:** Read this at the start of every Claude Code session. Under 200 lines. Everything you need to avoid breaking things.

---

## Project State

### ✅ What Works (Production-Ready)

**Data Scraping** ([src/etl/scraper.py](src/etl/scraper.py))
- Scrapes ufcstats.com: 8,520 fights (1994-2026), 760 events, ~3,000 fighters
- 77 columns of fight statistics (strikes, takedowns, control time)
- Rate-limited (2s default), retry logic, validation
- **Recently Fixed:** Winner detection bug (Feb 4, 2026) — all fights were "Draw" due to wrong CSS selector
- Run with: `python scrape_full_data_v1.py` (~7-8 hours for full scrape)

**Feature Engineering** ([src/features/engineer.py](src/features/engineer.py))
- Transforms 8,520 fights → 145 features
- Career stats (win rate, KO/sub rates), rolling windows (last 3/5/10 fights), matchup differentials
- Handles missing pre-2009 data (2,406 fights lack detailed stats)
- Run with: `python run_feature_engineering.py` (~28 seconds)
- Output: [data/processed/ufc_fights_features_v1.csv](data/processed/ufc_fights_features_v1.csv) — 8,520 × 145

**Model Training** ([src/models/train.py](src/models/train.py)) ✅ **IMPLEMENTED**
- HistGradientBoostingClassifier with hyperparameter tuning
- Temporal split: 7,457 train (pre-2024) / 1,063 test (2024+)
- Accuracy: 58.9%, Log Loss: 0.665, Brier: 0.236
- Run with: `python src/models/train.py` (or `--skip-tuning` for quick)
- Model saved: [data/models/ufc_model_v1.pkl](data/models/ufc_model_v1.pkl)

**Prediction Serving** ([src/predict/serve.py](src/predict/serve.py)) ✅ **IMPLEMENTED**
- Scrapes upcoming events from ufcstats.com
- Matches fighters to historical data (exact + fuzzy matching)
- Generates 136-feature vectors, outputs win probabilities
- Run with: `python src/predict/serve.py` (live) or `--backtest DATE` (historical)
- Backtest: UFC 325 — 9/13 correct (69.2%)
- Predictions saved to: [data/predictions/](data/predictions/)

**Web Interface** ([src/web/app.py](src/web/app.py)) ✅ **IMPLEMENTED**
- Flask app serving predictions at localhost:5000
- Fight card UI with center-out probability bars
- Routes: `/` (home), `/event/<slug>`, `/archive`, `/backtest/<date>`
- JSON API at `/api/predict/next` and `/api/predict/<slug>`
- Run with: `python src/web/app.py`

### ⚠️ What's Broken / Stubbed

**Old predict.py** ([src/models/predict.py](src/models/predict.py)) — Legacy stubs, use src/predict/serve.py instead

### 🚧 In Progress

**Phase 5b:** Deploy to cloud (currently localhost only)

---

## Conventions

### File Naming
- **Scripts:** `snake_case.py` (e.g., `scrape_full_data_v1.py`, `run_feature_engineering.py`)
- **Modules:** `snake_case.py` (e.g., `scraper.py`, `engineer.py`)
- **Data Files:** `snake_case_v1.csv` with version suffix (e.g., `ufc_fights_v1.csv`)
- **Docs:** `UPPERCASE.md` for root-level docs, `Title_Case.md` for [docs/](docs/)

### Code Style
- **Formatting:** Black (not enforced but preferred)
- **Docstrings:** Google style (triple quotes, Args/Returns sections)
- **Type Hints:** Used in function signatures where beneficial
- **Logging:** Use `logging` module, not `print()` (except in standalone scripts)

### Config & Secrets
- **Configuration:** [config.yaml](config.yaml) — Central config for paths, model params, web server settings
- **Secrets:** Use environment variables (no API keys in git). Access via `os.getenv()` or `python-dotenv`
- **Git:** Never commit `.env` files (already in [.gitignore](.gitignore))

### Data File Storage
- **Raw Data:** [data/raw/](data/raw/) — Scraped CSVs, never modified after scraping
- **Processed Data:** [data/processed/](data/processed/) — Feature-engineered CSVs, artifacts
- **Models:** [data/models/](data/models/) — Saved .pkl model files (ufc_model_v1.pkl, evaluation_results_v1.json)
- **Predictions:** [data/predictions/](data/predictions/) — Generated prediction CSVs (empty currently)
- **Git Policy:** ALL data files excluded via [.gitignore](.gitignore) (too large for git)

### Testing
- **Test Directory:** [tests/](tests/) contains smoke tests for feature engineering
- **Test File:** [tests/test_feature_engineering.py](tests/test_feature_engineering.py) — Validates pipeline outputs, target variable, regression tests
- **Run Tests:** `pytest tests/`

### Git Conventions
- **Commit Messages:** Descriptive (50 char summary + detailed body if complex)
- **Branching:** Working directly on `main` (small team, fast iteration)
- **No Force Push:** Never `git push --force` on `main`

---

## Gotchas & Landmines

### Things That Look Wrong But Are Intentional

1. **Differential Features Outside [0,1]:**
   Features like `diff_career_win_rate` can be negative (e.g., -0.5 if Fighter 1 has 30% win rate, Fighter 2 has 80%). This is CORRECT — it represents Fighter 1 being worse. Validation warnings are expected but not errors.

2. **17,040 Records from 8,520 Fights:**
   Feature engineering doubles data by creating 1 row per fighter-performance (Fighter A's view + Fighter B's view). This is intentional for chronological career stat calculation.

3. **Zeros for Pre-2009 Fight Stats:**
   2,406 fights (1994-2008) have zeros for detailed strike/grappling stats. Not data corruption — these columns didn't exist in ufcstats.com back then. Flag `has_detailed_stats = False` handles this.

4. **Fighter 1 vs Fighter 2 Arbitrary:**
   Red/Blue corner or Fighter 1/Fighter 2 assignment is ARBITRARY (not predictive). Model should be symmetric — swapping fighters shouldn't change prediction. Use differential features to enforce this.

### Fragile Areas

1. **Scraper CSS Selectors:**
   Winner detection uses `a.b-flag_style_green` (line 416-437 in [scraper.py](src/etl/scraper.py:416-437)). If ufcstats.com changes HTML, scraper breaks. Always validate on recent event after site updates.

2. **Rolling Window Calculation:**
   Uses pandas `.shift(1).rolling(window)` with chronological ordering. If data is NOT sorted by date, calculations will be wrong. Always verify `df = df.sort_values('fight_date')` before feature engineering.

3. **Data Leakage Risk:**
   Career stats MUST exclude current fight (`.shift(1)` before aggregation). Never use current fight's outcome to predict current fight. Easy to break if modifying feature engineering.

4. **Feature Name Consistency:**
   Model expects exact feature names. If you change feature engineering (rename columns), re-train model or it will crash. Feature names stored in `feature_config_v1.json`.

### Common Mistakes

1. **Don't Run Scraper Without Rate Limiting:**
   Default 2s rate limit is MANDATORY. Running faster = risk getting IP banned from ufcstats.com. Test with `limit_events=5`, not full scrape.

2. **Don't Use `git add .` Blindly:**
   Data files are large (7-30 MB CSVs). Always check `.gitignore` before committing. Use `git add <specific files>`.

3. **README.md Reflects Actual State:**
   [README.md](README.md) was updated 2026-02-05 to reflect current project status. Keep it in sync with ARCHITECTURE.md and CLAUDE.md.

### Data Assumptions

- **Dates:** `event_date` is string "Month DD, YYYY" (e.g., "January 15, 2026"). Parse with `pd.to_datetime()` before date math.
- **Nulls:** Missing values are empty strings `""` or `NaN`. Both exist. Handle both when checking nulls.
- **Timezones:** No timezone info in data. All dates are UTC-assumed.
- **Encoding:** All CSVs are UTF-8. Fighter names may have non-ASCII characters (e.g., "José Aldo").

---

## Current Priorities

**Fill this in each session with specific tasks:**

- [ ] **Priority 1:** _________________________________________
- [ ] **Priority 2:** _________________________________________
- [ ] **Priority 3:** _________________________________________

---

## Rules of the Road

### Adding Features

1. **Always chronological:** New features must respect fight date ordering. Use `.shift(1)` to avoid data leakage.
2. **Add to both fighters:** If you add `f1_new_feature`, MUST add `f2_new_feature` and `diff_new_feature`.
3. **Update config:** Add feature name to `feature_config_v1.json` for model compatibility.
4. **Re-run pipeline:** After modifying [engineer.py](src/features/engineer.py), run `python run_feature_engineering.py` to regenerate [ufc_fights_features_v1.csv](data/processed/ufc_fights_features_v1.csv).

### Ordering Constraints

1. **Scraping → Feature Engineering → Model Training:**
   Must run in this order. Can't train model without features, can't engineer features without raw data.

2. **Feature Engineering Re-Run:**
   If you change [scraper.py](src/etl/scraper.py) and re-scrape data, MUST re-run feature engineering (features depend on raw data).

3. **Model Re-Training:**
   If you change feature engineering (add/remove features), MUST re-train model (feature mismatch will crash prediction).

### Never Touch Directly

1. **[data/raw/](data/raw/) Files:** Once scraped, consider immutable. Re-scrape if you need fresh data (don't hand-edit CSVs).
2. **Winner Column After Scrape:** Never manually fix winner in raw data. Fix scraper bug and re-scrape instead.
3. **Git History:** No rewriting git history on `main`. No `git rebase -i`, `git commit --amend` after push.

### Pipeline Stage Dependencies

```
[Scraper] → [Raw Data] → [Feature Engineer] → [Processed Data] → [Model Training] → [Saved Model] → [Prediction] → [Web App]
    ✅           ✅              ✅                  ✅                ✅               ✅             ✅            ✅
```

**Dependency Chain:**
- Model Training depends on Feature Engineering completion
- Prediction depends on Model Training completion (now complete)
- Web App depends on Prediction completion

**Side Effects:**
- Changing scraper requires re-running everything downstream
- Changing feature engineering requires re-training model
- Don't skip steps or pipeline breaks

---

## Flags & Issues

### Known Issues (Non-Blocking)

1. **Validation Warnings in Feature Engineering:**
   Report shows 33,719 missing values and 13 features outside expected ranges. This is expected due to early-career fighters (0 previous fights → division by zero). Should add bounds clamping but not urgent.

### Historical Artifacts

- **[FEATURE_ENGINEERING_IMPLEMENTATION_PROMPT.md](FEATURE_ENGINEERING_IMPLEMENTATION_PROMPT.md):** 440-line planning doc from Phase 2. Historical artifact, can archive.

### Breaking Changes to Watch For

1. **ufcstats.com HTML Structure:**
   If site redesigns, scraper CSS selectors break. Check lines 416-437 (winner detection), 386-389 (fight table), 529-646 (detailed stats tables).

2. **Pandas API Changes:**
   Feature engineering uses `.rolling()` with datetime index. Pandas 2.x changed behavior. If upgrading pandas, test rolling window calculations.

3. **Scikit-Learn Model Compatibility:**
   Saved .pkl models are version-specific. If upgrading sklearn/xgboost, re-train models (loading old models may crash).

---

## Quick Reference

### Run Scraper (Full Data)
```bash
python scrape_full_data_v1.py
# Runtime: ~7-8 hours
# Output: data/raw/ufc_fights_v1.csv (8,520 fights, 7 MB)
```

### Run Feature Engineering
```bash
python run_feature_engineering.py
# Runtime: ~28 seconds
# Output: data/processed/ufc_fights_features_v1.csv (145 features)
```

### Run Model Training
```bash
python src/models/train.py                # Full hyperparameter tuning (~1 min)
python src/models/train.py --skip-tuning  # Quick training with defaults (~10s)
# Output: data/models/ufc_model_v1.pkl, evaluation_results_v1.json
```

### Run Predictions
```bash
python src/predict/serve.py                    # Predict next upcoming event
python src/predict/serve.py --backtest 2026-01-31  # Backtest against historical event
python src/predict/serve.py --format json      # Output JSON instead of CSV
# Output: data/predictions/predictions_[event]_[date].csv
```

### Run Web App
```bash
python src/web/app.py
# → http://127.0.0.1:5000
# Routes: / (home), /event/<slug>, /archive, /backtest/<date>
# API: /api/predict/next, /api/predict/<slug>
```

### Test Scraper (Small Sample)
```python
from src.etl.scraper import UFCDataScraper

with UFCDataScraper() as scraper:
    df = scraper.scrape_historical_fights(limit_events=5)
    print(f"Scraped {len(df)} fights")
```

### Load Processed Data
```python
import pandas as pd

df = pd.read_csv('data/processed/ufc_fights_features_v1.csv')
print(df.shape)  # (8520, 145)
print(df['f1_is_winner'].value_counts())  # Target distribution
```

### Check Git Status Before Commit
```bash
git status
# Verify NO data files are staged (should show "Untracked files" if new data)
git add src/features/engineer.py  # Add specific files only
```

---

## When Things Break

1. **Scraper returns empty dataframes:**
   → Check ufcstats.com is online
   → Verify CSS selectors haven't changed (use browser DevTools)
   → Check rate limiting (too fast = blocked)

2. **Feature engineering crashes with KeyError:**
   → Raw data columns missing (re-scrape)
   → Check [ufc_fights_v1.csv](data/raw/ufc_fights_v1.csv) has expected 58 columns

3. **Feature values look wrong (all zeros, all NaNs):**
   → Check data is sorted by date BEFORE feature engineering
   → Verify `.shift(1)` is used in rolling calculations
   → Check for data leakage (using current fight to predict itself)

4. **Model training will crash with "feature mismatch":**
   → Feature names changed after model was saved
   → Re-train model with new features
   → Check `feature_config_v1.json` matches current feature names

---

**Read ARCHITECTURE.md for comprehensive project overview.**
**Read DATA_DICTIONARY.md for feature definitions.**
**Read CHANGELOG.md for recent changes.**
