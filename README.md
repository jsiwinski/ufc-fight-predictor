# UFC Fight Predictor

A machine learning system for predicting UFC fight outcomes using historical fight data from ufcstats.com. Built with Python, XGBoost, and Flask.

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1: Data Collection** | **Complete** | Scraper collects 8,520 fights from 760 events (1994-2026) |
| **Phase 2: Feature Engineering** | **Complete** | 145 ML-ready features generated per fight |
| **Phase 3: Model Training** | Planned | XGBoost classifier with temporal train/test split |
| **Phase 4: Web Interface** | Planned | Flask app displaying predictions with explanations |

## Quick Start

### Installation

```bash
# Clone and setup
cd ufc-fight-predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Feature Engineering

```bash
python run_feature_engineering.py
# Output: data/processed/ufc_fights_features_v1.csv (8,520 fights × 145 features)
```

### Scrape Fresh Data (Optional)

```bash
python scrape_full_data_v1.py
# Runtime: ~7-8 hours for complete UFC history
```

## Documentation

| Document | Purpose |
|----------|---------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture, data flow diagrams, module map |
| **[CLAUDE.md](CLAUDE.md)** | Session starter guide — conventions, gotchas, rules |
| **[DATA_DICTIONARY.md](DATA_DICTIONARY.md)** | All 145 features with definitions, types, rationale |
| **[CHANGELOG.md](CHANGELOG.md)** | Development history and known issues |
| **[docs/SCRAPER_DOCUMENTATION.md](docs/SCRAPER_DOCUMENTATION.md)** | Complete scraper API reference |
| **[docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)** | Raw data schema (77 columns) |
| **[docs/ufc_fight_model_methodology.html](docs/ufc_fight_model_methodology.html)** | Visual methodology document |

## Project Structure

```
ufc-fight-predictor/
├── src/
│   ├── etl/
│   │   └── scraper.py          # UFC data scraper (1,253 lines)
│   ├── features/
│   │   └── engineer.py         # Feature engineering (836 lines)
│   ├── models/
│   │   ├── train.py            # Model training (planned)
│   │   └── predict.py          # Prediction generation (planned)
│   └── web/
│       └── app.py              # Flask app (planned)
├── data/
│   ├── raw/                    # Scraped data (not in git)
│   ├── processed/              # Feature-engineered data (not in git)
│   ├── models/                 # Saved models (not in git)
│   └── predictions/            # Generated predictions (not in git)
├── docs/                       # Documentation
├── tests/                      # Unit tests
├── examples/                   # Usage examples
├── notebooks/                  # Jupyter notebooks
├── .githooks/                  # Git hooks (auto-changelog)
├── config.yaml                 # Central configuration
├── requirements.txt            # Python dependencies
└── run.py                      # CLI entry point
```

## Data Pipeline

```
ufcstats.com → Scraper → Raw CSV → Feature Engineer → Processed CSV → Model → Predictions
                          ↓                              ↓
                    8,520 fights                   145 features
                    77 columns                     per fight
```

### Features Generated

- **Career Statistics (16)**: Win rate, KO rate, submission rate, fight count
- **Rolling Windows (40)**: Last 3/5/10 fight performance metrics
- **Matchup Differentials (37)**: Fighter A vs Fighter B comparisons
- **Style Indicators (6)**: Striker vs grappler tendencies
- **Temporal (4)**: Year, month, layoff indicators
- **Weight Class (15)**: One-hot encoded divisions

### Target Variable

- `f1_is_winner`: Binary (0/1), 48.9%/51.1% class balance

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  raw_path: 'data/raw'
  processed_path: 'data/processed'

scraper:
  rate_limit: 2  # seconds between requests
  max_retries: 3

model:
  type: 'xgboost'
  test_split_year: 2024
```

## Development

### Enable Auto-Changelog Hook

```bash
git config core.hooksPath .githooks
chmod +x .githooks/post-commit
```

### Run Tests

```bash
pytest tests/
```

### Commit Convention

```bash
git commit -m "[CATEGORY] Description"
# Categories: [DATA], [FEATURE], [MODEL], [FIX], [INFRA], [DOCS]
```

## Known Limitations

- **No live data**: Historical fights only, no real-time odds
- **Pre-2009 gaps**: 2,406 early fights lack detailed strike statistics
- **Scraping time**: Full data collection takes 7-8 hours (rate-limited)

## License

MIT License

## Acknowledgments

- [ufcstats.com](http://www.ufcstats.com) for fight statistics
- [UFC-DataLab](https://github.com/komaksym/UFC-DataLab) for schema reference
- Built with Claude Opus 4.5
