# UFC Fight Predictor - Data Scraper

## Quick Start

### Installation
```bash
pip install requests beautifulsoup4 pandas urllib3
```

### Run Test (2 minutes)
```bash
python test_scraper.py
```

### Scrape Sample Data (5-10 minutes)
```bash
python examples/scraper_usage.py
# Choose option 4 (test scraping)
```

### Scrape All Data (20-30 hours)
```bash
python examples/scraper_usage.py
# Choose option 1 (comprehensive scraping)
```

## What Gets Scraped

### Fight Data (77+ columns)
- **Event Info**: Name, date, location
- **Fighters**: Names, nicknames, results
- **Outcome**: Method, round, time, referee
- **Statistics**: 60+ detailed performance metrics
  - Strikes by target (head, body, leg)
  - Strikes by position (distance, clinch, ground)
  - Takedowns, submissions, knockdowns, control time

### Fighter Profiles (14 columns)
- **Physical**: Height, weight, reach, stance, DOB
- **Career Stats**: Strike accuracy, takedown stats, defense metrics

## Quick Usage

### Example 1: Scrape Everything
```python
from src.etl.scraper import UFCDataScraper

with UFCDataScraper() as scraper:
    files = scraper.scrape_all_data(output_dir='data/raw')
    # Creates: ufc_fights_raw.csv and ufc_fighters_raw.csv
```

### Example 2: Scrape Recent Fights
```python
with UFCDataScraper() as scraper:
    df = scraper.scrape_historical_fights(start_date='2024-01-01')
    scraper.export_to_csv(df, 'data/raw/recent_fights.csv')
```

### Example 3: Test with 5 Events
```python
with UFCDataScraper() as scraper:
    df = scraper.scrape_historical_fights(limit_events=5)
    print(f"Scraped {len(df)} fights")
```

### Example 4: Get Specific Fighter
```python
with UFCDataScraper() as scraper:
    stats = scraper.scrape_fighter_stats("Conor McGregor")
    print(f"Reach: {stats['Reach']}, Accuracy: {stats['Str_Acc']}")
```

## Output Files

### Location
```
data/raw/
├── ufc_fights_raw.csv      # All fight data
└── ufc_fighters_raw.csv    # All fighter profiles
```

### File Sizes
- Full fight history: ~20-30 MB
- All fighter profiles: ~500 KB

## Configuration

```python
config = {
    'rate_limit': 2,      # Seconds between requests (default: 2)
    'timeout': 10,        # Request timeout (default: 10)
    'max_retries': 3,     # Retry attempts (default: 3)
}

scraper = UFCDataScraper(config)
```

## Time Estimates

With default rate limit (2 seconds):
- **5 events**: ~5-10 minutes
- **50 events**: ~50-100 minutes
- **All events (~700)**: ~20-30 hours
- **All fighters (~3000)**: ~2-3 hours

## Features

✅ Comprehensive data (77+ columns for fights, 14 for fighters)
✅ Automatic retry logic with exponential backoff
✅ Progress tracking and detailed logging
✅ Data validation and quality scoring
✅ CSV export with optional validation
✅ Context manager for automatic cleanup
✅ Configurable rate limiting
✅ Modular design (events, fights, fighters)

## Data Quality

The scraper includes built-in validation:

```python
validated_df, report = scraper.validate_fight_data(df)

print(f"Quality score: {report['data_quality_score']:.1f}%")
print(f"Duplicates: {report['duplicates']}")
print(f"Missing values: {report['missing_values']}")
```

## Use Cases

### For Machine Learning
- Train fight prediction models
- Feature engineering from historical data
- Fighter style analysis
- Performance trend analysis

### For Analysis
- Fight statistics exploration
- Fighter comparison
- Historical trends
- Event-level analytics

## Important Notes

### Respectful Scraping
- Default 2-second rate limit (be respectful to ufcstats.com)
- Automatic retry with exponential backoff
- Request timeout handling
- Session management

### Data Limitations
- ✅ Complete historical fight data
- ✅ Fighter biographical and career stats
- ❌ No upcoming/future fight data
- ❌ No live odds or betting lines
- ❌ No training camp information

### Best Practices
1. **Testing**: Use `limit_events=5` during development
2. **Updates**: Use `start_date` to scrape only new events
3. **Monitoring**: Enable logging to track progress
4. **Storage**: Use CSV for initial storage, database for production
5. **Validation**: Always validate data before training models

## Documentation

- **[Full Documentation](SCRAPER_DOCUMENTATION.md)** - Comprehensive guide
- **[Data Schema](DATA_SCHEMA.md)** - Complete field descriptions
- **[Usage Examples](../examples/scraper_usage.py)** - Code examples
- **[Test Script](../test_scraper.py)** - Verification tests

## Reference

Based on analysis of [UFC-DataLab](https://github.com/komaksym/UFC-DataLab) repository with enhancements:
- Better error handling
- Progress tracking
- Data validation
- Comprehensive documentation
- Test coverage

## Data Source

All data scraped from [ufcstats.com](http://www.ufcstats.com) - Official UFC statistics provider.

## Support

For issues or questions:
1. Check documentation files in `docs/`
2. Run test script: `python test_scraper.py`
3. Try examples: `python examples/scraper_usage.py`
4. Review UFC-DataLab repository for reference

## Next Steps

After scraping data:
1. **Clean data**: Remove duplicates, handle missing values
2. **Feature engineering**: Create ML-ready features
3. **Exploratory analysis**: Understand patterns and distributions
4. **Model training**: Build fight prediction models
5. **Validation**: Test model performance

See `notebooks/` directory for analysis examples.
