# UFC Data Scraper Documentation

## Overview

The UFC Data Scraper is a comprehensive tool for collecting historical UFC fight data and fighter profiles from ufcstats.com. It's designed specifically for building machine learning models for fight prediction.

## Features

### Data Collection
- **Historical Fight Data**: Complete fight records with 77+ statistical columns
- **Fighter Profiles**: Biographical data and career statistics for all fighters
- **Event Information**: Event names, dates, locations, and bout details

### Technical Features
- **Respectful Scraping**: Configurable rate limiting (default: 2s between requests)
- **Robust Error Handling**: Automatic retry logic with exponential backoff
- **Progress Tracking**: Detailed logging for long scraping sessions
- **Data Validation**: Built-in validation and cleaning functionality
- **Modular Design**: Separate functions for events, fights, and fighters

## Data Schema

### Fight Data (77 columns)

#### Event Information
- `event_name`: Name of the UFC event
- `event_date`: Date of the event (format: "Month DD, YYYY")
- `event_location`: City and venue location

#### Fighter Information
- `red_fighter_name`: Name of the red corner fighter
- `blue_fighter_name`: Name of the blue corner fighter
- `red_fighter_nickname`: Red corner fighter's nickname
- `blue_fighter_nickname`: Blue corner fighter's nickname

#### Fight Outcome
- `red_fighter_result`: Win/Loss/Draw for red corner
- `blue_fighter_result`: Win/Loss/Draw for blue corner
- `method`: Method of victory (KO/TKO, Submission, Decision, etc.)
- `round`: Round in which fight ended
- `time`: Time of finish (MM:SS format)
- `time_format`: Format (3x5, 5x5 for title fights)
- `referee`: Name of the referee
- `bout_type`: Type of bout (title, main event, etc.)
- `bonus`: Performance bonuses awarded
- `details`: Additional fight details
- `weight_class`: Weight class of the fight
- `fight_id`: Unique identifier for the fight
- `fight_url`: URL to the fight details page

#### Total Statistics (per fighter)
- `*_fighter_KD`: Knockdowns
- `*_fighter_sig_str`: Significant strikes (landed of attempted)
- `*_fighter_sig_str_pct`: Significant strike accuracy percentage
- `*_fighter_total_str`: Total strikes (landed of attempted)
- `*_fighter_TD`: Takedowns (successful of attempted)
- `*_fighter_TD_pct`: Takedown accuracy percentage
- `*_fighter_sub_att`: Submission attempts
- `*_fighter_rev`: Reversals
- `*_fighter_ctrl`: Control time (MM:SS)

#### Significant Strikes by Target (per fighter)
- `*_fighter_sig_str_head`: Head strikes (landed of attempted)
- `*_fighter_sig_str_head_pct`: Head strike accuracy
- `*_fighter_sig_str_body`: Body strikes (landed of attempted)
- `*_fighter_sig_str_body_pct`: Body strike accuracy
- `*_fighter_sig_str_leg`: Leg strikes (landed of attempted)
- `*_fighter_sig_str_leg_pct`: Leg strike accuracy

#### Significant Strikes by Position (per fighter)
- `*_fighter_sig_str_distance`: Distance strikes (landed of attempted)
- `*_fighter_sig_str_distance_pct`: Distance strike accuracy
- `*_fighter_sig_str_clinch`: Clinch strikes (landed of attempted)
- `*_fighter_sig_str_clinch_pct`: Clinch strike accuracy
- `*_fighter_sig_str_ground`: Ground strikes (landed of attempted)
- `*_fighter_sig_str_ground_pct`: Ground strike accuracy

*Note: Replace `*` with `red` or `blue` for respective fighter stats*

### Fighter Profile Data (14 columns)

#### Biographical Information
- `fighter_name`: Fighter's full name
- `Height`: Height (e.g., "5'10\"")
- `Weight`: Weight class (e.g., "155 lbs")
- `Reach`: Reach in inches (e.g., "74\"")
- `Stance`: Fighting stance (Orthodox, Southpaw, Switch)
- `DOB`: Date of birth (format: "Month DD, YYYY")

#### Career Statistics
- `SLpM`: Significant Strikes Landed per Minute
- `Str_Acc`: Striking Accuracy (percentage)
- `SApM`: Significant Strikes Absorbed per Minute
- `Str_Def`: Striking Defense (percentage)
- `TD_Avg`: Average Takedowns Landed per 15 minutes
- `TD_Acc`: Takedown Accuracy (percentage)
- `TD_Def`: Takedown Defense (percentage)
- `Sub_Avg`: Average Submissions Attempted per 15 minutes

## Installation

```python
# The scraper uses standard libraries plus:
pip install requests beautifulsoup4 pandas urllib3
```

## Usage

### Basic Usage - Scrape All Data

```python
from src.etl.scraper import UFCDataScraper

# Initialize scraper
scraper = UFCDataScraper()

# Scrape everything (fights + fighters)
output_files = scraper.scrape_all_data(output_dir='data/raw')

print(f"Fights: {output_files['fights']}")
print(f"Fighters: {output_files['fighters']}")

scraper.close()
```

### Using Context Manager (Recommended)

```python
from src.etl.scraper import UFCDataScraper

with UFCDataScraper() as scraper:
    # Scraper automatically closes when done
    output_files = scraper.scrape_all_data()
```

### Scrape Recent Fights Only

```python
with UFCDataScraper() as scraper:
    df = scraper.scrape_historical_fights(
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    scraper.export_to_csv(df, 'data/raw/fights_2024.csv')
```

### Test with Limited Data

```python
with UFCDataScraper() as scraper:
    # Only scrape 5 most recent events
    df = scraper.scrape_historical_fights(limit_events=5)
    print(f"Scraped {len(df)} fights")
```

### Scrape Specific Fighter

```python
with UFCDataScraper() as scraper:
    stats = scraper.scrape_fighter_stats("Conor McGregor")
    if stats:
        print(f"Height: {stats['Height']}")
        print(f"Reach: {stats['Reach']}")
        print(f"Striking Accuracy: {stats['Str_Acc']}")
```

### Scrape All Fighters

```python
with UFCDataScraper() as scraper:
    fighters_df = scraper.scrape_all_fighters()
    scraper.export_to_csv(fighters_df, 'data/raw/all_fighters.csv')
```

### Custom Configuration

```python
config = {
    'rate_limit': 3,      # Wait 3 seconds between requests
    'timeout': 15,        # 15 second timeout
    'max_retries': 5,     # Retry 5 times on failure
    'user_agent': 'Custom User Agent String'
}

scraper = UFCDataScraper(config)
```

## Data Validation

```python
with UFCDataScraper() as scraper:
    df = scraper.scrape_historical_fights(limit_events=10)

    # Validate and get report
    validated_df, report = scraper.validate_fight_data(df)

    print(f"Total rows: {report['total_rows']}")
    print(f"Duplicates removed: {report['duplicates']}")
    print(f"Data quality score: {report['data_quality_score']:.1f}%")
    print(f"Missing values: {report['missing_values']}")
```

## Utility Functions

The scraper includes utility functions for data processing:

```python
from src.etl.scraper import (
    parse_strike_data,
    calculate_strike_percentage,
    parse_control_time,
    clean_fighter_name
)

# Parse strike data
landed, attempted = parse_strike_data("25 of 50")  # Returns (25, 50)

# Calculate accuracy
accuracy = calculate_strike_percentage("25 of 50")  # Returns 50.0

# Parse control time
seconds = parse_control_time("3:45")  # Returns 225

# Clean fighter names
name = clean_fighter_name("  CONOR McGREGOR  ")  # Returns "Conor McGregor"
```

## Performance Considerations

### Scraping Time Estimates

With default rate limit of 2 seconds between requests:

- **Single Event**: ~30-60 seconds (10-15 fights)
- **10 Events**: ~5-10 minutes
- **100 Events**: ~50-100 minutes
- **All Events** (~700): ~20-30 hours
- **All Fighters** (~3000): ~2-3 hours

### Recommendations

1. **For Development/Testing**: Use `limit_events=5` parameter
2. **For Production**: Run overnight or in batches
3. **For Updates**: Use `start_date` to only scrape recent events
4. **Rate Limiting**: Keep at 2+ seconds to be respectful to ufcstats.com

### Memory Usage

- Fight data: ~500KB per 100 fights
- Fighter data: ~200KB per 1000 fighters
- Total for all UFC history: ~20-30 MB

## Error Handling

The scraper includes comprehensive error handling:

```python
try:
    with UFCDataScraper() as scraper:
        df = scraper.scrape_historical_fights()
except KeyboardInterrupt:
    print("Scraping interrupted by user")
except Exception as e:
    print(f"Error: {e}")
    # Partial data may still be available
```

### Common Issues

1. **Request Timeout**: Increase `timeout` in config
2. **Rate Limiting**: Server may block if rate_limit is too low
3. **Network Errors**: Automatic retry logic handles transient failures
4. **Invalid Data**: Validation functions detect and report issues

## ML Model Integration

The scraped data is designed for machine learning models:

### Feature Engineering Ideas

1. **Fighter Statistics**
   - Win/loss ratio
   - Average strikes per minute
   - Takedown success rate
   - Submission tendency

2. **Fight Metrics**
   - Strike differential
   - Control time advantage
   - Strike distribution (head/body/leg)
   - Position preference (distance/clinch/ground)

3. **Physical Attributes**
   - Height differential
   - Reach advantage
   - Weight class
   - Age at time of fight

4. **Historical Performance**
   - Recent form (last 3-5 fights)
   - Opponent quality
   - Method of victories/defeats
   - Performance by position

### Example: Create Training Features

```python
import pandas as pd

# Load data
fights_df = pd.read_csv('data/raw/ufc_fights_raw.csv')
fighters_df = pd.read_csv('data/raw/ufc_fighters_raw.csv')

# Merge fighter data into fights
fights_with_stats = fights_df.merge(
    fighters_df,
    left_on='red_fighter_name',
    right_on='fighter_name',
    suffixes=('', '_red')
).merge(
    fighters_df,
    left_on='blue_fighter_name',
    right_on='fighter_name',
    suffixes=('', '_blue')
)

# Create features
fights_with_stats['reach_advantage'] = (
    fights_with_stats['Reach'].str.extract('(\d+)').astype(float) -
    fights_with_stats['Reach_blue'].str.extract('(\d+)').astype(float)
)

# Calculate strike success rates
def extract_landed(strike_str):
    if pd.isna(strike_str):
        return 0
    match = strike_str.split(' of ')
    return int(match[0]) if len(match) == 2 else 0

fights_with_stats['red_sig_str_landed'] = (
    fights_with_stats['red_fighter_sig_str'].apply(extract_landed)
)
```

## API Reference

### UFCDataScraper Class

#### `__init__(config: Optional[Dict] = None)`
Initialize the scraper with optional configuration.

#### `scrape_all_data(output_dir: str, limit_events: Optional[int] = None) -> Dict[str, str]`
Main entry point. Scrapes fights and fighters, returns file paths.

#### `scrape_historical_fights(start_date: Optional[str], end_date: Optional[str], limit_events: Optional[int]) -> pd.DataFrame`
Scrape historical fight data with optional filtering.

#### `scrape_all_fighters() -> pd.DataFrame`
Scrape biographical and career stats for all fighters.

#### `scrape_fighter_stats(fighter_name: str) -> Optional[Dict]`
Get stats for a specific fighter by name.

#### `validate_fight_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]`
Validate and clean fight data, return cleaned data and report.

#### `export_to_csv(df: pd.DataFrame, filepath: str, validate: bool = True) -> str`
Export DataFrame to CSV with optional validation.

#### `close()`
Close session and cleanup resources.

## Data Source

All data is scraped from [ufcstats.com](http://www.ufcstats.com), the official UFC statistics provider.

### Data Limitations

1. **Historical Only**: No upcoming fight data available
2. **Completed Events**: Only shows finished fights
3. **Official Stats**: May not include unofficial/removed fights
4. **Updates**: New events appear after completion (usually 1-2 days)

## Best Practices

1. **Rate Limiting**: Always use appropriate rate limiting (2+ seconds)
2. **Error Handling**: Wrap scraping in try/except blocks
3. **Testing**: Use `limit_events` parameter during development
4. **Incremental Updates**: Use date filters to only scrape new data
5. **Data Validation**: Always validate scraped data before training models
6. **Context Manager**: Use `with` statement for automatic cleanup
7. **Logging**: Enable logging to monitor progress and debug issues

## Example Workflow

```python
import logging
from src.etl.scraper import UFCDataScraper
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)

# 1. Initial full scrape
print("Step 1: Scraping all historical data...")
with UFCDataScraper() as scraper:
    files = scraper.scrape_all_data(output_dir='data/raw')

# 2. Load and validate
print("Step 2: Loading and validating data...")
fights_df = pd.read_csv(files['fights'])
fighters_df = pd.read_csv(files['fighters'])

print(f"Loaded {len(fights_df)} fights and {len(fighters_df)} fighters")

# 3. Basic data exploration
print(f"\nFight data columns: {fights_df.shape[1]}")
print(f"Fighter data columns: {fighters_df.shape[1]}")

# 4. Check data quality
print(f"\nMissing values in fight data:")
print(fights_df.isnull().sum().sort_values(ascending=False).head(10))

# 5. Ready for feature engineering and model training!
```

## Support and References

- **Repository**: [UFC-DataLab](https://github.com/komaksym/UFC-DataLab) (reference implementation)
- **Data Source**: [UFCStats.com](http://www.ufcstats.com)
- **UFC API**: No official API; this scraper uses web scraping

## License

This scraper is for educational and research purposes. Always respect the terms of service of data sources.
