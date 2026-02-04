# UFC Data Schema Documentation

## Overview

This document describes the data schema used by the UFC scraper, based on analysis of the [UFC-DataLab repository](https://github.com/komaksym/UFC-DataLab).

## Data Sources

### Primary Source: ufcstats.com
- **Historical Fight Data**: Completed UFC events with round-by-round statistics
- **Fighter Profiles**: Biographical data and career averages
- **Coverage**: All UFC events from UFC 1 (1993) to present

### Data Limitations
- ✅ Complete historical fight data
- ✅ Fighter biographical and career stats
- ❌ No upcoming fight data (historical only)
- ❌ No live odds or betting data
- ❌ No training camp or injury information

## Fight Data Schema

### Complete Column List (77+ columns)

```
Event Information:
├── event_name               # UFC 308: Topuria vs Holloway
├── event_date               # October 26, 2024
├── event_location           # Abu Dhabi, United Arab Emirates
└── fight_url                # URL to fight details page

Fighter Identification:
├── red_fighter_name         # Fighter in red corner
├── red_fighter_nickname     # Fighter nickname
├── blue_fighter_name        # Fighter in blue corner
└── blue_fighter_nickname    # Fighter nickname

Fight Outcome:
├── red_fighter_result       # W/L/D/NC
├── blue_fighter_result      # W/L/D/NC
├── method                   # KO/TKO, Submission, Decision, etc.
├── round                    # Round number (1-5)
├── time                     # Time of finish (MM:SS)
├── time_format              # 3x5 or 5x5 (title fights)
├── referee                  # Referee name
├── bout_type                # Title Bout, Main Event, etc.
├── bonus                    # FOTN, POTN, etc.
├── details                  # Additional details
├── weight_class             # Featherweight, Welterweight, etc.
└── fight_id                 # Unique identifier

Total Fight Statistics (per fighter):
├── *_fighter_KD             # Knockdowns
├── *_fighter_sig_str        # Significant strikes (e.g., "75 of 144")
├── *_fighter_sig_str_pct    # Sig. strike accuracy (e.g., "52%")
├── *_fighter_total_str      # Total strikes (e.g., "100 of 200")
├── *_fighter_TD             # Takedowns (e.g., "3 of 8")
├── *_fighter_TD_pct         # Takedown accuracy (e.g., "37%")
├── *_fighter_sub_att        # Submission attempts
├── *_fighter_rev            # Reversals
└── *_fighter_ctrl           # Control time (e.g., "3:45")

Significant Strikes by Target (per fighter):
├── *_fighter_sig_str_head       # Head strikes (landed of attempted)
├── *_fighter_sig_str_head_pct   # Head strike accuracy
├── *_fighter_sig_str_body       # Body strikes (landed of attempted)
├── *_fighter_sig_str_body_pct   # Body strike accuracy
├── *_fighter_sig_str_leg        # Leg strikes (landed of attempted)
└── *_fighter_sig_str_leg_pct    # Leg strike accuracy

Significant Strikes by Position (per fighter):
├── *_fighter_sig_str_distance       # Distance strikes
├── *_fighter_sig_str_distance_pct   # Distance strike accuracy
├── *_fighter_sig_str_clinch         # Clinch strikes
├── *_fighter_sig_str_clinch_pct     # Clinch strike accuracy
├── *_fighter_sig_str_ground         # Ground strikes
└── *_fighter_sig_str_ground_pct     # Ground strike accuracy
```

*Note: `*` represents either `red` or `blue` for respective fighter statistics*

### Data Format Examples

#### Event Information
```
event_name: "UFC 308: Topuria vs Holloway"
event_date: "October 26, 2024"
event_location: "Abu Dhabi, United Arab Emirates"
```

#### Fighter Information
```
red_fighter_name: "ILIA TOPURIA"
red_fighter_nickname: "El Matador"
blue_fighter_name: "MAX HOLLOWAY"
blue_fighter_nickname: "Blessed"
```

#### Fight Outcome
```
method: "KO/TKO"
round: "3"
time: "1:34"
red_fighter_result: "W"
blue_fighter_result: "L"
```

#### Statistics (Format: "landed of attempted")
```
red_fighter_sig_str: "75 of 144"
red_fighter_sig_str_pct: "52%"
red_fighter_TD: "3 of 8"
red_fighter_TD_pct: "37%"
red_fighter_ctrl: "3:45"
```

## Fighter Profile Schema

### Complete Column List (14 columns)

```
Biographical Data:
├── fighter_name     # Full name (e.g., "Conor McGregor")
├── Height           # Height (e.g., "5'9\"")
├── Weight           # Weight class (e.g., "155 lbs")
├── Reach            # Reach in inches (e.g., "74\"")
├── Stance           # Orthodox/Southpaw/Switch
└── DOB              # Date of birth (e.g., "July 14, 1988")

Career Statistics:
├── SLpM             # Significant Strikes Landed per Minute
├── Str_Acc          # Striking Accuracy (e.g., "43%")
├── SApM             # Significant Strikes Absorbed per Minute
├── Str_Def          # Striking Defense (e.g., "52%")
├── TD_Avg           # Average Takedowns per 15 minutes
├── TD_Acc           # Takedown Accuracy (e.g., "47%")
├── TD_Def           # Takedown Defense (e.g., "84%")
└── Sub_Avg          # Submission Attempts per 15 minutes
```

### Data Format Examples

```
fighter_name: "Conor McGregor"
Height: "5'9\""
Weight: "155 lbs"
Reach: "74\""
Stance: "Southpaw"
DOB: "July 14, 1988"
SLpM: "5.32"
Str_Acc: "43%"
SApM: "3.57"
Str_Def: "52%"
TD_Avg: "0.64"
TD_Acc: "47%"
TD_Def: "84%"
Sub_Avg: "0.2"
```

## Data Quality Considerations

### Missing Values
Some fields may be empty for various reasons:
- **Nicknames**: Not all fighters have nicknames
- **Reach**: Historical data may be incomplete
- **DOB**: Older fighters may not have recorded birthdates
- **Position Stats**: Very short fights may have incomplete positional data

### Data Cleaning Recommendations

1. **Strike Data**: Parse "landed of attempted" format
   ```python
   "75 of 144" → landed=75, attempted=144, accuracy=52.08%
   ```

2. **Control Time**: Convert to seconds for analysis
   ```python
   "3:45" → 225 seconds
   ```

3. **Height/Reach**: Standardize units
   ```python
   "5'9\"" → 69 inches or 175.26 cm
   "74\"" → 74 inches or 187.96 cm
   ```

4. **Percentages**: Remove '%' and convert to float
   ```python
   "52%" → 0.52 or 52.0
   ```

5. **Fighter Names**: Normalize capitalization
   ```python
   "CONOR McGREGOR" → "Conor McGregor"
   ```

## Schema Comparison with UFC-DataLab

### Similarities ✓
- ✅ Same data source (ufcstats.com)
- ✅ Same 77-column structure for fight data
- ✅ Same 14-column structure for fighter profiles
- ✅ Identical field names and formats
- ✅ Same "landed of attempted" format for strikes

### Enhancements in Our Implementation ⭐
- ⭐ Better error handling with retry logic
- ⭐ Progress tracking and logging
- ⭐ Data validation functionality
- ⭐ Context manager support
- ⭐ Configurable rate limiting
- ⭐ Utility functions for data parsing
- ⭐ Comprehensive documentation
- ⭐ Test scripts and examples

## Feature Engineering for ML Models

### Primary Features (Directly Available)

#### Physical Attributes
```python
features = [
    'Height',           # Raw height value
    'Weight',           # Weight class
    'Reach',            # Arm reach
    'height_diff',      # Height differential between fighters
    'reach_diff',       # Reach advantage
]
```

#### Striking Metrics
```python
features = [
    'SLpM',             # Strikes landed per minute
    'Str_Acc',          # Striking accuracy
    'SApM',             # Strikes absorbed per minute
    'Str_Def',          # Striking defense
    'sig_str_head_pct', # Head strike accuracy
    'sig_str_body_pct', # Body strike accuracy
    'sig_str_leg_pct',  # Leg strike accuracy
]
```

#### Grappling Metrics
```python
features = [
    'TD_Avg',           # Takedowns per 15 min
    'TD_Acc',           # Takedown accuracy
    'TD_Def',           # Takedown defense
    'Sub_Avg',          # Submission attempts per 15 min
]
```

#### Fight-Level Stats
```python
features = [
    'sig_str_distance', # Distance striking preference
    'sig_str_clinch',   # Clinch work
    'sig_str_ground',   # Ground striking
    'ctrl',             # Control time
    'KD',               # Knockdown power
]
```

### Derived Features (Calculated)

#### Historical Performance
```python
# Calculate from fight history
features = [
    'win_rate',              # Overall win percentage
    'finish_rate',           # KO/Sub percentage
    'recent_form',           # Last 3-5 fights
    'ko_power',              # KO/TKO rate
    'sub_threat',            # Submission rate
    'decision_wins',         # Decision win rate
]
```

#### Matchup-Specific
```python
# Calculate for each fight
features = [
    'striking_differential',  # SLpM difference
    'defense_differential',   # Defense stat differences
    'grappling_advantage',    # TD stats comparison
    'experience_gap',         # Number of UFC fights
    'age_at_fight',          # Calculate from DOB and event_date
]
```

#### Position Preferences
```python
# Analyze strike distribution
features = [
    'distance_fighter',      # Prefers distance (>50% strikes)
    'clinch_fighter',        # Prefers clinch
    'ground_fighter',        # Prefers ground
    'volume_striker',        # High strike volume
    'accuracy_striker',      # High accuracy, lower volume
]
```

### Example Feature Calculation

```python
import pandas as pd
from datetime import datetime

def calculate_features(fight_row, fighter_profile, fight_history):
    """
    Calculate ML features for a single fight.

    Args:
        fight_row: Current fight data
        fighter_profile: Fighter's biographical data
        fight_history: Fighter's previous fights

    Returns:
        Dictionary of features
    """
    features = {}

    # Physical features
    features['reach'] = extract_inches(fighter_profile['Reach'])
    features['height'] = extract_inches(fighter_profile['Height'])

    # Career stats
    features['str_acc'] = parse_percentage(fighter_profile['Str_Acc'])
    features['td_acc'] = parse_percentage(fighter_profile['TD_Acc'])

    # Historical performance
    recent_fights = fight_history.head(5)
    features['recent_win_rate'] = (recent_fights['result'] == 'W').mean()
    features['recent_finish_rate'] = (
        recent_fights['method'].isin(['KO/TKO', 'Submission'])
    ).mean()

    # Fight-level aggregates
    features['avg_sig_str'] = recent_fights['sig_str_landed'].mean()
    features['avg_takedowns'] = recent_fights['TD_landed'].mean()

    return features

def parse_percentage(pct_string):
    """Convert '52%' to 0.52"""
    return float(pct_string.strip('%')) / 100

def extract_inches(measurement):
    """Convert '5'9\"' or '74\"' to inches"""
    if 'ft' in measurement or "'" in measurement:
        # Handle feet and inches
        match = re.match(r"(\d+)'(\d+)", measurement)
        if match:
            return int(match.group(1)) * 12 + int(match.group(2))
    elif '"' in measurement:
        return int(measurement.strip('"'))
    return 0
```

## Data Update Strategy

### Initial Load
1. Scrape all historical fights (~700 events)
2. Scrape all fighter profiles (~3000 fighters)
3. Save to `data/raw/ufc_fights_raw.csv` and `ufc_fighters_raw.csv`

### Regular Updates
```python
from datetime import datetime, timedelta

# Get date of last scraped event
last_date = df['event_date'].max()

# Scrape only new events
new_fights = scraper.scrape_historical_fights(
    start_date=last_date
)

# Append to existing data
combined = pd.concat([df, new_fights], ignore_index=True)
```

### Recommended Update Frequency
- **Fight Data**: Weekly (new events typically every 1-2 weeks)
- **Fighter Profiles**: Monthly (stats update slowly)
- **Full Re-scrape**: Quarterly (to catch corrections)

## Data Storage Recommendations

### CSV Files (Current)
```
data/raw/
├── ufc_fights_raw.csv        # ~20-30 MB for full history
└── ufc_fighters_raw.csv      # ~500 KB for all fighters
```

### Database Schema (Recommended for Production)

```sql
-- Events table
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY,
    event_name TEXT,
    event_date DATE,
    event_location TEXT
);

-- Fighters table
CREATE TABLE fighters (
    fighter_id INTEGER PRIMARY KEY,
    fighter_name TEXT UNIQUE,
    height TEXT,
    weight TEXT,
    reach TEXT,
    stance TEXT,
    dob DATE,
    slpm REAL,
    str_acc REAL,
    sapm REAL,
    str_def REAL,
    td_avg REAL,
    td_acc REAL,
    td_def REAL,
    sub_avg REAL
);

-- Fights table
CREATE TABLE fights (
    fight_id INTEGER PRIMARY KEY,
    event_id INTEGER,
    red_fighter_id INTEGER,
    blue_fighter_id INTEGER,
    winner_id INTEGER,
    method TEXT,
    round INTEGER,
    time TEXT,
    -- ... (all statistics columns)
    FOREIGN KEY (event_id) REFERENCES events(event_id),
    FOREIGN KEY (red_fighter_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (blue_fighter_id) REFERENCES fighters(fighter_id)
);
```

## Usage Examples

### Load and Explore Data
```python
import pandas as pd

# Load data
fights = pd.read_csv('data/raw/ufc_fights_raw.csv')
fighters = pd.read_csv('data/raw/ufc_fighters_raw.csv')

# Basic exploration
print(f"Total fights: {len(fights)}")
print(f"Date range: {fights['event_date'].min()} to {fights['event_date'].max()}")
print(f"Total fighters: {len(fighters)}")

# Most common methods
print(fights['method'].value_counts())

# Average strikes per fight
fights['red_sig_str_landed'] = fights['red_fighter_sig_str'].str.split(' of ').str[0].astype(int)
print(f"Average significant strikes: {fights['red_sig_str_landed'].mean():.1f}")
```

### Merge Fighter Profiles with Fights
```python
# Add fighter profiles to fight data
fights_enhanced = fights.merge(
    fighters,
    left_on='red_fighter_name',
    right_on='fighter_name',
    how='left',
    suffixes=('', '_red_profile')
).merge(
    fighters,
    left_on='blue_fighter_name',
    right_on='fighter_name',
    how='left',
    suffixes=('', '_blue_profile')
)

# Calculate reach advantage
fights_enhanced['reach_advantage'] = (
    fights_enhanced['Reach'].str.extract('(\d+)').astype(float) -
    fights_enhanced['Reach_blue_profile'].str.extract('(\d+)').astype(float)
)
```

## Related Documentation

- [Scraper Documentation](SCRAPER_DOCUMENTATION.md) - How to use the scraper
- [UFC-DataLab Repository](https://github.com/komaksym/UFC-DataLab) - Reference implementation
- [UFCStats.com](http://www.ufcstats.com) - Original data source

## Notes

- All dates use "Month DD, YYYY" format (e.g., "October 26, 2024")
- Fighter names are in UPPERCASE in raw data (should be cleaned)
- Strike data uses "landed of attempted" format consistently
- Percentages include '%' symbol (should be parsed)
- Missing values represented as empty strings or '--'
- Control time uses "MM:SS" format
