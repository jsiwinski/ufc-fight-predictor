# UFC Fight Predictor

A machine learning system for predicting UFC fight outcomes with win probabilities. The system collects historical and upcoming fight data, trains predictive models, and displays predictions through a web interface.

## Project Status

This project is in **initial setup phase**. The structure and scaffolding are complete, with each component ready for phased implementation.

## Features (Planned)

- **Data ETL Pipeline**: Automated collection and processing of UFC fight data
  - Historical fight results and statistics
  - Upcoming event schedules and fight cards
  - Fighter statistics and career records
  - Automated data updates

- **Machine Learning Model**: Predictive model for fight outcomes
  - Feature engineering from fighter and fight data
  - Multiple model types (XGBoost, Random Forest, Gradient Boosting)
  - Hyperparameter tuning and cross-validation
  - Model evaluation and performance tracking

- **Prediction Engine**: Generate win probabilities for upcoming fights
  - Individual fight predictions
  - Event-level predictions
  - Confidence levels and prediction explanations
  - Historical prediction tracking

- **Web Interface**: Display predictions and statistics
  - Upcoming fight predictions
  - Event-specific fight cards
  - Fighter profiles and history
  - Prediction accuracy tracking

## Project Structure

```
ufc-fight-predictor/
├── data/
│   ├── raw/              # Raw scraped data
│   ├── processed/        # Cleaned and processed data
│   ├── models/           # Saved model files
│   └── predictions/      # Generated predictions
├── src/
│   ├── etl/              # Data collection and processing
│   │   ├── scraper.py    # Data scraping
│   │   ├── processor.py  # Data cleaning
│   │   └── updater.py    # Data updates
│   ├── features/         # Feature engineering
│   │   └── engineer.py   # Feature creation
│   ├── models/           # ML models
│   │   ├── train.py      # Model training
│   │   └── predict.py    # Prediction generation
│   └── web/              # Web interface
│       ├── app.py        # Flask application
│       ├── static/       # CSS, JS files
│       └── templates/    # HTML templates
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit tests
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
└── run.py                # Main entry point
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (venv or conda)

### Installation

1. Clone or navigate to the project directory:
```bash
cd ufc-fight-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the application:
   - Review `config.yaml` and adjust settings as needed
   - Set any required environment variables (API keys, etc.)

## Usage

The project uses a CLI interface through `run.py`:

### ETL Commands (Phase 1 - Coming Soon)

```bash
# Collect historical fight data
python run.py etl --historical

# Collect upcoming fight data
python run.py etl --upcoming

# Update all data
python run.py etl --update
```

### Training Commands (Phase 2 - Coming Soon)

```bash
# Train the model
python run.py train

# Train with hyperparameter tuning
python run.py train --tune
```

### Prediction Commands (Phase 3 - Coming Soon)

```bash
# Generate predictions for all upcoming fights
python run.py predict

# Generate predictions for a specific event
python run.py predict --event "UFC 300"
```

### Web Interface (Phase 4 - Coming Soon)

```bash
# Launch the web interface
python run.py web
```

Then visit `http://127.0.0.1:5000` in your browser.

## Development Workflow

This project will be implemented in phases:

### Phase 1: ETL Pipeline ⏳
**Status**: Not Started

Implement data collection and processing:
- [ ] Identify and configure data sources
- [ ] Implement historical fight data scraper
- [ ] Implement upcoming events scraper
- [ ] Create data cleaning and validation pipeline
- [ ] Implement data update scheduler
- [ ] Test data quality and completeness

**Files to implement**: `src/etl/scraper.py`, `src/etl/processor.py`, `src/etl/updater.py`

### Phase 2: Model Training ⏳
**Status**: Not Started (requires Phase 1)

Build and train the prediction model:
- [ ] Implement feature engineering
- [ ] Create training pipeline
- [ ] Train initial models
- [ ] Perform hyperparameter tuning
- [ ] Evaluate model performance
- [ ] Select and save best model

**Files to implement**: `src/features/engineer.py`, `src/models/train.py`

### Phase 3: Prediction Engine ⏳
**Status**: Not Started (requires Phase 2)

Generate predictions for upcoming fights:
- [ ] Implement prediction pipeline
- [ ] Add probability calibration
- [ ] Create prediction explanations
- [ ] Implement batch prediction for events
- [ ] Add confidence level calculations
- [ ] Set up prediction storage and versioning

**Files to implement**: `src/models/predict.py`

### Phase 4: Web Interface ⏳
**Status**: Not Started (requires Phase 3)

Build the web interface:
- [ ] Implement Flask routes
- [ ] Create HTML templates
- [ ] Style with CSS
- [ ] Add JavaScript for interactivity
- [ ] Implement API endpoints
- [ ] Add prediction update triggers
- [ ] Deploy application

**Files to implement**: `src/web/app.py`, templates, static files

## Configuration

The `config.yaml` file contains all configurable parameters:

- **Data sources**: URLs and paths for data collection
- **Model parameters**: Training hyperparameters
- **Storage settings**: Data and model storage locations
- **Web server**: Host, port, and debug settings
- **API keys**: External service credentials (use environment variables)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## Contributing

Each phase should be implemented in a separate session:
1. Complete implementation for the phase
2. Write tests for the new functionality
3. Update documentation as needed
4. Move to the next phase

## License

MIT License (or your preferred license)

## Acknowledgments

- UFC for fight data
- scikit-learn and XGBoost for ML frameworks
- Flask for web framework

## Notes

- This project is for educational/research purposes
- Predictions are probabilistic and should not be used for gambling
- Data scraping should respect robots.txt and terms of service
- Consider rate limiting when collecting data

## Next Steps

To begin development:
1. Start with **Phase 1: ETL Pipeline**
2. Identify specific data sources for UFC fight data
3. Implement scraping logic in `src/etl/scraper.py`
4. Test data collection and quality

See TODO comments in each module for specific implementation details.
