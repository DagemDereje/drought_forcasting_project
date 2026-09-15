# Meteorological Drought Forecasting

An interactive Streamlit app that predicts short-term drought severity — the
3-month Standardized Precipitation Index (SPI-3) — from recent meteorological
station data, using a trained XGBoost regression model.

## Problem

Drought monitoring agencies and agricultural planners need early signals of
worsening dry conditions, not just after-the-fact confirmation. SPI-3 is a
standard meteorological drought indicator: negative values indicate
drier-than-normal conditions relative to the historical record, with more
negative values indicating more severe drought. This project forecasts SPI-3
from recent temperature, humidity, and rainfall history, so a forecast can be
produced before the full 3-month window of data needed to compute SPI
directly is available.

## Data

Monthly station-level meteorological records: temperature (max/min),
relative humidity, rainfall, elevation, and location, spanning multiple
years at a single station. The dataset reflects real recorded conditions
rather than synthetic values.

## Methodology

1. **Feature engineering** — lagged SPI values (t-1, t-2, t-3), lagged
   rainfall (t-1, t-2, t-3), and rolling 3-month mean/standard deviation of
   rainfall, alongside current-month temperature, humidity, and calendar
   month.
2. **Preprocessing** — `StandardScaler` applied to all numeric features
   before model input, since gradient-boosted trees still benefit from
   consistent, well-behaved feature scales during training/validation
   comparisons.
3. **Model** — `XGBRegressor`, trained to predict the continuous SPI-3
   value directly (a regression problem, not a classification of
   "drought / no drought").
4. **Interface** — a Streamlit app takes current-month readings and recent
   history as input and returns a predicted SPI-3 value, translated into a
   plain-language drought severity category (e.g. near-normal, moderate,
   severe).

## Tech stack

Python, pandas, scikit-learn, XGBoost, Streamlit.

## Results

*(Add your model's validation metrics here — e.g. RMSE/MAE on a held-out
time period, or R² — from `notebooks/01_data_preprocessing.ipynb`. Reporting
error on a time-based split, not a random split, is the more honest number
for a forecasting task like this, since it reflects predicting the future
from the past rather than interpolating within known periods.)*

## Limitations

- Trained on data from a single meteorological station; performance on
  other locations or climates is untested.
- SPI-3 forecasting from lagged features assumes recent patterns are
  informative about the near future — this breaks down around abrupt
  regime changes (e.g. an unprecedented weather event).
- This is a portfolio/research prototype, not an operational early-warning
  system — it has not been validated against real-world drought outcomes
  or reviewed by domain meteorologists.

## Run locally

```bash
cd drought_forecasting_app
pip install -r requirements.txt
streamlit run app.py
```

## Future improvements

- Multi-station training for broader geographic generalization
- Formal backtesting against historical drought events
- Confidence intervals on predictions, not just a point estimate
