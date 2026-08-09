# About Project

## Stages in Project :
### Stage 1 : Fetch Data

Pulls 5 years of daily OHLCV data for Nifty50 stocks from NSE via yfinance.
Part of: Stock Return Prediction using LSTM (Quant Research roadmap, Tier 2)

### Stage 2 : Feature Engineering :

Builds all input features for the LSTM stock return model.
Reads CSVs from data/raw/, writes enriched CSVs to data/features/.
 
Feature groups:
  A. Return features       — what the stock actually did
  B. Momentum indicators   — RSI, MACD (is it trending?)
  C. Volatility indicators — Bollinger Bands, ATR (how much is it moving?)
  D. Volume signals        — z-score, OBV (is smart money moving?)
  E. Market context        — beta-adjusted return vs Nifty50 (stock vs market)
  F. Target variable       — next-day log return (what we want to predict)
 
CRITICAL — Lookahead bias rule:
  Every feature at row t uses ONLY data from rows 0..t.
  Rolling windows, shifts, and indicator lookback periods all respect this.
  This is the single most common mistake in quant fresher projects —
  make sure you can explain this in an interview.

### Stage 3 : Sliding Window Sequence

This stage converts the engineered feature data from Stage 2 into sliding-window sequences suitable for LSTM training.

What it does
- Loads all *_features.csv files from data/features/.
- Uses a 60-day lookback window to create sequential input samples.
- Uses the next-day return as the prediction target (HORIZON = 1).
- Performs a strict date-based train/test split using 2024-01-01 as the cutoff.
- Builds train and test sequences independently to prevent sequences from crossing the train/test boundary.
- Validates the generated data for:
- NaN values
- Correct window size
- Consistent feature dimensions
- Saves the resulting sequences and corresponding dates to data/sequences/.
- Sequence Structure
- For each ticker, the following files are generated:
data/sequences/
  - RELIANCE_X_train.npy
  - RELIANCE_y_train.npy
  - RELIANCE_X_test.npy
  - RELIANCE_y_test.npy
  - RELIANCE_train_dates.csv
  - RELIANCE_test_dates.csv
    
The saved sequences are used as input for Stage 4 — LSTM Training.
