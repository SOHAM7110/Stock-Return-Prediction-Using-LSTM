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

### Stage 4 — LSTM Model Architecture + Training
Stock Return Prediction | Quant Research Roadmap — Tier 2
 
Loads sequences from Stage 3 (main.py → load_sequences())   
Builds, trains, and saves two models per ticker:   
  - Regression  : predicts next-day log return (float)    
  - Classifier  : predicts direction (1=up, 0=down)   
 
Both models share the same LSTM backbone — only the output head differs.   
Train both and compare: sometimes direction accuracy matters more than   
minimising return error, depending on your trading strategy.  
 
Output:   
    models/TICKER_regression.keras    
    models/TICKER_classifier.keras   
    models/TICKER_history.csv       ← loss curves for plotting

### Stage 5 — Quant Evaluation Suite
Stock Return Prediction | Quant Research Roadmap — Tier 2
 
Loads trained models from Stage 4, runs predictions on the test set,  
and evaluates them the way a quant desk would — NOT just RMSE.
 
Metrics computed:  
  ML metrics    : RMSE, MAE, R²   
  Quant metrics : Sharpe ratio, Sortino ratio, Max drawdown,  
                  Calmar ratio, Directional accuracy, Hit rate by quintile,   
                  Information Coefficient (IC), Annualised return   
 
Outputs:   
  results/TICKER_metrics.csv       ← all metrics in one row per ticker   
  results/TICKER_equity_curve.csv  ← daily P&L for plotting   
  results/summary.csv              ← cross-ticker comparison table   
 
Why these metrics matter more than RMSE:   
  A model can have low RMSE but still be useless for trading if it   
  gets direction wrong on the big move days. Sharpe ratio, directional  
  accuracy, and IC tell you whether the signal is actually tradeable.


### Stage 6 — Backtesting Engine
Stock Return Prediction | Quant Research Roadmap — Tier 2
 
Three layers of increasing realism:
 
  Layer 1 — Naive backtest         : signal × return, flat position sizing   
  Layer 2 — Realistic backtest     : transaction costs + slippage + signal threshold   
  Layer 3 — Kelly backtest         : volatility-scaled position sizing (Kelly Criterion)   
 
Walk-forward validation:   
  Rolls a training window forward in 6-month steps.   
  Retrains the model on each fold to prevent the model from using   
  a parameter set tuned on data it never should have seen.  
 
Outputs:  
  backtest/TICKER_naive.csv   
  backtest/TICKER_realistic.csv   
  backtest/TICKER_kelly.csv   
  backtest/TICKER_walkforward.csv   
  backtest/summary.csv
 
Key realistic frictions modelled:   
  - Bid-ask spread        : 0.05% per trade (NSE liquid large-caps)      
  - Market impact         : 0.03% × sqrt(order_size / avg_volume)   
  - Brokerage commission  : 0.03% per trade (Zerodha/discount broker)   
  - STT (Securities Tx Tax): 0.025% on sell side (Indian equity specific)   
  - SEBI charges          : 0.0001% per trade   
  - Slippage              : 0.02% assumed execution slippage   
  Total one-way cost estimate: ~0.13–0.15% per trade
 
Position limits:
  - Max position size     : 100% of capital (no leverage at fresher stage)   
  - Min signal threshold  : 0.0 (trade on any non-zero prediction)   
  - Kelly fraction        : half-Kelly (full Kelly is too aggressive)   
