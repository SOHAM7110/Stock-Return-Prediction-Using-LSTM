import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import date
import os
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import fetch_data
import feature_engineering
import sliding_window_sequence
import lstm
import evaluation
import backtest


NIFTY50_SAMPLE = [                                                      
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
]   
    # Each item is a ticker Symbol
    # NSE tickers need '.NS' suffix for yfinance
    # BSE would use '.BO' instead

TICKERS     = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
]

NIFTY50_INDEX = "^NSEI"     # used as a market-wide feature

START_DATE = "2020-01-01"
END_DATE = date.today()
TEST_CUTOFF    = "2024-01-01"    # train = before, test = on/after

OUTPUT_DIR = "data/raw"     # folder where csv files will be saved

WINDOW = 60
HORIZON = 1
FETCH_PAUSE = 1.0

RAW_DIR        = "data/raw"
FEATURE_DIR    = "data/features"
SEQ_DIR        = "data/sequences"
INDEX_FILE     = os.path.join(RAW_DIR, "NSEI.csv")

FEATURE_COLS = [
    'log_return_1d', 'log_return_5d', 'log_return_10d', 'log_return_21d',
    'rsi_14', 
    'macd_line', 'macd_signal', 'macd_histogram',
    'close_to_sma20', 'close_to_sma50',
    'roc_5', 'roc_21',
    'bb_width', 'bb_position',
    'atr_14_norm', 'realised_vol_21',
    'volume_zscore', 'volume_ratio_5d', 'obv_zscore',
    'rolling_beta', 'index_ret', 'relative_return'
]
NORM_COLS      = [f"{c}_norm" for c in FEATURE_COLS]
TARGET_COL     = f"target_return_{HORIZON}d"

TRADING_DAYS  = 252
TX_COST       = 0.001        # 0.1% per trade (realistic NSE estimate)
RISK_FREE     = 0.065        # 6.5% annual (approx Indian 10-yr G-Sec yield)
RF_DAILY      = RISK_FREE / TRADING_DAYS

BACKTEST_DIR = "backtest"
MODEL_DIR = "models"
RESULTS_DIR = "results"

BID_ASK_SPREAD = 0.0005     # 0.05% - large cap NSE stocks
BROKERAGE = 0.0003          # 0.03% - discount broker 
STT_SELL = 0.00025          # 0.025% - securities transaction tax (SELL ONLY)
SEBI_CHARGE = 0.000001      # 0.0001%
SLIPPAGE = 0.0002            # 0.02% execution slippage estimate
MARKET_IMPACT_K = 0.0003    # scales with sqrt(order/avg_volume)

BASE_ONE_WAY_COST = BID_ASK_SPREAD / 2 + BROKERAGE + SEBI_CHARGE + SLIPPAGE

HALF_KELLY = 0.5            # fractional Kelly — conservative
MAX_POSITION = 1.0          # cap at 100% of capital (no leverage)
MIN_POSITION = 0.05         # ignore Kelly sizes below 5% (noise)
SIGNAL_THRESHOLD  = 0.0     # minimum |predicted return| to trade



def run_stage(name, func):
    print("_" * 50)
    print(f"\tRunning : {name}")
    print("_" * 50)

    start = time.time()
    func()
    end = time.time()
    print("-" * 50)
    print(f"Completed {name}")
    print(f"Time Taken : {(end-start):.2f} seconds \n")

def main():
    pipeline = [
        ("Stage 1 - Data Collection", fetch_data.main),
        ("Stage 2 - Feature Engineering", feature_engineering.main),
        ("Stage 3 - Sliding Window Sequence", sliding_window_sequence.main),
        ("Stage 4 - LSTM", lstm.main),
        ("Stage 5 - Evaluation", evaluation.main),
        ("Stage 6 - Backtesting", backtest.main),
    ]
    for stage_name, stage_function in pipeline:
        run_stage(stage_name, stage_function)



