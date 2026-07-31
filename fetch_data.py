import yfinance as yf
import pandas as pd
import time
from datetime import date
import os

NIFTY50_SAMPLE = [                                                      
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
]   
    # Each item is a ticker Symbol
    # NSE tickers need '.NS' suffix for yfinance
    # BSE would use '.BO' instead

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




def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull OHLCV data for a single ticker.
    auto_adjust=True (default in yfinance >=0.2.something) means Close is
    already split/dividend-adjusted — important so a stock split doesn't
    look like a -50% return crash in your training data.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,    # adjusts OHLC for splits & dividends
        progress=False,
        multi_level_index=False,   # keeps columns flat: Open, High, Low, Close, Volume
    )

    if df.empty:
        print(f"  WARNING: no data returned for {ticker}")
        return df

    df["Ticker"] = ticker
    df.index.name = "Date"
    return df


def fetch_all(tickers: list[str], start: str, end: str, pause: float = 1.0) -> dict[str, pd.DataFrame]:
    """
    Pull data for a list of tickers with a small delay between calls
    to stay polite to Yahoo Finance's servers and avoid rate limiting.
    """
    data = {}
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Fetching {ticker} ...")
        df = fetch_stock_data(ticker, start, end)
        if not df.empty:
            data[ticker] = df
        time.sleep(pause)
    return data


def save_data(data: dict[str, pd.DataFrame], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for ticker, df in data.items():
        # Strip ".NS" for a cleaner filename
        clean_name = ticker.replace(".NS", "").replace("^", "")
        path = os.path.join(output_dir, f"{clean_name}.csv")
        df.to_csv(path)
        print(f"  Saved {path}  ({len(df)} rows)")

def main():
    stock_data = fetch_all(NIFTY50_SAMPLE, START_DATE, END_DATE)
    print("\nFetched indiviual stock data")
    index_data = {
        NIFTY50_INDEX : fetch_stock_data(NIFTY50_INDEX, START_DATE, END_DATE)
    }
    print(f"\nFetched benchmark index {NIFTY50_INDEX}")

    save_data(stock_data, OUTPUT_DIR)
    save_data(index_data, OUTPUT_DIR)
    print("\nData Saved\n")

    print("Sanity check — RELIANCE.NS sample:")
    if "RELIANCE.NS" in stock_data:
        print(stock_data["RELIANCE.NS"].head())
        print(f"\nDate range: {stock_data['RELIANCE.NS'].index.min()} to {stock_data['RELIANCE.NS'].index.max()}")
        print(f"Total trading days: {len(stock_data['RELIANCE.NS'])}")
    
    print("\nDone. Data saved to:", OUTPUT_DIR)      