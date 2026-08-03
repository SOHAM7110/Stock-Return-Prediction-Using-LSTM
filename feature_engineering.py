import pandas as pd
import numpy as np
import os

RAW_DIR = "data/raw"
FEATURE_DIR = "data/features"
INDEX_FILE = "data/raw/NSEI.csv"

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
HORIZON = 1
NORM_COLS      = [f"{c}_norm" for c in FEATURE_COLS]
TARGET_COL     = f"target_return_{HORIZON}d"

# FEATURE VARIABLES

""" 
1. Log Returns (4 features)

    You use log returns instead of percentage returns for two reasons:  
    they're additive across time periods (log_return_5d ≈ sum of 5 daily log returns), and they're closer to normally distributed —  
    both good properties for neural network inputs.
"""
def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Log returns are preferred over simple percentage returns in quant finance
        because they're additive over time and more normally distributed —
        both useful properties for neural network inputs.
    
        log_return_1d  = log(Close_t / Close_{t-1})  ← most important feature
        log_return_5d  = log(Close_t / Close_{t-5})  ← weekly momentum
        log_return_10d = log(Close_t / Close_{t-10}) ← two-week momentum
        log_return_21d = log(Close_t / Close_{t-21}) ← monthly momentum
    """
    close = df['Close']

    df['log_return_1d'] = np.log(close/close.shift(1))
    df['log_return_5d'] = np.log(close/close.shift(5))
    df['log_return_10d'] = np.log(close/close.shift(10))
    df['log_return_21d'] = np.log(close/close.shift(21))

    return df


"""
2. Momentum Indicators (7 features)

    RSI, MACD line, MACD signal, MACD histogram, price-to-SMA ratios, and rate-of-change.  
    The key decision is feeding all three MACD components separately rather than just using a  
    crossover signal — the LSTM can discover its own crossover logic from the raw components, which  
    is almost always more expressive.

     * **RSI**  
        RSI compares Average recent gains against Average recent losses  
        If gains dominate, RSI becomes large; If losses dominate, RSI becomes small.  

        Traditionally,  
            RSI > 70 => Overbought, stock has risen quickly  
            RSI < 30 => Oversold, stock has fallen rapidly  

        Notice these are not guarantees.

    * **MACD**
        Moving Average Convergence Divergence
        - measures the strength of price movements,  
        MACD measures the relationship between two moving averages.
        - But questions arise:  
            - Is the trend getting stronger?  
            - Is it slowing down?  
            - Is momentum increasing?  
            - Is the trend about to reverse?  
        MACD attempts to answer these questions.
        

"""

# Momentum Indicators :
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
        RSI (Relative Strength Index) — measures speed and magnitude of price moves.
        Range: 0–100. Above 70 = overbought, below 30 = oversold.
        We return the raw RSI value (not a signal) so the LSTM can learn
        its own thresholds from data rather than using our hardcoded ones.
    
        We implement manually here to avoid a TA-Lib dependency.
    """

    # input => series = close prises
    # output => RSI values
    # period => time interval = 14days, industry standard

    delta = series.diff()               # Computes current price - perivous
    
    gain = delta.clip(lower = 0)        # clip(lower = 0) => if less than zero => becomes zero      
    loss = delta.clip(upper = 0)        # clip(upper = 0) => this keeps only negative moments

    # RSI compares avg recent gain vs avg recent loss
    avg_gain = gain.ewm(        # this computes EMA of Gains
        alpha = 1 / period,     
        # This controls how quickly older observations lose influence
        # small alpha => smooth RSI, large alpha more sensitive RSI
        min_periods = period,       # first 13 rows become NAN, becuase there is no historical data
        adjust = False 
        ).mean()
    
    avg_loss = loss.ewm(        # this computes EMA for Loss
        alpha = 1 / period,
        min_periods = period,
        adjust = False
        ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1+ rs))
    return rsi

def compute_macd(series: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9):
    """
        MACD (Moving Average Convergence Divergence).
        Returns three series:
        macd_line      = EMA(fast) - EMA(slow)
        macd_signal    = EMA(macd_line, signal)
        macd_histogram = macd_line - macd_signal (crossover signal)

        We feed all three to the LSTM separately so it can learn
        which component of MACD is predictive for each stock.
    """
    # Fast EMA (typically 12 days)
    ema_fast = series.ewm(span = fast, adjust = False).mean()

    # Slow EMA (typically 26 days)
    ema_slow = series.ewm(span = slow, adjust = False).mean()

    # MACD Line => difference between short term and long term trends
    macd_line = ema_fast - ema_slow

    # Signal Line is 9 day EMA of the MACD Line
    macd_sig = macd_line.ewm(span = signal, adjust = False).mean()

    """Histogram => measure of distance between MACD and Signal
    Interpretation:
    Positive Histogram: MACD is above the Signal Line → bullish momentum is strengthening.
    Negative Histogram: MACD is below the Signal Line → bearish momentum is strengthening.
    Histogram near zero: Momentum is weakening or changing direction."""
    macd_hist = macd_line - macd_sig

    return macd_line, macd_sig, macd_hist

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df['rsi_14'] = compute_rsi(df['Close'], period = 14)

    macd_line, macd_sig, macd_hist = compute_macd(df['Close'])
    df['macd_line'] = macd_line
    df['macd_signal'] = macd_sig
    df['macd_histogram'] = macd_hist

    # Price relative to its own moving averages — captures trend state
    df['close_to_sma20'] = df['Close']/df['Close'].rolling(20).mean() - 1
    df['close_to_sma50'] = df['Close']/df['Close'].rolling(50).mean() - 1

    # Rateof change - normalized momentum over N days
    df['roc_5'] = df['Close'].pct_change(5)
    df['roc_21'] = df['Close'].pct_change(21)

    return df




"""
3. Volatility Indicators (4 features)

    Bollinger Band width (how much is the stock compressing before a breakout?), BB position (where is  
    price within the band?), ATR normalised by price (raw vol scaled to be comparable across stocks at  
    different price levels), and 21-day realised vol annualised. Volatility features matter because LSTM  
    return predictions are significantly more reliable during low-vol regimes than high-vol ones.

    * **Bollinger Bands** 
        - How far is today's price from its recent average?
        - Bollinger Bands :
          - Middle Band => SMA20
          - Upper Band => SMA20 + 2 * std20
          - Lower Band => SMA20 - 2 * std20
"""


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """ 
        Bollinger Bands — price relative to its own volatility envelope.
        bb_width    = how wide the bands are (realized vol proxy)
        bb_position = where price sits within the band (-1 to +1 roughly)

        ATR (Average True Range) — raw volatility in price terms.
        Normalised by Close so it's comparable across different price levels.

        Realised vol — rolling std of log returns, annualised.
        A direct measure of how much the stock is moving day-to-day.
    """

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Bollinger Bands (20-dyas, 2 std)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    # Bollinger Band width => measures the width of volatility envolope
    df['bb_width'] = (bb_upper - bb_lower)/ sma20
    
    # Bollinger Position => Where the current price sits within the bands
    # Near 0 = close to the lower band, near 1 = close to the upper band.
    df['bb_position'] = (close - bb_lower)/(bb_upper - bb_lower)

    # ATR
    # ATR (Average True Range) is a technical indicator that measures how much a stock typically moves in a day,
    # regardless of whether it moves up or down.
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis = 1).max(axis = 1)
    df['atr_14_norm'] = true_range.ewm(span = 14, adjust = False).mean() / close

    # Realised volatility (annualised)
    log_ret = np.log(close / close.shift(1))
    df['realised_vol_21'] = log_ret.rolling(21).std()*np.sqrt(252)

    return df



"""
4. Volume Signals (3 features)

    Volume z-score (the core signal — how abnormal is today's volume?), volume ratio vs 5-day average  
    (simpler version), and OBV z-score (On-Balance Volume, a directional accumulation proxy). Volume is  
    your best cheap proxy for institutional activity without buying expensive order flow data.
"""

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """ 
        Volume z-score: how unusual is today's volume vs the past 20 days?
        Values above +2 or below -2 signal abnormal activity.
        This is a simple proxy for institutional flow / news events.

        OBV (On-Balance Volume): cumulates volume with sign based on price direction.
        Captures whether volume is accumulating (bullish) or distributing (bearish).
        We normalise OBV by its own rolling mean so it's stationary enough for LSTM.
    """

    volume = df['Volume']
    close = df['Close']

    # Volume z-score (rolling 20-day)
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    df['volume_zscore'] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    # Volume ratio — today vs 5-day average (simpler, more stable)
    df['volume_ratio_5d'] = volume/ volume.rolling(5).mean()

    # OBV (normalised)
    direction = np.sign(close.diff())
    obv = (volume * direction).cumsum()
    obv_mean = obv.rolling(20).mean()
    obv_std = obv.rolling(20).std()
    df['obv_zscore'] = (obv - obv_mean) / obv_std.replace(0, np.nan)

    return df




"""
5. Market Context (3 features)

    This is the most quant-specific feature group. Rolling 60-day beta tells you how much the stock  
    co-moves with Nifty. The relative return (stock return minus beta-adjusted index return) is the  
    idiosyncratic component — what the market doesn't explain. This is what quant models are actually  
    trying to predict, not the raw return.
"""

def add_market_features(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """ 
        Beta-adjusted relative return:
        = stock's log return - (rolling_beta × index log return)

        This tells you how much the stock moved BEYOND what the market explains.
        Positive = stock is outperforming the market today (idiosyncratic strength).
        Negative = stock is underperforming even accounting for market movement.

        Rolling beta (60-day) is recalculated every day using only past data —
        no lookahead bias.
    """
    index_ret = np.log(index_df['Close'] / index_df['Close'].shift(1))
    index_ret.name = 'index_ret'

    stock_ret = df['log_return_1d']

    # Align on date index
    aligned = pd.concat([stock_ret, index_ret], axis = 1).dropna()

    # Rolling 60-day beta = cov(stock, index) / var(index)
    rolling_cov = aligned['log_return_1d'].rolling(60).cov(aligned['index_ret'])
    rolling_var = aligned['index_ret'].rolling(60).var()
    rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)

    df['rolling_beta'] = rolling_beta
    df['index_ret'] = index_ret
    df['relative_return'] = stock_ret - (rolling_beta * index_ret)

    return df






# TARGET VARIABLE

def add_target(df : pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """ 
    Target = log return N days ahead.
    We use shift(-horizon) to peek forward — this is intentional ONLY for the
    target column. All feature columns must never use future data.

    For classification (direction): target_direction = 1 if return > 0 else 0
    We keep both — you can switch loss functions to test regression vs clf.
    """

    df[f"target_return_{horizon}d"] = df["log_return_1d"].shift(-horizon)
    df[f"target_direction_{horizon}"] = (df[f"target_return_{horizon}d"] > 0).astype(int)

    return df


# Rolling Z-Score Normalisation
"""avoids lookhead bias vs global min-max"""

def rolling_zscore_normalise(df : pd.DataFrame, cols : list[str], window : int = 252) -> pd.DataFrame:
    """ 
        Normalise each feature using its own rolling 252-day (1yr) mean and std.
        This is the correct approach for time-series ML:
        - Global normalisation uses future data to compute mean/std → LOOKAHEAD BIAS
        - Rolling normalisation only uses past data → SAFE

        Values more than 3 std from the mean are clipped to prevent extreme
        values destabilising LSTM training.
    """

    for col in cols:
        if col not in df.columns:
            continue
        roll_mean = df[col].rolling(window, min_periods = 60).mean()
        roll_std = df[col].rolling(window, min_periods = 60).std()
        z = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
        df[f"{col}_norm"] = z.clip(-3,3)
    return df


# ADD FEATURES


def build_features(ticker_csv : str,
                   index_df : pd.DataFrame,
                   horizon : int = 1) -> pd.DataFrame:
    df = pd.read_csv(ticker_csv, index_col = 'Date', parse_dates = True)

    df = add_return_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_market_features(df, index_df)
    df = add_target(df, horizon = horizon)
    df = rolling_zscore_normalise(df, FEATURE_COLS)

    df = df.dropna(subset = [f"{c}_norm" for c in FEATURE_COLS if f"{c}_norm" in df.columns])

    return df


def add_features():

    os.makedirs(FEATURE_DIR, exist_ok = True)
    index_df = pd.read_csv(INDEX_FILE, index_col = "Date", parse_dates = True)
    raw_files = [f for f in os.listdir(RAW_DIR)
                 if f.endswith(".csv") and f != "NSEI.csv"]
    
    for fname in sorted(raw_files):
        ticker = fname.replace(".csv", "")
        df_feat = build_features(
            ticker_csv = os.path.join(RAW_DIR, fname),
            index_df = index_df,
            horizon = HORIZON,
        )
        out_path = os.path.join(FEATURE_DIR, f"{ticker}_features.csv")
        df_feat.to_csv(out_path)

    for i, col in enumerate(NORM_COLS, 1):
        print(f" {i:02d}.{col}")

def main():
    os.makedirs(FEATURE_DIR, exist_ok = True)
    index_df = pd.read_csv(INDEX_FILE, index_col = "Date", parse_dates = True)
    raw_files = [f for f in os.listdir(RAW_DIR)
                 if f.endswith(".csv") and f != "NSEI.csv"]

    for fname in raw_files:
        ticker = fname.replace(".csv", "")
        df_feat = build_features(
            ticker_csv = os.path.join(RAW_DIR,  fname),
            index_df = index_df,
            horizon = 1,
        )

        out_path = os.path.join(FEATURE_DIR, f"{ticker}_features.csv")
        df_feat.to_csv(out_path)

        norm_cols = [
            f"{c}_norm" for c in FEATURE_COLS
        ]
        for i,  col in enumerate (norm_cols, 1):
            print(f" {i: 02d}. {col}")
