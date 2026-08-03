import pandas as pd
import numpy as np
import os

HORIZON = 1
WINDOW = 60
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
FEATURE_DIR = "data/features"
SEQ_DIR     = "data/sequences"
NORM_COLS   = [f"{c}_norm" for c in FEATURE_COLS]
TARGET_COL  = f"target_return_{HORIZON}d"
TEST_CUTOFF = "2024-01-01"


def make_sequences(feat_arr: np.ndarray,
                   target_arr: np.ndarray,
                   window: int) -> tuple:
    """
    Sliding window across (T, F) feature matrix → LSTM 3D input.

    Row i:
        X[i] = feat_arr[i : i+window]   shape (window, F)  ← 60-day history
        y[i] = target_arr[i + window]   scalar              ← next-day return

    NEVER shuffle these arrays — time ordering is the signal.
    """
    X, y = [], []
    for i in range(len(feat_arr) - window):
        X.append(feat_arr[i : i + window])
        y.append(target_arr[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def date_split(df: pd.DataFrame,
               cutoff: str,
               feature_cols: list,
               target_col: str,
               window: int) -> tuple:
    """
    Hard date split — test is strictly after train.
    Sequences built on each split independently to prevent
    boundary leakage (a sequence straddling the cutoff date).
    """
    train_df = df[df.index <  cutoff].copy()
    test_df  = df[df.index >= cutoff].copy()

    print(f"    Train : {train_df.index.min().date()} → "
          f"{train_df.index.max().date()}  ({len(train_df)} days)")
    print(f"    Test  : {test_df.index.min().date()}  → "
          f"{test_df.index.max().date()}  ({len(test_df)} days)")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X_train, y_train = make_sequences(
        train_df[feature_cols].values, train_df[target_col].values, window)
    X_test, y_test   = make_sequences(
        test_df[feature_cols].values,  test_df[target_col].values,  window)

    return (X_train, y_train, X_test, y_test,
            train_df.index[window:], test_df.index[window:])


def sanity_check(X_train, y_train, X_test, y_test, ticker):
    print(f"    Shapes → X_train {X_train.shape} | X_test {X_test.shape}")
    assert not np.isnan(X_train).any(), "NaNs in X_train"
    assert not np.isnan(X_test).any(),  "NaNs in X_test"
    assert not np.isnan(y_train).any(), "NaNs in y_train"
    assert not np.isnan(y_test).any(),  "NaNs in y_test"
    assert X_train.shape[1] == WINDOW,  f"Window mismatch (expected {WINDOW})"
    assert X_train.shape[2] == X_test.shape[2], "Feature count mismatch"
    print(f"    ALL OKAY")


def save_sequences(ticker, X_train, y_train, X_test, y_test,
                   train_dates, test_dates):
    os.makedirs(SEQ_DIR, exist_ok=True)
    base = os.path.join(SEQ_DIR, ticker)
    np.save(f"{base}_X_train.npy", X_train)
    np.save(f"{base}_y_train.npy", y_train)
    np.save(f"{base}_X_test.npy",  X_test)
    np.save(f"{base}_y_test.npy",  y_test)
    pd.Series(train_dates, name="date").to_csv(f"{base}_train_dates.csv", index=False)
    pd.Series(test_dates,  name="date").to_csv(f"{base}_test_dates.csv",  index=False)


def load_sequences(ticker: str, seq_dir: str = SEQ_DIR):
    """
    Load saved sequences for a ticker.
    Usage:
        X_train, y_train, X_test, y_test, train_dates, test_dates = load_sequences("RELIANCE")
    """
    base = os.path.join(seq_dir, ticker)
    return (
        np.load(f"{base}_X_train.npy"),
        np.load(f"{base}_y_train.npy"),
        np.load(f"{base}_X_test.npy"),
        np.load(f"{base}_y_test.npy"),
        pd.read_csv(f"{base}_train_dates.csv", parse_dates=["date"])["date"],
        pd.read_csv(f"{base}_test_dates.csv",  parse_dates=["date"])["date"],
    )


def main():
    print("=" * 50)
    print("Stage 3 — Sequence Windowing")
    print(f"Window     : {WINDOW} days")
    print(f"Test cutoff: {TEST_CUTOFF}")
    print("=" * 50)

    # Find all feature CSVs
    if not os.path.isdir(FEATURE_DIR):
        raise FileNotFoundError(
            f"Feature directory '{FEATURE_DIR}' not found. Run Stage 2 first."
        )

    feature_files = sorted([
        f for f in os.listdir(FEATURE_DIR) if f.endswith("_features.csv")
    ])

    if not feature_files:
        raise FileNotFoundError(
            f"No *_features.csv files found in '{FEATURE_DIR}'. Run Stage 2 first."
        )

    print(f"Found {len(feature_files)} tickers to process.\n")

    # Process each ticker
    summary = []

    for fname in feature_files:
        ticker = fname.replace("_features.csv", "")
        print(f"\n  {ticker} ...")

        df = pd.read_csv(
            os.path.join(FEATURE_DIR, fname),
            index_col="Date",
            parse_dates=True,
        ).dropna(subset=[TARGET_COL])

        X_train, y_train, X_test, y_test, train_dates, test_dates = date_split(
            df=df,
            cutoff=TEST_CUTOFF,
            feature_cols=NORM_COLS,
            target_col=TARGET_COL,
            window=WINDOW,
        )

        sanity_check(X_train, y_train, X_test, y_test, ticker)
        save_sequences(ticker, X_train, y_train,
                       X_test, y_test, train_dates, test_dates)

        summary.append({
            "Ticker"     : ticker,
            "Train seqs" : len(X_train),
            "Test seqs"  : len(X_test),
            "Features"   : X_train.shape[2],
            "Window"     : WINDOW,
        })

        print(f"    Saved → {SEQ_DIR}/{ticker}_[X/y]_[train/test].npy")

    print("\n" + "=" * 60)
    print("Stage 3 Complete — Summary")
    print("=" * 60)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nAll sequences saved to: {SEQ_DIR}/")
    print("Next → Stage 4: LSTM training")
    print("  Load with: X_train, y_train, X_test, y_test, tr_d, te_d = load_sequences('RELIANCE')")


if __name__ == "__main__":
    main()