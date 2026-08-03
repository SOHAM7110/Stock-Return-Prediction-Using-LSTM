# Transctional Cost Model

import numpy as np
import pandas as pd
from scipy import stats

from sliding_window_sequence import load_sequences
from lstm import load_model


import os
import sys

TICKERS     = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
]

TRADING_DAYS  = 252
TX_COST       = 0.001        # 0.1% per trade (realistic NSE estimate)
RISK_FREE     = 0.065        # 6.5% annual (approx Indian 10-yr G-Sec yield)
RF_DAILY      = RISK_FREE / TRADING_DAYS

BACKTEST_DIR = "backtest"
SEQ_DIR        = "data/sequences"
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


def transaction_cost(position_change : float, is_sell : bool,  avg_volume_ratio : float = 1.0) -> float:
    """
        Computes realistic total transaction cost for one position change.
    
        position_change : how much capital is invested (0.0 to 1.0 of capital)
        is_sell : True if closing a long or opening a short
        avg_volume_ratio : order_size / avg_daily_volume (default 1x = small order)
    
        Market impact grows with sqrt(order_size / volume) — the Almgren-Chriss
        approximation. For a small retail/prop account trading Nifty50 large-caps,
        this is negligible. Include it anyway so the model is deployable at scale.
    
        Returns total cost as a fraction of position value.
    """

    if abs(position_change) < 1e-6:
        return 0.0

    base_cost = BASE_ONE_WAY_COST * abs(position_change)
    stt = STT_SELL * abs(position_change) if is_sell else 0.0
    mkt_impact = MARKET_IMPACT_K * np.sqrt(abs(position_change) * avg_volume_ratio)

    return base_cost + stt + mkt_impact


# LAYER 1 : Naive Backtest

def naive_backtest(y_true : np.ndarray,
                   y_pred : np.ndarray,
                   dates : pd.DatetimeIndex) -> pd.DataFrame:
    """
    Simplest possible backtest.
    Position = sign of prediction : +1 long, -1 short, no costs.

    Purpose : establishes the upper bound on strategy performance
    Any realistic backtest must score below this.

    If your realistic backtest beats your naive one => BUG in cost MODEL
    """

    positions = np.sign(y_pred)
    strategy = positions * y_true
    buyhold = y_true.copy()

    df = pd.DataFrame({
        "actual" : y_true,
        "predicted" : y_pred,
        "position" : positions,
        "strategy_ret" : strategy,
        "buyhold_ret" : buyhold,
    }, index = dates)

    df["strategy_equity"] = (1 + df["strategy_ret"]).cumprod()
    df["buyhold_equity"] = (1 + df["buyhold_ret"].cumprod())
    return df

# LAYER 2 : Realistic Backtest 

def realistic_backtest(y_true : np.ndarray, y_pred : np.ndarray,
                       dates : pd.DatetimeIndex, threshold : float  = SIGNAL_THRESHOLD) -> pd.DataFrame:
    """ 
    This function asks :
    "If I actually traded using those predictions, how much money would I make?"

    LSTM does not know anythin about trading, it can only predict + 0.8%, -0.4%, etc
    this function converts those predictions into BUY, SELL or HOLD 
    and then simulates what would happen with real money

    Adds three real-world frictions the naive backtest ignores :
    1. Signal threshold
        Only trade when |predicted_return| > threshold.
        Avoids chasing tiny predicted moves that are within noise.
        Reduces turnover (number of trades) substantially.
    
    2. Realistic transaction costs
        Bid-ask, brokerage, STT, SEBI charges, slippage, market impact.
        Applied on position CHANGES only — holding a position costs nothing.
    
    3. Position persistence
        Track the current held position explicitly.
        If position doesn't change (model stays long two days in a row),
        no cost is charged on day 2.
    """

    n = len(y_true)
    records = []                # will store everyday's result
    position = 0.0              # initial position, no position
    capital = 1.0               # initial capital => 1 or 100 or 1000 or ...
    for i in range(n):
        pred = y_pred[i]
        actual = y_true[i]

        if pred > threshold:        # if predicted retur is sufficiently +ve then
            desired = 1.0
        elif pred < -threshold:     # if prediction is strongly negative then
            desired = -1.0
        else:
            desired = 0.0

        delta = desired - position      # How much did we change our position?
        is_sell  = (delta < 0)          # flag is passed to the cost function.
        cost = transaction_cost(delta, is_sell)

        gross_ret = position * actual
        net_ret = gross_ret - cost

        capital *= (1 + net_ret)
        position = desired

        records.append({
                "actual"        : actual,
                "predicted"     : pred,
                "desired_pos"   : desired,
                "position"      : position,
                "delta"         : delta,
                "cost"          : cost,
                "gross_ret"     : gross_ret,
                "net_ret"       : net_ret,
                "capital"       : capital,
            })

        df = pd.DataFrame(records, index=dates)
        df["strategy_equity"] = (1 + df["net_ret"]).cumprod()
        df["buyhold_equity"]  = (1 + y_true).cumprod()
        df["buyhold_ret"]     = y_true
        df["n_trades"]        = (df["delta"].abs() > 0.01).cumsum()

    return df



# LAYER 3 : Kelly Criterion Backtest

def kelly_fraction(pred_return : float, rolling_vol : float) -> float:
    """ 
    Kelly Criterion: size the position proportional to your edge.
    Not every prediction deserves the same investment amount.
    How confident are we? Should we invest 5%, 20%, or 100% of our capital?
    Kelly decides how much money to risk.
    
    Full Kelly formula for a continuous return distribution:
        f* = μ / σ²
    """
    if rolling_vol < 1e-6:
        return 0.0

    # sigma => volitality => measures uncertainty
    sigma_sq = rolling_vol ** 2         # variance
    full_kelly = pred_return / sigma_sq
    half_kelly = full_kelly * HALF_KELLY    # sacrifies little growth but dramatically reduces risk

    # Clip to [-MAX_POSITION< + MAX_POSITION]
    sized = np.clip(half_kelly, -MAX_POSITION, MAX_POSITION)
    # Never allow the value outside a range.

    # Zero out positions below minimum threshold (noise suppression)
    if abs(sized) < MIN_POSITION:
        return 0.0

    return float(sized)

def kelly_backtest(y_true : np.ndarray, y_pred : np.ndarray, dates : pd.DatetimeIndex, vol_window : int = 21) -> pd.DataFrame:
    """
    This Kelly backtest asks two questions:
        - Should I trade?
        - If yes, how much of my capital should I invest?
    """

    n         = len(y_true)
    records   = []
    position  = 0.0
    capital   = 1.0

    # Initialise rolling return buffer for vol estimation
    ret_buffer = list(y_true[:vol_window])

    for i in range(vol_window, n):
        pred   = y_pred[i]
        actual = y_true[i]

        # Rolling realised vol (past only — no lookahead)
        rolling_vol = np.std(ret_buffer[-vol_window:]) * np.sqrt(TRADING_DAYS)

        # Kelly-sized desired position
        # "Given today's prediction and today's market risk, how large should my position be?"
        # Kelly_fraction returns discrete values
        desired  = kelly_fraction(pred, rolling_vol)

        # Transaction cost on position change
        delta    = desired - position
        is_sell  = (delta < 0)
        cost     = transaction_cost(delta, is_sell)

        # P&L
        gross_ret = position * actual
        net_ret   = gross_ret - cost
        capital  *= (1 + net_ret)
        position  = desired

        # Update rolling buffer
        ret_buffer.append(actual)

        records.append({
            "actual"        : actual,
            "predicted"     : pred,
            "rolling_vol"   : rolling_vol,
            "desired_pos"   : desired,
            "position"      : position,
            "delta"         : delta,
            "cost"          : cost,
            "gross_ret"     : gross_ret,
            "net_ret"       : net_ret,
            "capital"       : capital,
        })

    df = pd.DataFrame(records, index=dates[vol_window:])
    df["strategy_equity"] = (1 + df["net_ret"]).cumprod()
    df["buyhold_equity"]  = (1 + y_true[vol_window:]).cumprod()
    df["buyhold_ret"]     = y_true[vol_window:]
    df["n_trades"]        = (df["delta"].abs() > 0.01).cumsum()
    return df


# LAYER 4 : Walk-Forward Validation



def walk_forward_backtest(y_true : np.ndarray,
                          y_pred : np.ndarray,
                          dates : pd.DatetimeIndex,
                          train_window : int = 504, # 2 Years
                          test_window : int = 126,
                          step : int = 63
                          ) -> pd.DataFrame:
        """
        Instead of one fixed train/test split, we roll a window forward:

        Fold 1: Train [0:504]    → Test [504:630]
        Fold 2: Train [63:567]   → Test [567:693]
        Fold 3: Train [126:630]  → Test [630:756]
        ...

        Why this matters:
        A single train/test split might get lucky or unlucky depending
        on which market regime falls in the test period.
        Walk-forward averages over multiple regimes — bull, bear, sideways —
        giving a more reliable estimate of live performance.
        """

        n = len(y_true)
        results = []
        fold = 0

        start = train_window    # first test window starts after initial train window

        while start + test_window <= n:
                test_slice = slice(start, start + test_window)
                # test_slice is slice(504, 504 + test_window)
                fold_true = y_true[test_slice]
                fold_pred = y_pred[test_slice]
                fold_dates = dates[test_slice]

                fold_df     = realistic_backtest(fold_true, fold_pred, fold_dates)
                fold_rets   = fold_df["net_ret"].values

                ann_factor  = TRADING_DAYS / len(fold_rets)
                ann_ret     = float((1 + fold_rets).prod() ** ann_factor - 1)
                sharpe      = float(
                    (fold_rets.mean() - RF_DAILY) / fold_rets.std() * np.sqrt(TRADING_DAYS)
                ) if fold_rets.std() > 0 else 0.0
                dir_acc     = float(((fold_pred > 0) == (fold_true > 0)).mean())
                ic, _       = stats.spearmanr(fold_pred, fold_true)
                eq          = (1 + fold_rets).cumprod()
                mdd         = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
                n_trades    = int((fold_df["delta"].abs() > 0.01).sum())
                
                results.append({
                    "fold"          : fold,
                    "test_start"    : fold_dates[0].date(),
                    "test_end"      : fold_dates[-1].date(),
                    "n_days"        : len(fold_rets),
                    "ann_return"    : ann_ret,
                    "sharpe"        : sharpe,
                    "dir_accuracy"  : dir_acc,
                    "ic"            : float(ic),
                    "max_drawdown"  : mdd,
                    "n_trades"      : n_trades,
                    "total_cost"    : float(fold_df["cost"].sum()),
                })
                
                fold  += 1
                start += step
                
                summary = pd.DataFrame(results)
        return summary




# Performance Metrics (Shared across all layers)

def compute_metrics(df: pd.DataFrame,
                    ret_col: str = "net_ret",
                    equity_col: str = "strategy_equity") -> dict:
    """Computes full metric suite for any backtest DataFrame."""
    rets      = df[ret_col].values
    equity    = df[equity_col].values
    n         = len(rets)

    ann_ret   = float((1 + rets).prod() ** (TRADING_DAYS / n) - 1)
    excess    = rets - RF_DAILY
    sharpe    = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) \
                if excess.std() > 0 else 0.0
    downside  = excess[excess < 0]
    sortino   = float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS)) \
                if len(downside) > 0 and downside.std() > 0 else np.nan
    roll_max  = np.maximum.accumulate(equity)
    drawdowns = (equity - roll_max) / roll_max
    mdd       = float(drawdowns.min())
    calmar    = float(ann_ret / abs(mdd)) if mdd != 0 else np.nan

    n_trades  = int((df["delta"].abs() > 0.01).sum()) \
                if "delta" in df.columns else 0
    total_cost = float(df["cost"].sum()) if "cost" in df.columns else 0.0

    bh_rets    = df["buyhold_ret"].values if "buyhold_ret" in df.columns else rets
    bh_ann     = float((1 + bh_rets).prod() ** (TRADING_DAYS / n) - 1)
    bh_excess  = bh_rets - RF_DAILY
    bh_sharpe  = float(bh_excess.mean() / bh_excess.std() * np.sqrt(TRADING_DAYS)) \
                 if bh_excess.std() > 0 else 0.0

    return {
        "ann_return"      : ann_ret,
        "sharpe"          : sharpe,
        "sortino"         : sortino,
        "max_drawdown"    : mdd,
        "calmar"          : calmar,
        "n_trades"        : n_trades,
        "total_cost_paid" : total_cost,
        "bh_ann_return"   : bh_ann,
        "bh_sharpe"       : bh_sharpe,
        "beats_bh"        : int(sharpe > bh_sharpe),
    }


# Comparison Table :


def print_layer_comparison(ticker: str,
                           naive_m: dict,
                           real_m:  dict,
                           kelly_m: dict):
    """
    Side-by-side comparison of all three layers.
    The gap between naive and realistic is the most important number —
    it tells you how much friction costs you in this strategy.
    """
    print(f"\n  {'─'*68}")
    print(f"  {ticker}  —  Layer comparison")
    print(f"  {'─'*68}")
    print(f"  {'Metric':<22} {'Naive':>12} {'Realistic':>12} {'Kelly':>12}")
    print(f"  {'─'*68}")

    rows = [
        ("Ann. return",     "ann_return",      "{:>+11.2%}"),
        ("Sharpe ratio",    "sharpe",           "{:>12.3f}"),
        ("Sortino ratio",   "sortino",          "{:>12.3f}"),
        ("Max drawdown",    "max_drawdown",     "{:>+11.2%}"),
        ("Calmar ratio",    "calmar",           "{:>12.3f}"),
        ("N trades",        "n_trades",         "{:>12d}"),
        ("Total cost paid", "total_cost_paid",  "{:>+11.4f}"),
        ("B&H Sharpe",      "bh_sharpe",        "{:>12.3f}"),
        ("Beats B&H",       "beats_bh",         "{:>12d}"),
    ]

    def fmt(val, template):
        try:
            return template.format(val)
        except (ValueError, TypeError):
            return f"{'—':>12}"

    for label, key, template in rows:
        nv = naive_m.get(key, np.nan)
        rv = real_m.get(key,  np.nan)
        kv = kelly_m.get(key, np.nan)
        print(f"  {label:<22}{fmt(nv, template)}{fmt(rv, template)}{fmt(kv, template)}")

    cost_drag = (real_m.get("ann_return", 0) - naive_m.get("ann_return", 0))
    print(f"  {'─'*68}")
    print(f"  Cost drag (naive→real): {cost_drag:>+.2%} annualised")
    print(f"  {'─'*68}")



# Walk-Forward Summary :

def print_walkforward(wf_df: pd.DataFrame, ticker: str):
    """Prints fold-by-fold walk-forward results."""
    print(f"\n  Walk-forward results — {ticker}")
    print(f"  {'Fold':<6} {'Period':<26} {'Ann Ret':>9} "
          f"{'Sharpe':>8} {'Dir Acc':>8} {'IC':>7} {'Trades':>7}")
    print(f"  {'─'*74}")

    for _, row in wf_df.iterrows():
        period = f"{row['test_start']} → {row['test_end']}"
        print(
            f"  {int(row['fold']):<6}"
            f"{period:<26}"
            f"{row['ann_return']:>+9.2%}"
            f"{row['sharpe']:>8.3f}"
            f"{row['dir_accuracy']:>8.2%}"
            f"{row['ic']:>7.4f}"
            f"{int(row['n_trades']):>7}"
        )

    print(f"  {'─'*74}")
    print(
        f"  {'Average':<32}"
        f"{wf_df['ann_return'].mean():>+9.2%}"
        f"{wf_df['sharpe'].mean():>8.3f}"
        f"{wf_df['dir_accuracy'].mean():>8.2%}"
        f"{wf_df['ic'].mean():>7.4f}"
    )

    # Consistency check
    positive_sharpe = (wf_df["sharpe"] > 0).sum()
    total_folds     = len(wf_df)
    print(f"\n  Positive Sharpe in {positive_sharpe}/{total_folds} folds "
          f"({'consistent ✓' if positive_sharpe >= total_folds * 0.7 else 'inconsistent'})")


def main():
    import tensorflow as tf

    print(f"One-way cost breakdown:")
    print(f"  Bid-ask spread  : {BID_ASK_SPREAD/2:.4%}")
    print(f"  Brokerage       : {BROKERAGE:.4%}")
    print(f"  STT (sell only) : {STT_SELL:.4%}")
    print(f"  SEBI charge     : {SEBI_CHARGE:.6%}")
    print(f"  Slippage        : {SLIPPAGE:.4%}")
    print(f"  Base one-way    : {BASE_ONE_WAY_COST:.4%}")

    os.makedirs(BACKTEST_DIR, exist_ok = True)
    all_summary = []

    for ticker in TICKERS:
        model_path = os.path.join(MODEL_DIR, f"{ticker}_regression.keras")
        seq_path = os.path.join(SEQ_DIR, f"{ticker}_X_test.npy")

        if not os.path.exists(model_path) or not os.path.exists(seq_path):
            print(f"\n SKIP {ticker}, model or sequence not found")
            continue
        print(f"\n{'-'*25}")
        print(f"  {ticker}")
        print(f"{'-'*25}")

        # Load Sequences
        X_train, y_train, X_test, y_test, train_dates, test_dates = load_sequences(ticker)

        # Load model and predict
        model = tf.keras.models.load_model(model_path)
        y_pred = model.predict(X_test, verbose = 0).flatten()
        y_true = y_test.flatten()
        dates = pd.DatetimeIndex(test_dates)

        print(f"  Test period : {dates[0].date()} → {dates[-1].date()}")
        print(f"  Samples     : {len(y_true)}")

        # Naive backtest
        naive_df = naive_backtest(y_true, y_pred, dates)
        naive_df.to_csv(os.path.join(BACKTEST_DIR, f"{ticker}_naive.csv"))
        naive_m = compute_metrics(
            naive_df.assign(delta = 0, cost = 0, buyhold_ret = y_true),
            ret_col = "strategy_ret", equity_col = "strategy_equity"
        )

        # Realistic
        real_df = realistic_backtest(y_true, y_pred, dates)
        real_df.to_csv(os.path.join(BACKTEST_DIR, f"{ticker}_realistic.csv"))
        real_m = compute_metrics(real_df)

        # Kelly
        kelly_df = kelly_backtest(y_true, y_pred, dates)
        kelly_df.to_csv(os.path.join(BACKTEST_DIR, f"{ticker}_kelly.csv"))
        kelly_m = compute_metrics(kelly_df)

        # Comparison
        print_layer_comparison(ticker, naive_m, real_m, kelly_m)

        # Walk forward
        wf_df = walk_forward_backtest(y_true, y_pred, dates)
        wf_df.to_csv(
            os.path.join(BACKTEST_DIR, f"{ticker}_walkforward.csv")
        )
        print_walkforward(wf_df, ticker)

        all_summary.append({
            "ticker"             : ticker,
            "naive_sharpe"       : naive_m["sharpe"],
            "realistic_sharpe"   : real_m["sharpe"],
            "kelly_sharpe"       : kelly_m["sharpe"],
            "realistic_mdd"      : real_m["max_drawdown"],
            "kelly_mdd"          : kelly_m["max_drawdown"],
            "cost_drag"          : real_m["ann_return"] - naive_m["ann_return"],
            "n_trades"           : real_m["n_trades"],
            "total_cost"         : real_m["total_cost_paid"],
            "wf_avg_sharpe"      : wf_df["sharpe"].mean(),
            "wf_consistency"     : float((wf_df["sharpe"] > 0).mean()),
            "beats_bh"           : real_m["beats_bh"],
        })

    # Cross-Ticker Summary
    if all_summary:
        summary_df = pd.DataFrame(all_summary).sort_values("realistic_sharpe",  ascending = False)
        summary_df.to_csv(
            os.path.join(BACKTEST_DIR, "summary.csv"), index = False
        )

        print("\n" + "-" * 25)
        print("FINAL SUMMARY  (sorted by Realistic Sharpe)")
        print("-" * 25)
        print(f"{'Ticker':<14}{'Naive Sh':>9}{'Real Sh':>9}{'Kelly Sh':>9}"
                f"{'Real MDD':>9}{'Cost Drag':>10}{'WF Cons':>8}{'BH':>5}")
        print("─" * 78)
        for _, row in summary_df.iterrows():
            print(
                f"{row['ticker']:<14}"
                f"{row['naive_sharpe']:>9.3f}"
                f"{row['realistic_sharpe']:>9.3f}"
                f"{row['kelly_sharpe']:>9.3f}"
                f"{row['realistic_mdd']:>9.2%}"
                f"{row['cost_drag']:>+9.2%}"
                f"{row['wf_consistency']:>8.0%}"
                f"{'✓' if row['beats_bh'] else '✗':>5}"
            )
        print("=" * 78)
        print(f"\nAll results saved to {BACKTEST_DIR}/")



