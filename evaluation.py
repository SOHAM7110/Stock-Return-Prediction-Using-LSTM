# Quantitative Evaluation:
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

from sliding_window_sequence import load_sequences
from lstm import load_model


sys.path.append(os.getcwd())

TICKERS     = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
]

RESULTS_DIR   = "results"
MODEL_DIR     = "models"

TRADING_DAYS = 252
TX_COST = 0.001         # tax
RISK_FREE = 0.065        
""" It is the return you can expect from an investment that is 
considered to have very little default risk.

In India, a common approximation is the yield 
on long-term Government Securities (G-Secs).
"""
RF_DAILY      = RISK_FREE / TRADING_DAYS

# ML Metrics :
def ml_metrics(y_true : np.ndarray, y_pred : np.ndarray) -> dict:
    """ 
    R² < 0 means the model is worse than predicting the mean every day.
    For financial returns, R² of 0.01 - 0.05 is actually considered useful
    (markets are hard to predict — don't expect 0.9).
    """

    residuals = y_true - y_pred
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()

    return {
        "rmse" : float(np.sqrt(np.mean(residuals ** 2))),
        "mae"  : float(np.mean(np.abs(residuals))),
        "r2"   : float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }

# Directional Accuracy :

def directional_accuracy(y_true : np.ndarray, y_pred : np.ndarray) -> dict:
    """ 
    This function comapres actual returns and predicted returns
    and asks did the model correctly predict whether the market would go UP or DOWN
    
    It only cares about the direction 
    """

    true_dir = (y_true > 0).astype(int)     # eg : [0.9, 0.3, 0.4, -0.01, -0.05 ...] => [True, True, True, False, False ...]
    pred_dir = (y_pred > 0).astype(int)     # eg : [0.9, -0.3, 0.4, 0.01, -0.05 ...] => [True, False, True, True, False ...]
    correct = (true_dir == pred_dir)

    large_mask = np.abs(y_true) > 0.01
    large_acc = correct[large_mask].mean() if large_mask.sum() > 0 else np.nan

    up_mask = y_true > 0
    up_acc = correct[up_mask].mean() if up_mask.sum() > 0 else np.nan

    down_mask = y_true < 0
    down_acc = correct[down_mask].mean() if down_mask.sum() > 0 else np.nan

    return {
            "directional_accuracy"  : float(correct.mean()),
            "large_move_accuracy"   : float(large_acc),
            "up_day_accuracy"       : float(up_acc),
            "down_day_accuracy"     : float(down_acc),
            "n_large_moves"         : int(large_mask.sum()),
        }

def simulate_strategy(y_true : np.ndarray, y_pred : np.ndarray, dates : pd.DatetimeIndex, threshold : float = 0.0) -> pd.DataFrame:
    """ 
    This function awnsers to : 
    If I actually traded based on my model's predictions, how much money would I have made?
        Simulates a simple long/short strategy driven by the model's signal.
    
        Rules:
          If predicted return > +threshold  → long  (+1)
          If predicted return < -threshold  → short (-1)
          Otherwise                         → flat  (0)
    
        Transaction cost applied on every position change only.
        A 5-day long position pays tx cost once on entry and once on exit.
    
        strategy_return[t] = position[t] x actual_return[t] - tx_cost_if_traded
        """
    positions = []
    for pred in y_pred:
        if pred > threshold:
            positions.append(1)
        elif pred < -threshold:
            positions.append(-1)
        else:
            positions.append(0)

    position_changes = np.diff(positions, prepend = 0)
    # np.diff() => substracts consecutive values
    # prepend = 0? => the model correctly counts that as entering a trade.
    tx_costs = np.abs(position_changes) * TX_COST

    strategy_ret = positions * y_true - tx_costs

    buyhold_ret = y_true.copy()
    # y_true already contains the actual market return for each day.
    # so it says that if we dont sell then, y_true is equivalent to buyhold_ret

    df = pd.DataFrame({
            "date"          : dates,
            "actual_return" : y_true,
            "pred_return"   : y_pred,
            "position"      : positions,
            "strategy_ret"  : strategy_ret,
            "buyhold_ret"   : buyhold_ret,
        }).set_index("date")


    # If I started with ₹1 (or ₹100,000), how would my money grow over time?
    # Think of the number 1 as representing 100% of your existing money.
    df["strategy_equity"] = (1 + df["strategy_ret"]).cumprod()
    df["buyhold_equity"] = (1 + df["buyhold_ret"]).cumprod()
    # cumprod() => cummulative product

    return df   

# Risk Return Matrix

def sharpe_ratio(returns : np.ndarray, rf_daily : float  = RF_DAILY) -> float:
    """
        Sharpe = (mean_daily_excess_return / std_daily_return) x sqrt(252)
    
        Interpretation:
          < 0    : losing money after risk-free rate
          0 - 0.5  : weak
          0.5 - 1  : acceptable
          1 - 2    : good (most hedge funds target this)
          > 2    : excellent (check carefully for overfitting)
        """
    excess = returns  - rf_daily
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS))

def sortino_ratio(returns : np.ndarray, rf_daily : float = RF_DAILY) -> float:
    """
        It calculates how much excess return a fund generates for every unit of downside risk taken. 
        A higher Sortino ratio indicates better, more efficient performance.
    """
    excess = returns - rf_daily
    downside = excess[excess < 0]

    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    return float ((excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS))


def max_drawdown(equity_curve : np.ndarray) -> float:
    # max_drawdown => The largest percentage drop from any previous peak to the next lowest point.
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - rolling_max) / rolling_max

    return float(drawdowns.min())

def calmar_ratio(annualised_return : float, mdd: float) ->float:
    """
        Calmar = annualised_return / |max_drawdown|
        Measures return per unit of worst-case drawdown risk.
        > 1 means you're generating more return than your worst drawdown.
    """

    if mdd == 0:
        return np.nan
    return float(annualised_return / abs(mdd))


# Information Coefficient

def information_coefficient(y_true : np.ndarray, y_pred : np.ndarray) -> dict:
    """
    Information Coefficient = IC measures whether your model correctly ranks assets or returns.

    Inputs:
        y_true → actual future returns
        y_pred → model predictions

    Outputs :
        How predictive the model is
        Whether the prediction is statistically significant
        Whether the signal is consistent through time

    Why Spearman not Pearson:
        We care whether the model correctly RANKS days — predicts that
        a high-return day is higher than a low-return day.
        Rank correlation captures this. Linear correlation doesn't.
    
    Interpretation:
        IC = 0    : no predictive power
        IC = 0.05 : weak but usable (common in live quant funds)
        IC = 0.10 : strong
        IC > 0.15 : exceptional (check for lookahead bias)
    
    ICIR = mean(rolling IC) / std(rolling IC)
        High ICIR = consistent signal. More valuable than a high but
        volatile IC that averages well but is unreliable day-to-day.    
    """


    ic, p_value = stats.spearmanr(y_pred, y_true)

    n = len(y_true)
    rolling_ic = []

    for i in range(21, n):
        window_ic, _ = stats.spearmanr(y_pred[i-21 : i], y_true[i-21 : i])
        rolling_ic.append(window_ic)          # float, not tuple

    rolling_ic = np.array(rolling_ic)
    icir = float(rolling_ic.mean() / rolling_ic.std()) if rolling_ic.std() > 0 else np.nan

    return {
        "ic"      : float(ic),
        "ic_pval" : float(p_value),
        "icir"    : float(icir),
    }


# Quantile Analysis :

def quintile_analysis(y_true : np.ndarray, y_pred : np.ndarray) -> dict:
    """ 
    It awnsers to :
        "If I trusted my model and only bought the highest-ranked predictions,
        would I actually make more money than buying the lowest-ranked predictions?"
    
    A quintile simply means :
        Divide the ranked predictions into 5 equal groups.

    Rank predictions into 5 buckets. Check if top quintile outperforms bottom.
    
    This is how factor researchers validate signals at quant funds.
        A good signal shows a monotonic pattern:
        Q1 (lowest pred) → Q5 (highest pred) returns increase steadily.
    
    Spread = Q5_mean - Q1_mean
        > 0      : model distinguishes high from low return days
        > 0.001  : ~25% annualised spread — strong alpha
    """

    df = pd.DataFrame({"pred": y_pred, "actual": y_true})


    df["quintile"] = pd.qcut(df["pred"], q=5,
                             labels=["Q1", "Q2", "Q3", "Q4", "Q5"])


    qmeans = df.groupby("quintile", observed=True)["actual"].mean()

    result = {f"quintile_{q}_mean_ret": float(qmeans.get(q, np.nan))
              for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]}

    q1 = qmeans.get("Q1", np.nan)
    q5 = qmeans.get("Q5", np.nan)
    result["quintile_spread"] = float(q5 - q1) if not (np.isnan(q1) or np.isnan(q5)) else np.nan
    result["monotonic"]       = bool(qmeans.is_monotonic_increasing)

    return result




# Full evaluation on one ticker

def evaluate_ticker(ticker : str) -> dict:
    print(f" Evaluating : {ticker}")

    X_train, y_train, X_test, y_test, train_dates, test_dates = load_sequences(ticker)

    model = load_model(ticker, "regression")
    y_pred_raw = model.predict(X_test, verbose = 0).flatten()
    y_true = y_test.flatten()


    print(f"Test samples : {len(y_true)}")
    print(f"Test period {test_dates.iloc[0].date()} to {test_dates.iloc[-1].date()} ")


    ml = ml_metrics(y_true, y_pred_raw)
    print(f"\n  ML metrics:")
    print(f"RMSE : {ml['rmse']:.6f}")
    print(f"MAE  : {ml['mae']:.6f}")
    print(f"R²   : {ml['r2']:.4f}  {'⚠ worse than mean' if ml['r2'] < 0 else ''}")


    da = directional_accuracy(y_true, y_pred_raw)
    print(f"\n  Directional accuracy:")
    print(f"Overall          : {da['directional_accuracy']:.2%}")
    print(f"Large moves only : {da['large_move_accuracy']:.2%}  (n={da['n_large_moves']})")
    print(f"Up days          : {da['up_day_accuracy']:.2%}")
    print(f"Down days        : {da['down_day_accuracy']:.2%}")


    ic_metrics = information_coefficient(y_true, y_pred_raw)
    print(f"\n  Information Coefficient:")
    print(f"IC   : {ic_metrics['ic']:.4f}  (p={ic_metrics['ic_pval']:.4f})")
    print(f"ICIR : {ic_metrics['icir']:.4f}")


    q_metrics = quintile_analysis(y_true, y_pred_raw)
    print(f"\n  Quintile analysis:")
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        print(f"    {q}: {q_metrics[f'quintile_{q}_mean_ret']:+.5f}")
    print(f"    Spread (Q5-Q1) : {q_metrics['quintile_spread']:+.5f}")
    print(f"    Monotonic      : {q_metrics['monotonic']}")


    equity_df    = simulate_strategy(y_true, y_pred_raw, test_dates)
    strat_rets   = equity_df["strategy_ret"].values
    buyhold_rets = equity_df["buyhold_ret"].values

    n_days        = len(strat_rets)
    strat_ann_ret = float((1 + strat_rets).prod() ** (TRADING_DAYS / n_days) - 1)
    bh_ann_ret    = float((1 + buyhold_rets).prod() ** (TRADING_DAYS / n_days) - 1)

    strat_sharpe  = sharpe_ratio(strat_rets)
    strat_sortino = sortino_ratio(strat_rets)
    strat_mdd     = max_drawdown(equity_df["strategy_equity"].values)
    strat_calmar  = calmar_ratio(strat_ann_ret, strat_mdd)
    bh_sharpe     = sharpe_ratio(buyhold_rets)
    bh_mdd        = max_drawdown(equity_df["buyhold_equity"].values)

    print(f"\n  Strategy vs Buy-and-Hold:")
    print(f"{'':30}{'Strategy':>12}{'Buy & Hold':>12}")
    print(f"  {'Annualised return':<28}{strat_ann_ret:>+11.2%}{bh_ann_ret:>+11.2%}")
    print(f"  {'Sharpe ratio':<28}{strat_sharpe:>12.3f}{bh_sharpe:>12.3f}")
    print(f"  {'Sortino ratio':<28}{strat_sortino:>12.3f}{'—':>12}")
    print(f"  {'Max drawdown':<28}{strat_mdd:>+11.2%}{bh_mdd:>+11.2%}")
    print(f"  {'Calmar ratio':<28}{strat_calmar:>12.3f}{'—':>12}")


    return {
            "ticker"               : ticker,
            "test_start"           : str(test_dates.iloc[0].date()),
            "test_end"             : str(test_dates.iloc[-1].date()),
            "n_test_days"          : len(y_true),
            **ml,
            **da,
            **ic_metrics,
            **q_metrics,
            "strat_ann_return"     : strat_ann_ret,
            "strat_sharpe"         : strat_sharpe,
            "strat_sortino"        : strat_sortino,
            "strat_max_drawdown"   : strat_mdd,
            "strat_calmar"         : strat_calmar,
            "bh_ann_return"        : bh_ann_ret,
            "bh_sharpe"            : bh_sharpe,
            "bh_max_drawdown"      : bh_mdd,
            "beats_buyhold_sharpe" : int(strat_sharpe > bh_sharpe),
        }  



# Cross-Ticker Summary

"""Suppose you tested 20 different stocks,
    this is how you compare them
"""
def print_summary(summary_df : pd.DataFrame):
    ranked = summary_df.sort_values("strat_sharpe", ascending = False)
    print("\n" + "=" * 78)
    print("CROSS-TICKER SUMMARY  (ranked by Strategy Sharpe)")
    print("=" * 78)
    print(f"{'Ticker':<14}{'Dir Acc':>8}{'IC':>7}{'Sharpe':>8}"
          f"{'Sortino':>9}{'MDD':>8}{'Ann Ret':>9}{'> B&H':>6}")
    print("─" * 78)
    for _, row in ranked.iterrows():
        beats = "✓" if row["beats_buyhold_sharpe"] else "✗"
        print(
            f"{row['ticker']:<14}"
            f"{row['directional_accuracy']:>8.2%}"
            f"{row['ic']:>7.4f}"
            f"{row['strat_sharpe']:>8.3f}"
            f"{row['strat_sortino']:>9.3f}"
            f"{row['strat_max_drawdown']:>8.2%}"
            f"{row['strat_ann_return']:>+9.2%}"
            f"{beats:>6}"
        )
    print("─" * 78)
    print(
        f"{'Average':<14}"
        f"{ranked['directional_accuracy'].mean():>8.2%}"
        f"{ranked['ic'].mean():>7.4f}"
        f"{ranked['strat_sharpe'].mean():>8.3f}"
        f"{ranked['strat_sortino'].mean():>9.3f}"
        f"{ranked['strat_max_drawdown'].mean():>8.2%}"
        f"{ranked['strat_ann_return'].mean():>+9.2%}"
    )
    print("=" * 78)



    

def main():
    all_results = []

    for ticker in TICKERS:
        model_path = os.path.join(MODEL_DIR, f"{ticker}_regression.keras")

        if not os.path.exists(model_path):
            print(f"  SKIP {ticker} — model not found. Run Stage 4 first.")
            continue                           # was missing — fell through to evaluate_ticker

        try:
            metrics = evaluate_ticker(ticker)
            all_results.append(metrics)
        except Exception as e:
            print(f"\n ERROR on {ticker} : {e}")

    if not all_results:
        print("No results. check model and sequence files")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print(f"\nFull metrics → results/summary.csv")
    print_summary(summary_df)


if __name__ == "__main__":
    main()


