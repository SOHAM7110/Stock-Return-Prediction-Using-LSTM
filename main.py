import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import date

import fetch_data
import feature_engineering
import sliding_window_sequence
import lstm
import evaluation
import backtest

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")


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
        # ("Stage 6 - Backtesting", backtest.main),
    ]
    for stage_name, stage_function in pipeline:
        run_stage(stage_name, stage_function)


if __name__ == "__main__":
    main()
