import numpy as np 
import pandas as pd
import os


import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import(
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional
)

from tensorflow.keras.callbacks import(
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sliding_window_sequence import load_sequences
import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = "models"
TICKERS     = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
]

# Training Hyperparameters :
BATCH_SIZE = 32
MAX_EPOCHS = 100
LR = 1e-3
SEED = 42
SEQ_DIR = "data/sequences"

tf.random.set_seed(SEED)
np.random.seed(SEED)



# REGRESSION MODEL --------------------------------------------------------------------------------------
def build_regression_model(timesteps : int, n_features : int) -> Model:
    # timesteps =  window size(prevous trading days)
    # n_features = no. of columns
    # return type = Model

    inputs = Input(
        shape = (timesteps, n_features),    # tensorflow automatically handles batch size
        name = 'ohlcv_sequence'
    )

    # Layer 1 : Bidirectional LSTM
    """
    Why use bidirectional window?

    During training, the model already has the complete 60-day window.
    Reading it both forward and backward helps it learn richer relationships within that window.
    """
    x = Bidirectional(
        # creates two LSTMs 
                # one reads forward day 1-> day2 -> ... day 60
                # one reads backward day 60-> day59 -> ... day 1
        LSTM(
            # whatever learned from the the input sequence, gets converted to 64 values
            # Hidden state size = 64(comman starting number)
            64,
            return_sequences= True,
            # instead of returning only the output from last day, return everyday's output
            # why? => the 2nd LSTM layer needs the entire sequence
            dropout = 0.1,              # input dropout, 10% of input connections are randomly ignored
            recurrent_dropout = 0.1     # dropout 10% of memory connections        
            ),
            name = 'bilstm_1'
    )(inputs)
    
    # Layer 2 : Unidirectional LSTM (extracts final state)
    x = LSTM(
        64,
        return_sequences = False,
        dropout = 0.1,
        recurrent_dropout = 0.1,
        name = 'lstm_2'
    )(x)

    # Layer 3 : Dense Head
    x = BatchNormalization(name = 'batch_norm')(x)
    x = Dense(32, activation = 'relu', kernel_regularizer = l2(1e-4), name = 'dense_1')(x)
    x = Dropout(0.2, name = 'drop_3')(x)

    # Layer 4 : Output Layer
    outputs = Dense(1, activation = 'linear', name = 'return_output')(x)

    model = Model(inputs, outputs, name = 'LSTM_Regression')
    model.compile(
        optimizer = Adam(learning_rate = LR, clipnorm = 1.0),
        loss = 'huber',
        metrics = ["mae"]
    )

    return model

# CLASSIFICATION MODEL --------------------------------------------------------------------------------------
def build_classifier_model(timesteps : int, n_features : int) -> Model:
    """
        Classification LSTM — predicts direction (1=up, 0=down).

        Same backbone as regression but with a sigmoid output and binary
        cross-entropy loss. We add class_weight in training to handle the
        slight imbalance between up/down days in bull markets.

        Sigmoid output gives a probability — you can tune the threshold
        (default 0.5) to trade off precision vs recall in backtesting.
        E.g. only go long when probability > 0.6 for a more conservative signal.
    """
    input = Input(shape = (timesteps, n_features), name = 'ohlcv_sequenc')

    # Layer 1 : Bidirectional LSTM Layer
    x = Bidirectional(
        LSTM(64, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1),
        name = 'bilstm_1'
    )(input)
    x = Dropout(0.3, name = 'drop_1')(x)

    # Layer 2 : Unidirectional LSTM Layer
    x = LSTM(64, return_sequences = False, dropout = 0.1, recurrent_dropout = 0.1,
             name = 'lstm_2')(x)
    x = Dropout(0.3, name = 'drop_2')(x)

    # Layer 3 : Dense Head
    x = BatchNormalization(name = 'batch_norm')(x)
    x = Dense(32, activation = 'relu',
              kernel_regularizer = l2(1e-4), name = 'dense_1')(x)
    x = Dropout(0.2, name = 'drop_3')(x)

    # Layer 4 : Output Layer
    output = Dense(1, activation = 'sigmoid', name = 'direction_output')(x)

    model = Model(input, output, name = 'lstm_classifier')
    model.compile(
        optimizer = Adam(learning_rate = LR, clipnorm = 1.0),
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )

    return model


# CALLBACKS :

def get_callbacks(model_path: str, monitor: str = "val_loss") -> list:
    """
    Three callbacks that prevent wasted training time and overfitting:

    EarlyStopping:
        Stops training when val_loss hasn't improved for 15 epochs.
        restore_best_weights=True means you get the best checkpoint,
        not the last (overfitted) one.

    ReduceLROnPlateau:
        Halves learning rate when val_loss plateaus for 7 epochs.
        Lets the model fine-tune after the big initial learning phase.
        min_lr=1e-6 prevents the LR from becoming uselessly tiny.

    ModelCheckpoint:
        Saves the best model to disk during training.
        If training crashes at epoch 80, you don't lose everything.
    """
    return[
        EarlyStopping(
            monitor = monitor,
            patience = 15,
            restore_best_weights = True,
            verbose = 1
        ),

        ReduceLROnPlateau(
            monitor = monitor,
            factor = 0.5,
            patience = 7,
            min_lr = 1e-6,
            verbose = 1
        ),

        ModelCheckpoint(
            filepath = model_path,
            monitor = monitor,
            save_best_only = True,
            verbose = 0
        )
    ]


# CLASS WEIGHTS (ONLY FOR CLASSIFIER)

def compute_class_weights(y_train : np.ndarray) -> dict:
    """
    In a bull market, up days outnumber down days (~55/45).
    Without class weights, the classifier learns to predict 'up' always
    and achieves 55% accuracy trivially — not useful for trading.
    Weighting penalises misclassifying the minority class more heavily.
    """
    n_total = len(y_train)
    n_up = y_train.sum()
    n_down = n_total - n_up
    w_up = n_total / (2* n_up) if n_up > 0 else 1.0
    w_down = n_total / (2 * n_down) if n_down > 0 else 1.0
    
    # Why divide by 2? => Weight = Total Samples / (Number of classes x Samples in class)

    print(f"    Class weights → up: {w_up:.3f}  down: {w_down:.3f}  "
          f"(up days: {n_up/n_total:.1%})")
    return {1: w_up, 0: w_down}


# TRAINING

def train_regression(X_train, y_train, X_test, y_test, ticker) -> dict:
    os.makedirs(MODEL_DIR, exist_ok = True)

    timesteps, n_features = X_train.shape[1], X_train.shape[2]
    model = build_regression_model(timesteps, n_features)       # Function with model architecture

    if ticker == TICKERS[0]:
        model.summary()

    model_path = os.path.join(MODEL_DIR, f"{ticker}_regression.keras")
    history = model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs = MAX_EPOCHS,
        batch_size = BATCH_SIZE,
        callbacks = get_callbacks(model_path, monitor = 'val_loss'),
        verbose = 0,
        shuffle = False,    # NEVER shuffle time-series batches
    )

    print(f"    Stopped at epoch {len(history.history['loss'])}")
    print(f"    Best val_loss : {min(history.history['val_loss']):.6f}")
    print(f"    Best val_mae  : {min(history.history['val_mae']):.6f}")

    return history.history

def train_classifier(X_train, y_train_dir, X_test, y_test_dir, ticker) -> dict:
    timesteps, n_features = X_train.shape[1], X_train.shape[2]
    model = build_classifier_model(timesteps, n_features)

    model_path = os.path.join(MODEL_DIR, f"{ticker}_classifer.keras")
    class_weight = compute_class_weights(y_train_dir)

    history = model.fit(
        X_train, y_train_dir,
        validation_data = (X_test, y_test_dir),
        epochs = MAX_EPOCHS,
        batch_size = BATCH_SIZE,
        callbacks = get_callbacks(model_path, monitor = "val_accuracy"),
        class_weight = class_weight,
        verbose = 0,
        shuffle = False,    # NEVER shuffle time-series batches
    )

    print(f"    Stopped at epoch {len(history.history['loss'])}")
    print(f"    Best val_accuracy : {max(history.history['val_accuracy']):.4f}")

    return history.history

def save_history(reg_hist: dict, clf_hist: dict, ticker: str):
    max_len = max(len(reg_hist['loss']), len(clf_hist['loss']))

    def pad(lst):
        return lst + [np.nan] * (max_len - len(lst))
    
    df = pd.DataFrame({
        "reg_loss" : pad(reg_hist["loss"]),
        "reg_val_loss" : pad(reg_hist["val_loss"]),
        "reg_mae" : pad(reg_hist["mae"]),
        "clf_loss" : pad(clf_hist["loss"]),
        "clf_val_loss" : pad(clf_hist["val_loss"]),
        "clf_accuracy" : pad(clf_hist["accuracy"]),
        "clf_val_accuracy" : pad(clf_hist["val_accuracy"])
    })
    path = os.path.join(MODEL_DIR, f"{ticker}_history.csv")
    df.to_csv(path, index_label = "epoch")
    print(f"    History saved -> {path}")





def load_model(ticker: str, model_type: str = "regression") -> Model:
    """
    model_type: 'regression' or 'classifier'

    Usage in Stage 5:
        model = load_model("RELIANCE", "regression")
        y_pred = model.predict(X_test)
    """
    path = os.path.join(MODEL_DIR, f"{ticker}_{model_type}.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No save modelat {path}. Run Stage 4 first")
    return tf.keras.models.load_model(path)





def main():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f" GPU detcted : {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("No GPU detected")

    os.makedirs(MODEL_DIR, exist_ok = True)

    for ticker in TICKERS:
        seq_path = os.path.join(SEQ_DIR, f"{ticker}_X_train.npy")
        if not os.path.exists(seq_path):
            print(f"\n SKIP {ticker} — sequences not found")
            continue

        print(f"\n{'─'*30}")
        print(f"  {ticker}")
        print(f"{'─'*30}")

        # Load sequences from Stage 3
        X_train, y_train, X_test, y_test, train_dates, test_dates = load_sequences(ticker)
        # print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

        y_train_dir = (y_train > 0).astype(np.float32)
        y_test_dir = (y_test > 0).astype(np.float32)

        reg_hist = train_regression(X_train, y_train, X_test, y_test, ticker)
        clf_hist = train_classifier(X_train, y_train_dir, X_test, y_test_dir, ticker)

        save_history(reg_hist, clf_hist, ticker)

        print("Stage 4 complete.")