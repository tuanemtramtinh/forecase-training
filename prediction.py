import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse


# ─────────────────────────────────────────────
# FIX 1: Đổi tên biến accumulator Xs/ys để không
#         shadow tham số X/y truyền vào.
# ─────────────────────────────────────────────
def create_sequences(X: np.ndarray, y: np.ndarray, window: int):
    Xs, ys = [], []
    for i in range(len(y) - window):
        Xs.append(X[i : i + window, :])
        ys.append(y[i + window])
    return np.asarray(Xs), np.asarray(ys)


def prepare_product_timeseries(
    df: pd.DataFrame, product_id: str
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Product ID"] == product_id].sort_values("Date")

    if df.empty:
        raise ValueError(f"Product ID '{product_id}' not found in the CSV file.")

    # Nếu 1 Product xuất hiện nhiều dòng trong cùng 1 ngày,
    # gộp lại thành 1 bản ghi/ngày để có chuỗi thời gian rõ ràng.
    def mode_or_nan(s: pd.Series):
        s = s.dropna()
        return s.mode().iloc[0] if not s.mode().empty else np.nan

    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            {
                "Units Sold": "sum",
                "Seasonality": mode_or_nan,
                "Category": mode_or_nan,
                "Holiday/Promotion": "max",
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # One-hot cho biến phân loại
    daily = pd.get_dummies(
        daily,
        columns=["Seasonality", "Category"],
        prefix=["Season", "Cat"],
        dtype=float,
    )
    feature_cols = [c for c in daily.columns if c not in ["Date", "Units Sold"]]
    return daily, feature_cols


def align_feature_columns(
    train_daily: pd.DataFrame, test_daily: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # Căn cột one-hot giữa train & test (test có thể thiếu/thừa category/season)
    train_cols = set(train_daily.columns)
    test_cols = set(test_daily.columns)
    all_cols = sorted((train_cols | test_cols) - {"Date", "Units Sold"})

    train_aligned = train_daily.copy()
    test_aligned = test_daily.copy()

    for c in all_cols:
        if c not in train_aligned.columns:
            train_aligned[c] = 0.0
        if c not in test_aligned.columns:
            test_aligned[c] = 0.0

    base_cols = ["Date", "Units Sold"]
    train_aligned = train_aligned[base_cols + all_cols]
    test_aligned = test_aligned[base_cols + all_cols]
    return train_aligned, test_aligned, all_cols


def build_model(window: int, n_features: int) -> keras.Model:
    # model = keras.Sequential(
    #     [
    #         keras.layers.Input(shape=(window, n_features)),
    #         keras.layers.LSTM(128),  # Chỉ dùng 1 lớp LSTM với 64 units
    #         keras.layers.Dropout(0.5),
    #         keras.layers.Dense(16, activation="relu"),
    #         keras.layers.Dense(1),
    #     ]
    # )
    # Build the Model
    model = keras.models.Sequential()

    # First Layer
    model.add(
        keras.layers.LSTM(64, return_sequences=True, input_shape=(window, n_features))
    )

    # Second Layer
    model.add(keras.layers.LSTM(64, return_sequences=False))

    # 3rd Layer (Dense)
    model.add(keras.layers.Dense(128, activation="relu"))

    # 4th Layer
    model.add(keras.layers.Dropout(0.5))

    # Final Output Layer
    model.add(keras.layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product_id", type=str, default="P0003")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_path", type=str, default="./dataset/train.csv")
    parser.add_argument("--test_path", type=str, default="./dataset/test.csv")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    # Bỏ _ vì align_feature_columns sẽ tính lại feature_cols sau khi align
    train_daily, _ = prepare_product_timeseries(train_df, args.product_id)
    test_daily, _ = prepare_product_timeseries(test_df, args.product_id)

    train_daily, test_daily, feature_cols = align_feature_columns(
        train_daily, test_daily
    )

    print(train_daily)

    # X = [lag1 UnitSold, exogenous features], y = Units Sold
    y_train_raw = train_daily["Units Sold"].astype(float).values.reshape(-1, 1)
    X_train_exog = train_daily[feature_cols].astype(float).values

    y_test_raw = test_daily["Units Sold"].astype(float).values.reshape(-1, 1)
    X_test_exog = test_daily[feature_cols].astype(float).values

    # # Lag1 train: shift 1 bước, điểm đầu lấy chính nó (không có giá trị trước)
    # y_train_lag1 = np.roll(y_train_raw, 1)
    # y_train_lag1[0] = y_train_raw[0]
    # X_train_raw = np.concatenate([y_train_lag1, X_train_exog], axis=1)

    # Lag1
    y_train_lag1 = np.roll(y_train_raw, 1)
    y_train_lag1[0] = y_train_raw[0]

    # Lag7
    y_train_lag7 = np.roll(y_train_raw, 7)
    y_train_lag7[:7] = y_train_raw[:7]

    # Lag14
    y_train_lag14 = np.roll(y_train_raw, 14)
    y_train_lag14[:14] = y_train_raw[:14]

    # Combine
    X_train_raw = np.concatenate(
        [y_train_lag1, y_train_lag7, y_train_lag14, X_train_exog], axis=1
    )

    # ─────────────────────────────────────────────────────────────
    # FIX 2: Lag1 test — điểm đầu tiên lấy từ giá trị CUỐI của
    #         train (không phải y_test_raw[0]) để tránh data leakage
    #         và đảm bảo tính liên tục train → test.
    # ─────────────────────────────────────────────────────────────
    # y_test_lag1 = np.roll(y_test_raw, 1)
    # y_test_lag1[0] = y_train_raw[-1]  # ← last known value from train
    # X_test_raw = np.concatenate([y_test_lag1, X_test_exog], axis=1)

    # Lag1
    y_test_lag1 = np.roll(y_test_raw, 1)
    y_test_lag1[0] = y_train_raw[-1]

    # Lag7
    y_test_lag7 = np.roll(y_test_raw, 7)
    y_test_lag7[:7] = y_train_raw[-7:]

    # Lag14
    y_test_lag14 = np.roll(y_test_raw, 14)
    y_test_lag14[:14] = y_train_raw[-14:]

    # Combine
    X_test_raw = np.concatenate(
        [y_test_lag1, y_test_lag7, y_test_lag14, X_test_exog], axis=1
    )

    # Scale: fit chỉ trên train, transform cả train/test (tránh leakage)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_raw)
    y_train_scaled = y_scaler.fit_transform(y_train_raw).reshape(-1)

    # ─────────────────────────────────────────────────────────────
    # FIX 3: Clip sau khi transform test để tránh giá trị out-of-range
    #         (> 1 hoặc < 0) khi test có outlier ngoài range của train.
    # ─────────────────────────────────────────────────────────────
    X_test_scaled = np.clip(x_scaler.transform(X_test_raw), 0, 1)
    y_test_scaled = np.clip(y_scaler.transform(y_test_raw), 0, 1).reshape(-1)

    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled, y_train_scaled, args.window
    )
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, args.window)

    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        raise ValueError(
            "Not enough data after grouping by day. "
            "Try smaller --window or check product history length."
        )

    model = build_model(args.window, X_train_seq.shape[2])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    pred_scaled = model.predict(X_test_seq, verbose=0).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    y_true = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    r2 = float(r2_score(y_true, pred))
    print(
        f"Product={args.product_id} | Test RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}"
    )

    # Plot: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual", color="black")
    plt.plot(pred, label="Predicted", color="tomato")
    plt.title(f"Units Sold prediction (Product {args.product_id})")
    plt.xlabel("Time step (test)")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot: Training curves
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
