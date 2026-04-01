import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse

PRODUCT_IDS = ["P0001", "P0002", "P0003", "P0004", "P0005"]
HORIZONS = {"1 ngày": 1, "1 tuần": 7, "1 tháng": 30}


# ──────────────────────────────────────────────
# Tạo sequences cho từng horizon
# Input : X[i : i+window], Target: y[i+window+horizon-1]
# ──────────────────────────────────────────────
def create_sequences(X: np.ndarray, y: np.ndarray, window: int, horizon: int):
    Xs, ys = [], []
    for i in range(len(y) - window - horizon + 1):
        Xs.append(X[i : i + window, :])
        ys.append(y[i + window + horizon - 1])
    return np.asarray(Xs), np.asarray(ys)


def prepare_product_timeseries(df: pd.DataFrame, product_id: str):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Product ID"] == product_id].sort_values("Date")

    if df.empty:
        raise ValueError(f"Product ID '{product_id}' not found.")

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

    daily = pd.get_dummies(
        daily,
        columns=["Seasonality", "Category"],
        prefix=["Season", "Cat"],
        dtype=float,
    )
    feature_cols = [c for c in daily.columns if c not in ["Date", "Units Sold"]]
    return daily, feature_cols


def align_feature_columns(train_daily: pd.DataFrame, test_daily: pd.DataFrame):
    all_cols = sorted(
        (set(train_daily.columns) | set(test_daily.columns)) - {"Date", "Units Sold"}
    )
    train_aligned = train_daily.copy()
    test_aligned = test_daily.copy()
    for c in all_cols:
        if c not in train_aligned.columns:
            train_aligned[c] = 0.0
        if c not in test_aligned.columns:
            test_aligned[c] = 0.0
    base = ["Date", "Units Sold"]
    return train_aligned[base + all_cols], test_aligned[base + all_cols], all_cols


def build_model(window: int, n_features: int) -> keras.Model:
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(window, n_features)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mae")
    return model


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(
        np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100
    )


# ──────────────────────────────────────────────
# Train + evaluate 1 product trên 3 horizons
# ──────────────────────────────────────────────
def run_product(product_id, train_df, test_df, window, epochs, batch_size):
    train_daily, _ = prepare_product_timeseries(train_df, product_id)
    test_daily, _ = prepare_product_timeseries(test_df, product_id)
    train_daily, test_daily, feature_cols = align_feature_columns(
        train_daily, test_daily
    )

    # Features: UnitsSold (lịch sử) + Seasonality + Category + Holiday/Promotion
    all_feature_cols = ["Units Sold"] + feature_cols

    y_train_raw = train_daily["Units Sold"].astype(float).values
    y_test_raw = test_daily["Units Sold"].astype(float).values

    X_train_raw = train_daily[all_feature_cols].astype(float).values
    X_test_raw = test_daily[all_feature_cols].astype(float).values

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_scaled = x_scaler.fit_transform(X_train_raw)
    y_train_scaled = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    X_test_scaled = np.clip(x_scaler.transform(X_test_raw), 0, 1)
    y_test_scaled = np.clip(
        y_scaler.transform(y_test_raw.reshape(-1, 1)), 0, 1
    ).reshape(-1)

    results = {}
    predictions = {}

    for horizon_name, horizon in HORIZONS.items():
        X_tr, y_tr = create_sequences(X_train_scaled, y_train_scaled, window, horizon)
        X_te, _ = create_sequences(X_test_scaled, y_test_scaled, window, horizon)

        if len(X_tr) == 0 or len(X_te) == 0:
            print(f"  [{product_id}] {horizon_name}: không đủ data, bỏ qua.")
            results[horizon_name] = None
            continue

        print(f"  [{product_id}] Training horizon {horizon_name} ({horizon} bước)...")
        model = build_model(window, X_tr.shape[2])
        model.fit(
            X_tr,
            y_tr,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
            verbose=0,
        )

        pred_scaled = model.predict(X_te, verbose=0).reshape(-1, 1)
        pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
        y_true = y_test_raw[window + horizon - 1 :]
        n = min(len(y_true), len(pred))
        y_true, pred = y_true[:n], pred[:n]

        results[horizon_name] = {
            "WAPE": round(wape(y_true, pred), 2),
            "MAE": round(float(mean_absolute_error(y_true, pred)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, pred))), 2),
            "sMAPE": round(smape(y_true, pred), 2),
        }
        # Lưu raw arrays để gộp toàn bộ products khi tính metric tổng hợp
        predictions[horizon_name] = (y_true, pred)

    return results, predictions


# ──────────────────────────────────────────────
# Plot: actual vs predicted cho 1 product (3 subplots)
# ──────────────────────────────────────────────
def plot_product(product_id: str, predictions: dict):
    horizon_names = [h for h in predictions if predictions[h] is not None]
    if not horizon_names:
        return
    fig, axes = plt.subplots(1, len(horizon_names), figsize=(18, 4))
    if len(horizon_names) == 1:
        axes = [axes]
    fig.suptitle(f"Product {product_id} — Actual vs Predicted", fontsize=13)
    for ax, name in zip(axes, horizon_names):
        y_true, pred = predictions[name]
        ax.plot(y_true, label="Actual", color="black", linewidth=0.8)
        ax.plot(pred, label="Predicted", color="tomato", linewidth=1.2)
        ax.set_title(name)
        ax.set_xlabel("Time step (test)")
        ax.set_ylabel("Units Sold")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Plot: bar chart so sánh metrics tất cả products
# ──────────────────────────────────────────────
def plot_summary(all_results: dict):
    metrics = ["WAPE", "MAE", "RMSE", "sMAPE"]
    horizon_names = list(HORIZONS.keys())
    products = list(all_results.keys())

    fig, axes = plt.subplots(len(metrics), len(horizon_names), figsize=(16, 12))
    fig.suptitle("So sánh metrics theo Product & Horizon", fontsize=14)

    for col, horizon in enumerate(horizon_names):
        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            values = [
                all_results[p][horizon][metric] if all_results[p].get(horizon) else 0
                for p in products
            ]
            bars = ax.bar(products, values, color="steelblue")
            ax.set_title(f"{metric} — {horizon}", fontsize=9)
            ax.set_ylabel(metric, fontsize=8)
            for bar, v in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_path", type=str, default="./dataset/split/train.csv")
    parser.add_argument("--test_path", type=str, default="./dataset/split/test.csv")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    all_results = {}
    # Gộp raw predictions theo horizon: {horizon_name: ([y_true...], [y_pred...])}
    all_preds: dict[str, tuple[list, list]] = {h: ([], []) for h in HORIZONS}

    for product_id in PRODUCT_IDS:
        print(f"\n{'=' * 55}")
        print(f"  Product: {product_id}")
        print(f"{'=' * 55}")

        results, predictions = run_product(
            product_id,
            train_df,
            test_df,
            args.window,
            args.epochs,
            args.batch_size,
        )
        all_results[product_id] = results
        plot_product(product_id, predictions)

        # Tích lũy raw arrays để tính metric gộp sau
        for horizon_name, pair in predictions.items():
            y_true, y_pred = pair
            all_preds[horizon_name][0].append(y_true)
            all_preds[horizon_name][1].append(y_pred)

    # Tính metric gộp: nối toàn bộ y_true / y_pred của 5 products thành 1 mảng
    print("\n" + "=" * 65)
    print("TỔNG HỢP TOÀN BỘ MODEL — LSTM (gộp 5 products)")
    print("Cách tính: nối predictions của 5 products, tính metric 1 lần")
    print("=" * 65)

    summary_rows = []
    for horizon_name in HORIZONS:
        yt_all = np.concatenate(all_preds[horizon_name][0])
        yp_all = np.concatenate(all_preds[horizon_name][1])
        row = {
            "Horizon": horizon_name,
            "WAPE": round(wape(yt_all, yp_all), 2),
            "MAE": round(float(mean_absolute_error(yt_all, yp_all)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(yt_all, yp_all))), 2),
            "sMAPE": round(smape(yt_all, yp_all), 2),
            "N": len(yt_all),
        }
        summary_rows.append(row)

    final_df = pd.DataFrame(summary_rows).set_index("Horizon")
    print(final_df.to_string())

    plot_summary(all_results)


if __name__ == "__main__":
    main()
