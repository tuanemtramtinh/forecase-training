from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import json


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


INPUT_DIR_PATH = "./dataset/"

# To make a quick run on laptop, you can cap number of series.
MAX_SERIES = 500  # e.g. 500 or None for all

SEQ_LENGTH = 28
TRAIN_RATIO = 0.67

EMB_DIM = 16
EPOCHS = 10
BATCH_SIZE = 256

# Save/load trained artifacts so you don't need to retrain every run.
ARTIFACT_DIR = "./artifacts_global_lstm"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.keras")
META_PATH = os.path.join(ARTIFACT_DIR, "meta.json")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.npz")
FORCE_TRAIN = False  # set True to retrain even if artifacts exist


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def read_data():
    calendar_df = pd.read_csv(INPUT_DIR_PATH + "calendar.csv")
    calendar_df = reduce_mem_usage(calendar_df)
    print(
        "Calendar has {} rows and {} columns".format(
            calendar_df.shape[0], calendar_df.shape[1]
        )
    )

    sales_train_validation_df = pd.read_csv(
        INPUT_DIR_PATH + "sales_train_validation.csv"
    )
    print(
        "Sales train validation has {} rows and {} columns".format(
            sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]
        )
    )
    return calendar_df, sales_train_validation_df


def make_date_index(calendar_df, n_days):
    date_index = calendar_df["date"]
    dates = date_index.iloc[:n_days].tolist()
    dates_list = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in dates]
    return pd.to_datetime(dates_list)


def make_series_matrix(sales_train_validation_df, n_days, max_series=None):
    df = sales_train_validation_df.copy()
    df["item_store_id"] = df["item_id"] + "_" + df["store_id"]
    if max_series is not None:
        df = df.iloc[:max_series].reset_index(drop=True)

    ids = df["item_store_id"].tolist()
    sales = df.loc[:, "d_1" : f"d_{n_days}"].to_numpy(dtype=np.float32)
    return ids, sales


def minmax_scale_per_series(values, eps=1e-6):
    vmin = values.min(axis=1, keepdims=True)
    vmax = values.max(axis=1, keepdims=True)
    denom = np.maximum(vmax - vmin, eps)
    scaled_01 = (values - vmin) / denom
    scaled = scaled_01 * 2.0 - 1.0
    return scaled.astype(np.float32), vmin.astype(np.float32), vmax.astype(np.float32)


def minmax_inverse_per_series(scaled_values, vmin, vmax):
    scaled_01 = (scaled_values + 1.0) / 2.0
    return scaled_01 * (vmax - vmin) + vmin


def build_windows_global(series_scaled, seq_length, train_ratio):
    n_series, n_days = series_scaled.shape
    windows_per_series = n_days - seq_length
    if windows_per_series <= 0:
        raise ValueError("Not enough days for given SEQ_LENGTH.")

    n_train_per_series = int(windows_per_series * train_ratio)
    if n_train_per_series <= 0 or n_train_per_series >= windows_per_series:
        raise ValueError("Bad TRAIN_RATIO; results in empty train or test.")

    x_train, id_train, y_train = [], [], []
    x_test, id_test, y_test = [], [], []

    for sid in range(n_series):
        s = series_scaled[sid]
        for t in range(seq_length, n_days):
            xw = s[t - seq_length : t]
            yw = s[t]
            idx = t - seq_length
            if idx < n_train_per_series:
                x_train.append(xw)
                y_train.append(yw)
                id_train.append(sid)
            else:
                x_test.append(xw)
                y_test.append(yw)
                id_test.append(sid)

    x_train = np.asarray(x_train, dtype=np.float32)[..., None]
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    id_train = np.asarray(id_train, dtype=np.int32).reshape(-1, 1)

    x_test = np.asarray(x_test, dtype=np.float32)[..., None]
    y_test = np.asarray(y_test, dtype=np.float32).reshape(-1, 1)
    id_test = np.asarray(id_test, dtype=np.int32).reshape(-1, 1)

    return (x_train, id_train, y_train), (x_test, id_test, y_test)


def make_model(seq_length, n_series, emb_dim):
    seq_in = keras.Input(shape=(seq_length, 1), name="seq_in")
    id_in = keras.Input(shape=(1,), dtype="int32", name="series_id_in")

    x = keras.layers.LSTM(64, return_sequences=True)(seq_in)
    x = keras.layers.LSTM(64, return_sequences=False)(x)

    emb = keras.layers.Embedding(input_dim=n_series, output_dim=emb_dim)(id_in)
    emb = keras.layers.Flatten()(emb)

    h = keras.layers.Concatenate()([x, emb])
    h = keras.layers.Dense(128, activation="relu")(h)
    h = keras.layers.Dropout(0.5)(h)
    out = keras.layers.Dense(1, name="y_out")(h)

    model = keras.Model(inputs=[seq_in, id_in], outputs=out)
    model.compile(
        optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()]
    )
    return model


def save_artifacts(model, series_ids, vmin, vmax, config):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "series_ids": series_ids,
                "config": config,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    np.savez(SCALER_PATH, vmin=vmin, vmax=vmax)


def load_artifacts():
    if not (
        os.path.exists(MODEL_PATH)
        and os.path.exists(META_PATH)
        and os.path.exists(SCALER_PATH)
    ):
        return None
    model = keras.models.load_model(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scaler = np.load(SCALER_PATH)
    vmin = scaler["vmin"]
    vmax = scaler["vmax"]
    return model, meta, vmin, vmax


def main():
    calendar_df, sales_df = read_data()
    n_days = 1913
    dates = make_date_index(calendar_df, n_days)

    loaded = None if FORCE_TRAIN else load_artifacts()
    if loaded is not None:
        model, meta, vmin, vmax = loaded
        series_ids = meta["series_ids"]
        cfg = meta.get("config", {})
        print(f"Loaded model from {MODEL_PATH}")
        print(f"Artifacts config: {cfg}")
    else:
        series_ids, sales = make_series_matrix(
            sales_df, n_days=n_days, max_series=MAX_SERIES
        )
        print(f"Series count: {len(series_ids)}, days: {sales.shape[1]}")

        sales_scaled, vmin, vmax = minmax_scale_per_series(sales)
        (x_train, id_train, y_train), (x_test, id_test, y_test) = build_windows_global(
            sales_scaled, seq_length=SEQ_LENGTH, train_ratio=TRAIN_RATIO
        )
        print("Train:", x_train.shape, id_train.shape, y_train.shape)
        print("Test :", x_test.shape, id_test.shape, y_test.shape)

        model = make_model(
            seq_length=SEQ_LENGTH, n_series=len(series_ids), emb_dim=EMB_DIM
        )
        model.summary()

        model.fit(
            {"seq_in": x_train, "series_id_in": id_train},
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.05,
            shuffle=True,
        )

        save_artifacts(
            model=model,
            series_ids=series_ids,
            vmin=vmin,
            vmax=vmax,
            config={
                "max_series": MAX_SERIES,
                "seq_length": SEQ_LENGTH,
                "train_ratio": TRAIN_RATIO,
                "emb_dim": EMB_DIM,
            },
        )
        print(f"Saved model + metadata to {ARTIFACT_DIR}")

    # Build test set for evaluation/plotting (always rebuild from raw data).
    series_ids_now, sales_now = make_series_matrix(
        sales_df, n_days=n_days, max_series=MAX_SERIES
    )
    if series_ids_now != series_ids:
        raise ValueError(
            "Current series_ids differ from saved artifacts. "
            "Keep MAX_SERIES consistent (or set FORCE_TRAIN=True)."
        )
    sales_scaled_now, _, _ = minmax_scale_per_series(sales_now)
    (_, _, _), (x_test, id_test, y_test) = build_windows_global(
        sales_scaled_now, seq_length=SEQ_LENGTH, train_ratio=TRAIN_RATIO
    )

    y_pred = model.predict({"seq_in": x_test, "series_id_in": id_test}, batch_size=4096)

    sample_sid = int(np.median(id_test))
    mask = id_test.reshape(-1) == sample_sid
    if not np.any(mask):
        sample_sid = int(id_test[0, 0])
        mask = id_test.reshape(-1) == sample_sid

    y_pred_s = y_pred[mask]
    y_true_s = y_test[mask]

    y_pred_inv = minmax_inverse_per_series(y_pred_s, vmin[sample_sid], vmax[sample_sid])
    y_true_inv = minmax_inverse_per_series(y_true_s, vmin[sample_sid], vmax[sample_sid])

    windows_per_series = n_days - SEQ_LENGTH
    n_train_per_series = int(windows_per_series * TRAIN_RATIO)
    test_start_t = SEQ_LENGTH + n_train_per_series
    n_points = y_true_inv.shape[0]
    date_slice = dates[test_start_t : test_start_t + n_points]

    plt.figure(figsize=(15, 5))
    plt.plot(date_slice, y_true_inv.reshape(-1), label="Actual")
    plt.plot(date_slice, y_pred_inv.reshape(-1), label="Predicted")
    plt.legend()
    plt.title(f"Global model - sample series: {series_ids[sample_sid]}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
