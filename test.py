from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime as dt
from sklearn.preprocessing import MinMaxScaler


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

INPUT_DIR_PATH = "./dataset/"


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
    sell_prices_df = pd.read_csv(INPUT_DIR_PATH + "sell_prices.csv")
    sell_prices_df = reduce_mem_usage(sell_prices_df)
    print(
        "Sell prices has {} rows and {} columns".format(
            sell_prices_df.shape[0], sell_prices_df.shape[1]
        )
    )

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

    submission_df = pd.read_csv(INPUT_DIR_PATH + "sample_submission.csv")
    return sell_prices_df, calendar_df, sales_train_validation_df, submission_df


_, calendar_df, sales_train_validation_df, _ = read_data()

# Create date index
date_index = calendar_df["date"]
dates = date_index[0:1913]
dates_list = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in dates]

sales_train_validation_df["item_store_id"] = sales_train_validation_df.apply(
    lambda x: x["item_id"] + "_" + x["store_id"], axis=1
)
DF_Sales = sales_train_validation_df.loc[:, "d_1":"d_1913"].T
DF_Sales.columns = sales_train_validation_df["item_store_id"].values

# Set Dates as index
DF_Sales = pd.DataFrame(DF_Sales).set_index([dates_list])
DF_Sales.index = pd.to_datetime(DF_Sales.index)
DF_Sales.head()

# Select arbitrary index and plot the time series
index = 6780
y = pd.DataFrame(DF_Sales.iloc[:, index])
y = pd.DataFrame(y).set_index([dates_list])
TS_selected = y
y.index = pd.to_datetime(y.index)
ax = y.plot(figsize=(30, 9), color="red")
ax.set_facecolor("lightgrey")
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.legend(fontsize=20)
plt.title(label="Sales Demand Selected Time Series Over Time", fontsize=23)
plt.ylabel(ylabel="Sales Demand", fontsize=21)
plt.xlabel(xlabel="Date", fontsize=21)
plt.show()

data = np.array(y)
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(data.reshape(-1, 1))


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length):
        _x = data[i : (i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


seq_length = 28
x, y = sliding_windows(train_data_normalized, seq_length)
print(x.shape)
print(y.shape)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = np.array(x)
dataY = np.array(y)

trainX = np.array(x[0:train_size])
trainY = np.array(y[0:train_size])

testX = np.array(x[train_size : len(x)])
testY = np.array(y[train_size : len(y)])

# Build the Model
model = keras.models.Sequential()

# First Layer
model.add(
    keras.layers.LSTM(64, return_sequences=True, input_shape=(trainX.shape[1], 1))
)

# Second Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# 3rd Layer (Dense)
model.add(keras.layers.Dense(128, activation="relu"))

# 4th Layer
model.add(keras.layers.Dropout(0.5))

# Final Output Layer
model.add(keras.layers.Dense(1))

model.summary()
model.compile(
    optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()]
)

training = model.fit(trainX, trainY, epochs=20, batch_size=32)

y_pred = model.predict(testX)
y_pred_inv = scaler.inverse_transform(y_pred)
testY_inv = scaler.inverse_transform(testY)

plt.figure(figsize=(15, 5))

plt.plot(testY_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted")

plt.legend()
plt.title("Prediction vs Actual")
plt.show()
