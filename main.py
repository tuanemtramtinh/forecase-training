from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

data = pd.read_csv("./MicrosoftStock.csv")

print(data.head())
print(data.info())
print(data.describe())


# Initial Data Visualization
# Plot 1 - Open and Close Prices Of time

plt.figure(figsize=(12, 6))
plt.plot(data["date"], data["open"], label="Open", color="blue")
plt.plot(data["date"], data["close"], label="Close", color="red")
plt.title("Open-Close Price over Time")
plt.legend()
# plt.show()

# Plot 2 - Trading Volume (Check for outliers)
plt.figure(figsize=(12, 6))
plt.plot(data["date"], data["volume"], label="Volume", color="orange")
plt.title("Stoc Volume over Time")
# plt.show()

# Drop non-numeric Columns
numeric_data = data.select_dtypes(include=["int64", "float64"])

# Plot 3 - Check correlation between features
plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Feature Correlation Heatmap")
# plt.show()


# Convẻt the Data into Date time then create a date filter
data["date"] = pd.to_datetime(data["date"])

prediction = data.loc[
    (data["date"] > datetime(2013, 1, 1)) & (data["date"] < datetime(2018, 1, 1))
]


plt.figure(figsize=(12, 6))
plt.plot(data["date"], data["close"], color="blue")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over Time")

# Prepare for LSTM Model (Sequential)
stock_close = data.filter(["close"])
dataset = stock_close.values  # convert to numpy array
training_data_len = int(np.ceil(len(dataset) * 0.95))

# Preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len]  # 95%

X_train, Y_train = [], []

# Create a sliding window for out stock (60 days)
for i in range(60, len(training_data)):
    X_train.append(training_data[i - 60 : i, 0])
    Y_train.append(training_data[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the Model
model = keras.models.Sequential()

# First Layer
model.add(
    keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1))
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

training = model.fit(X_train, Y_train, epochs=20, batch_size=32)

# Prep the test data
test_data = scaled_data[training_data_len - 60 :]
X_test, Y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60 : i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Male a Prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plotting data
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test["Predictions"] = predictions

plt.figure(figsize=(12, 8))
plt.plot(train["date"], train["close"], label="Train (Actual)", color="blue")
plt.plot(test["date"], test["close"], label="Test (Actual)", color="orange")
plt.plot(test["date"], test["Predictions"], label="Predictions", color="red")
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
