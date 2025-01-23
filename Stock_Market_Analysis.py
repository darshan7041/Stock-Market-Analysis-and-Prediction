# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Step 1: Data Collect
ticker = "MSFT"
data = yf.download(ticker, start="2010-01-01", end="2023-01-01")
if data.empty or 'Close' not in data:
    raise ValueError(f"No data found for ticker '{ticker}'. Please verify the ticker or date range.")
data = data[['Close']]
print(data.head())

# Step 2: Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare train and test data
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# sequences for LSTM
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)


# Reshape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Step 3: LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=20)


# Step 4: Predictions and Evaluation
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# Evaluate with MSE
mse = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions)
print(f"Mean Squared Error: {mse}")

# Step 5: Visualization
valid = data[train_size + seq_length:]
valid['Predictions'] = predictions

plt.figure(figsize=(14, 7))
train = data[:train_size]
plt.plot(train['Close'], label='Training Data')
plt.plot(valid['Close'], label='Actual Price')
plt.plot(valid['Predictions'], label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
