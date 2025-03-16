import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

ticker_symbol = "GOOGL"

ticker = yf.Ticker(ticker_symbol)
data_df = pd.DataFrame(ticker.history(period = "2y", interval = "1d"))
print(data_df)

print(data_df.info())
# From the print realized there's no missing data in the closing price

# Compute daily returns
data_df["Daily Return"] = data_df["Close"].pct_change()

# Drop the first row because the first return will be NaN
data_df.dropna(inplace=True)
print(data_df.head(), "\n")  

# Plot hist of daily return
data_df["Daily Return"].plot.hist(bins=50)
plt.show()

# Prepare data
x, y = [], []

# Ammount of past days to use for prediction
window_size = 10

returns = data_df["Daily Return"].values

# Loop with rolling window
for i in range(len(returns) - window_size):
    x.append(returns[i:i+window_size])
    y.append(returns[i+window_size])

# Convert to numpy array
x, y = np.asarray(x), np.asarray(y)

# Split data into train test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# Initialize model
model = LinearRegression()

# Train linear regression model
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

# Annual Return, 252 is the approximate number of trading days in a year.
cumulative_return = (data_df['Daily Return'] + 1).prod() - 1
num_days = len(data_df)
annual_return = (1 + cumulative_return) ** (252 / num_days) - 1  # Annualized
print(f"Annual Return: {annual_return:.6f}")

# Assume risk-free rate is 4% per year → Convert to daily
risk_free_rate = 0.04 / 252 

# Sharpe Ratio 
daily_mean = data_df['Daily Return'].mean()
daily_std = data_df['Daily Return'].std()
sharpe_ratio = ((daily_mean - risk_free_rate) / daily_std) * np.sqrt(252)  # Annualized
print(f"Sharpe Ratio: {sharpe_ratio:.6f}")

# Sortino Ratio 
# Filter only negative returns for downside risk
downside_returns = data_df["Daily Return"][data_df["Daily Return"] < 0]

# Compute downside standard deviation
downside_std = downside_returns.std()
sortino_ratio = ((daily_mean - risk_free_rate) / downside_std) * np.sqrt(252)  # Annualized
print(f"Sortino Ratio: {sortino_ratio:.6f}")