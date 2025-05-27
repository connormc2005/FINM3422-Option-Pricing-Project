import datetime as dt
import yfinance as yf
import pandas as pd

# ASX tickers (".AX" suffix)
tickers = ["BHP.AX", "CSL.AX", "WDS.AX", "MQG.AX"]

# Define your end date (last day of pricing) and compute start date = end - 60 days
end_date = dt.datetime(2025, 5, 17)
start_date = end_date - dt.timedelta(days=62)

# Download data between start_date and end_date (inclusive of end_date)
data = yf.download(tickers, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

# Extract closing prices
close_prices = data["Close"]

# Show first few rows of closing prices
print(f"Closing Prices (from {start_date.date()} to {end_date.date()}):")
print(close_prices.head())
print(close_prices.tail())
# Calculate the correlation matrix
corr_matrix = close_prices.corr()

# Display the correlation matrix
print(f"\nCorrelation Matrix (based on Close prices):")
print(corr_matrix)
