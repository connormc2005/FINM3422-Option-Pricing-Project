import yfinance as yf
import datetime
import pandas as pd

def get_spot_prices(date="2025-05-16"):
    """
    Fetches the closing prices on a specific date from Yahoo Finance.
    Returns a dictionary of ticker -> closing price.
    """
    target_date = datetime.datetime.strptime(date, "%Y-%m-%d")

    tickers = {
        'BHP': 'BHP.AX',
        'CSL': 'CSL.AX',
        'WDS': 'WDS.AX',
        'MQG': 'MQG.AX'
    }

    spot_prices = {}

    for name, ticker in tickers.items():
        data = yf.download(
            ticker,
            start=target_date.strftime("%Y-%m-%d"),
            end=(target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False
        )

        if data.empty:
            print(f"Warning: No data for {name} ({ticker}) on {date}")
            spot_prices[name] = None
        else:
            spot_prices[name] = round(data['Close'].iloc[0], 2)

    return spot_prices

def get_correlation_matrix(date="2025-05-16", window=30):
    """
    Fetches adjusted close prices for the 4 stocks over a rolling window prior to the given date
    and calculates the correlation matrix of daily returns.
    """
    end_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    start_date = end_date - datetime.timedelta(days=window * 2)  # Extra buffer for weekends/holidays

    tickers = {
        'BHP': 'BHP.AX',
        'CSL': 'CSL.AX',
        'WDS': 'WDS.AX',
        'MQG': 'MQG.AX'
    }

    df = yf.download(
        list(tickers.values()),
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False
    )['Close']

    # Rename columns to match your internal tickers
    df.columns = tickers.keys()

    # Drop rows with missing data (e.g. public holidays)
    df = df.dropna()

    # Restrict to the last `window` rows
    df = df.tail(window)

    # Calculate daily returns
    returns = df.pct_change().dropna()

    # Compute correlation matrix
    corr_matrix = returns.corr()

    return corr_matrix

if __name__ == "__main__":
    # Step 1: Spot Prices
    prices = get_spot_prices("2025-05-16")
    print("Spot Prices as of 16 May 2025:")
    for ticker, price in prices.items():
        print(f"{ticker}: ${price}")

    print("\n30-Day Correlation Matrix (Daily Returns):")
    correlation_matrix = get_correlation_matrix("2025-05-16", window=30)
    print(correlation_matrix.round(4))
