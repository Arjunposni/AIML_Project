import yfinance as yf
import pandas as pd
import numpy as np

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series)-n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

# Example usage
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-12-31'
n_steps = 60  # Using 60 days to predict next day

data = get_stock_data(ticker, start_date, end_date)
X, y = prepare_data(data.values, n_steps)

# Save data for model training
np.save('X.npy', X)
np.save('y.npy', y)
print("Data saved successfully!")