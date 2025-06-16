# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 21:47:30 2025

@author: ACER
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model('stock_lstm.h5')
scaler = joblib.load('scaler.save')

st.title('Stock Price Prediction')
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL):', 'AAPL')
days = st.slider('Days of History to Show:', 30, 365, 180)

if st.button('Predict'):
    # Get recent data
    data = yf.download(ticker, period=f'{days}d')['Close'].values
    scaled_data = scaler.transform(data.reshape(-1, 1))
    
    # Prepare last sequence
    last_sequence = scaled_data[-60:].reshape(1, 60, 1)
    
    # Make prediction
    predicted_price = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data, label='Historical Prices')
    ax.axhline(y=predicted_price[0][0], color='r', linestyle='--', label='Predicted Next Day Price')
    ax.set_title(f'{ticker} Stock Price Prediction')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price ($)')
    ax.legend()
    st.pyplot(fig)
    
    st.success(f'Predicted next day closing price: ${predicted_price[0][0]:.2f}')