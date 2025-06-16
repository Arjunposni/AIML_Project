# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 21:44:22 2025

@author: ACER
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Load preprocessed data
X = np.load('X.npy')
y = np.load('y.npy')

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[1]))
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Reshape for LSTM [samples, timesteps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model and scaler
model.save('stock_lstm.h5')
joblib.dump(scaler, 'scaler.save')
print("Model saved successfully!")