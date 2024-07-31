import streamlit as st
import requests
import json
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import mysql.connector
from mysql.connector import Error

# Define constants
def train(login_id, server, password, symbol):
    TIMEFRAME = mt5.TIMEFRAME_H4
    HISTORY_BARS = 10000

    # Initialize MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
        
    authorized = mt5.login(login_id, password=password, server=server)
    data =  mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, HISTORY_BARS)

    if authorized:
        print(f"Connected to login_id {login_id}")
    else:
        print(f"Failed to connect to login_id {login_id}, error code: {mt5.last_error()}")
        mt5.shutdown()
        quit()

    # Convert data to DataFrame
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['tick_volume'] = df['tick_volume'].astype(float)
    df.sort_values('time', inplace=True)

    # Calculate indicators
    df['RSI'] = ta.rsi(df['close'], length=9)
    df['EMAF'] = ta.ema(df['close'], length=10)
    df['EMAM'] = ta.ema(df['close'], length=26)
    df['EMAS'] = ta.ema(df['close'], length=50)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=10)

    df['Adj Close'] = df['close']
    df['TargetNextClose'] = df['close'].shift(-1)
    df.drop(['close', 'time', 'tick_volume', 'spread', 'real_volume', 'close'], axis=1, inplace=True)
    df.dropna(inplace=True)

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

    # Define features (X) and target (y)
    X = df_scaled.drop('TargetNextClose', axis=1)
    y = df_scaled['TargetNextClose']
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # Split the data into training and testing sets
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build the LSTM model
    model = Sequential([
        LSTM(150, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(150, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    # Predict using the test data
    predictions = model.predict(X_test)
    test_data_with_predictions = np.concatenate([X_test.reshape(X_test.shape[0], X_test.shape[2]), predictions], axis=1)
    test_data_with_actual = np.concatenate([X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(-1, 1)], axis=1)
    scaled_predictions = scaler.inverse_transform(test_data_with_predictions)[:, -1]
    scaled_actual = scaler.inverse_transform(test_data_with_actual)[:, -1]

    # Plotting the results with Streamlit
    plt.figure(figsize=(14, 7))
    plt.plot(scaled_actual, label='Actual')
    plt.plot(scaled_predictions, label='Predicted', linestyle='--')
    plt.title('LSTM Predictions vs Actual Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Next Close')
    plt.legend()
    
    st.pyplot(plt.gcf())  # Display the plot in Streamlit

    # Print the actual values and predicted values
    results = pd.DataFrame({'Actual': scaled_actual, 'Predicted': scaled_predictions})
    st.write(results)

    # Save the model to a file
    model_name = f'{login_id}_{symbol}.h5'
    model.save(model_name)

    # Save the model name in the database
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='CMLTB_DB',
            user='root',
            password=''
        )

        if connection.is_connected():
            cursor = connection.cursor()
            update_query = """
            UPDATE parameters 
            SET model = %s 
            WHERE login_id = %s AND symbol = %s
            """
            cursor.execute(update_query, (model_name, login_id, symbol))
            connection.commit()
            st.write(f"Model {model_name} saved to database.")

    except Error as e:
        st.write(f"Error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
