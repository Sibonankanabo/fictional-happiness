import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt

def show(login_id, server, password, symbol, model_name):
    TIMEFRAME = mt5.TIMEFRAME_H4
    HISTORY_BARS = 100

    # Initialize MetaTrader 5 and login
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # Attempt to log in
    authorized = mt5.login(login_id, password=password, server=server)
    if not authorized:
        print(f"Failed to connect to account {login_id}, error code: {mt5.last_error()}")
        mt5.shutdown()
        quit()

    print(f"Connected to account {login_id}")

    # Load the trained model
    model = load_model(model_name)

    # Fetch latest market data
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, HISTORY_BARS)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['tick_volume'] = df['tick_volume'].astype(float)

    # Calculate indicators
    df['RSI'] = ta.rsi(df['close'], length=9)
    df['EMAF'] = ta.ema(df['close'], length=10)
    df['EMAM'] = ta.ema(df['close'], length=26)
    df['EMAS'] = ta.ema(df['close'], length=50)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=10)

    df['Adj Close'] = df['close']
    df['TargetNextClose'] = df['close'].shift(-1)
    df.drop(['tick_volume', 'spread', 'time', 'real_volume', 'close'], axis=1, inplace=True)
    df.dropna(inplace=True)

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

    # Prepare data for prediction
    X_new_data = df_scaled.drop('TargetNextClose', axis=1)
    X_new_data = np.reshape(X_new_data, (X_new_data.shape[0], 1, X_new_data.shape[1]))

    # Make predictions
    predictions = model.predict(X_new_data)
    prediction_scaled = predictions[-1][0]

    # Inverse scale the prediction to get the original scale
    prediction_df = pd.DataFrame([prediction_scaled] * df_scaled.shape[1], index=df.columns).T
    prediction = scaler.inverse_transform(prediction_df).flatten()[df.columns.get_loc('TargetNextClose')]

    current_price = df['Adj Close'].iloc[-1]

    # Plot the actual and predicted values using matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(df['Adj Close'].index, df['Adj Close'], label='Actual', color='blue')
    plt.plot([df.index[-1]], [prediction], label='Predicted', marker='o', color='red')
    plt.title('Actual vs Predicted Close Prices')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    
    # Display the plot in Streamlit
    st.pyplot(plt)
    plt.close()

    # Print the actual and predicted values in a table
    actual_predicted_df = pd.DataFrame({
        'Actual': [current_price],
        'Predicted': [prediction]
    })

    st.write("Actual and Predicted Values")
    st.dataframe(actual_predicted_df)

    # Shutdown MT5
    mt5.shutdown()

# Example usage
# if __name__ == "__main__":
#     # Load the parameters
#     login_id = 123456  # Replace with your MT5 login ID
#     server = "YourServerName"  # Replace with your server name
#     password = "YourPassword"  # Replace with your password
#     symbol = "EURUSD"  # Replace with your symbol
#     model_name = "your_model.h5"  # Replace with your model file name
    
#     # Call the show function
#     show(login_id, server, password, symbol, model_name)
