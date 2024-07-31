import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
import mysql.connector
from mysql.connector import Error
import streamlit as st
import matplotlib.pyplot as plt

def traderbot(login_id, server, password, symbol, lot_size, risk_percentage):
    TIMEFRAME = mt5.TIMEFRAME_M15
    HISTORY_BARS = 100
    
    lot_size = float(lot_size)
    # Fetch the model name from the database
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='CMLTB_DB',
            user='root',
            password=''
        )

        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            query = "SELECT model FROM parameters WHERE login_id = %s AND symbol = %s"
            cursor.execute(query, (login_id, symbol))
            result = cursor.fetchone()

            if result and 'model' in result:
                model_name = result['model']
            else:
                print("Model not found in database.")
                return
    except Error as e:
        print(f"Database error: {e}")
        return
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

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

    # Function to close open positions
    def close_position(position):
        order_type = mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "magic": position.magic,
            "comment": "LSTM Close Position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(close_request)
        print("Position closed, result:", result)

    # Start continuous prediction
    try:
        while True:
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

            # Define SL level
            ATR_MULTIPLIER = 2  # Multiplier for ATR-based SL
            atr = ta.atr(df['high'], df['low'], df['Adj Close'], length=14).iloc[-1]

            if prediction > current_price:
                sl_level = current_price - atr * ATR_MULTIPLIER
            else:
                sl_level = current_price + atr * ATR_MULTIPLIER

            # Close existing positions if any
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for position in positions:
                    close_position(position)

            # Determine the order type
            order_type = mt5.ORDER_TYPE_BUY if prediction > current_price else mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

            # Place the new order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl_level,
                "deviation": 10,
                "magic": 0,
                "comment": "LSTM Order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            result = mt5.order_send(request)
            print(f"{'Buy' if order_type == mt5.ORDER_TYPE_BUY else 'Sell'} Order sent, result:", result, prediction)

            # Plot the actual and predicted values using matplotlib
            plt.figure(figsize=(12, 6))
            plt.plot(df['Adj Close'].index, df['Adj Close'], label='Actual', color='blue')
            plt.plot(df['Adj Close'].index[-1], prediction, label='Predicted', marker='o', color='red')
            plt.title('Actual vs Predicted Close Prices')
            plt.xlabel('Index')
            plt.ylabel('Price')
            plt.legend()
            
            # Display the plot in Streamlit
            st.pyplot(plt)
            plt.close()

            # Sleep for a while before the next iteration (e.g., 60 minutes)
            time.sleep(60 * 60)

    except Exception as e:
        print("Error occurred:", e)

    finally:
        # Ensure MT5 is properly shut down
        mt5.shutdown()
