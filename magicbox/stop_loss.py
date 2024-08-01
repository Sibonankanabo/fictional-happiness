import MetaTrader5 as mt5
import pandas as pd
import time

# Define constants

 # 5 minutes in seconds


def stoploss(login_id, server, password, symbol):
    NEW_SL_OFFSET = 1000  # Example offset for the new SL level
    MAX_LOSS = -10  # Maximum acceptable loss in USD
    SLEEP_TIME = 60 
    while True:
        # Initialize MetaTrader 5 and login
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()

        # Attempt to log in
        authorized = mt5.login(login_id, password=password, server=server)
        if authorized:
            print(f"Connected to login_id {login_id}")
        else:
            print(f"Failed to connect to login_id {login_id}, error code: {mt5.last_error()}")
            mt5.shutdown()
            quit()

        # Retrieve current positions
        current_orders = mt5.positions_get(symbol=symbol)

        # Check if there are any open orders
        if not current_orders:
            print(f"No open orders found for the symbol '{symbol}', error code: {mt5.last_error()}")
        else:
            # Convert to DataFrame for easier inspection
            orders_df = pd.DataFrame(list(current_orders), columns=current_orders[0]._asdict().keys())
            
            print(f"Found {len(orders_df)} open orders.")
            for index, order in orders_df.iterrows():
                order_ticket = order['ticket']
                order_type = order['type']
                order_price = order['price_open']
                sl = order['sl']
                tp = order['tp']
                profit = order['profit']  # Current profit of the order

                print(f"Order Ticket: {order_ticket}, Type: {order_type}, Open Price: {order_price}, SL: {sl}, TP: {tp}, Profit: {profit}")

                # Check if the order is in profit before adjusting SL
                if profit > 0:
                    # Calculate new SL level based on order type
                    if order_type == mt5.ORDER_TYPE_BUY:
                        new_sl = order_price + NEW_SL_OFFSET
                        if sl is None or new_sl > sl:
                            new_sl = sl  # Keep existing SL if it is better
                    elif order_type == mt5.ORDER_TYPE_SELL:
                        new_sl = order_price - NEW_SL_OFFSET
                        if sl is None or new_sl < sl:
                            new_sl = sl  # Keep existing SL if it is better
                    else:
                        continue

                    # Modify the order with the new SL
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "volume": order['volume'],
                        "type": order_type,
                        "position": order_ticket,
                        "sl": new_sl,
                        "magic": order['magic'],
                        "comment": "Modified SL",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_RETURN,
                    }

                    result = mt5.order_send(request)

                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"Order {order_ticket} modified successfully: New SL={new_sl}")
                    else:
                        print(f"Failed to modify order {order_ticket}, error code: {mt5.last_error()}")

                # Close the order if the loss is greater than MAX_LOSS
                if profit < MAX_LOSS:
                    # Determine order type to set close price
                    if order_type == mt5.ORDER_TYPE_BUY:
                        close_price = mt5.symbol_info_tick(symbol).bid
                    elif order_type == mt5.ORDER_TYPE_SELL:
                        close_price = mt5.symbol_info_tick(symbol).ask
                    else:
                        continue

                    # Close the order
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": order['volume'],
                        "type": mt5.ORDER_ACTION_CLOSE_BY,
                        "position": order_ticket,
                        "price": close_price,
                        "magic": order['magic'],
                        "comment": "Closed due to loss",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_FOK,
                    }

                    result = mt5.order_send(close_request)

                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        return(f"Order {order_ticket} closed successfully.")
                    else:
                        return(f"Failed to close order {order_ticket}, error code: {mt5.last_error()}")

        # Shutdown MT5
        mt5.shutdown()
        
        # Wait for the next iteration
        return(f"Waiting for {SLEEP_TIME / 60} minutes before the next check.")
        time.sleep(SLEEP_TIME)
