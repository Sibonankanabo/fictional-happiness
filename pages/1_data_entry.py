import streamlit as st
import MetaTrader5 as mt5
import mysql.connector
from mysql.connector import Error

# Function to get symbols from MetaTrader5
def get_symbols():
    # Initialize MetaTrader5 and login
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    symbols = mt5.symbols_get()
    symbol_names = [s.name for s in symbols]
    return symbol_names

# Function to save data to MySQL
def save_to_mysql(login_id, server, password, symbol, lot_size, risk_percentage):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='CMLTB_DB',
            user='root',
            password=''
        )

        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            
            # Check if login_id already exists
            cursor.execute("SELECT * FROM `account_data` WHERE `login_id` = %s", (login_id,))
            account_exists = cursor.fetchone()

            if account_exists:
                st.write(f"Account with Login ID {login_id} already exists. Adding new parameters.")
                
                # Save parameters only
                cursor.execute(
                    "INSERT INTO `parameters` (`login_id`, `symbol`, `lot_size`, `risk_percentage`) VALUES (%s, %s, %s, %s)",
                    (login_id, symbol, lot_size, risk_percentage)
                )
            else:
                st.write(f"New account. Saving account data and parameters.")
                
                # Save account data
                cursor.execute(
                    "INSERT INTO `account_data` (`login_id`, `server`, `password`) VALUES (%s, %s, %s)",
                    (login_id, server, password)
                )

                # Save parameters
                cursor.execute(
                    "INSERT INTO `parameters` (`login_id`, `symbol`, `lot_size`, `risk_percentage`) VALUES (%s, %s, %s, %s)",
                    (login_id, symbol, lot_size, risk_percentage)
                )

            connection.commit()
            st.write("Data saved to MySQL database successfully.")
    except Error as e:
        st.write(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            st.write("MySQL connection is closed.")

# Streamlit form for user input
with st.form("my_form"):
    st.write("Login to your account")
    login_id = st.text_input("Login ID")
    server = st.text_input("Server")
    password = st.text_input("Password", type='password')
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        # Check if all required fields are filled
        if not login_id or not server or not password:
            st.write("Please fill in all fields.")
        else:
            if mt5.initialize():
                login_id_int = int(login_id)
                authorized = mt5.login(login_id_int, password=password, server=server)
                
                if authorized:
                    st.write("You have logged in successfully. Continue.")
                    symbols = get_symbols()
                    selected_symbol = st.selectbox(
                        "Choose symbol",
                        symbols,
                        index=0
                    )
                    lot_size = st.text_input("Lot size")
                    risk_percentage = st.text_input("Risk percentage")
                    
                    if not lot_size or not risk_percentage:
                        st.write("Please enter the lot size and risk percentage.")
                    else:
                        save_to_mysql(login_id, server, password, selected_symbol, lot_size, risk_percentage)
                else:
                    st.write("Wrong credentials")
            else:
                st.write("Failed to initialize MetaTrader5")
