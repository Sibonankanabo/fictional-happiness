import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
from magicbox import train_model as tr

# Function to connect to MySQL database and fetch joined data
def fetch_joined_data():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='CMLTB_DB',
            user='root',
            password=''
        )

        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            
            # Fetch joined account data and parameters
            query = """
                        SELECT 
                account_data.login_id, 
                account_data.server,
                account_data.password,
                parameters.symbol, 
                parameters.lot_size, 
                parameters.risk_percentage 
            FROM 
                account_data 
            JOIN 
                parameters 
            ON 
                account_data.login_id = parameters.login_id

            """
            cursor.execute(query)
            joined_data = cursor.fetchall()
            
            return joined_data

    except Error as e:
        st.write(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Function to format the display in the dropdown list
def format_option(option):
    return f"Account ID: {option['login_id']} | Server: {option['server']} | Symbol: {option['symbol']}"

# Streamlit UI
st.title("Select Account and Parameters for Model Training")

# Fetch joined data from the database
joined_data = fetch_joined_data()

# Check if data is available
if joined_data:
    # Create a dropdown with formatted options
    selected_option = st.selectbox(
        "Choose Account and Parameters",
        joined_data,
        format_func=format_option
    )

    # Display selected data
    st.write("Selected Account and Parameters:")
    st.write(f"Login ID: {selected_option['login_id']}")
    st.write(f"Server: {selected_option['server']}")
    st.write(f"Symbol: {selected_option['symbol']}")
    st.write(f"Lot Size: {selected_option['lot_size']}")
    st.write(f"Risk Percentage: {selected_option['risk_percentage']}")

    # Button to proceed with the selected option (e.g., training the model)
    if st.button("Train Model"):
        st.write("Model training initiated with the selected data.")
        tr.train(selected_option['login_id'], selected_option['server'], selected_option['password'], selected_option['symbol'])
        # Add your model training code here
else:
    st.write("No data found.")

