import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd

# Function to connect to MySQL database and fetch data
def fetch_data():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='CMLTB_DB',
            user='root',
            password=''
        )

        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            
            # Fetch account data
            cursor.execute("SELECT * FROM `account_data`")
            account_data = cursor.fetchall()
            
            # Fetch parameters
            cursor.execute("SELECT * FROM `parameters`")
            parameters = cursor.fetchall()
            
            return account_data, parameters

    except Error as e:
        st.write(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Streamlit UI
st.title("User Accounts and Trading Parameters")

# Fetch data from the database
account_data, parameters = fetch_data()

# Display account data
st.subheader("Account Data")
if account_data:
    account_df = pd.DataFrame(account_data)
    st.dataframe(account_df)
else:
    st.write("No account data found.")

# Display parameters
st.subheader("Trading Parameters")
if parameters:
    parameters_df = pd.DataFrame(parameters)
    st.dataframe(parameters_df)
else:
    st.write("No parameters found.")
