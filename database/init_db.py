from pandas._libs import properties
import os
import time
import mysql.connector
from mysql.connector import Error
import csv
from utils.fetch_tweets import fetch_and_store_tweets
from ml.train_model import train_and_save_model

def init_database():
    host = os.getenv("DB_HOST", "localhost")
    user = os.getenv("DB_USER", "user")
    password = os.getenv("DB_PASSWORD", "password")
    database = os.getenv("DB_NAME", "socialmetrics")
    port = os.getenv("DB_PORT", "3307")

    # Retry connection for docker-compose (wait for db to be ready)
    retries = 10
    conn = None
    for i in range(retries):
        try:
            print(f"Attempting to connect to MySQL (Attempt {i+1}/{retries})...")
            # Connect without database first to create it
            conn = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                port=port,
                database=database   
            )
            if conn.is_connected():
                break
        except Error as e:
            print(f"Database not ready yet: {e}")
            time.sleep(5)
    
    if not conn or not conn.is_connected():
        print("Failed to connect to MySQL.")
        return

    try:
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        cursor.execute(f"USE {database}")
        
        # Create tweets table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS tweets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            text TEXT NOT NULL,
            positive TINYINT NOT NULL,
            negative TINYINT NOT NULL
        )
        """
        cursor.execute(create_table_query)
        
        # Check if table is empty to insert dummy data
        cursor.execute("SELECT COUNT(*) FROM tweets")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("Table is empty. Fetching tweets and training model...")
            
            try:
                fetch_and_store_tweets(5000, "tweet_eval_sentiment")
                train_and_save_model()
            except Exception as e:
                print(f"Error fetching tweets or training model: {e}")
        else:
            print(f"Table already contains {count} records.")
            
    except Error as e:
        print(f"Error during database initialization: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    init_database()
