import os
import mysql.connector
from mysql.connector import Error

def get_connection():
    try:
        # Use env vars or defaults for docker
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "db"),
            user=os.getenv("DB_USER", "user"),
            password=os.getenv("DB_PASSWORD", "password"),
            database=os.getenv("DB_NAME", "socialmetrics")
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def fetch_all_tweets():
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, text, positive, negative FROM tweets")
        rows = cursor.fetchall()
        return rows
    except Error as e:
        print(f"Error fetching tweets: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def insert_tweets(tweets_data):
    # tweets_data should be a list of tuples: (text, positive, negative)
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        sql = "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)"
        cursor.executemany(sql, tweets_data)
        conn.commit()
        print(f"{cursor.rowcount} record(s) inserted.")
        return True
    except Error as e:
        print(f"Error inserting tweets: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
