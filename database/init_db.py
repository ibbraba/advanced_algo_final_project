import os
import time
import mysql.connector
from mysql.connector import Error

def init_database():
    host = os.getenv("DB_HOST", "db")
    user = os.getenv("DB_USER", "user")
    password = os.getenv("DB_PASSWORD", "password")
    database = os.getenv("DB_NAME", "socialmetrics")

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
                password=password
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
            print("Table is empty. Inserting dummy dataset...")
            dummy_data = [
                ("I love this product, it is amazing!", 1, 0),
                ("This is the worst service I have ever experienced.", 0, 1),
                ("Absolutely fantastic, would buy again.", 1, 0),
                ("Terrible support and broken item.", 0, 1),
                ("I am very happy with my purchase.", 1, 0),
                ("I hate this, total waste of money.", 0, 1),
                ("Great quality and fast shipping.", 1, 0),
                ("Awful experience, never again.", 0, 1),
                ("Superb! Exceeded my expectations.", 1, 0),
                ("Disgusting behavior from the staff.", 0, 1)
            ]
            insert_query = "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)"
            cursor.executemany(insert_query, dummy_data)
            conn.commit()
            print(f"Inserted {cursor.rowcount} dummy records.")
        else:
            print(f"Table already contains {count} records. No dummy data inserted.")
            
    except Error as e:
        print(f"Error during database initialization: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    init_database()
