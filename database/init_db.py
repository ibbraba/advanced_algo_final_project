import os
import time
import mysql.connector
from mysql.connector import Error
import csv

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
            print("Table is empty. Inserting dataset from tweets.csv...")
            
            csv_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tweets.csv')
            
            try:
                with open(csv_file_path, mode='r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    next(csv_reader) # Skip header (text,positive,negative)
                    data_to_insert = []
                    for row in csv_reader:
                        if len(row) == 3:
                            data_to_insert.append((row[0], int(row[1]), int(row[2])))
                
                insert_query = "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)"
                # Insert in chunks to avoid overwhelming the database if the file is large
                chunk_size = 1000
                for i in range(0, len(data_to_insert), chunk_size):
                    cursor.executemany(insert_query, data_to_insert[i:i+chunk_size])
                
                conn.commit()
                print(f"Successfully inserted {len(data_to_insert)} records from tweets.csv.")
            except Exception as e:
                print(f"Error reading or inserting from tweets.csv: {e}")
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
