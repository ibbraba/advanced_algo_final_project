import os
import sys
import json
from mysql.connector import Error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import insert_tweets, get_connection

CURSOR_FILE = "fetch_cursor.json"

def get_last_offset(source_name="tweet_eval_sentiment"):
    if not os.path.exists(CURSOR_FILE):
        return 0
    with open(CURSOR_FILE, "r") as f:
        data = json.load(f)
    return data.get(source_name, 0)

def update_last_offset(source_name, new_offset):
    data = {}
    if os.path.exists(CURSOR_FILE):
        with open(CURSOR_FILE, "r") as f:
            data = json.load(f)
    data[source_name] = new_offset
    with open(CURSOR_FILE, "w") as f:
        json.dump(data, f)

def filter_new_tweets(tweets_data):
    """
    Retourne uniquement les tweets dont le texte n'existe pas déjà en DB.
    tweets_data : liste de tuples (text, positive, negative)
    """
    if not tweets_data:
        return []

    conn = get_connection()
    if not conn:
        print("Could not connect to database, skipping duplicate check.")
        return tweets_data  # fallback : on ne bloque pas l'insertion si la DB est injoignable

    existing_texts = set()

    try:
        cursor = conn.cursor(dictionary=True)

        texts = [t[0] for t in tweets_data]
        chunk_size = 1000

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            placeholders = ", ".join(["%s"] * len(chunk))
            query = f"SELECT text FROM tweets WHERE text IN ({placeholders})"
            cursor.execute(query, chunk)
            rows = cursor.fetchall()
            existing_texts.update(row["text"] for row in rows)

    except Error as e:
        print(f"Error checking for duplicate tweets: {e}")
        return tweets_data  # même logique de fallback en cas d'erreur
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

    new_tweets = [t for t in tweets_data if t[0] not in existing_texts]

    print(f"{len(tweets_data) - len(new_tweets)} duplicates skipped, {len(new_tweets)} new tweets to insert.")
    return new_tweets


def fetch_and_store_tweets(num_samples, source_name="tweet_eval_sentiment"):
   
    print("Fetching tweets...")
    
    try:
        # pyrefly: ignore [missing-import]
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency: pip install datasets")
        return False

    offset = get_last_offset(source_name)

    # Sur-fetcher pour compenser les doublons éventuels et le filtrage des neutres
    fetch_size = num_samples * 2
    print(f"Fetching up to {fetch_size} tweets starting at offset {offset}...")

    try:
        ds = load_dataset(
            "cardiffnlp/tweet_eval",
            "sentiment",
            split=f"train[{offset}:{offset + fetch_size}]"
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return False

    if len(ds) == 0:
        print("No more new tweets available at this offset (end of dataset reached).")
        return False

    tweets_data = []
    for example in ds:
        text = example["text"]
        label = example["label"]

        if label == 2:
            positive, negative = 1, 0
        elif label == 0:
            positive, negative = 0, 1
        else:
            continue  # skip neutral

        tweets_data.append((text, positive, negative))

    if not tweets_data:
        print("No valid (non-neutral) tweets found in this batch.")
        return False

    # Filtrage des doublons déjà présents en DB (via la colonne text existante)
    new_tweets = filter_new_tweets(tweets_data)
    new_tweets = new_tweets[:num_samples]  # on ne garde que ce qu'il faut

    if not new_tweets:
        print("All fetched tweets were already in the database.")
        # on avance quand même le curseur pour ne pas retomber sur le même lot au run suivant
        update_last_offset(source_name, offset + len(ds))
        return False

    print(f"Inserting {len(new_tweets)} new tweets (out of {len(tweets_data)} fetched)...")

    try:
        success = insert_tweets(new_tweets)
    except Exception as e:
        print(f"Database insert failed: {e}")
        return False

    if success:
        update_last_offset(source_name, offset + len(ds) // 2)
        print("Successfully populated database.")
        return True
    else:
        print("Database insert returned failure.")
        return False

if __name__ == "__main__":
    fetch_and_store_tweets()




