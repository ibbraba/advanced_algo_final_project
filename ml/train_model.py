import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import fetch_all_tweets

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

def train_and_save_model():
    print("Fetching data for training...")
    tweets = fetch_all_tweets()
    
    if not tweets:
        print("No data found to train the model.")
        return False
        
    df = pd.DataFrame(tweets)
    
    if len(df) < 2:
        print("Not enough data to train the model (need at least 2 samples).")
        return False
        
    # Extract labels (1 for positive, 0 for negative)
    # If positive is 1, label is 1. If negative is 1, label is 0.
    # We assume binary classification for this example based on the db schema
    df['label'] = df['positive'].apply(lambda x: 1 if x == 1 else 0)
    
    print(f"Training on {len(df)} samples...")
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save the model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print("Model trained and saved successfully.")
    return True

if __name__ == "__main__":
    train_and_save_model()
