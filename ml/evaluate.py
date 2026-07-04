import os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import fetch_all_tweets
from ml.train_model import MODEL_PATH, VECTORIZER_PATH

def evaluate_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model or vectorizer not found. Train the model first.")
        return

    print("Loading data and model for evaluation...")
    tweets = fetch_all_tweets()
    if not tweets:
        print("No data found for evaluation.")
        return
        
    df = pd.DataFrame(tweets)
    df['label'] = df['positive'].apply(lambda x: 1 if x == 1 else 0)
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    X = vectorizer.transform(df['text'])
    y_true = df['label']
    y_pred = model.predict(X)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n--- Confusion Matrix ---")
    print(cm)
    
    # Metrics
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print("\n--- Performance Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    evaluate_model()
