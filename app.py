import os
import threading
import time
import joblib
import schedule
from flask import Flask, request, jsonify
from flasgger import Swagger

from database.init_db import init_database
from ml.train_model import train_and_save_model, MODEL_PATH, VECTORIZER_PATH
from utils.fetch_tweets import fetch_and_store_tweets

app = Flask(__name__)
swagger = Swagger(app)

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            return model, vectorizer
        except Exception as e:
            print(f"Error loading model: {e}")
    return None, None

def fetch_and_train_job():
    fetch_and_store_tweets(2000, "tweet_eval_sentiment")
    train_and_save_model()

def run_fetch_scheduler():
    print("Scheduler thread started")

    schedule.every(1).weeks.do(fetch_and_train_job)
    print(schedule.jobs)
    while True:
        print("Checking pending jobs...")
        schedule.run_pending()
        time.sleep(86400)  # Sleep for 1 day


@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    """
    Analyze sentiment of a list of strings.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: array
          items:
            type: string
          example: ["I love this!", "This is terrible."]
    responses:
      200:
        description: Sentiment scores for the input strings
        schema:
          type: object
          additionalProperties:
            type: number
            format: float
          example: {"I love this!": 0.8, "This is terrible.": -0.9}
      400:
        description: Invalid format
      503:
        description: Model not trained yet
      500:
        description: Server error
    """
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid or missing JSON payload."}), 400
            
        if not isinstance(data, list):
            return jsonify({"error": "Invalid format. Expected a JSON array (list)."}), 400
            
        if len(data) == 0:
            return jsonify({"error": "The list is empty. Please provide at least one string."}), 400
            
        if not all(isinstance(item, str) for item in data):
            return jsonify({"error": "Invalid format. All elements in the list must be strings."}), 400
        
        model, vectorizer = load_model()
        
        if not model or not vectorizer:
            return jsonify({"error": "Model not trained yet."}), 503
            
        # Transform text
        X = vectorizer.transform(data)
        
        # Predict probability
        # LogisticRegression output is [prob_negative, prob_positive]
        probs = model.predict_proba(X)
        
        results = {}
        for i, text in enumerate(data):
            # Calculate a score between -1 and 1
            # prob_positive is probability of class 1, prob_negative is probability of class 0
            prob_positive = probs[i][1]
            prob_negative = probs[i][0]
            score = prob_positive - prob_negative
            results[text] = float(score)
            
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize DB and train model on startup if needed
    print("Initializing database...")
    init_database()
    
    model, _ = load_model()
    if not model:
        print("Initial model training...")
        train_and_save_model()
        
    # Start background scheduler thread
    scheduler_thread = threading.Thread(target=run_fetch_scheduler, daemon=True)
    scheduler_thread.start()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000)
