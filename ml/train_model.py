from sklearn.metrics import multilabel_confusion_matrix
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import fetch_all_tweets

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline

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
    df['label'] = df['positive'].apply(lambda x: 1 if x == 1 else 0)
    
    print(f"Training on {len(df)} samples...")

    # Vérifier que les deux classes sont présentes
    class_counts = df['label'].value_counts()
    if len(class_counts) < 2:
        print("Not enough class diversity to train the model (need both classes).")
        return False

    # Adapter le nombre de folds à la taille du plus petit groupe de classe
    n_splits = min(5, class_counts.min())
    if n_splits < 2:
        print("Not enough samples per class for cross-validation.")
        return False

    X_text = df['text']
    y = df['label']

    # Pipeline : le vectorizer est refit à chaque fold, uniquement sur les
    # données d'entraînement de ce fold (évite la fuite de données)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression())
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Running {n_splits}-fold cross-validation...")
    cv_results = cross_validate(
        pipeline,
        X_text, y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1']
    )

    # Matrices de confusion 
    # Prédictions obtenues par validation croisée
    y_pred = cross_val_predict(
        pipeline,
        X_text,
        y,
        cv=cv
    )

    """
    Matrices de confusion : block désactivé dans le CRON job
     # Une matrice de confusion par classe
    mcm = multilabel_confusion_matrix(y, y_pred)

    # ===============================
    # Classe négative (label = 0)
    # ===============================
    cm_negative = mcm[0]

    print("\nConfusion Matrix - Negative class")
    print(cm_negative)

    tn, fp, fn, tp = cm_negative.ravel()

    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_negative,
        display_labels=["Not Negative", "Negative"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Negative")
    plt.show()


    # ===============================
    # Classe positive (label = 1)
    # ===============================
    cm_positive = mcm[1]

    print("\nConfusion Matrix - Positive class")
    print(cm_positive)

    tn, fp, fn, tp = cm_positive.ravel()

    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_positive,
        display_labels=["Not Positive", "Positive"]
    )
    disp.plot(cmap="Greens")
    plt.title("Confusion Matrix - Positive")
    plt.show()
 """
    print("Cross-validation results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        scores = cv_results[f'test_{metric}']
        print(f"  {metric}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Une fois la performance validée par la CV, on entraîne le modèle final
    # sur l'intégralité des données disponibles
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(X_text)

    model = LogisticRegression()
    model.fit(X, y)
    
    # Save the model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print("Model trained and saved successfully.")
    return True
    

if __name__ == "__main__":
    train_and_save_model()
