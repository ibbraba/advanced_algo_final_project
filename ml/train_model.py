import os
import sys
import json
from datetime import datetime, timezone

import joblib
import pandas as pd

# Backend non-interactif : indispensable pour un script lancé par un CRON
# (pas d'écran => plt.show() planterait ou bloquerait le job).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
    StratifiedKFold,
)
from sklearn.metrics import (
    multilabel_confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import fetch_all_tweets

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


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
    df["label"] = df["positive"].apply(lambda x: 1 if x == 1 else 0)

    print(f"Training on {len(df)} samples...")

    # Vérifier que les deux classes sont présentes
    class_counts = df["label"].value_counts()
    if len(class_counts) < 2:
        print("Not enough class diversity to train the model (need both classes).")
        return False

    # --- Constat de l'analyse : le dataset est déséquilibré (plus de positifs
    # que de négatifs), ce qui faisait sur-prédire "positif" par le modèle et
    # faisait chuter le rappel sur la classe négative à 54,8 %. ---
    print(f"Répartition des classes : {class_counts.to_dict()}")

    # Adapter le nombre de folds à la taille du plus petit groupe de classe
    n_splits = min(5, class_counts.min())
    if n_splits < 2:
        print("Not enough samples per class for cross-validation.")
        return False

    X_text = df["text"]
    y = df["label"]

    # Pipeline : le vectorizer est refit à chaque fold, uniquement sur les
    # données d'entraînement de ce fold (évite la fuite de données).
    # class_weight="balanced" pondère automatiquement l'inverse de la
    # fréquence de chaque classe dans la fonction de coût, pour empêcher le
    # modèle de favoriser la classe majoritaire (positive).
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # On garde les métriques globales (calculées sur la classe positive par
    # défaut dans scikit-learn) ET on ajoute des métriques explicites par
    # classe, pour pouvoir suivre le rappel négatif d'une semaine à l'autre
    # plutôt que de se fier uniquement à l'accuracy globale.
    scoring = {
        "accuracy": "accuracy",
        "precision_pos": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall_pos": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1_pos": make_scorer(f1_score, pos_label=1, zero_division=0),
        "precision_neg": make_scorer(precision_score, pos_label=0, zero_division=0),
        "recall_neg": make_scorer(recall_score, pos_label=0, zero_division=0),
        "f1_neg": make_scorer(f1_score, pos_label=0, zero_division=0),
    }

    print(f"Running {n_splits}-fold cross-validation...")
    cv_results = cross_validate(pipeline, X_text, y, cv=cv, scoring=scoring)

    # Prédictions obtenues par validation croisée (une prédiction "out of
    # fold" par tweet), utilisées pour les matrices de confusion.
    y_pred = cross_val_predict(pipeline, X_text, y, cv=cv)

    print("\nCross-validation results:")
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        print(f"  {metric}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    print("\nClassification report (out-of-fold predictions) :")
    print(classification_report(
        y, y_pred, target_names=["Negative", "Positive"], zero_division=0
    ))

    # --- Matrices de confusion : réactivées, et sauvegardées en PNG plutôt
    # qu'affichées à l'écran (nécessaire pour tourner sans souci dans un
    # CRON). Les fichiers générés peuvent être réutilisés tels quels dans le
    # rapport d'évaluation. ---
    os.makedirs(REPORTS_DIR, exist_ok=True)
    mcm = multilabel_confusion_matrix(y, y_pred)

    # Classe négative (label = 0)
    cm_negative = mcm[0]
    tn, fp, fn, tp = cm_negative.ravel()
    print(f"\nConfusion Matrix - Negative class\n{cm_negative}")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_negative, display_labels=["Not Negative", "Negative"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Negative")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "negative_confusion_matrix.png"), dpi=150)
    plt.close()

    # Classe positive (label = 1)
    cm_positive = mcm[1]
    tn, fp, fn, tp = cm_positive.ravel()
    print(f"\nConfusion Matrix - Positive class\n{cm_positive}")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_positive, display_labels=["Not Positive", "Positive"]
    )
    disp.plot(cmap="Greens")
    plt.title("Confusion Matrix - Positive")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "postive_confusion_matrix.png"), dpi=150)
    plt.close()

    # Sauvegarde d'un résumé texte/JSON des métriques de cette exécution,
    # pour garder un historique des réentraînements hebdomadaires du CRON
    # et repérer une éventuelle dérive du modèle dans le temps.
    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "class_distribution": class_counts.to_dict(),
        "cv_metrics": {
            metric: {
                "mean": float(cv_results[f"test_{metric}"].mean()),
                "std": float(cv_results[f"test_{metric}"].std()),
            }
            for metric in scoring
        },
    }
    with open(os.path.join(REPORTS_DIR, "last_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Une fois la performance validée par la CV, on entraîne le modèle final
    # sur l'intégralité des données disponibles (même pondération de classe
    # que pendant la validation croisée, pour rester cohérent).
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(X_text)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)

    # Save the model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nModel trained and saved successfully.")
    return True


if __name__ == "__main__":
    train_and_save_model()