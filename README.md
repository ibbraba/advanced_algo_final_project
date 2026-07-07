# API d'analyse de sentiments

Cette application permet de détecter les discours haineux sur X (anciennement Twitter) en attribuant à chaque tweet un score de sentiment.

## Comment utiliser l'API

### Installation

#### Sans Docker — Prérequis

* Avoir Python 3.12 installé
* Avoir MySQL 8.0 installé **ou** une image Docker MySQL lancée

#### Clonage du projet

```bash
# Cloner le dépôt
git clone <repo>
cd advanced_algo_final_project

# Installer les dépendances
pip install -r requirements.txt
```

#### Paramétrages du serveur SQL

Il est nécessaire de configurer un fichier `.env` dans le dossier du projet avec les informations de connexion à la base de données :

```bash
DB_HOST=localhost ou [IP_DE_LA_MACHINE_MYSQL]
DB_USER=[Nom d'utilisateur MySQL]
DB_PASSWORD=[Mot de passe MySQL]
DB_NAME=socialmetrics
DB_PORT=[PORT d'hébergement du serveur MySQL]
```

### Démarrage

```bash
python app.py
```

### Test

Il est possible de tester l'API via l'interface Swagger : http://localhost:5000/apidocs/

**Endpoint : `POST /sentiment`**

Requête :

```bash
curl -X POST "http://localhost:5000/sentiment" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "[ \"I love this!\", \"This is terrible.\"]"
```

Réponse (200) :

```json
{
  "I love this!": 0.9969312730613182,
  "This is terrible.": -0.9945610262674309
}
```

La réponse est renvoyée au format JSON avec le payload suivant : `{ "tweet": score }`, où `score` est compris entre -1 (très négatif) et 1 (très positif).

**Gestion des erreurs**

L'endpoint `/sentiment` valide le payload reçu et retourne un code `400` accompagné d'un message d'erreur explicite dans les cas suivants :

```python
if data is None:
    return jsonify({"error": "Invalid or missing JSON payload."}), 400

if not isinstance(data, list):
    return jsonify({"error": "Invalid format. Expected a JSON array (list)."}), 400

if len(data) == 0:
    return jsonify({"error": "The list is empty. Please provide at least one string."}), 400

if not all(isinstance(item, str) for item in data):
    return jsonify({"error": "Invalid format. All elements in the list must be strings."}), 400
```

## Entraînement du modèle

Pour juger la qualité du modèle, on utilise la validation croisée. Le modèle utilise une `LogisticRegression` pour effectuer un apprentissage de classification supervisé :

```python
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X_text)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X, y)
```

Le paramètre `class_weight="balanced"` pondère automatiquement les classes pendant l'apprentissage, afin de compenser le déséquilibre entre tweets positifs et négatifs (voir la section [Analyse des performances](#analyse-des-performances)).

Pour cela, on utilise la fonction `evaluate_model` qui permet de :
- Entraîner le modèle sur l'ensemble des données
- Valider le modèle sur l'ensemble des données
- Afficher les résultats de la validation croisée
- Sauvegarder le modèle

Notre but est d'avoir une précision et un rappel supérieurs à 0,8, mais principalement une meilleure précision (pour éviter de valider des tweets insultants comme étant positifs).

Nous partons d'un jeu de données venant du dataset `tweet_eval_sentiment` sur HuggingFace ; 5 000 tweets sont chargés en base de données lors de l'initialisation de l'application, et la base atteint aujourd'hui environ 25 000 tweets grâce au réentraînement hebdomadaire (voir [CRON](#cron)).

### Résultats

| Métrique  | Valeur | Écart-type (±) |
|-----------|--------|-----------------|
| Accuracy  | 0,809  | 0,006           |
| Précision | 0,820  | 0,006           |
| Rappel    | 0,937  | 0,007           |
| F1-score  | 0,875  | 0,004           |

Après ajout de la pondération des classes et réentraînement sur l'ensemble de la base (~25 000 tweets), les métriques par classe sont les suivantes :

| Classe   | Précision | Rappel | F1-score |
|----------|-----------|--------|----------|
| Positive | 0,904     | 0,810  | 0,854    |
| Négative | 0,620     | 0,782  | 0,692    |

**Analyse des performances**

La pondération des classes améliore nettement la détection des tweets négatifs (rappel négatif : 54,8 % → 78,2 %) et réduit le risque de valider un tweet insultant comme positif (précision positive : 83,9 % → 90,4 %), au prix d'une accuracy globale légèrement plus faible (82,5 % → 80,2 %) et de plus de fausses alertes sur la classe négative (précision négative : 77,0 % → 62,0 %). Ce compromis est cohérent avec l'objectif du projet, qui privilégie la détection des contenus insultants plutôt que l'accuracy brute.

Les deux matrices de confusion, le détail de la comparaison avant/après pondération et les recommandations d'amélioration sont disponibles dans le [rapport d'évaluation PDF](./rapport_evaluation.pdf) joint au dépôt.

## CRON

Afin de mieux peupler la base de données, on ajoute une tâche planifiée qui récupère les derniers tweets sur les réseaux sociaux et les ajoute à la base de données.

- Toutes les semaines, le modèle récupère 2 000 nouveaux tweets depuis le dataset `tweet_eval_sentiment` sur HuggingFace.
- Seuls les tweets non présents en doublon sont chargés en base de données.
- Le modèle est ensuite automatiquement réentraîné avec les nouvelles données.

```python
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
```

```bash
# Entraîner le modèle manuellement
python ml/evaluate.py
```

```bash
# Ajouter les derniers tweets à la base de données manuellement
python utils/fetch_tweets.py
```

## Idées d'améliorations

* Nettoyage du texte avant vectorisation
* Vérification des valeurs vides
* Augmentation du jeu de données : seuls 46 000 tweets sont actuellement disponibles, il faudrait l'étendre pour obtenir un meilleur entraînement du modèle
* Importer des jeux de données de tweets difficiles à classer (mentions, emojis, fautes d'orthographe, langage propre aux tweets) — des sous-jeux existent dans `tweet_eval_sentiment` (voir https://huggingface.co/datasets/cardiffnlp/tweet_eval)
* Traduction / prise en charge de tweets en différentes langues
* Ajout d'un niveau d'intensité : les colonnes en base indiquent aujourd'hui « positif » ou « négatif » ; il faudrait nuancer avec une échelle de -2 à 2 (très négatif, négatif, neutre, positif, très positif)
* Ajuster le seuil de décision du modèle (au lieu de 0,5 par défaut) pour continuer à améliorer le rappel négatif sans perdre davantage de précision
* Envisager un ré-échantillonnage complémentaire (ex. SMOTE) si le rappel négatif doit encore progresser au-delà de 78,2 %