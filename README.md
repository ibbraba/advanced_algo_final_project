# API d'analyse de sentiments

Cette application permet de détecter les discours haineux.

## Comment utiliser l'API

### Prérequis

- Avoir Docker et Docker Compose installés
- Avoir Python 3.12 installé

### Installation

#### Avec Docker

Ouvrir un terminal dans le dossier du projet et lancer les commandes suivantes :

```bash
docker-compose up --build
```

#### Sans Docker

#### Prérequis

* Avoir Python 3.12 installé
* Avoir MySQL 8.0 installé

```bash
# Cloner le dépôt
git clone <repo>
cd advanced_algo_final_project

# Installer les dépendances
pip install -r requirements.txt


```




### Démarrage

```bash
# Démarrer le conteneur Docker
./run-api.sh
```

### Test

Il est possible de tester l'API via l'interface Swagger : http://localhost:5000/apidocs/


ou en ligne de commandes 

```bash
# Tester l'API
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "J'aime beaucoup cette vidéo, merci."}'
```


# Entrainement du modèle 

Pour juger la qualité du modèle, on utilise la validation croisée. 
Pour cela, on utilise la fonction `evaluate_model` qui permet de :
- Entrainer le modèle sur l'ensemble des données
- Valider le modèle sur l'ensemble des données
- Afficher les résultats de la validation croisée
- Sauvegarder le modèle

Notre but est d'avoir une précision et un rappel supérieur à 0.8 mais principalement une meilleure précision. (pour éviter de valider des tweets insultants comme étant positifs)

Nous partons avec un jeu de données avec 50000 tweets au format .csv, les tweets sont chargés en base de données lors de l'initialisation de l'application. 

Après l'entrainement sur fichier csv nous obtenons les metriques suivantes 

Cross-validation results:

  accuracy: 1.000 (+/- 0.000)

  precision: 1.000 (+/- 0.000)

  recall: 1.000 (+/- 0.000)

  f1: 1.000 (+/- 0.000)

Mais nous nous appuyons sur un jeu de données simple, il faudrait complexifier les tweets 

# CRON  

Afin de mieux peupler la base de données, on ajoute un cron qui va récupérer les derniers tweets sur les réseaux sociaux et les ajouter à la base de données. 
Toutes les semaines, le modèle va fetch 2000 nouveaux tweets depuis le modèle tweet_eval_sentiment depuis HuggingFace 
Seuls les tweets n'étant pas ajoutés en doublons seront chargés en base de données 




```bash
# Entrainer le modèle
python ml/evaluate.py
```



```bash
# Ajouter les derniers tweets à la base de données
python utils/fetch_tweets.py
```

# Idées d'améliorations 

* Nettoyage texte avant vectorisation 
* Verification des valeurs vides 
* Importer des jeu de données de tweets difficile à classer (mentions, emojis, fautes d'orthographe, language de tweets)
* Traduction / tweets en différentes langues 
* Ajout d'intensités : Les colonnes en BDD indiquent positif ou négatif, il faudrait nuancer les données en ajoutant une colonne d'intensité allant de -2 à 2. (Très négatif, négatif, neutre, positif, très positif)
