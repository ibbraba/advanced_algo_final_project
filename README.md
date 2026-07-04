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

Il est possible de tester l'API via l'interface Swagger ou en ligne de commandes 

```bash
# Tester l'API
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "J'aime beaucoup cette vidéo, merci."}'
```