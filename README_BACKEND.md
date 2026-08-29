# Backend - API de Recommandation de Films

Ce document décrit l'architecture, le fonctionnement et le déploiement du backend FastAPI du système de recommandation.

---

## Vue d'ensemble

Le backend expose une API REST **FastAPI** qui agrège les prédictions de trois modèles de recommandation :

- **LightGCN** (Graph Neural Network)
- **SVD** (Factorisation de matrice)
- **Item-Item CF** (Filtrage collaboratif item-based)

Pour un utilisateur donné, l'API renvoie simultanément le Top-N des recommandations de chaque modèle, avec métadonnées de films et scores de pertinence.

### Stack technique

| Composant | Technologie |
|-----------|-------------|
| Framework API | FastAPI 0.141+ |
| Serveur ASGI | Uvicorn |
| ML - LightGCN | PyTorch |
| ML - Baselines | Scikit-Surprise (SVD, Item-Item) |
| Données | Pandas / CSV |
| Sérialisation | Pickle (modèles), JSON (API) |
| Logs | Python `logging` |

---

## Prérequis

- **Python** : 3.9 à 3.11 (recommandé)
- **pip** : à jour
- **Git** : pour cloner le dépôt
- (Optionnel) **Node.js 18+** : si vous souhaitez lancer le frontend React en parallèle

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Prosper015/recommender-gnn-vs-cf
cd recommender-gnn-vs-cf
```

### 2. Créer et activer un environnement virtuel

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Dépendances clés** (déjà présentes dans `requirements.txt`) :

- `fastapi` : framework web
- `uvicorn` : serveur ASGI
- `pandas` : manipulation des données films
- `torch` : chargement et inférence LightGCN
- `pydantic` : validation (utilisé implicitement par FastAPI)

> **Note** : Si vous comptez entraîner ou utiliser les modèles SVD / Item-Item avec `scikit-surprise`, installez-le séparément :
> ```bash
> pip install scikit-surprise
> ```

---

## Configuration

### Variables d'environnement

Aucune variable d'environnement obligatoire n'est requise pour le mode développement. Les chemins sont définis en dur dans `src/services/model_service.py` :

```python
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

LIGHTGCN_MODEL_PATH = MODELS_DIR / "lightgcn_best.pt"
SVD_MODEL_PATH = MODELS_DIR / "svd_model.pkl"
ITEM_ITEM_MODEL_PATH = MODELS_DIR / "item_item_model.pkl"
MOVIES_CSV_PATH = DATA_DIR / "movies_cleaned.csv"
```

### Fichiers attendus (obligatoires en production)

Ces artefacts sont générés par `python -m scripts.run_pipeline` (voir plus bas).
Le backend **ne sert plus aucune donnée fictive** : tout est réel.

| Fichier | Rôle | Requis |
|---------|------|--------|
| `data/processed/movies_cleaned.csv` | Métadonnées des films (`movieId`, `title`, `genres`) | **Oui** |
| `models/id_mappings.json` | Mappings MovieLens brut ↔ index internes (LightGCN) | **Oui** |
| `models/lightgcn_best.pt` | Poids du modèle LightGCN (PyTorch) | Oui (sinon LightGCN vide) |
| `models/svd_model.pkl` | Modèle SVD sérialisé (pickle) | Oui (sinon SVD vide) |
| `models/item_item_model.pkl` | Modèle Item-Item CF sérialisé (pickle) | Oui (sinon Item-Item vide) |

Si `movies_cleaned.csv` ou `id_mappings.json` manquent, `is_loaded` reste `false` et
l'endpoint `/api/v1/recommendations/{user_id}` répond `503`. Un modèle manquant
renvoie simplement un tableau vide pour ce modèle (pas de faux de données).

### CORS

Le CORS est configuré dans `src/api/main.py` pour autoriser le frontend Vite :

```python
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
```

---

## Lancement

### Démarrer le serveur de développement

```bash
# Depuis la racine du projet
uvicorn src.api.main:app --reload --port 8000
```

- **API** : http://127.0.0.1:8000
- **Documentation interactive (Swagger UI)** : http://127.0.0.1:8000/docs
- **Documentation alternative (ReDoc)** : http://127.0.0.1:8000/redoc
- **Healthcheck** : http://127.0.0.1:8000/

Le flag `--reload` active le rechargement automatique à chaque modification du code Python (idéal en développement, à désactiver en production).

### Vérifier que le service est opérationnel

```bash
curl http://127.0.0.1:8000/
```

Réponse attendue :

```json
{
  "status": "online",
  "message": "API de Recommandation opérationnelle",
  "models_loaded": true
}
```

---

## Structure du code

```
src/
├── api/
│   ├── __init__.py
│   └── main.py              # Application FastAPI, endpoints, CORS, lifespan
└── services/
    ├── __init__.py
    └── model_service.py     # Logique métier : chargement des modèles, recommandations
```

### `src/api/main.py`

Point d'entrée de l'application. Responsabilités :

- Initialisation de l'application FastAPI
- Configuration du middleware CORS
- Gestion du cycle de vie (`lifespan`) : chargement des artefacts au démarrage
- Définition des routes et délégation au `ModelService`

### `src/services/model_service.py`

Couche service (métier). Responsabilités :

- Chargement des artefacts réels (modèles PyTorch / pickle, CSV films, mappings d'index)
- Génération des recommandations pour chaque modèle (LightGCN, SVD, Item-Item)
- Conversion d'index : LightGCN travaille en indices internes reindexés, le backend
  les convertit en `movieId` bruts MovieLens via `models/id_mappings.json`
- Formatage uniforme des résultats (mapping `movieId` → `title`, `genres`, `score`)
- **Plus de fallback mock** : un modèle non chargé renvoie `[]` ; un `user_id`
  inconnu déclenche une `UnknownUserError` → `404` côté API

---

## Génération des artefacts (démockage)

Avant de lancer l'API en production, générez les vrais artefacts depuis MovieLens :

```bash
# Depuis la racine du projet (environnement actif)
python -m scripts.run_pipeline --dataset 100k --epochs 50
```

Ce script unique : télécharge MovieLens, parse films + ratings, applique le split
LOO temporel, entraîne et exporte les 3 modèles + `movies_cleaned.csv` + `id_mappings.json`.
Ajoutez `--no-mlflow` si MLflow n'est pas installé (logging optionnel).

## Endpoints API

### `GET /`

Healthcheck de l'API.

**Réponse 200 :**

```json
{
  "status": "online",
  "message": "API de Recommandation opérationnelle",
  "models_loaded": true
}
```

---

### `GET /api/v1/users`

Retourne la liste des identifiants utilisateurs disponibles pour les tests.

**Réponse 200 :** liste réelle des `user_id` connus du dataset (MovieLens 100k → 943 utilisateurs).

```json
{
  "users": [1, 2, 3, 4, 5, "...", 943]
}
```

---

### `GET /api/v1/recommendations/{user_id}`

Génère les recommandations pour un utilisateur donné.

**Paramètres de requête :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `user_id` | `int` | - (obligatoire dans le chemin) | Identifiant de l'utilisateur |
| `top_n` | `int` | `5` | Nombre de recommandations par modèle (min: 1, max: 20) |

**Exemple de requête :**

```bash
curl "http://127.0.0.1:8000/api/v1/recommendations/1?top_n=5"
```

**Réponse 200 :**

```json
{
  "user_id": 1,
  "top_n": 5,
  "lightgcn": [
    {
      "movieId": 1,
      "title": "Inception",
      "genres": "Sci-Fi|Thriller",
      "score": 0.95
    }
  ],
  "svd": [
    {
      "movieId": 2,
      "title": "The Dark Knight",
      "genres": "Action|Crime|Drama",
      "score": 4.25
    }
  ],
  "item_item": [
    {
      "movieId": 3,
      "title": "Interstellar",
      "genres": "Sci-Fi|Adventure|Drama",
      "score": 0.88
    }
  ]
}
```

---

## Format des réponses

### Structure d'une recommandation

Chaque élément dans les tableaux `lightgcn`, `svd` et `item_item` suit ce schéma :

| Champ | Type | Description |
|-------|------|-------------|
| `movieId` | `integer` | Identifiant unique du film |
| `title` | `string` | Titre du film |
| `genres` | `string` | Genres séparés par `\|` (ex: `"Sci-Fi\|Thriller"`) |
| `score` | `float` | Score de pertinence prédit par le modèle |

### Codes d'erreur

| Code | Signification | Cas d'occurrence |
|------|---------------|------------------|
| `200` | Succès | La requête est valide et les recommandations ont été générées |
| `400` | Requête invalide | Paramètres mal formés (ex: `top_n` hors bornes) |
| `404` | Utilisateur non trouvé | `user_id` inexistant dans le dataset d'entraînement (cold-start) |
| `503` | Service indisponible | `movies_cleaned.csv` ou `id_mappings.json` manquants (`is_loaded = false`) |
| `500` | Erreur serveur | Exception non gérée pendant la génération des recommandations |

---

## Modèles de recommandation

### LightGCN (PyTorch)

- **Entrée** : `user_id` (int)
- **Sortie** : liste de `(movie_id, score)` triée par score décroissant
- **Chargement** : `torch.load()` sur `models/lightgcn_best.pt`
- **Inférence** : produit des scores de similarité entre l'embedding utilisateur et les embeddings items

### SVD (Surprise / Scikit-Learn)

- **Entrée** : `user_id`, `movie_id` (movieId **bruts** MovieLens)
- **Sortie** : note prédite (`est`) pour chaque film
- **Chargement** : `pickle.load()` sur `models/svd_model.pkl` (objet `surprise.SVD` brut)
- **Inférence** : prédit la note que l'utilisateur donnerait à chaque film, puis tri décroissant

### Item-Item CF

- **Entrée** : `user_id` (movieId brut)
- **Sortie** : dict précalculé `{user_id: [(item_id, score), ...]}` (movieId bruts)
- **Chargement** : `pickle.load()` sur `models/item_item_model.pkl`
- **Inférence** : agrège les items similaires à ceux déjà aimés par l'utilisateur

---

## Gestion des erreurs

### Au chargement (lifespan)

Si un artefact ne peut pas être chargé (fichier manquant, format invalide), l'erreur est loggée mais le serveur démarre quand même en mode dégradé. Le flag `models_loaded` dans le healthcheck reflète l'état.

### Pendant l'exécution

Chaque méthode de recommandation est encapsulée dans un `try/except`. En cas d'erreur :

1. L'erreur est loggée avec le contexte (`user_id`, `top_n`)
2. La méthode retourne une liste vide `[]`
3. L'endpoint renvoie un JSON valide avec des tableaux vides pour le modèle en erreur
4. Le frontend affiche "Aucune recommandation disponible" pour le modèle concerné

### Logs

Les logs sont configurables via le module Python `logging`. Le logger utilisé est `"recommender_api"`.

---

## CORS

Le middleware CORS est configuré pour autoriser les requêtes provenant du frontend React (Vite sur le port 5173).

```python
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
```

---

## Tests

### Test manuel avec curl

```bash
# Healthcheck
curl http://127.0.0.1:8000/

# Liste des utilisateurs
curl http://127.0.0.1:8000/api/v1/users

# Recommandations pour l'utilisateur 1
curl "http://127.0.0.1:8000/api/v1/recommendations/1?top_n=5"
```

### Test manuel avec Swagger UI

1. Ouvrir http://127.0.0.1:8000/docs
2. Cliquer sur `GET /api/v1/recommendations/{user_id}`
3. Cliquer sur **Try it out**
4. Entrer un `user_id` (ex: `1`)
5. Modifier `top_n`
6. Cliquer sur **Execute**

---

## Dépannage

### `ModuleNotFoundError: No module named 'torch'`

PyTorch n'est pas installé dans l'environnement virtuel actif.

```bash
pip install torch
```

### `ModuleNotFoundError: No module named 'surprise'`

Si vous utilisez les vrais modèles SVD / Item-Item :

```bash
pip install scikit-surprise
```

### Le frontend affiche "Aucune recommandation disponible"

Causes possibles :

1. **Les modèles ne sont pas chargés** : vérifiez `models_loaded` dans le healthcheck. Si `false`, vérifiez les chemins dans `model_service.py`.
2. **Le serveur FastAPI n'est pas démarré** : vérifiez que `uvicorn` tourne sur le port 8000.
3. **Erreur CORS** : vérifiez que l'origine du frontend est bien dans la liste `origins` de `main.py`.
4. **`movies_cleaned.csv` manquant** : le service ne peut pas mapper les `movieId` vers les titres sans ce fichier.

### `500 Internal Server Error` sur `/api/v1/recommendations/{user_id}`

Consultez les logs du serveur FastAPI. L'erreur exacte est loggée avant que l'exception ne soit remontée au client.

### Port 8000 déjà utilisé

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Puis relancer sur un autre port
uvicorn src.api.main:app --reload --port 8001
```

