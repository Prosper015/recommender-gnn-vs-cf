
# Système de Recommandation : GNN (LightGCN) vs Filtrage Collaboratif

Projet de fin d'année comparant une approche moderne par réseau de neurones sur graphes (**LightGCN**) aux approches classiques de filtrage collaboratif (**SVD**, **Item-Item CF**) sur le jeu de données MovieLens.

---

## Aperçu de l'Architecture

Le projet s'articule autour de 3 composants majeurs :

1. **Pipeline de Données (Graphes) :** Préparation du graphe biparti Utilisateur-Item et découpage temporel strict (*Leave-One-Out*).

2. **Entraînement & Ablation ML :** Implémentation des baselines et du modèle LightGCN avec étude d'ablation de la profondeur (over-smoothing).

3. **Application de Démonstration :** API Backend FastAPI et interface web Frontend pour comparer les recommandations côte à côte pour un utilisateur donné.

---

## Structure du Projet

```text
recommender-gnn-vs-cf/
├── .gitignore
├── README.md
├── requirements.txt         # Dépendances globales / backend
│
├── data/                    # Géré principalement par le DATA ENGINEER
│   ├── raw/                 # Datasets originaux -> à ignorer sur Git
│   └── processed/           # Datasets nettoyés, splits temporels, graphes
│
├── src/                     # Code source principal
│   ├── data_pipeline/       # Code du DATA ENGINEER (loading, splitting, graph creation)
│   ├── models/              # Code du ML ENGINEER (baselines, LightGCN, ablation)
│   ├── evaluation/          # Code des métriques (Precision, Recall, NDCG)
│   └── api/                 # Code BACKEND (FastAPI, schemas, endpoints)
│
├── saved_models/            # Modèles sauvegardés & embeddings (ML Engineer) -> à ignorer sur Git
│
└── frontend/                # Code FRONTEND
````

## Répartition des Rôles & Responsabilités

| Rôle                   | Missions Principales                                                                                                                                                                                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data Engineer          | • Téléchargement et parsing de MovieLens (100K / 1M).<br>• Implémentation du split temporel strict (Leave-One-Out).<br>• Construction du graphe biparti pour PyTorch Geometric / DGL.                                                                                       |
| ML Engineer            | • Entraînement des baselines (scikit-surprise : SVD, Item-Item CF).<br>• Implémentation de LightGCN (PyTorch).<br>• Étude d'ablation de la profondeur $K$ (phénomène d'over-smoothing).<br>• Évaluation offline ($\text{Precision}@K$, $\text{Recall}@K$, $\text{NDCG}@K$). |
| Dev Backend & Frontend | • Développement de l'API REST avec FastAPI.<br>• Chargement des modèles et orchestration du Top-$N$ côte à côte.<br>• Conception de l'interface web (sélection d'utilisateur, historique, comparaison côte à côte).                                                         |

---

## Guide d'Installation & Lancement Local

### 1. Prérequis

* Python : 3.9 à 3.11 recommandé
* Node.js : v18+ et npm (pour le frontend)
* Git

### 2. Cloner le projet et configurer Python

```bash
# Cloner le dépôt
git clone https://github.com/Prosper015/recommender-gnn-vs-cf.git
cd recommender-gnn-vs-cf

# Créer et activer l'environnement virtuel
python -m venv venv

# Sur Linux / macOS :
source venv/bin/activate

# Sur Windows (PowerShell) :
.\venv\Scripts\activate

# Installer les dépendances backend & ML
pip install -r requirements.txt
```

### 3. Exécuter le Backend (FastAPI)

```bash
# Depuis la racine du projet
uvicorn src.api.main:app --reload --port 8000
```

* API Documentation (Swagger UI) : http://localhost:8000/docs
* Healthcheck : http://localhost:8000/health

### 4. Configurer et exécuter le Frontend

```bash
# Dans un nouveau terminal, aller dans le dossier frontend
cd frontend

# Installer les dépendances Node.js
npm install

# Démarrer le serveur de développement
npm run dev
```

L'interface de démonstration sera accessible sur `http://localhost:5173` (ou l'URL indiquée par Vite).

---

## Docker - Démarrage en une commande

### Prérequis

- **Docker** : 20.10+
- **Docker Compose** : v2+

### Architecture des conteneurs

| Service | Image | Port exposé | Rôle |
|---------|-------|-------------|------|
| `api` | `python:3.10-slim` | 8000 | Backend FastAPI + modèles ML |
| `app` | `node:18-alpine` | 3000 | Frontend React |

Les deux services communiquent sur un réseau Docker interne `recommender-network`. Le frontend atteint le backend via le nom de service `api`.

### Fichiers Docker

```text
recommender-gnn-vs-cf/
├── Dockerfile.backend           # Image du backend Python/FastAPI
├── docker-compose.yml           # Orchestration des 2 services
└── frontend/
    └── Dockerfile.frontend      # Image du frontend React
```

### Volumes pour le Hot-Reloading

Le `docker-compose.yml` monte les dossiers de code source en volume pour que les modifications locales soient reflétées instantanément dans les conteneurs :

- **Backend** : `./src:/app/src`, `./models:/app/models`, `./data:/app/data`
- **Frontend** : `./frontend/src:/app/src`

### Démarrage complet

```bash
# Construire les images et démarrer les conteneurs en arrière-plan
docker compose up --build -d

# Suivre les logs en temps réel
docker compose logs -f

# Arrêter les conteneurs
docker compose down
```

### URLs d'accès

| Service | URL | Description |
|---------|-----|-------------|
| Frontend React | http://localhost:3000 | Interface de comparaison des modèles |
| Backend FastAPI | http://localhost:8000/docs | Documentation interactive Swagger UI |
| Backend FastAPI | http://localhost:8000/ | Healthcheck |

### Variables d'environnement

| Variable | Service | Valeur par défaut | Description |
|----------|---------|-------------------|-------------|
| `REACT_APP_API_URL` | `app` | `http://api:8000` | URL du backend vue depuis le frontend (utilise le DNS Docker Compose) |
| `PYTHONUNBUFFERED` | `api` | `1` | Désactive le buffer de sortie Python pour les logs Docker |
| `CHOKIDAR_USEPOLLING` | `app` | `true` | Active le polling pour le hot-reloading Vite sous Docker Desktop (Windows/macOS) |

> **Important** : Dans `docker-compose.yml`, `REACT_APP_API_URL` vaut `http://api:8000` (et non `localhost:8000`), car depuis le conteneur `app`, `localhost` désigne le conteneur lui-même. Le nom de service `api` est résolu par le réseau Docker interne.

### Reconstruire après modification du code

```bash
# Reconstruire uniquement le service modifié
docker compose build api
docker compose up -d api

#pour le frontend
docker compose build app
docker compose up -d app
```

### Nettoyage

```bash
# Arrêter et supprimer les conteneurs, réseaux et volumes anonymes
docker compose down

# Supprimer également les images construites
docker compose down --rmi all
```

---

## Endpoints API Principaux

| Méthode | Endpoint                            | Description                                                     |
| ------- | ----------------------------------- | --------------------------------------------------------------- |
| GET     | `/api/v1/users`                     | Récupère la liste des identifiants utilisateurs disponibles.    |
| GET     | `/api/v1/users/{user_id}/history`   | Renvoie les films historiques déjà vus/notés par l'utilisateur. |
| GET     | `/api/v1/recommendations/{user_id}` | Génère le Top-$N$ côte à côte (LightGCN vs SVD vs Item-Item).   |
| GET     | `/api/v1/metrics/ablation`          | Renvoie les résultats d'ablation sur la profondeur du GNN.      |

---

## Protocole d'Évaluation & Métriques Exigées

Le projet applique un protocole offline rigoureux :

### Split Temporel (Leave-One-Out)

Pour chaque utilisateur, l'interaction la plus récente sert de test, évitant tout risque de data leakage temporel.

### Métriques Ranking à $K$

* Precision@K : Proportion de recommandations pertinentes parmi les Top-$N$.
* Recall@K : Capacité du modèle à capturer les éléments réellement consultés.
* NDCG@K (Normalized Discounted Cumulative Gain) : Évalue la qualité du positionnement des bons films en haut de liste.

### Étude d'Ablation

Variation de la profondeur des couches de convolution $K \in {1, 2, 3, 4, 5}$ sur LightGCN pour observer l'émergence du problème d'over-smoothing (lorsque la profondeur excessive rend les embeddings uniformes et dégrade les performances).

---

## Git Workflow pour l'Équipe

### main

Branche stable contenant du code testé et fonctionnel.

### Branches de fonctionnalités

Toute modification doit faire l'objet d'une branche dédiée issue de `main` :

```text
feature/data-pipeline
feature/models-lightgcn
feature/fastapi-backend
feature/frontend-ui
```

Ouvrir une Pull Request (PR) vers `main` pour faire valider votre code avant fusion.

```
```
