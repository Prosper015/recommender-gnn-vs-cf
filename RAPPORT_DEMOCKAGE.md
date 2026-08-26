# Rapport de démockage — Passage en production (Backend ↔ Modèles/Data réels)

**Projet :** Système de recommandation de films (MovieLens) — LightGCN (GNN) vs SVD vs Item-Item CF
**Date :** 2026-08-22
**Auteur :** Revue & refactoring (Architecte Logiciel / Lead ML Engineer)

---

## 1. Objectif

Supprimer le mode *mock* (données fictives) du backend FastAPI et câbler le service sur les
**artefacts réellement entraînés** (modèles + métadonnées films + mappings d'index), puis
valider la chaîne bout-en-bout, la cohérence du ML/data pipeline et le dimensionnement Docker.

---

## 2. État initial — problèmes identifiés

| # | Composant | Problème |
|---|-----------|----------|
| 1 | `src/services/model_service.py` | **Fallback mock silencieux** : dès qu'un artefact manquait, le service renvoyait des recommandations fictives (Inception, The Dark Knight…). Impossible de garantir des données réelles. |
| 2 | `src/services/model_service.py` | **Bug d'indexation LightGCN** : `recommend_lightgcn` passait le `user_id` **brut** au modèle (qui attend un index interne reindexé) et interpretétait les indices d'items **internes** comme des `movieId` bruts → films attribués au mauvais utilisateur. |
| 3 | `data/processed/movies_cleaned.csv` | **Inexistant** : aucun code ne parsait/exportait les métadonnées films. Le backend ne pouvait mapper aucun film. |
| 4 | `models/*.pt` / `*.pkl` | **Inexistants** : aucun modèle n'était entraîné. |
| 5 | `GET /api/v1/users` | Retournait une liste codée en dur `[1,2,3,4,5]` au lieu des vrais utilisateurs. |
| 6 | Frontend `App.jsx` | Liste d'utilisateurs codée en dur ; `API_URL` via `process.env` (non disponible côté navigateur sous Vite) → le frontend Docker ne pouvait pas atteindre le backend via `api:8000`. |
| 7 | Gestion d'erreur | `user_id` inconnu (cold-start) non géré (aucun 404). |
| 8 | Dépendances | `scikit-surprise`, `requests`, `tqdm`, `pytest` non installés dans le venv ; `mlflow`/`pyarrow` requis en dur (lourds, inutiles à l'inférence). |

---

## 3. Changements apportés

### 3.1 Data Pipeline — `src/data_pipeline/download.py`
- Ajout de `parse_movies()` (gère **100k** `u.item` à 19 genres binaires **et** **1m** `movies.dat`) et
  `save_movies()` / `get_movies()` → produit **`data/processed/movies_cleaned.csv`**
  (`movieId`, `title`, `genres` séparés par `|`).
- `GENRES_100K` canonique pour reconstruction propre des genres.

### 3.2 Split temporel — `src/data_pipeline/temporal_split.py`
- `save_splits()` : passage du **parquet → CSV** (supprime la dépendance `pyarrow` ; portable).
- Ajout de `from pathlib import Path` (correction d'import).

### 3.3 ML — `scripts/train_lightgcn.py` & `scripts/train_baselines.py`
- **MLflow rendu optionnel** (`try/except ImportError` + garde `if mlflow is not None`) : l'entraînement
  produit les artefacts sans MLflow installé (`--no-mlflow`).
- `train_baselines.py` : utilisation de `nullcontext()` quand MLflow absent.

### 3.4 Nouvel orchestrateur — `scripts/run_pipeline.py`
Script unique de **démockage** :
```
python -m scripts.run_pipeline --dataset 100k --epochs 50 [--no-mlflow]
```
Télécharge MovieLens → parse films → split LOO → entraîne **SVD + Item-Item CF + LightGCN**
→ exporte les modèles, `movies_cleaned.csv` et `id_mappings.json`.

### 3.5 Backend — `src/services/model_service.py` (refonte)
- **Suppression totale du mock** (`_generate_mock_recommendations` retiré).
- Chargement de **`id_mappings.json`** + construction des mappings inverses
  (`user_raw_to_internal`, `item_internal_to_raw`).
- `recommend_lightgcn` : convertit `user_id` brut → index interne, appelle
  `get_user_item_scores`, puis **reconvertit les indices internes → `movieId` bruts** avant formatage.
- `recommend_svd` / `recommend_item_item` : conservent la logique réelle (movieId bruts),
  renvoient `[]` si le modèle n'est pas chargé (plus de faux).
- `is_known_user()` + `get_available_users()` (liste réelle des utilisateurs).
- `UnknownUserError` levée pour un `user_id` inconnu (→ 404).
- Flags d'état : `is_loaded`, `lightgcn_available`, `svd_available`, `item_item_available`.

### 3.6 API — `src/api/main.py`
- `GET /api/v1/users` → **vrais utilisateurs** (`model_service.get_available_users()`).
- `GET /api/v1/recommendations/{user_id}` : `503` si artefacts manquants ;
  **`404` explicite en cas de cold-start** (`is_known_user`).
- **CORS** : ajout des origines Docker `http://localhost:3000` / `127.0.0.1:3000`.

### 3.7 Frontend — `frontend/src/App.jsx`
- Récupération **dynamique** de la liste des utilisateurs via `GET /api/v1/users`.
- `API_URL` via `import.meta.env.VITE_API_URL` (correct sous Vite) avec repli `127.0.0.1:8000`.
- Messages d'erreur adaptés (404 cold-start, 503 modèles non chargés).

### 3.8 Docker — `docker-compose.yml`
- `app` : `VITE_API_URL=/` (URL **relative**) : le frontend appelle `/api/*` sur la même origine que le proxy.
- Volumes `./models:/app/models` et `./data:/app/data` conservés (les artefacts générés
  côté host sont montés dans le conteneur `api`).

### 3.9 Documentation
- `README_BACKEND.md` : suppression de la section « mode mock », doc de `run_pipeline`,
  comportement 404/503, endpoints mis à jour.
- `README_FRONTEND.md` : `VITE_API_URL`, chargement dynamique des utilisateurs.

---

## 4. Artefacts générés (vrais)

Exécuté via `python -m scripts.run_pipeline --dataset 100k --epochs 50 --no-mlflow` :

| Artefact | Taille | Contenu |
|----------|--------|---------|
| `data/processed/movies_cleaned.csv` | 1 682 films | `movieId, title, genres` |
| `data/processed/100k/{train,val,test}.csv` | 98 114 / 943 / 943 | Split LOO temporel |
| `models/lightgcn_best.pt` | 8,5 Mo | LightGCN (K=3, dim=64) + graphe |
| `models/svd_model.pkl` | 4,8 Mo | `surprise.SVD` brut |
| `models/item_item_model.pkl` | 137 Ko | dict `{user_id: [(item, score)]}` |
| `models/id_mappings.json` | 45 Ko | mappings brut ↔ interne |

---

## 5. Vérifications

### 5.1 Qualité du ML & data pipeline (sortie du pipeline)
- **Split LOO strict** : `train=98114 | val=943 | test=943`, **0 utilisateur sous le seuil**,
  **vérification anti-leakage OK** (aucune interaction de train ne suit le test).
- **Métriques full-ranking** (MovieLens 100k, LOO) :

  | Modèle | Recall@10 | NDCG@10 | Precision@5 |
  |--------|-----------|---------|-------------|
  | **LightGCN** | **0.0891** | **0.0451** | **0.0112** |
  | SVD | 0.0201 | 0.0093 | 0.0023 |
  | Item-Item CF | 0.0053 | 0.0032 | 0.0008 |

  → LightGCN (GNN) surpasse nettement les baselines, cohérent avec la littérature.
  Architecture LightGCN validée : propagation d'embeddings sur graphe biparti
  (moyenne des couches 0..K, pas de self-loops ni non-linéarité), loss BPR.

### 5.2 Bout-en-bout (API réelle)
Lancé via `uvicorn src.api.main:app` :
- `GET /` → `{"status":"online","models_loaded":true}`.
- `GET /api/v1/users` → **943 utilisateurs réels** (`[1,2,3,4,5,...]`).
- `GET /api/v1/recommendations/1?top_n=3` → titres **réels** :
  - LightGCN : *Star Wars (1977)* 0.589, *Contact (1997)* 0.577, *The English Patient (1996)* 0.559
  - SVD : *Chasing Amy (1997)*, *Amistad (1997)*, *Contact (1997)*
  - Item-Item : *Groundhog Day (1993)*, *The Terminator (1984)*, *The Professional (1994)*
- `GET /api/v1/recommendations/999` → **404** `user_id=999 est inconnu…` (cold-start géré).

### 5.3 Tests
`pytest tests/` → **18 passed** (métriques + LightGCN). Linter frontend (`oxlint`) → **OK**.

---

## 6. Comment lancer le projet (production)

```bash
# 1. Environnement
python -m venv venv && .\venv\Scripts\activate
pip install -r requirements.txt
pip install scikit-surprise requests tqdm pytest   # (si non présents)

# 2. Générer les vrais artefacts
python -m scripts.run_pipeline --dataset 100k --epochs 50

# 3. Backend
uvicorn src.api.main:app --reload --port 8000

# 4. Frontend (autre terminal)
cd frontend && npm install && npm run dev
```

Docker : `docker compose up --build -d` (reverse proxy nginx en entrée unique sur le port 80 ;
les artefacts `models/` et `data/` du host sont montés dans le conteneur `api` ;
le frontend appelle le backend en **URL relative** `/api/*` via le proxy → accessible
depuis n'importe quelle machine sur le réseau, sans DNS `api` ni CORS).

---

## 7. Points d'attention / limites
- **Cold-start** : un `user_id` hors dataset renvoie `404`. Pour la prod, ajouter une
  stratégie de repli (popularité / contenu) côté API si besoin.
- **Item-Item précalculé** : dict par utilisateur vu à l'entraînement ; un utilisateur
  cold-start côté Item-Item renvoie `[]` (géré, mais vide).
- **MLflow/pyarrow** : désormais optionnels (logging expérimental et parquet non requis à l'inférence).
- **Dataset par défaut** : 100k. Le pipeline gère aussi 1m (chemins/parsing adaptés).
- **Docker** : les artefacts sont montés depuis le host (dev). Pour une image autonome,
  ajouter l'étape `run_pipeline` dans `Dockerfile.backend` (téléchargement + torch à build).
