# Frontend - Interface de Comparaison de Modèles de Recommandation

Ce document décrit l'architecture, le fonctionnement et le déploiement du frontend React du système de recommandation.

---

---

## Vue d'ensemble

Le frontend est une **Single Page Application (SPA)** React qui consomme l'API FastAPI du backend pour afficher, pour un utilisateur donné, les recommandations générées par trois modèles côte à côte :

- **LightGCN (GNN)** — Réseau de neurones sur graphes
- **SVD (Matrice)** — Factorisation de matrice
- **Item-Item CF (Similarité)** — Filtrage collaboratif item-based

L'interface permet de :
- Sélectionner un utilisateur parmi une liste prédéfinie
- Rafraîchir les recommandations manuellement ou automatiquement au changement d'utilisateur
- Visualiser les scores et genres des films recommandés pour chaque modèle

### Stack technique

| Composant | Technologie |
|-----------|-------------|
| Framework UI | React 19 |
| Build tool | Vite 8 |
| HTTP Client | Axios |
| Icônes | Lucide React |
| Linting | Oxlint |
| Langage | JavaScript (ES Modules) |

---

## Prérequis

- **Node.js** : v18+ (v20+ recommandé)
- **npm** : v9+ (ou pnpm / yarn)
- **Git** : pour cloner le dépôt
- **Backend FastAPI** : doit être démarré sur `http://127.0.0.1:8000` (voir `README_BACKEND.md`)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Prosper015/recommender-gnn-vs-cf
cd recommender-gnn-vs-cf
```

### 2. Installer les dépendances

```bash
cd frontend
npm install
```

Cela installe :
- React et React DOM
- Axios pour les requêtes HTTP
- Lucide React pour les icônes
- Vite et ses plugins
- Oxlint pour le linting

---

## Lancement

### Démarrer le serveur de développement

```bash
# Depuis le dossier frontend/
npm run dev
```

Le serveur Vite démarre par défaut sur **http://localhost:5173**.

### Accéder à l'application

Ouvrir http://localhost:5173 dans le navigateur.

> **Prérequis** : Le backend FastAPI doit être démarré sur http://127.0.0.1:8000 pour que les recommandations s'affichent.

---

## Structure du projet

```
frontend/
├── index.html                 # Point d'entrée HTML
├── package.json               # Dépendances et scripts
├── vite.config.js             # Configuration Vite
├── .oxlintrc.json             # Configuration du linter
└── src/
     ├── main.jsx               # Bootstrap React (montage sur #root)
     ├── App.jsx                # Composant principal
     └── index.css              # Styles globaux et thème
 ```

### Fichiers clés

| Fichier | Rôle |
|---------|------|
| `src/main.jsx` | Monte l'application React dans le DOM |
| `src/App.jsx` | Composant racine : état, fetch API, rendu des 3 colonnes |
| `src/index.css` | Variables CSS, layout, cartes, badges, responsive |
| `vite.config.js` | Configuration du bundler (plugin React) |
| `package.json` | Scripts, dépendances, métadonnées du projet |

---

## Scripts disponibles

```bash
# Développement avec rechargement automatique (HMR)
npm run dev

# Build de production (optimisé dans dist/)
npm run build

# Prévisualiser le build de production localement
npm run preview

# Linter le code
npm run lint
```

### Détails

| Script | Commande | Description |
|--------|----------|-------------|
| `dev` | `vite` | Démarre le serveur de dev avec HMR |
| `build` | `vite build` | Build optimisé pour la production dans `dist/` |
| `preview` | `vite preview` | Sert le build de production localement |
| `lint` | `oxlint` | Analyse le code JS/JSX |

---

## Architecture de l'application

### Composant `App.jsx`

C'est le **seul composant** de l'application. Il encapsule toute la logique :

```
App
├── Header
│   ├── Titre + icône Film
│   └── Sélecteur d'utilisateur + bouton Actualiser
├── Message d'erreur (conditionnel)
└── Grille de 3 colonnes
    ├── Colonne LightGCN (GNN)
    ├── Colonne SVD (Matrice)
    └── Colonne Item-Item CF (Similarité)
```

### États gérés

| State | Type | Rôle |
|-------|------|------|
| `userId` | `number \| null` | Utilisateur sélectionné (défaut: `null`, défini depuis la liste au chargement) |
| `users` | `number[]` | Liste des utilisateurs disponibles, **chargée dynamiquement** depuis `GET /api/v1/users` (943 users réels MovieLens 100k) |
| `recommendations` | `{ lightgcn: [], svd: [], item_item: [] }` | Données des 3 modèles |
| `loading` | `boolean` | Indicateur de chargement lors du fetch |
| `error` | `string \| null` | Message d'erreur si le backend est inaccessible |

---

## Intégration avec le backend

### Point d'entrée API

```javascript
// src/App.jsx — Vite n'expose que les variables prefixees VITE_ (cf. docker-compose).
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
```

Ce chemin pointe vers le serveur FastAPI. En local, défaut `http://127.0.0.1:8000`.
Sous Docker (reverse proxy nginx), `VITE_API_URL=/` (URL **relative**) : le frontend est
servi par nginx et appelle `/api/v1/...` sur la même origine, donc accessible depuis
n'importe quelle machine sans résoudre le nom de service `api`.

### Endpoint consommé

```
GET /api/v1/recommendations/{user_id}?top_n=5
```

### Flux de données

```text
Utilisateur change le sélecteur
         ↓
useEffect détecte le changement de userId
         ↓
fetchData() est appelée
         ↓
GET http://127.0.0.1:8000/api/v1/recommendations/{userId}?top_n=5
         ↓
Réponse JSON parsée et mappée vers le state recommendations
         ↓
React re-render les 3 colonnes avec les nouvelles données
```

## Dépannage

### `Failed to fetch` / Erreur réseau

Le frontend affiche "Impossible de contacter l'API FastAPI".

**Causes possibles :**
1. Le backend FastAPI n'est pas démarré
2. Le backend écoute sur un port différent de 8000
3. Pare-feu bloque la connexion
4. `API_URL` est incorrecte dans `App.jsx`

**Solution :**
```bash
# Vérifier que le backend répond
curl http://127.0.0.1:8000/

# Si le backend est sur un autre port, modifier API_URL dans App.jsx
```

### `Aucune recommandation disponible` pour tous les modèles

Le backend répond mais renvoie des tableaux vides.

**Causes possibles :**
1. Les modèles ne sont pas chargés (voir `models_loaded` dans le healthcheck)
2. `user_id` n'existe pas dans les données
3. Les artefacts de modèles sont corrompus

**Solution :** Vérifier les logs du backend FastAPI.

### Port 5173 déjà utilisé

```bash
# Vite tentera automatiquement le port suivant (5174, 5175, etc.)
# Ou forcer un port spécifique
npm run dev -- --port 3000
```
