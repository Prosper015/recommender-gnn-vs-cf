# Modèles & Évaluation — Documentation (rôle ML Engineer)

**Auteur :** TCHABODI Sadikou
**Branche :** `feature/models-lightgcn`
**Dossiers :** `src/models/`, `src/evaluation/`, `scripts/`

Ce document explique ce que fait le code de modélisation et d'évaluation,
pourquoi chaque choix a été fait, et comment l'utiliser — dans le même
esprit que `RAPPORT_DATA_PIPELINE.md` (Data Engineer), pour que le rapport
final et la présentation orale s'appuient sur une base écrite claire.

## En résumé, avant de rentrer dans les détails

| # | Étape | Ce qu'elle produit | Pourquoi elle existe |
|---|---|---|---|
| 1 | Protocole d'évaluation (full-ranking LOO) | `precision/recall/ndcg@{5,10,20}` | Comparer objectivement les 3 méthodes, sans biais |
| 2 | Baselines SVD + Item-Item CF | 2 modèles entraînés, comparés | Exigence minimale du sujet (≥2 baselines classiques) |
| 3 | LightGCN | Un GNN codé à la main, sans PyG | Cœur du sujet |
| 4 | Étude d'ablation (profondeur K) | Courbe qualité vs over-smoothing | Livrable obligatoire du sujet |
| 5 | Métadonnées films (`movies_cleaned.csv`) | Titres/genres lisibles | Débloque la démo (sinon aucun titre affichable) |

Chaque étape est détaillée ci-dessous avec un exemple simple, puis le
fonctionnement technique du code.

---

## 1. Étape 1 — Protocole d'évaluation (`src/evaluation/metrics.py`)

**L'idée sur un exemple ultra simple :**
Un utilisateur a un seul film "caché" à deviner (le test, issu du split
temporel d'Anne). Le modèle classe ce film parmi **tous** les films non vus
(pas un échantillon). S'il ressort en position 3 : `precision@5 = 1/5`,
`recall@5 = 1` (trouvé), `ndcg@5 = 1/log2(3+1) = 0.5` (trouvé, mais pas en
tête de liste). S'il ressortait en position 15, à `k=5` tout vaudrait 0
(raté), mais `k=20` le capterait.

**Pourquoi le "full-ranking" (classer parmi TOUS les films) plutôt qu'un
échantillon de négatifs :**
Krichene & Rendle (2020) montrent que les métriques calculées sur un
échantillon de négatifs peuvent être trompeuses et sur-estimer la qualité
réelle d'un modèle. Le sujet exige un protocole "rigoureux" — le
full-ranking est le choix le plus strict, celui utilisé par le papier
LightGCN lui-même.

**Ce que le code fait, techniquement :**
`evaluate_full_ranking()` prend une fonction générique `score_fn` (pas un
modèle particulier), un jeu de test, les films déjà vus par chaque
utilisateur (à exclure des candidats), et l'univers complet des item_id
candidats. **Point important** : cet univers n'est jamais supposé être
`[0, N)` — les baselines travaillent sur des movieId MovieLens bruts (non
contigus), LightGCN sur des indices réindexés. Chaque appelant passe
l'espace qui lui correspond.

**Pourquoi la même fonction sert aux 3 méthodes :**
Elle ne connaît aucun modèle en particulier — seulement une fonction
`(user_id, candidats) -> scores`. C'est ce qui permet de comparer SVD,
Item-Item CF et LightGCN avec un code d'évaluation strictement identique,
donc une comparaison honnête.

**Vérifié en conditions réelles :** 18 tests unitaires (`tests/test_metrics.py`),
dont un qui compare le résultat de `evaluate_full_ranking()` à une moyenne
calculée à la main sur un cas jouet — pas juste "ça tourne sans erreur".

**Sortie :** dict `{"precision@5": ..., "recall@10": ..., "ndcg@20": ...}`

---

## 2. Étape 2 — Baselines classiques (`src/models/baselines.py`, `scripts/train_baselines.py`)

**SVD, l'idée sur un exemple ultra simple :**
Chaque utilisateur et chaque film sont résumés par un vecteur de 100
nombres appris automatiquement (pas des catégories qu'on aurait définies).
Note prédite = produit scalaire des deux vecteurs. Ces vecteurs démarrent
aléatoires et s'ajustent sur 20 passages complets des données (`n_epochs`),
par descente de gradient.

**Item-Item CF, l'idée sur un exemple ultra simple :**
Deux films sont dits "similaires" si les utilisateurs qui ont noté les
deux leur donnent des notes cohérentes (similarité cosinus). Pour prédire
la note d'un film jamais vu, on fait la moyenne pondérée des notes que
l'utilisateur a données aux `k=40` films les plus similaires.

**Pourquoi ces deux techniques précisément :**
Ce sont les baselines explicitement citées par le sujet, tirées de
Koren et al. (2009) — la technique qui a gagné le Netflix Prize — et du
filtrage collaboratif classique.

**Pourquoi scikit-surprise (pas hand-rolled) :**
Contrairement à LightGCN, ces méthodes sont censées être des techniques
**classiques et éprouvées**, réutilisées telles quelles — pas la partie
que le sujet demande d'implémenter à la main. `scikit-surprise` s'installe
sans souci sous Windows (wheel précompilée pour Python 3.11, testé avant
d'engager la conception dessus).

**Contrat de sauvegarde (important pour l'intégration backend) :**
`save_svd()` persiste l'algorithme scikit-surprise **brut** (pas un
wrapper à nous), et `save_item_item()` persiste un dictionnaire précalculé
`{user_id: [(item_id, score), ...]}` — format découvert en lisant
directement le code du service backend (`model_service.py`, branche
`feature/fastapi-backend`, pas encore mergée) avant d'écrire la fonction de
sauvegarde, pour être certain de matcher exactement ce qu'il attend.

**Vérifié en conditions réelles (MovieLens 100K, seed fixe = résultats
reproductibles à l'identique à chaque run) :**

| Méthode | precision@10 | recall@20 | ndcg@10 |
|---|---|---|---|
| SVD | 0.00201 | 0.04030 | 0.00931 |
| Item-Item CF | 0.00053 | 0.00848 | 0.00317 |

SVD bat Item-Item CF sur les 9 métriques, systématiquement.

**Courbes de diagnostic (`scripts/analyze_baselines.py`), preuve que les
hyperparamètres par défaut ne sont pas arbitraires :**
En faisant varier `n_epochs` (SVD) et `k_neighbors` (Item-Item CF) et en
mesurant le RMSE sur validation, les deux courbes montrent un vrai
sur-apprentissage : le RMSE descend puis **remonte légèrement** après le
minimum (SVD : minimum ≈ 15 époques ; Item-Item CF : minimum ≈ k=40) —
voir `results/baselines_hyperparameter_curves.png`.

**Sorties :** `models/svd_model.pkl`, `models/item_item_model.pkl`,
`results/baselines_metrics.csv`, `results/baselines_comparison.png`,
`results/baselines_hyperparameter_curves.png`.

---

## 3. Étape 3 — LightGCN (`src/models/lightgcn.py`, `scripts/train_lightgcn.py`)

**L'idée sur un exemple ultra simple :**
L'embedding de l'utilisateur 0 se recalcule à chaque couche comme la
moyenne pondérée des embeddings de ses voisins (les films qu'il a notés) à
la couche précédente. Après K couches, on fait la moyenne des K+1
embeddings obtenus (couche 0 à K) — c'est l'embedding final. Le score
prédit = produit scalaire entre l'embedding final utilisateur et
l'embedding final film.

**Pourquoi codé à la main, sans PyTorch Geometric :**
Le service backend recharge les modèles avec `torch.load()` dans un
environnement qui n'a que `torch` installé, pas PyG. Un modèle construit
avec des couches PyG ne s'exécuterait pas côté API sans ajouter PyG au
backend aussi. Le sujet autorise explicitement un "GNN ou équivalent
simplifié" — rester en torch pur garde une empreinte de dépendances
identique entre entraînement et inférence, et reste totalement explicite
pour le rapport (pas de boîte noire).

**Ce que le code fait, techniquement (`propagate()`) :**
```python
x = self.embedding.weight        # couche 0, brute
layers = [x]
for _ in range(self.num_layers):
    x = torch.sparse.mm(adjacency, x)   # une couche = une multiplication
    layers.append(x)
return torch.stack(layers, dim=0).mean(dim=0)   # combinaison finale
```
`adjacency` est la matrice normalisée `D⁻¹ᐟ² A D⁻¹ᐟ²` (`build_normalized_adjacency()`),
construite une fois à partir du graphe d'Anne (`graph_builder.py`).

**Entraînement (`bpr_loss()`) :** BPR (Bayesian Personalized Ranking,
Rendle et al., 2009) — pour chaque interaction connue, tire un film que
l'utilisateur n'a **pas** vu, et pousse le score du film aimé à être plus
haut que celui du film négatif. `sample_bpr_triplets()` fait cet
échantillonnage négatif par rejet.

**Vérifié en conditions réelles :** un run complet, K=3, 50 époques,
**2 min 38 s** sur CPU (mesuré) → `ndcg@10 = 0.045`, déjà nettement au-dessus
de SVD (0.00931) sur ce seul point de profondeur.

**Sorties :** `models/lightgcn_best.pt`, `models/id_mappings.json`
(correspondance movieId brut ↔ indice interne, nécessaire pour que le
backend traduise les scores LightGCN en vrais films).

---

## 4. Étape 4 — Étude d'ablation / over-smoothing (`scripts/run_ablation.py`)

**L'idée sur un exemple ultra simple :**
Plus on ajoute de couches de propagation, plus chaque embedding se
rapproche de la moyenne de ses voisins — répété trop de fois, tous les
embeddings du graphe finissent par se ressembler. Un modèle qui ne
distingue plus les utilisateurs entre eux ne peut plus faire de
recommandation personnalisée : c'est l'over-smoothing.

**Ce que le code fait, techniquement :**
Entraîne un modèle **séparé, from scratch**, pour chaque profondeur K = 1,
2, 3, 4, 5. Pour chacun, mesure à la fois la qualité (precision/recall/ndcg)
et `embedding_similarity_diagnostic()` — la similarité cosinus moyenne
entre 2000 paires de nœuds tirées au hasard dans tout le graphe (proche de
0 = nœuds distincts, proche de 1 = over-smoothing confirmé).

**Statut actuel — important, à ne pas confondre :**
Un test de plomberie (3 époques par profondeur) a été fait pour valider
que le pipeline tourne de bout en bout — **pas un résultat scientifique
exploitable**. Le run réel (50 époques × 5 profondeurs, ~13 minutes mesurées
sur ce projet) reste à lancer avant de pouvoir tirer une vraie conclusion
sur l'over-smoothing pour le rapport.

**Sorties (une fois le run réel fait) :** `results/ablation_depth.csv`,
`results/ablation_depth.png`, et le meilleur modèle (meilleur NDCG@10)
sauvegardé automatiquement dans `models/lightgcn_best.pt`.

---

## 5. Métadonnées films (`src/data_pipeline/movies.py`)

**Le trou identifié :** le backend attend `data/processed/movies_cleaned.csv`
(`movieId, title, genres`) pour afficher de vrais titres dans la démo — rien
ne le générait. Techniquement du ressort du Data Engineer (parsing de
données brutes), mais bloquant pour toute l'équipe, donc traité ici.

**Ce que le code fait :** parse `u.item` (séparateur `|`, encodage
`latin-1`), transforme les 19 colonnes de genre binaires en une chaîne
lisible (`"Animation|Children's|Comedy"`), avec un fallback `"Genre inconnu"`
plutôt qu'une chaîne vide (une chaîne vide redevient `NaN` quand pandas
relit le CSV — sans ce fallback, la démo afficherait littéralement "nan").

**Vérifié en conditions réelles :** `data/processed/movies_cleaned.csv`
généré, 1682 films.

---

## 6. Bug d'intégration trouvé (pas corrigé ici — hors de mon périmètre de fichiers)

En simulant l'appel exact du backend (`torch.load(path, map_location=device)`,
sans `weights_only=False`) sur `lightgcn_best.pt`, le chargement **échoue** :
PyTorch 2.6+ bloque par défaut la désérialisation de classes personnalisées
pour des raisons de sécurité. Correctif d'une ligne
(`weights_only=False`), à appliquer dans `model_service.py`
(`feature/fastapi-backend`) — remonté au Dev Backend, pas corrigé
directement puisque ce n'est pas mon fichier.

---

## 7. Comment lancer le pipeline complet

```powershell
# Depuis la racine du projet, environnement virtuel active
.\.venv\Scripts\python.exe -m pytest tests/ -q                          # 24 tests

.\.venv\Scripts\python.exe -m scripts.train_baselines --dataset 100k     # ~1-2 min
.\.venv\Scripts\python.exe -m scripts.analyze_baselines --dataset 100k   # courbes + comparaison

.\.venv\Scripts\python.exe -m scripts.train_lightgcn --dataset 100k --depth 3 --epochs 20   # essai rapide
.\.venv\Scripts\python.exe -m scripts.run_ablation --dataset 100k --epochs 50               # ~13 min, le vrai run

.\.venv\Scripts\python.exe -m scripts.build_comparison_table
```

Des configurations de lancement/débogage VS Code équivalentes existent dans
`.vscode/launch.json` (panneau *Run and Debug*, `F5`).

---

## 8. Ce qui n'est PAS encore fait (pistes pour la suite)

- **Le run réel de l'ablation** (50 époques × 5 profondeurs) — priorité n°1,
  débloque le vrai tableau comparatif et l'analyse over-smoothing pour le
  rapport.
- Regénérer `results/comparison_table.csv`/`.md` une fois ce run terminé.
- Extension optionnelle à MovieLens 1M (fait pour l'instant sur 100K
  uniquement).
- La démo web (sélection utilisateur → top-N par méthode, côte à côte) —
  hors de ce périmètre (Backend/Frontend), mais bloquée tant que
  `model_service.py` n'applique pas le correctif `weights_only=False`.
- Le rapport scientifique (8-20 pages).

---

## 9. Notes techniques Windows (pour l'équipe)

- `scikit-surprise` : une wheel précompilée existe pour Python 3.11,
  aucun souci d'installation malgré la réputation de la librairie sous
  Windows.
- Lancer un script en mode debug VS Code (`F5`) peut occasionnellement
  planter au tout premier démarrage avec un `KeyboardInterrupt` provenant
  de `debugpy` lui-même (pas de notre code) — relancer une seconde fois
  résout le problème dans la quasi-totalité des cas.
