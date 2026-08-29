# Data Pipeline - Documentation (rôle Data Engineer)

**Auteure :** Anne AGBOTA
**Branche :** `feature/data-pipeline`
**Dossier :** `src/data_pipeline/`

Ce document explique ce que fait le pipeline de données, pourquoi chaque
choix a été fait, et comment l'utiliser - pour que le ML Engineer et le Dev
puissent s'appuyer dessus sans avoir à relire tout le code.

## En résumé, avant de rentrer dans les détails

Le rôle de Data Engineer se résume à 4 grandes étapes, qui s'enchaînent :

| # | Étape | Ce qu'elle produit | Pourquoi elle existe |
|---|---|---|---|
| 1 | Récupérer et nettoyer les données MovieLens | Un tableau propre (`user_id`, `item_id`, `rating`, `timestamp`) | Sans données propres et bien indexées, rien de fiable ne peut être construit derrière |
| 2 | Découper en train / validation / test (Leave-One-Out) | 3 fichiers `.parquet` | Évaluer un modèle de recommandation sur le vrai "futur" de chaque utilisateur, sans tricher |
| 3 | Construire le graphe utilisateur-film | Un fichier `.npz` (graphe biparti) | LightGCN a besoin de ce graphe pour propager l'information entre utilisateurs et films |
| 4 | Tracer le tout avec MLflow | Un run MLflow (params + métriques + artefacts) | Exigence du professeur : tout doit être reproductible et vérifiable |

Chaque étape est détaillée ci-dessous avec un exemple simple, puis le
fonctionnement technique du code.

---

## 1. Objectif général

Avant d'entraîner ou de comparer LightGCN (GNN) et le Filtrage Collaboratif,
il faut des données **propres, découpées correctement dans le temps, et
transformées en graphe**. C'est le rôle de ce module : transformer les
fichiers bruts MovieLens en artefacts directement exploitables par le ML
Engineer, tout en traçant chaque étape avec MLflow (exigence du professeur).

Le pipeline se déroule en 4 étapes, dans cet ordre strict :

```
MovieLens brut
      │
      ▼
1. Téléchargement + parsing + reindexation      (download.py)
      │
      ▼
2. Split temporel Leave-One-Out (train/val/test) (temporal_split.py)
      │
      ▼
3. Construction du graphe biparti (train only)   (graph_builder.py)
      │
      ▼
4. Traçage MLflow (params, métriques, artefacts)  (mlflow_tracking.py)
```

---

## 2. Étape 1 - Téléchargement, parsing, reindexation (`download.py`)

**L'idée en une phrase :** on part d'un fichier brut téléchargé sur
Internet, et on en ressort un tableau propre, avec des identifiants
utilisables par n'importe quel modèle.

Ce fichier fait 4 choses, dans l'ordre :

1. **`download_zip()`** - télécharge l'archive `.zip` de MovieLens depuis
   le site officiel GroupLens, si elle n'est pas déjà sur le disque (sinon
   il ne re-télécharge pas, pour gagner du temps).
2. **`extract_zip()`** - dézippe l'archive.
3. **`parse_ratings()`** - lit le fichier de notes et le transforme en
   tableau pandas avec 4 colonnes propres (`user_id`, `item_id`, `rating`,
   `timestamp`). Point important : MovieLens 100K et 1M n'ont pas le même
   format de fichier (le premier utilise des tabulations, le second des
   `::`), donc cette fonction gère les deux cas.
4. **`reindex_ids()`** - c'est la plus importante à comprendre : MovieLens
   donne des identifiants utilisateur/film qui ne sont pas forcément
   continus (par exemple user 1, 2, 5, 9...). Or, pour construire un
   graphe ensuite, PyTorch a besoin d'identifiants qui se suivent sans
   trou : 0, 1, 2, 3... Cette fonction fait cette conversion.

**Pourquoi la reindexation est indispensable :**
MovieLens ne garantit pas que les IDs se suivent sans trou. Or PyTorch
Geometric / DGL, ainsi que les couches `nn.Embedding` du ML Engineer, ont
besoin d'indices de nœuds allant de `0` à `N-1` sans discontinuité. Sans
cette étape, la construction du graphe ou l'entraînement planterait ou
donnerait des résultats faux.

**Pourquoi gérer 100K et 1M différemment :**
Les deux versions de MovieLens n'ont pas le même format de fichier
(`u.data` séparé par tabulations pour 100K, `ratings.dat` séparé par `::`
pour 1M). `parse_ratings()` gère les deux cas pour que le reste du pipeline
n'ait jamais à s'en soucier.

**Vérifié en conditions réelles (MovieLens 100K) :**
100 000 interactions, 943 utilisateurs, 1682 films - reindexés en `[0, 942]`
et `[0, 1681]` sans trou.

---

## 3. Étape 2 - Split temporel Leave-One-Out (`temporal_split.py`)

**L'idée sur un exemple ultra simple :**
Imagine un utilisateur qui a noté 5 films, dans cet ordre chronologique :
Film A, B, C, D, E (E = le plus récent). On découpe comme ça :

- **Test = Film E** (le tout dernier noté) → sert à vérifier si le modèle
  final devine bien ce que l'utilisateur regarderait *maintenant*.
- **Validation = Film D** (l'avant-dernier) → sert à régler les réglages
  du modèle pendant l'entraînement.
- **Train = Films A, B, C** → c'est sur ça que le modèle apprend.

**Ce que ça fait, techniquement :**
Pour chaque utilisateur, trie son historique par ordre chronologique et
répartit :
- la **dernière** interaction → `test`
- l'**avant-dernière** → `validation`
- **tout le reste** → `train`

Un utilisateur avec moins de `min_interactions` (5 par défaut) interactions
va **entièrement** dans le train : on ne peut pas évaluer fiablement sur
trop peu de données, et l'exclure complètement fausserait le graphe.

**Pourquoi dans cet ordre précis et pas au hasard ?**
Parce qu'en vrai, un système de recommandation prédit le futur à partir du
passé. Si on mélangeait au hasard (comme le ferait un `train_test_split`
classique), le modèle pourrait "apprendre" avec des films que
l'utilisateur n'a vus qu'*après* - ce qui n'a aucun sens en pratique et
fausserait complètement l'évaluation.

**Pourquoi un split temporel et pas un split aléatoire, vu autrement :**
Un système de recommandation prédit le futur à partir du passé. Un split
aléatoire classique (ex: `train_test_split` de scikit-learn) pourrait
laisser un film "futur" dans le train et un film "passé" dans le test -
ce qui n'a aucun sens en usage réel et fausse complètement l'évaluation
(le modèle semblerait meilleur qu'il ne l'est vraiment).

**Garde-fou anti-leakage :**
`_validate_no_leakage()` vérifie **automatiquement**, après chaque split,
qu'aucune interaction de train n'est postérieure à l'interaction de test du
même utilisateur. Si une violation existait (bug), le programme s'arrête
avec une erreur explicite plutôt que de laisser passer des données
corrompues silencieusement.

**Vérifié en conditions réelles (MovieLens 100K, `min_interactions=5`) :**
train=98 114 / val=943 / test=943 (943+943+98114 = 100 000, rien perdu).
0 utilisateur exclu (tous les utilisateurs MovieLens 100K ont ≥5 notes).
Vérification anti-leakage : **OK**.

**Sortie :** `data/processed/<dataset>/{train,val,test}.parquet`

---

## 4. Étape 3 - Graphe biparti Utilisateur-Item (`graph_builder.py`)

**L'idée sur un exemple ultra simple :**
Imagine 3 utilisateurs (U1, U2, U3) et 2 films (F1, F2) :
- U1 a vu F1 et F2
- U2 a vu F1
- U3 a vu F2

Le graphe biparti, c'est juste ça : des points ("nœuds") pour chaque
utilisateur et chaque film, reliés par un trait ("arête") à chaque fois
qu'il y a eu une note. Deux films ne sont jamais reliés directement entre
eux, ni deux utilisateurs - seulement utilisateur ↔ film.

**Pourquoi LightGCN a besoin de ça, concrètement ?**
L'idée derrière le GNN, c'est que "les goûts se propagent à travers le
graphe" : si U1 et U2 ont tous les deux aimé F1, et que U1 a aussi aimé F2,
alors F2 devient une bonne suggestion pour U2 - même si U2 n'a jamais vu
F2 - parce qu'ils sont "proches" dans le graphe. C'est cette propagation
que les couches de convolution de LightGCN calculent.

**Ce que le code fait, techniquement :**
Construit un graphe où chaque utilisateur et chaque film est un nœud, relié
par une arête à chaque interaction de **train uniquement**. Fournit une
représentation générique (numpy), indépendante de toute librairie de GNN.

**Convention d'indexation des nœuds (à respecter partout dans le projet) :**
- nœuds `[0, num_users - 1]` → utilisateurs
- nœuds `[num_users, num_users + num_items - 1]` → items

C'est la convention attendue par LightGCN, qui utilise un seul espace
d'indices unifié pour tous les nœuds du graphe (contrairement à un graphe
hétérogène classique séparant explicitement les deux types de nœuds).

**Pourquoi construire le graphe uniquement sur le train :**
Même logique anti-leakage que pour le split : si les arêtes de test étaient
présentes dans le graphe, le modèle "verrait" par la structure même du
graphe les interactions qu'il est censé prédire. L'évaluation serait
invalide.

**Pourquoi les arêtes sont bidirectionnelles (u→i ET i→u) :**
Les couches de convolution de LightGCN propagent l'information dans les
deux sens du graphe biparti (des utilisateurs vers les films et
inversement) à chaque passe de message-passing.

**Vérifié en conditions réelles (MovieLens 100K) :**
`num_nodes=2625` (943 users + 1682 items), `num_edges_directed=196228`
(2 × 98 114 interactions de train), degré moyen ≈ 74.75.

**Sortie :** `data/processed/<dataset>/bipartite_graph.npz`

**Pour le ML Engineer - comment charger ce graphe :**
```python
from src.data_pipeline.graph_builder import load_graph
graph = load_graph("data/processed/100k/bipartite_graph.npz")
# graph.edge_index -> numpy array (2, num_edges), prêt à convertir en
# torch.Tensor puis en torch_geometric.data.Data ou dgl.graph
```

---

## 5. Étape 4 - Traçage MLflow (`mlflow_tracking.py`)

**L'idée en une phrase :** MLflow, c'est un carnet de bord automatique.
Chaque fois qu'on relance le pipeline (peut-être avec un
`min_interactions` différent, ou sur le dataset 1M), MLflow enregistre
quels paramètres on a utilisés, quelles statistiques on a obtenues, et
garde une copie des fichiers produits. Ça permet, des semaines après, de
répondre à "avec quels réglages exacts a-t-on obtenu ce graphe précis ?" -
exactement ce qu'un jury strict peut vérifier.

**Ce que ça fait, techniquement :**
Log un run MLflow complet à chaque exécution du pipeline : paramètres
(dataset utilisé, seuil `min_interactions`), métriques descriptives
(tailles des splits, densité de la matrice, statistiques du graphe), et
artefacts (fichiers `.parquet` et `.npz` produits).

**Pourquoi tracer la préparation des données, pas seulement l'entraînement
du modèle :**
Le professeur exige MLflow sur l'ensemble du projet. Les choix faits ici
(seuil `min_interactions`, stratégie de split, version du dataset)
influencent directement les résultats finaux de comparaison GNN vs CF. Les
tracer permet de répondre précisément à "avec quelles données exactes ce
modèle a-t-il été entraîné ?" - un point que le jury peut demander.

**Backend choisi : SQLite (`mlflow.db`)**, pas le simple dossier `mlruns/`
en fichiers plats - c'est le backend recommandé par les versions récentes
de MLflow (le backend fichier pur est en maintenance limitée).

⚠️ **`mlflow.db` et `mlruns/` ne sont PAS poussés sur GitHub** (voir
`.gitignore`) : ce sont des historiques locaux, propres à chaque machine.
Chaque membre de l'équipe régénère son propre run en relançant le pipeline
chez lui.

**Comment consulter les runs (interface web parfois capricieuse sous
Windows) :**
```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Si la page reste blanche dans le navigateur, vérifier en Python à la place :
```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
print(mlflow.search_runs(experiment_names=["data_pipeline"]))
```

---

## 6. Comment lancer le pipeline complet

Étape par étape (pour bien voir ce qui se passe) :

```python
from src.data_pipeline.download import get_ratings, reindex_ids
from src.data_pipeline.temporal_split import leave_one_out_split, save_splits
from src.data_pipeline.graph_builder import build_bipartite_graph, save_graph
from src.data_pipeline.mlflow_tracking import init_mlflow, log_pipeline_run

df = get_ratings("100k")                      # ou "1m"
df, user_map, item_map = reindex_ids(df)
train_df, val_df, test_df = leave_one_out_split(df, min_interactions=5)
save_splits(train_df, val_df, test_df, "100k")

graph = build_bipartite_graph(train_df, num_users=len(user_map), num_items=len(item_map))
save_graph(graph, "data/processed/100k/bipartite_graph.npz")

init_mlflow()
log_pipeline_run(
    dataset="100k", min_interactions=5, ratings_df=df,
    train_df=train_df, val_df=val_df, test_df=test_df,
    graph_summary=graph.summary(),
    artifact_paths=["data/processed/100k/bipartite_graph.npz"],
)
```

**Installation préalable :**
```powershell
python -m pip install -r requirements.txt
```

---

## 7. Ce qui n'est PAS encore fait (pistes pour la suite)

- Script unique `pipeline.py` regroupant les 4 étapes en une seule commande
  (`python -m src.data_pipeline.pipeline --dataset 100k`) - pratique mais
  pas indispensable, chaque brique fonctionne déjà indépendamment.
- Exécution complète sur **MovieLens 1M** (fait pour l'instant sur 100K
  uniquement, pour itérer rapidement).
- Fonctions de conversion du graphe générique vers `torch_geometric.data.Data`
  et `dgl.graph()` - à ajouter dans `graph_builder.py` une fois que le ML
  Engineer confirme quelle(s) librairie(s) il utilise.
- Tests automatisés (`pytest`) validant la logique du split et du graphe.
- Fichier `id_mappings.json` (correspondance ID MovieLens original ↔ ID
  interne reindexé) pour que le Dev puisse traduire les IDs côté API.

---

## 8. Notes techniques Windows (pour l'équipe)

Sur certaines configurations Windows, les commandes `pip` et `mlflow`
peuvent échouer avec `Accès refusé`. Solution : toujours préfixer par
`python -m`, ex. `python -m pip install ...` et `python -m mlflow ui ...`,
qui contourne le blocage en laissant Python exécuter le module directement.
