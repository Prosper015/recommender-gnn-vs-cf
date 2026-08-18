"""
LightGCN (He et al., 2020) -- implementation "equivalent simplifie" en pur
PyTorch (torch.sparse), SANS PyTorch Geometric.

Pourquoi pas PyG : le service backend (feature/fastapi-backend,
src/services/model_service.py) recharge les modeles avec torch.load() dans
un environnement qui n'a que `torch` installe (pas PyG). Un modele construit
avec des couches PyG (MessagePassing, LGConv) ne se rechargerait/executerait
pas cote API sans ajouter PyG au backend aussi. Rester en torch pur garde une
empreinte de dependances identique entre entrainement et inference, et reste
totalement explicite pour le rapport (pas de "boite noire"). Le sujet
autorise explicitement un "GNN ou equivalent simplifie".

Convention d'indexation des noeuds (heritee de src/data_pipeline/graph_builder.py) :
    noeuds [0, num_users)                     -> utilisateurs
    noeuds [num_users, num_users + num_items) -> items
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.common import Recommender


def build_normalized_adjacency(edge_index: np.ndarray, num_nodes: int) -> torch.Tensor:
    """
    Construit D^-1/2 A D^-1/2 (tenseur sparse COO), tel qu'utilise par
    LightGCN pour la propagation. Pas de self-loops : contrairement a un GCN
    classique, LightGCN n'en ajoute pas -- la couche 0 (embedding brut) joue
    deja ce role via la combinaison finale (moyenne des couches).
    """
    row = torch.from_numpy(edge_index[0]).long()
    col = torch.from_numpy(edge_index[1]).long()

    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
    deg_inv_sqrt = degree.clamp(min=1.0).pow(-0.5)

    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    indices = torch.stack([row, col], dim=0)
    return torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

        # Adjacence normalisee : stockee comme buffers (suit le modele lors de
        # .to(device) / torch.save) mais reconstruite en tenseur sparse a la
        # volee (les tenseurs sparse eux-memes se serialisent mal historiquement).
        self.register_buffer("adjacency_indices", torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("adjacency_values", torch.zeros(0, dtype=torch.float32))
        self._adjacency_cache: torch.Tensor | None = None

    def set_graph(self, edge_index: np.ndarray) -> None:
        """A appeler une fois apres construction, avec le edge_index du TRAIN set uniquement."""
        adjacency = build_normalized_adjacency(edge_index, self.num_nodes)
        self.adjacency_indices = adjacency.indices()
        self.adjacency_values = adjacency.values()
        self._adjacency_cache = None

    def _adjacency(self) -> torch.Tensor:
        device = self.embedding.weight.device
        if self._adjacency_cache is None or self._adjacency_cache.device != device:
            self._adjacency_cache = torch.sparse_coo_tensor(
                self.adjacency_indices,
                self.adjacency_values,
                size=(self.num_nodes, self.num_nodes),
            ).coalesce().to(device)
        return self._adjacency_cache

    def propagate(self) -> torch.Tensor:
        """
        Propage les embeddings a travers les K couches et retourne la
        combinaison finale (moyenne des couches 0..K, couche 0 = embedding
        brut) -- formule exacte de LightGCN (pas de transformation
        non-lineaire ni de poids appris entre les couches).
        """
        adjacency = self._adjacency()
        x = self.embedding.weight
        layers = [x]
        for _ in range(self.num_layers):
            x = torch.sparse.mm(adjacency, x)
            layers.append(x)
        return torch.stack(layers, dim=0).mean(dim=0)

    def forward(self) -> torch.Tensor:
        return self.propagate()

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float = 1e-4,
    ) -> torch.Tensor:
        """Bayesian Personalized Ranking loss (Rendle et al., 2009), standard pour LightGCN."""
        final = self.propagate()
        u = final[users]
        pos = final[self.num_users + pos_items]
        neg = final[self.num_users + neg_items]

        pos_scores = (u * pos).sum(dim=-1)
        neg_scores = (u * neg).sum(dim=-1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # Regularisation L2 sur les embeddings de la couche 0 (seule couche
        # entrainable), comme dans l'implementation de reference LightGCN.
        reg = (
            self.embedding(users).pow(2).sum()
            + self.embedding(self.num_users + pos_items).pow(2).sum()
            + self.embedding(self.num_users + neg_items).pow(2).sum()
        ) / users.shape[0]
        return loss + l2_reg * reg

    @torch.no_grad()
    def get_user_item_scores(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Nom de methode EXACT attendu par model_service.recommend_lightgcn
        (feature/fastapi-backend) : torch.load(path) doit retourner un objet
        exposant get_user_item_scores(user_tensor) -> Tensor[batch, num_items]
        (indices internes 0..num_items-1, PAS des movieId bruts -- voir
        models/id_mappings.json pour la conversion cote backend/demo).
        """
        final = self.propagate()
        user_emb = final[user_ids.long()]
        item_emb = final[self.num_users : self.num_users + self.num_items]
        return user_emb @ item_emb.T

    @torch.no_grad()
    def embedding_similarity_diagnostic(self, sample_size: int = 2000, seed: int = 42) -> float:
        """
        Similarite cosinus moyenne entre paires d'embeddings finaux
        (echantillonnees aleatoirement pour rester tractable) -- diagnostic
        quantitatif de l'over-smoothing pour l'etude d'ablation : plus la
        profondeur K augmente, plus cette valeur devrait se rapprocher de 1
        (embeddings indiscernables) si l'over-smoothing survient.
        """
        final = self.propagate()
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randint(0, final.shape[0], (sample_size, 2), generator=generator)
        a, b = final[idx[:, 0]], final[idx[:, 1]]
        return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


class LightGCNRecommender(Recommender):
    """
    Adaptateur Recommender autour d'un LightGCN deja entraine, pour reutiliser
    evaluate_full_ranking() et le meme chemin de code que les baselines.
    Travaille en indices INTERNES (reindexes), pas en movieId bruts -- le
    dataframe de test passe a evaluate_full_ranking() doit donc lui aussi
    etre reindexe (voir src.data_pipeline.download.reindex_ids).
    """

    def __init__(self, model: LightGCN):
        self.model = model
        self.model.eval()
        self.all_item_ids_ = np.arange(model.num_items, dtype=np.int64)

    def fit(self, train_df: pd.DataFrame) -> "LightGCNRecommender":
        raise NotImplementedError(
            "LightGCN s'entraine via la boucle BPR de scripts/train_lightgcn.py, pas via fit()."
        )

    def score_items(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        device = next(self.model.parameters()).device
        user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
        scores = self.model.get_user_item_scores(user_tensor).squeeze(0)
        item_tensor = torch.from_numpy(item_ids).long().to(device)
        return scores[item_tensor].cpu().numpy()


def sample_bpr_triplets(
    train_df: pd.DataFrame, num_items: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genere un triplet (user, item_positif, item_negatif) par interaction de
    train_df. Echantillonnage negatif par rejet : un negatif tire au hasard
    est retire s'il fait partie des positifs connus de l'utilisateur.
    """
    users = train_df["user_id"].to_numpy()
    pos_items = train_df["item_id"].to_numpy()

    positives_per_user: dict[int, set[int]] = {}
    for u, i in zip(users, pos_items):
        positives_per_user.setdefault(int(u), set()).add(int(i))

    neg_items = np.empty_like(pos_items)
    for idx, u in enumerate(users):
        u_positives = positives_per_user[int(u)]
        while True:
            candidate = int(rng.integers(0, num_items))
            if candidate not in u_positives:
                neg_items[idx] = candidate
                break
    return users, pos_items, neg_items
