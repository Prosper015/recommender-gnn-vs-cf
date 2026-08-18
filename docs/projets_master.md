# Cours 14 - Projets Master : 10 sujets de recherche appliquée

> **Auteur :** TCHAYE-KONDI Jude, Ph.D.  
> **Derniere mise a jour :** 2026-07-22


## Objectif

Chaque projet doit démontrer une **démarche de recherche appliquée rigoureuse** :
- Implémentation maîtrisée d'une architecture moderne (pas seulement l'appel d'une bibliothèque)
- Une **étude d'ablation** substantielle (au moins 2 variantes comparées)
- Une **comparaison chiffrée** à une baseline ou à un résultat de la littérature
- Une application de démonstration (exigence allégée par rapport à la Licence)

---

## 1. Transformer entraîné from scratch vs. fine-tuning BERT

* **Type** : NLP - Transformers, étude d'ablation architecturale
* **Description** : Entraîner un petit Transformer encodeur from scratch pour une tâche de classification de texte, et comparer systématiquement ses performances au fine-tuning d'un BERT pré-entraîné, en étudiant l'effet de la taille du modèle, du nombre de têtes d'attention et du pré-entraînement.
* **Papier(s) de référence** :
  - Vaswani et al., 'Attention Is All You Need' (2017) - https://arxiv.org/abs/1706.03762
  - Devlin et al., 'BERT: Pre-training of Deep Bidirectional Transformers' (2018) - https://arxiv.org/abs/1810.04805
* **Dataset(s)** :
  - IMDB Reviews - https://huggingface.co/datasets/stanfordnlp/imdb
  - AG News - https://huggingface.co/datasets/fancyzhx/ag_news
* **Technos suggérées** : PyTorch (implémentation Transformer maison), transformers (HuggingFace) pour la baseline BERT, wandb ou tensorboard pour le suivi d'expériences
* **Requirements techniques (incluant l'étude d'ablation exigée)** :
- Implémenter un Transformer encodeur (self-attention, positional encoding, feed-forward) sans bibliothèque haut niveau pour la partie architecture.
- Entraîner ce modèle from scratch sur le dataset choisi.
- Fine-tuner un BERT pré-entraîné (ex. `bert-base-uncased`) sur la même tâche.
- Étude d'ablation : faire varier au moins 2 paramètres parmi {nombre de couches, nombre de têtes, présence/absence de pré-entraînement, taille du vocabulaire}.
- Comparaison chiffrée : accuracy/F1, nombre de paramètres, temps d'entraînement, courbes d'apprentissage.
* **Application de démonstration attendue** : Application web simple : saisie de texte → prédiction par les 2 modèles comparés côte à côte.
* **Livrables spécifiques attendus** :
- Tableau d'ablation complet (≥ 3 configurations comparées).
- Discussion : le pré-entraînement compense-t-il un modèle plus petit ? À partir de quelle taille de données le from-scratch devient-il compétitif ?

---

## 2. VAE conditionnel pour génération d'images

* **Type** : Génératif - VAE / CVAE, étude de l'espace latent
* **Description** : Implémenter un autoencodeur variationnel (VAE) et sa variante conditionnelle (CVAE) pour la génération d'images, et comparer leur capacité à générer des images contrôlées par une étiquette de classe.
* **Papier(s) de référence** :
  - Kingma & Welling, 'Auto-Encoding Variational Bayes' (2013) - https://arxiv.org/abs/1312.6114
  - Sohn et al., 'Learning Structured Output Representation using Deep Conditional Generative Models' (2015) - https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html
* **Dataset(s)** :
  - MNIST - https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
  - Fashion-MNIST - https://github.com/zalandoresearch/fashion-mnist
  - CelebA (sous-échantillon, attributs comme condition) - https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* **Technos suggérées** : PyTorch, matplotlib/UMAP pour la visualisation de l'espace latent
* **Requirements techniques (incluant l'étude d'ablation exigée)** :
- Implémenter le VAE (encodeur, reparamétrisation, décodeur, loss ELBO = reconstruction + KL).
- Implémenter la variante conditionnelle (label injecté à l'encodeur et au décodeur).
- Étude d'ablation : effet du poids du terme KL (β-VAE) sur la qualité de reconstruction vs. la structure de l'espace latent.
- Visualiser l'espace latent (projection 2D/t-SNE) et l'interpolation entre deux exemples.
- Évaluer quantitativement la qualité de génération (reconstruction loss, score FID si les ressources le permettent).
* **Application de démonstration attendue** : Application web : sélection d'une classe cible → génération d'images ; slider d'interpolation dans l'espace latent.
* **Livrables spécifiques attendus** :
- Tableau d'ablation sur le poids β (au moins 3 valeurs).
- Comparaison qualitative VAE vanilla vs. CVAE (contrôlabilité de la génération).

---

## 3. Modèle de diffusion (DDPM) entraîné from scratch vs. GAN

* **Type** : Génératif - Diffusion (DDPM), comparaison à un GAN
* **Description** : Implémenter un modèle de diffusion débruitant (DDPM) from scratch à petite échelle, et le comparer à un GAN entraîné sur le même dataset en termes de qualité et de diversité des images générées.
* **Papier(s) de référence** :
  - Ho et al., 'Denoising Diffusion Probabilistic Models' (2020) - https://arxiv.org/abs/2006.11239
  - Goodfellow et al., 'Generative Adversarial Networks' (2014) - https://arxiv.org/abs/1406.2661
* **Dataset(s)** :
  - Fashion-MNIST - https://github.com/zalandoresearch/fashion-mnist
  - CIFAR-10 downscalé (16x16 ou 32x32) - https://www.cs.toronto.edu/~kriz/cifar.html
* **Technos suggérées** : PyTorch, U-Net simple pour le réseau de débruitage, torchmetrics (FID) ou implémentation FID simplifiée
* **Requirements techniques (incluant l'étude d'ablation exigée)** :
- Implémenter le processus de diffusion direct (bruitage progressif) et inverse (débruitage appris par un petit U-Net).
- Entraîner un DCGAN (ou variante) sur le même dataset et budget de calcul comparable.
- Étude d'ablation sur le DDPM : nombre de pas de diffusion (ex. 100 vs 1000) et son effet sur qualité/temps de génération.
- Comparer DDPM vs GAN sur qualité (FID ou inspection visuelle qualitative rigoureuse) et diversité des échantillons générés.
- Documenter la stabilité d'entraînement observée (GAN notoirement instable) vs. celle du DDPM.
* **Application de démonstration attendue** : Application web : génération d'images par les deux modèles, affichage côte à côte pour comparaison visuelle.
* **Livrables spécifiques attendus** :
- Tableau comparatif FID (ou score qualitatif structuré) et temps de génération DDPM vs GAN.
- Discussion sur le compromis qualité/coût de calcul des deux familles génératives.

---

## 4. Recommandation par Graph Neural Network vs. filtrage collaboratif

* **Type** : Recommandation - GNN, évaluation offline rigoureuse
* **Description** : Modéliser les interactions utilisateur-item comme un graphe biparti et entraîner un GNN (type LightGCN) pour la recommandation, en comparant rigoureusement à des baselines de filtrage collaboratif classique.
* **Papier(s) de référence** :
  - He et al., 'LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation' (2020) - https://arxiv.org/abs/2002.02126
  - Koren et al., 'Matrix Factorization Techniques for Recommender Systems' (2009) - https://ieeexplore.ieee.org/document/5197422
* **Dataset(s)** :
  - MovieLens 100K/1M - https://grouplens.org/datasets/movielens/
* **Technos suggérées** : PyTorch Geometric (ou DGL), scikit-surprise (baselines SVD/ALS), FastAPI
* **Requirements techniques (incluant l'étude d'ablation exigée)** :
- Construire le graphe biparti utilisateur-item à partir des interactions.
- Implémenter un GNN de recommandation (LightGCN ou équivalent simplifié).
- Implémenter au moins 2 baselines classiques (SVD, filtrage collaboratif item-item).
- Protocole d'évaluation offline rigoureux : split temporel (pas aléatoire), leave-one-out, métriques precision@k/recall@k/NDCG@k.
- Étude d'ablation : nombre de couches de convolution de graphe (profondeur) et son effet sur l'over-smoothing.
* **Application de démonstration attendue** : Application web : sélection d'un utilisateur → top-N recommandé par chaque méthode, affiché côte à côte.
* **Livrables spécifiques attendus** :
- Tableau comparatif GNN vs baselines sur precision@k/recall@k/NDCG@k.
- Analyse de l'effet de la profondeur du GNN (ablation) sur la qualité des recommandations.

---

## 5. Federated learning avec hétérogénéité de données (non-IID)

* **Type** : Federated Learning - FedAvg vs FedProx sur données non-IID
* **Description** : Simuler un entraînement fédéré entre plusieurs clients dont les données sont distribuées de façon non-IID (déséquilibre de classes par client), et comparer l'algorithme FedAvg à FedProx, conçu pour mieux gérer cette hétérogénéité.
* **Papier(s) de référence** :
  - McMahan et al., 'Communication-Efficient Learning of Deep Networks from Decentralized Data' (FedAvg, 2017) - https://arxiv.org/abs/1602.05629
  - Li et al., 'Federated Optimization in Heterogeneous Networks' (FedProx, 2020) - https://arxiv.org/abs/1812.06127
* **Dataset(s)** :
  - CIFAR-10 - https://www.cs.toronto.edu/~kriz/cifar.html
  - MNIST - https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
  - (partitionné en non-IID via partition de Dirichlet par classe)
* **Technos suggérées** : PyTorch, simulation multi-process ou séquentielle (pas d'infra distribuée réelle), Flower (optionnel, framework de federated learning) ou implémentation maison
* **Requirements techniques (incluant l'étude d'ablation exigée)** :
- Partitionner les données en non-IID entre 5-10 clients simulés (partition de Dirichlet, paramètre α contrôlant l'hétérogénéité).
- Implémenter FedAvg (agrégation simple des poids) et FedProx (terme de régularisation proximal).
- Étude d'ablation : faire varier α (degré d'hétérogénéité) et observer l'écart de performance FedAvg vs FedProx.
- Comparer aussi à un entraînement centralisé (borne supérieure théorique) et à un entraînement purement local sans agrégation (borne inférieure).
- Mesurer le nombre de rounds de communication nécessaires pour converger.
* **Application de démonstration attendue** : Dashboard de suivi : courbes de convergence par méthode (FedAvg/FedProx/centralisé/local) et par niveau d'hétérogénéité.
* **Livrables spécifiques attendus** :
- Tableau croisé algorithme × degré d'hétérogénéité (α) avec accuracy finale et rounds de convergence.
- Discussion : à partir de quel niveau d'hétérogénéité FedProx surpasse-t-il significativement FedAvg ?

---

## 6. Vision Transformer (ViT) vs CNN sur classification fine-grained

* **Type** : Computer Vision - ViT, étude d'ablation (patch size, pré-entraînement)
* **Description** : Implémenter/adapter un Vision Transformer pour une tâche de classification fine-grained (distinction de sous-catégories visuellement proches) et le comparer rigoureusement à un CNN classique, en étudiant l'effet de la taille des patches et du pré-entraînement.
* **Papier(s) de référence** :
  - Dosovitskiy et al., 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' (ViT, 2020) - https://arxiv.org/abs/2010.11929
  - He et al., 'Deep Residual Learning for Image Recognition' (ResNet, 2015) - https://arxiv.org/abs/1512.03385
* **Dataset(s)** :
  - CUB-200-2011 (oiseaux) - https://www.vision.caltech.edu/datasets/cub_200_2011/
  - Stanford Cars (fine-grained classification) - https://ai.stanford.edu/~jkrause/cars/car_dataset.html
* **Technos suggérées** : timm ou transformers (HuggingFace) pour ViT, torchvision (ResNet), wandb/tensorboard
* **Requirements techniques (incluant l'étude d'ablation exigée)** :
- Entraîner un ViT from scratch ET en fine-tuning depuis un pré-entraînement ImageNet.
- Entraîner un CNN (ResNet) comparable en budget de calcul comme baseline.
- Étude d'ablation : taille des patches (ex. 16x16 vs 32x32) et effet de l'augmentation de données sur le ViT from scratch.
- Analyser spécifiquement le comportement data-hungry du ViT (courbe performance vs taille du jeu d'entraînement).
* **Application de démonstration attendue** : Application web : upload d'image → prédiction par ViT et CNN, avec visualisation des cartes d'attention du ViT.
* **Livrables spécifiques attendus** :
- Tableau d'ablation patch size × pré-entraînement (≥ 4 configurations).
- Courbe performance vs quantité de données d'entraînement (ViT vs CNN) et discussion du besoin en données du ViT.

---

## 7. Segmentation sémantique : U-Net vs DeepLab, étude d'ablation

* **Type** : Computer Vision - Segmentation, ablation architecture et loss
* **Description** : Implémenter et comparer deux architectures de segmentation sémantique (U-Net et DeepLabV3) sur un cas d'usage médical ou agricole, avec une étude d'ablation sur le choix de la fonction de perte.
* **Papier(s) de référence** :
  - Ronneberger et al., 'U-Net: Convolutional Networks for Biomedical Image Segmentation' (2015) - https://arxiv.org/abs/1505.04597
  - Chen et al., 'Rethinking Atrous Convolution for Semantic Image Segmentation' (DeepLabV3, 2017) - https://arxiv.org/abs/1706.05587
* **Dataset(s)** :
  - Leaf Disease Segmentation - https://www.kaggle.com/datasets/kumaresanmanickavelu/leaf-disease-segmentation
  - Ou un dataset de segmentation médicale libre (ex. ISIC skin lesion segmentation) - https://challenge.isic-archive.com/data/
* **Technos suggérées** : PyTorch (segmentation_models_pytorch pour DeepLab), albumentations (augmentation), métriques IoU/Dice
* **Requirements techniques (incluant l'étude d'ablation exigée)** :
- Implémenter/adapter U-Net et DeepLabV3 sur le dataset choisi (voir notebooks/01_computer_vision_unet.ipynb pour un squelette U-Net minimal commenté).
- Étude d'ablation sur la fonction de perte : Dice loss vs Cross-Entropy vs combinaison, et effet sur les classes minoritaires.
- Évaluer avec IoU et Dice score par classe (pas seulement une moyenne globale).
- Analyser qualitativement les erreurs de segmentation (bords, petites régions).
* **Application de démonstration attendue** : Application web : upload d'image → masque de segmentation superposé, comparaison des deux architectures.
* **Livrables spécifiques attendus** :
- Tableau IoU/Dice par architecture × fonction de perte (ablation).
- Analyse qualitative des cas d'échec les plus fréquents pour chaque architecture.

---

## 8. Traduction/résumé automatique : entraînement vs modèle pré-entraîné

* **Type** : NLP - Seq2Seq (encodeur-décodeur), comparaison à un modèle pré-entraîné
* **Description** : Entraîner un modèle encodeur-décodeur (Transformer) pour la traduction ou le résumé automatique sur un corpus donné, et le comparer à un modèle pré-entraîné de référence (ex. T5, mBART) sur les mêmes données de test.
* **Papier(s) de référence** :
  - Vaswani et al., 'Attention Is All You Need' (2017) - https://arxiv.org/abs/1706.03762
  - Raffel et al., 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer' (T5, 2020) - https://arxiv.org/abs/1910.10683
* **Dataset(s)** :
  - Opus/WMT (traduction) - https://opus.nlpl.eu/ ou https://huggingface.co/datasets/wmt/wmt19
  - CNN/DailyMail (résumé) - https://huggingface.co/datasets/abisee/cnn_dailymail
* **Technos suggérées** : PyTorch (implémentation seq2seq), transformers (HuggingFace) pour la baseline pré-entraînée, sacrebleu/rouge-score
* **Requirements techniques (incluant l'étude d'ablation exigée)** :
- Implémenter un Transformer encodeur-décodeur from scratch (ou fortement simplifié) et l'entraîner sur un sous-ensemble du corpus.
- Évaluer un modèle pré-entraîné (T5/mBART) en zero-shot puis fine-tuné sur le même sous-ensemble.
- Étude d'ablation : taille du corpus d'entraînement (ex. 10%, 50%, 100%) et son effet sur le modèle from-scratch vs pré-entraîné.
- Évaluer avec BLEU (traduction) ou ROUGE (résumé), et une analyse qualitative sur des exemples.
* **Application de démonstration attendue** : Application web : saisie de texte source → sortie des deux modèles comparée côte à côte avec scores.
* **Livrables spécifiques attendus** :
- Courbe performance vs taille du corpus d'entraînement (from-scratch vs pré-entraîné).
- Discussion : à partir de quelle taille de corpus le from-scratch devient-il compétitif ?

---

## 9. GAN pour génération/augmentation de données : étude de stabilité

* **Type** : Génératif - GAN, ablation de stabilité d'entraînement (DCGAN vs WGAN-GP)
* **Description** : Implémenter un DCGAN et sa variante WGAN-GP (avec gradient penalty) pour la génération d'images ou de données tabulaires, avec une étude systématique de la stabilité d'entraînement des deux variantes.
* **Papier(s) de référence** :
  - Radford et al., 'Unsupervised Representation Learning with Deep Convolutional GANs' (DCGAN, 2015) - https://arxiv.org/abs/1511.06434
  - Gulrajani et al., 'Improved Training of Wasserstein GANs' (WGAN-GP, 2017) - https://arxiv.org/abs/1704.00028
* **Dataset(s)** :
  - Fashion-MNIST - https://github.com/zalandoresearch/fashion-mnist
  - CIFAR-10 - https://www.cs.toronto.edu/~kriz/cifar.html
  - Ou un dataset tabulaire déséquilibré (ex. Credit Card Fraud pour l'augmentation) - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* **Technos suggérées** : PyTorch, torchmetrics (FID), suivi des courbes de perte générateur/discriminateur
* **Requirements techniques (incluant l'étude d'ablation exigée)** :
- Implémenter DCGAN (loss binaire classique) et WGAN-GP (distance de Wasserstein + gradient penalty).
- Étude d'ablation : comparer la stabilité d'entraînement (mode collapse, oscillations de la loss) entre les deux variantes sur plusieurs seeds.
- Mesurer la qualité de génération (FID ou score qualitatif structuré) et la diversité des échantillons.
- Si dataset tabulaire choisi : évaluer l'augmentation de données générées sur la performance d'un classifieur en aval.
* **Application de démonstration attendue** : Application web : génération d'échantillons par les deux modèles + visualisation des courbes de loss en direct.
* **Livrables spécifiques attendus** :
- Tableau de stabilité (nombre d'essais ayant convergé sur N runs) DCGAN vs WGAN-GP.
- Analyse critique du compromis stabilité/qualité/coût de calcul entre les deux approches.

---

## 10. Self-supervised learning (contrastif) pour pré-entraînement visuel

* **Type** : Self-supervised Learning - Apprentissage contrastif (type SimCLR simplifié)
* **Description** : Implémenter une méthode d'apprentissage contrastif simplifiée (type SimCLR) pour pré-entraîner un encodeur d'images sans labels, puis évaluer l'apport de ce pré-entraînement en fine-tuning sur une tâche de classification avec peu de labels.
* **Papier(s) de référence** :
  - Chen et al., 'A Simple Framework for Contrastive Learning of Visual Representations' (SimCLR, 2020) - https://arxiv.org/abs/2002.05709
  - He et al., 'Momentum Contrast for Unsupervised Visual Representation Learning' (MoCo, 2020) - https://arxiv.org/abs/1911.05722 (pour contexte comparatif)
* **Dataset(s)** :
  - CIFAR-10 - https://www.cs.toronto.edu/~kriz/cifar.html
  - STL-10 (adapté au SSL, peu de labels disponibles) - https://cs.stanford.edu/~acoates/stl10/
* **Technos suggérées** : PyTorch, augmentations fortes (albumentations/torchvision.transforms), NT-Xent loss (perte contrastive)
* **Requirements techniques (incluant l'étude d'ablation exigée)** :
- Implémenter le pipeline SimCLR simplifié : paires augmentées, encodeur partagé, tête de projection, perte NT-Xent.
- Pré-entraîner l'encodeur sur l'intégralité des images du dataset SANS utiliser les labels.
- Étude d'ablation : comparer le fine-tuning avec peu de labels (ex. 1%, 10%, 100% des labels) avec et sans pré-entraînement SSL.
- Étudier l'effet du choix des augmentations (crop, color jitter, blur) sur la qualité des représentations apprises.
* **Application de démonstration attendue** : Application web : visualisation des représentations apprises (projection t-SNE/UMAP colorée par classe réelle, non utilisée à l'entraînement).
* **Livrables spécifiques attendus** :
- Courbe accuracy vs % de labels disponibles, avec et sans pré-entraînement SSL (démonstration du principal bénéfice du SSL : peu de labels).
- Ablation sur les augmentations : quelle combinaison est la plus critique pour la qualité des représentations ?

---

# LIVRABLES ET MODALITÉS DE SOUMISSION (NIVEAU MASTER)

## Composition des équipes

* Les projets doivent être réalisés en **équipe de 2 à 3 étudiants maximum**.
* Aucun groupe de 1 étudiant n'est autorisé (sauf exception validée).
* Chaque membre doit avoir un **rôle clairement défini** :
  * Data / Experiment Engineer (pipeline, protocole expérimental)
  * Model / Research Engineer (implémentation du/des modèles, ablation)
  * Reporting / Backend Developer (API, application, rédaction)

## Rapport final - style article scientifique (OBLIGATOIRE)

Le rapport doit être soumis **avant la dernière séance** à l'adresse suivante :
**tchaye59@gmail.com**

### Format du rapport

* Format PDF
* Minimum : **8 pages** - Maximum : **20 pages** (hors annexes)
* Structure de type article scientifique court, figures et tableaux numérotés et référencés dans le texte

### Structure attendue du rapport

1. **Introduction** - contexte, problématique, contribution du projet.
2. **Travaux liés** - synthèse courte des papiers de référence du sujet et de 1-2 travaux
   complémentaires trouvés par l'équipe.
3. **Données** - source(s), analyse exploratoire, prétraitement.
4. **Méthode** - description précise du/des modèle(s), justification des choix
   d'hyperparamètres, description du protocole expérimental (métriques, split, seeds).
5. **Étude d'ablation** (OBLIGATOIRE, section dédiée) - au moins 2 variantes comparées
   scientifiquement, tableau de résultats, discussion de chaque variante.
6. **Résultats** - comparaison chiffrée à la baseline / à l'état de l'art (papier de référence
   ou méthode classique), analyse critique (le modèle est-il réellement meilleur, à quel coût ?).
7. **Déploiement** - API, application de démonstration, Docker (attendu allégé par rapport à la
   Licence : la priorité reste la rigueur expérimentale).
8. **Discussion et limites** - limites méthodologiques, biais, reproductibilité, pistes
   d'amélioration.
9. **Répartition du travail** (TRÈS IMPORTANT) - nom, rôle et contributions précises de
   chaque membre. Toute absence de répartition claire entraîne une pénalité.

## Code source (OBLIGATOIRE)

* Un **repository GitHub**, code propre, documenté, **reproductible** (seeds fixés, script
  unique pour reproduire chaque résultat du tableau d'ablation).
* README avec instructions d'installation, d'exécution, et description de l'architecture.

## Application de démonstration (OBLIGATOIRE, exigence allégée)

Une application simple (Web/Mobile/Desktop) permettant de tester qualitativement le modèle
retenu. L'effort attendu ici est moindre qu'en Licence : elle sert à **illustrer** les résultats,
pas à constituer un produit fini.

## Présentation finale

Chaque groupe présente pendant **20 minutes**, avec emphase sur la **méthode, l'étude
d'ablation et les résultats comparatifs** (la démonstration applicative est secondaire).

## Règles importantes

* Plagiat interdit (code ou rapport).
* Reprise d'une implémentation existante sans adaptation ni compréhension démontrée = 0.
* Absence d'étude d'ablation ou de comparaison chiffrée à une baseline = pénalité lourde.
* Modèle seul sans aucune application de démonstration = REFUSÉ.

## Deadline

Soumission du rapport + code : **avant la dernière séance**. Aucun retard accepté sans
justification valable.
