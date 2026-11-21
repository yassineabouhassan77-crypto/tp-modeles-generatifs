# TP — Modèles génératifs avec PyTorch (Partie 1)

TP d'IA générative multi-modale : du modèle discriminatif au Variational Autoencoder, sur MNIST.

Tout le travail est dans le notebook [`modeles_generatifs.ipynb`](modeles_generatifs.ipynb) (les sorties sont enregistrées, il peut être lu sans être exécuté).

## Contenu

| Étape | Contenu | Résultat |
|---|---|---|
| 0. Environnement | Projet géré avec [uv](https://docs.astral.sh/uv/), PyTorch, torchvision, torcheval, matplotlib | — |
| 1. Données | Chargement de MNIST (60 000 train / 10 000 test), visualisation, répartition des classes | — |
| 2. Classifieur | CNN (2 couches conv + 2 denses), cross-entropy, 3 époques | Accuracy test ≈ 98,8 % |
| 3. Auto-encodeur | MLP encodeur/décodeur, latent 2D, BCE | Bonnes reconstructions, mais l'échantillonnage depuis $\mathcal{N}(0, I)$ ne couvre pas toutes les classes |
| 4. VAE | Même architecture + KL, astuce de reparamétrisation, ELBO | Latent ≈ $\mathcal{N}(0, I)$, génération de toutes les classes, variété latente continue, interpolation |

Le notebook se termine par une comparaison des trois approches (discriminatif vs auto-encodeur vs VAE).

## Lancer le projet

Prérequis : Python 3.13 et [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/yassineabouhassan77-crypto/tp-modeles-generatifs.git
cd tp-modeles-generatifs
uv sync
```

Puis ouvrir `modeles_generatifs.ipynb` dans VS Code / Jupyter avec le kernel `.venv`, ou l'exécuter en ligne de commande :

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace modeles_generatifs.ipynb
```

Le dataset MNIST est téléchargé automatiquement dans `data/` (ignoré par git) au premier lancement. L'exécution complète prend quelques minutes sur CPU ; un GPU CUDA est utilisé automatiquement s'il est disponible.
