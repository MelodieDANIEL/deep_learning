.. slide::
Résumé des concepts clés du chapitre 5
================

.. slide::

📖 1. MLP vs CNN : pourquoi les convolutions ?
-----------------

**Problèmes des MLP pour les images** :

- Trop de paramètres (77M pour une image $$224×224$$ RGB)
- Perte de structure spatiale lors de l'aplatissement
- Pas d'invariance par translation

**Avantages des CNN** :

- **Partage de poids** : même filtre appliqué partout → réduction drastique des paramètres
- **Invariance par translation** : détecte les motifs quelle que soit leur position
- **Préservation de la structure spatiale** : traite les régions locales

**Les filtres de convolution** :

- Petites matrices apprenables ($$3×3$$, $$5×5$$, $$7×7$$)
- **Calcul** : le filtre glisse sur l'image et calcule à chaque position la **somme pondérée** des pixels recouverts
- Les valeurs du filtre déterminent ce qu'il détecte : **flou** (somme = 1), **contours** avec Sobel (somme = 0), etc.
- Apprennent automatiquement : contours, formes, objets complexes, etc.

.. slide::

📖 2. Couches de convolution
-------------------

**Paramètres clés de ``nn.Conv2d``** :

- ``in_channels`` : nombre de canaux en entrée (3 pour RGB)
- ``out_channels`` : nombre de filtres à apprendre
- ``kernel_size`` : taille du filtre (3×3, 5×5, etc.)
- ``stride`` : pas de déplacement (1 par défaut)
- ``padding`` : zéros ajoutés autour (préserve la taille si =1)

**Calcul de la taille de sortie** : $$H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{padding} - \text{kernel_size}}{\text{stride}} \right\rfloor + 1$$

**Le padding** : essentiel pour ne pas perdre d'information sur les bords.

**Cartes de caractéristiques** (*feature maps*) : chaque filtre produit une carte qui indique où son motif apparaît dans l'image. Une couche avec 64 filtres produit 64 cartes, qui deviennent les 64 canaux d'entrée de la couche suivante.

.. slide::

📖 3. Pooling
-------------------

**Max Pooling** (le plus utilisé) :

- Prend le maximum dans chaque région (kernel $$2×2$$ typiquement)
- Divise la taille spatiale par 2

**Avantages** :

- Diminue le nombre de paramètres et le temps de calcul
- Apporte une invariance aux petites translations
- Augmente le champ réceptif

**Architecture d'un CNN** (avec ``nn.Sequential``) :

- un bloc ``convolutions`` qui extrait les caractéristiques : (``nn.Conv2d`` → ``nn.ReLU`` → ``nn.MaxPool2d``) répété plusieurs fois,
- un bloc ``classification`` qui calcule les scores des classes : ``nn.Flatten`` → couches ``nn.Linear``.

⚠️ La taille d'entrée de la première couche ``nn.Linear`` dépend de la taille des images : nombre de filtres × $$H_{final}$$ × $$W_{final}$$.

.. slide::

📖 4. Datasets et transformations d'images
-------------------

🧠 **Rappel (chapitre 3)** : mini-batchs, ``Dataset``, ``DataLoader`` et séparation train/validation/test.

**Dataset d'images** :

- Doit implémenter ``__len__`` et ``__getitem__``
- Charge les images depuis le disque (ou depuis la mémoire avec ``preload``) et applique les transformations dans ``__getitem__``, pour que les augmentations aléatoires changent à chaque époque
- ``torchvision.datasets`` fournit des datasets prêts à l'emploi (MNIST, CIFAR-10, etc.)

**Transformations** (chapitre 4, section 5.4) : on passe un ``transforms.Compose`` au paramètre ``transform`` du ``Dataset`` :

- pour l'entraînement : augmentation (``RandomHorizontalFlip``, ``RandomRotation``, ``ColorJitter``…) + prétraitement,
- pour la validation et le test : prétraitement seul (``Resize``, ``ToTensor``, ``Normalize``).

💡 **Pourquoi pas d'augmentation pour val/test ?** On veut évaluer le modèle sur les vraies images.

.. slide::

📖 5. Sauvegarde de modèles
-------------------

**Trois méthodes** :

1. **Tout le modèle** : ``torch.save(model, 'model.pth')`` (éviter si possible ; pour le recharger, ``torch.load(..., weights_only=False)``)

2. **Poids uniquement** (recommandé ✓) : ``torch.save(model.state_dict(), 'weights.pth')``

3. **État complet** (pour reprendre l'entraînement) :

   - Sauvegarde : epoch, model_state_dict, optimizer_state_dict, loss, métriques
   - Permet de reprendre exactement où on s'est arrêté

✓ **Bonnes pratiques** : Préférer ``state_dict()`` au modèle complet, sauvegarder le meilleur modèle basé sur validation loss, inclure epoch et optimizer dans les checkpoints
