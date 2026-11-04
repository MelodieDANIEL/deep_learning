.. slide::
Résumé des concepts clés du chapitre 5
================

.. slide::

📖 1. MLP vs CNN : pourquoi les convolutions ?
-----------------

**Problèmes des MLP pour les images → solutions des CNN** :

- Trop de paramètres (77 millions de poids pour une seule couche de 512 neurones, sur une image $$224×224$$ RGB) → **partage de poids** : le même petit filtre est appliqué sur toute l'image
- Perte de la structure spatiale (l'image est aplatie) → **traitement local** : le filtre travaille sur des pixels voisins
- Pas de généralisation spatiale → **invariance par translation** : un motif appris à un endroit est détecté partout

**Les filtres de convolution** :

- Petites matrices de poids ($$3×3$$, $$5×5$$, $$7×7$$)
- **Calcul** : le filtre glisse sur l'image et calcule à chaque position la **somme pondérée** des pixels recouverts
- Les valeurs du filtre déterminent ce qu'il détecte : **flou** (somme = 1), **contours** avec Sobel (somme = 0), etc.
- Dans un CNN, les filtres sont initialisés au hasard puis **appris** pendant l'entraînement : contours, formes, objets complexes, etc.

.. slide::

📖 2. Couches de convolution
-------------------

- ``F.conv2d(x, filtre)`` : on donne soi-même le filtre (flou, Sobel…)
- ``nn.Conv2d`` : une **couche** qui crée ses filtres et les apprend, comme ``nn.Linear``

**Paramètres clés de ``nn.Conv2d``** :

- ``in_channels`` : nombre de canaux en entrée (1 en niveaux de gris, 3 en RGB)
- ``out_channels`` : nombre de filtres à apprendre
- ``kernel_size`` : taille $$k$$ du filtre ($$k×k$$)
- ``stride`` : pas de déplacement du filtre (1 par défaut)
- ``padding`` : nombre de pixels de zéros ajoutés autour de l'image (0 par défaut)

**Filtres appris** : une couche ``nn.Conv2d`` range les valeurs de ses filtres dans son attribut ``weight``, de forme $$(C_{out}, C_{in}, k_H, k_W)$$. Avec un biais par filtre, elle a $$C_{out} \times (C_{in} \times k_H \times k_W + 1)$$ paramètres : par exemple $$64 \times (3 \times 3 \times 3 + 1) = 1792$$ pour ``nn.Conv2d(3, 64, 3)``.

**Taille de sortie** : $$H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{padding} - \text{kernel_size}}{\text{stride}} \right\rfloor + 1$$ ($$\lfloor \; \rfloor$$ : arrondi à l'entier inférieur)

**Padding** : sans padding (et avec ``stride=1``), on perd $$k - 1$$ lignes et $$k - 1$$ colonnes. Pour conserver la taille : ``padding`` $$= \frac{k-1}{2}$$, soit 1 pour un filtre $$3×3$$ et 2 pour un filtre $$5×5$$.

**Cartes de caractéristiques** (*feature maps*) : chaque filtre produit une carte qui indique où son motif apparaît dans l'image. Une couche avec 64 filtres produit 64 cartes, qui deviennent les 64 canaux d'entrée de la couche suivante.

.. slide::

📖 3. Pooling et architecture d'un CNN
-------------------

- ``nn.MaxPool2d(2)`` : garde le **maximum** de chaque région $$2×2$$ (le plus utilisé)
- ``nn.AvgPool2d(2)`` : garde la **moyenne** de chaque région $$2×2$$
- La hauteur et la largeur sont divisées par 2, arrondies à l'entier inférieur ($$13×13$$ → $$6×6$$)

**Avantages** : moins de calculs et de paramètres, invariance aux petites translations, champ réceptif plus grand.

**Architecture d'un CNN** (avec ``nn.Sequential``) :

- un bloc ``convolutions`` qui extrait les caractéristiques : (``nn.Conv2d`` → ``nn.ReLU`` → ``nn.MaxPool2d``) répété plusieurs fois,
- un bloc ``classification`` qui calcule, pour chaque image, un score par classe : ``nn.Flatten`` → couches ``nn.Linear``.

⚠️ La taille d'entrée de la première couche ``nn.Linear`` dépend de la taille des images : nombre de filtres × $$H_{final}$$ × $$W_{final}$$.

.. slide::

📖 4. Datasets et transformations d'images
-------------------

🧠 **Rappel (chapitre 3)** : mini-batchs, ``Dataset``, ``DataLoader`` et séparation train/validation/test.

**Dataset d'images** :

- Doit implémenter ``__len__`` et ``__getitem__``
- Charge les images depuis le disque (ou depuis la mémoire avec ``preload``) et applique les transformations dans ``__getitem__``, pour que les augmentations aléatoires changent à chaque époque
- ``torchvision.datasets`` fournit des datasets prêts à l'emploi (MNIST, CIFAR-10, etc.)
- Le ``Dataset`` renvoie **une** image et son label, le ``DataLoader`` les regroupe en **batchs** (``shuffle=True`` pour l'entraînement, ``False`` pour la validation et le test)

**Transformations** (chapitre 4, section 5.4) : on passe un ``transforms.Compose`` au paramètre ``transform`` du ``Dataset`` :

- pour l'entraînement : augmentation (``RandomHorizontalFlip``, ``RandomRotation``, ``ColorJitter``…) + prétraitement,
- pour la validation et le test : prétraitement seul (``Resize``, ``ToTensor``, ``Normalize``).

💡 **Pourquoi pas d'augmentation pour val/test ?** On veut évaluer le modèle sur les vraies images. Pour n'augmenter que le train : deux ``Dataset`` sur les mêmes images (avec et sans augmentation), découpés par ``random_split`` avec la même graine (section 6.1).

.. slide::

📖 5. Sauvegarder et charger les poids d'un modèle
-------------------

- **Sauvegarder les poids** : ``torch.save(model.state_dict(), 'poids.pth')``
- **Les recharger** : recréer le modèle (mêmes couches), puis ``model.load_state_dict(torch.load('poids.pth'))``
- **Reprendre un entraînement** : sauvegarder aussi ``optimizer.state_dict()`` et le numéro de l'époque dans un dictionnaire

.. slide::

📖 6. Pipeline complet
-------------------

- **Loss** : ``nn.CrossEntropyLoss()``, qui applique elle-même le softmax aux scores
- **À chaque époque** : ``model.train()`` pour l'entraînement, puis ``model.eval()`` et ``torch.no_grad()`` pour la validation
- **Classe prédite** : avec ``outputs = model(images)`` (un score par classe), ``_, predicted = torch.max(outputs, 1)`` donne la position du plus grand score ; accuracy = 100 × bonnes prédictions / nombre d'images
- **Meilleur modèle** : on le sauvegarde d'après la validation, puis on le recharge pour l'évaluation finale sur le test, faite **une seule fois**, à la fin
