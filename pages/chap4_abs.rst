.. slide::
Résumé des concepts clés du chapitre 4
================

.. slide::

📖 1. Qu'est-ce qu'une image ?
-----------------

**Une image = une grille de pixels = un tenseur** :

- Niveaux de gris : 1 valeur par pixel (0 = noir, 255 = blanc)
- Couleur **RGB** : 3 valeurs par pixel (rouge, vert, bleu), les couleurs s'additionnent
- **RGBA** : un 4e canal alpha pour la transparence

**Ordre des dimensions** :

- NumPy, PIL, Matplotlib : $$(H, W, C)$$ (*channel-last*)
- PyTorch : $$(C, H, W)$$ (*channel-first*), et $$(N, C, H, W)$$ pour un batch

**Type des valeurs** :

- ``uint8`` : entiers dans [0, 255] (fichiers image)
- ``float32`` : réels dans [0, 1] (réseaux de neurones)

⚠️ **Dépassements** : en ``uint8``, 200 + 100 = 44 ! Convertir en ``float32`` avant les calculs.

.. slide::

📖 2. Charger, afficher, sauvegarder
-------------------

- ``Image.open('x.jpg')`` (PIL) : ⚠️ ``.size`` donne **(largeur, hauteur)**
- ``np.array(img_pil)`` : (H, W, C) en ``uint8``
- ``plt.imread`` : ⚠️ ``uint8`` pour un JPEG mais ``float32`` dans [0, 1] pour un PNG
- ``plt.imshow(img)`` : affichage, avec ``cmap='gray', vmin=0, vmax=255`` pour les niveaux de gris
- ``transforms.ToTensor()`` : PIL → tenseur $$(C, H, W)$$ en ``float32`` dans [0, 1]
- Pour afficher un tenseur : ``plt.imshow(img_t.permute(1, 2, 0))``

✓ **Bonnes pratiques** : toujours vérifier ``shape``, ``dtype``, ``min()`` et ``max()`` après un chargement.

.. slide::

📖 3. Slicing
-------------------

- Origine (0, 0) **en haut à gauche**, et accès par ``img[y, x]`` : **la ligne d'abord** !
- Syntaxe ``debut:fin:pas`` (début inclus, fin exclue)

**Exemples sur une image (H, W, C)** :

- Recadrage : ``img[95:205, 405:520]``
- Un canal : ``img[:, :, 0]``
- Sous-échantillonnage : ``img[::8, ::8]``
- Miroirs : ``img[:, ::-1]`` (horizontal) et ``img[::-1, :]`` (vertical)
- **Canal alpha** (transparence) : ``img_rgba[:, :, 3] = 60``

⚠️ En PyTorch $$(C, H, W)$$ : ``img_t[:, 95:205, 405:520]``. Le slicing crée une **vue** : utiliser ``.copy()`` / ``.clone()`` pour une vraie copie.

.. slide::

📖 4. Modifier les pixels
-------------------

- Opérations sur toute l'image **sans boucle** : luminosité ``img + 60``, négatif ``255 - img``, puis ``np.clip(..., 0, 255)``
- Niveaux de gris : $$0.299 R + 0.587 G + 0.114 B$$
- **Masque booléen** : ``masque = (img[:, :, 0] > 150) & ...`` puis ``img[masque] = [255, 0, 0]`` (``&``, ``|``, ``~`` et non ``and``, ``or``, ``not``)
- **Histogramme** : ``plt.hist(img[:, :, c].ravel(), bins=256)`` pour étudier la luminosité et le contraste

.. slide::

📖 5. Transformations torchvision
-------------------

- ``Resize((H, W))`` : ⚠️ (hauteur, largeur), alors que ``img_pil.resize((W, H))``
- ``CenterCrop``, ``RandomCrop``, ``RandomResizedCrop``, ``RandomHorizontalFlip``, ``RandomRotation``, ``ColorJitter``, ``Grayscale``, ``GaussianBlur``
- ``Normalize(mean, std)`` : ⚠️ c'est une **standardisation** par canal, avec ``mean`` et ``std`` calculés sur le jeu d'entraînement
- ``Compose([...])`` : enchaîne les transformations (``Normalize`` après ``ToTensor``)

**Prétraitement** (toutes les images) vs **augmentation** (aléatoire, entraînement uniquement).

⚠️ Une augmentation ne doit pas changer l'étiquette, et jamais d'augmentation pour la validation et le test.

.. slide::

📖 6. Batch d'images
-------------------

- ``torch.stack(liste_d_images)`` → $$(N, C, H, W)$$
- ⚠️ Toutes les images doivent avoir la même taille et le même nombre de canaux (``.convert('RGB')``)
- Les opérations s'appliquent à tout le batch d'un coup : ``batch.mean(dim=(0, 2, 3))``, ``batch.flip(dims=[3])``, ``F.interpolate(batch, size=(64, 64))``
