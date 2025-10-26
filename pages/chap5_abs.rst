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
- Apprennent automatiquement : contours, formes, objets complexes, etc.

.. slide::

📖 2. Couches de convolution
-------------------

**Paramètres clés de ``conv2d``** :

- ``in_channels`` : nombre de canaux en entrée (3 pour RGB)
- ``out_channels`` : nombre de filtres à apprendre
- ``kernel_size`` : taille du filtre (3×3, 5×5, etc.)
- ``stride`` : pas de déplacement (1 par défaut)
- ``padding`` : zéros ajoutés autour (préserve la taille si =1)

**Calcul de la taille de sortie** : $$H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{padding} - \text{kernel_size}}{\text{stride}} \right\rfloor + 1$$

**Le padding** : essentiel pour ne pas perdre d'information sur les bords.

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

.. slide::

📖 4. Mini-batchs
-------------------

**Trois approches** :

1. **Batch Gradient Descent** : tout le dataset (lent mais stable)
2. **SGD** : un exemple à la fois (rapide mais bruité)
3. **Mini-Batch** : compromis idéal (32 ou 64 exemples) ✓

**Avantages** : exploite le GPU, estime bien le gradient, régularisation naturelle.

.. slide::

📖 5. Datasets et DataLoaders
-------------------

**Dataset** : classe pour organiser vos données

- Doit implémenter ``__len__`` et ``__getitem__``
- Peut charger des images depuis le disque et appliquer des transformations si nécessaire.

**DataLoader** : automatise le chargement

- Découpage en mini-batchs
- Mélange des données (``shuffle=True`` pour train, ``False`` pour val/test)
- Chargement parallèle (``num_workers``)

✓ **Bonnes pratiques** : Toujours utiliser ``Dataset`` et ``DataLoader`` pour gérer les données

.. slide::

📖 6. Train/Val/Test
-------------------

**Proportions recommandées** :

- **Train** (70-80%) : entraînement du modèle
- **Validation** (10-15%) : surveillance et sélection du meilleur modèle pendant l'entraînement
- **Test** (10-15%) : évaluation finale uniquement

⚠️ **Règle d'or** : Ne JAMAIS utiliser le test set pendant l'entraînement !

✓ **Bonnes pratiques** : Utiliser ``random_split`` pour diviser automatiquement et toujours séparer les trois ensembles

.. slide::

📖 7. Transformations d'images
-------------------

**Prétraitement (toujours nécessaire)** :

- ``ToTensor()`` : convertit en tenseur PyTorch
- ``Normalize(mean, std)`` : centre les valeurs autour de 0

**Augmentation (train uniquement)** :

- ``RandomHorizontalFlip()`` : retourne horizontalement
- ``RandomRotation()`` : rotation aléatoire
- ``ColorJitter()`` : modifie luminosité/contraste

💡 **Pourquoi pas d'augmentation pour val/test ?** On veut évaluer sur les vraies images.

✓ **Bonnes pratiques** : Augmentation uniquement pour l'entraînement, jamais pour validation/test

.. slide::

📖 8. Sauvegarde de modèles
-------------------

**Trois méthodes** :

1. **Tout le modèle** : ``torch.save(model, 'model.pth')`` (éviter si possible)

2. **Poids uniquement** (recommandé ✓) : ``torch.save(model.state_dict(), 'weights.pth')``

3. **État complet** (pour reprendre l'entraînement) :

   - Sauvegarde : epoch, model_state_dict, optimizer_state_dict, loss, métriques
   - Permet de reprendre exactement où on s'est arrêté

✓ **Bonnes pratiques** : Préférer ``state_dict()`` au modèle complet, sauvegarder le meilleur modèle basé sur validation loss, inclure epoch et optimizer dans les checkpoints






