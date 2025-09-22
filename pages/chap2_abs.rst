.. slide::
Résumé des concepts clés du chapitre 2
======================================

.. slide::
📖 1. Perceptron simple
-----------------------

- Formule : $$y = \sigma(Wx + b)$$
- Les **poids** ($$w_i$$) mesurent l’importance des entrées, le **biais** ($$b$$) déplace la frontière de décision.
- **Mise à jour des paramètres** via la descente de gradient.
- Limite : ne résout que les problèmes **linéairement séparables** (ex. porte logique ET).
- Exemple : **XOR** n’est pas linéairement séparable → besoin d’un MLP.

.. slide::
📖 2. Fonctions d’activation
----------------------------

Introduisent la **non-linéarité** et influencent la convergence :

- **Sigmoïde** : sortie [0,1], utile pour la classification binaire (probabilité).    
- **Tanh** : sortie [-1,1], centrée, parfois plus stable que sigmoïde.  
- **ReLU** : rapide, efficace pour les couches cachées, évite vanishing gradient.  
- **Softmax** : pour la classification multi-classes, renvoie une distribution de probabilités.

👉 En résumé :  
- Pour une sortie **binaire**, on utilise la **sigmoïde** car elle produit une probabilité entre 0 et 1.  
- Pour une sortie **multi-classes**, on utilise la **softmax** qui renvoie une distribution de probabilités.  
- Pour les **couches cachées**, on privilégie **ReLU** (ou Tanh) car elles favorisent une meilleure convergence.  


.. slide::
📖 3. Epochs, Batchs, Itérations
--------------------------------

- **Itération** = une mise à jour après un batch.
- **Batch** = sous-ensemble des données.
- **Epoch** = passage complet sur toutes les données.

⚠️ Plusieurs epochs sont nécessaires pour que le modèle **converge**.


.. slide::
4. Normalisation vs. Standardisation
~~~~~~~~~~~~~~~~~~

- **Objectif** : préparer les données pour que le modèle apprenne plus efficacement et converge plus rapidement en évitant qu'une variable domine les autres. 

- **Normalisation** : ramène les valeurs dans [0,1].
- **Standardisation** : centre autour de 0 et réduit par l’écart-type. 

- **Limites de la normalisation** :
  - Dépend des valeurs min et max : les valeurs extrêmes ou aberrantes peuvent fausser l’échelle.
  - Ne centre pas les données : la moyenne n’est pas proche de 0, ralentissant la convergence.
  - Ne standardise pas l’écart-type : certaines variables peuvent dominer le gradient.

- **Avantages de la standardisation** :
  - Centrage et réduction des variables → moyenne proche de 0 et écart-type proche de 1.
  - Plus robuste aux valeurs aberrantes.
  - Accélère et stabilise la convergence du modèle.


.. slide::
📖 5. Perceptron multi-couches (MLP)
------------------------------------

- Un **MLP** = plusieurs couches linéaires + fonctions d’activation.
- **Couches** :
  - Entrée (features),
  - Cachées (non-linéarité),
  - Sortie (prédiction finale).

- Construction en PyTorch :
  - Avec ``nn.Sequential`` (rapide).
  - Avec une **classe** (plus flexible).

- Exemple : résolution du **XOR** possible avec au moins une couche cachée.

.. slide::
📖 6. Broadcasting
------------------

- Permet à PyTorch de faire des calculs avec des tenseurs de dimensions différentes sans passer par des boucles explicites.

.. slide::
📖 7. Observer la loss
----------------------

- La **loss** mesure l’erreur du modèle à chaque étape.  
- Si la loss **diminue et se stabilise** → le modèle converge.  
- Si la loss reste **élevée ou diverge** → le modèle n’apprend pas correctement.  
- En observant le graphique de la loss, on peut choisir le **nombre d’epochs suffisant** :  
  quand la courbe se stabilise, il est inutile de continuer l’entraînement pour éviter le surapprentissage.  
- **Early stopping** : arrêter automatiquement si la loss ne s’améliore plus.

.. slide::
📖 8. Inspection & Profiling
----------------------------

- **torch-summary** : affiche l’architecture, les dimensions et le nombre de paramètres.
- **torch.autograd.profiler** : mesure temps et mémoire des opérations → utile pour optimiser.

