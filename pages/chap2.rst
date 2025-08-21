
.. slide::

Chapitre 2 - Introduction 
================

🎯 Objectifs du Chapitre
----------------------


.. important::

   À la fin de ce chapitre, vous saurez : 

   - Créer et manipuler des tenseurs PyTorch sur CPU et GPU.
   - Calculer automatiquement les gradients à l’aide de ``autograd``.
   - Définir une fonction de perte.
   - Utiliser un optimiseur pour ajuster les paramètres d’un modèle.
   - Implémenter une boucle d'entraînement simple.

.. slide::

📖 1. Qu'est-ce que PyTorch ? 
----------------------


################################ Activation fonction ######################################
Parler de la softmax, relu , etc.
##########################################################################################



.. slide::
📖 20. Optimisation dans PyTorch
-----------------------

PyTorch fournit le module ``torch.optim`` qui implémente plusieurs algorithmes d’optimisation (SGD, Adam, etc.).  

Exemple avec la descente de gradient stochastique (SGD) :

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Exemple : un modèle très simple (une seule couche linéaire)
    model = nn.Linear(1, 1)

    # Fonction de perte
    loss_fn = nn.MSELoss()

    # Optimiseur : SGD avec un learning rate de 0.01
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Exemple de données
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([[2.0], [4.0], [6.0]])  # y = 2x

    # Étape d’entraînement
    y_pred = model(x)            # 1. prédiction
    loss = loss_fn(y_pred, y)    # 2. calcul de la perte

    optimizer.zero_grad()        # 3. réinitialise les gradients
    loss.backward()              # 4. rétropropagation
    optimizer.step()             # 5. mise à jour des poids

---










