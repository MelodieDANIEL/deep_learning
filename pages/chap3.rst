.. slide::
Chapitre 2 — Perceptron multi-couches 
===========================================

🎯 Objectifs du Chapitre
----------------------


.. important::

    À la fin de cette section,  vous saurez :  

    - Le fonctionnement du perceptron simple.
    - Utiliser une fonction d'activation adaptée.  
    - L’importance de la normalisation / standardisation des données et l'usage des epochs.  
    - Construire un réseau de neurones avec ``torch.nn``. 
    - Faire un entraînement simple d’un MLP pour un problème de régression.   
    - Suivre l’évolution de la loss et interpréter les résultats.  
    - Utiliser ``torch-summary`` pour inspecter l’architecture du réseau.  


.. slide::

📖 1. Rappels sur les perceptrons
----------------------

Le perceptron multi-couches (MLP de Multi-Layers Perceptron en anglais) est la brique de base des réseaux de neurones modernes. Dans ce chapitre, nous allons l’appliquer à des problèmes de régression simple. Avant de commencer, voici quelques rappels.

1.1. Perceptron simple
~~~~~~~~~~~~~~~~~~~~~~

Le perceptron est le bloc de base d’un réseau de neurones. Il réalise une transformation linéaire suivie (ou pas) d’une fonction d’activation telle que :  

  .. math::

     y = \sigma(Wx + b)

  où :  
   - $$y$$ est la sortie du perceptron, 
   - $$\sigma$$ est une fonction d’activation, 
   - $$W$$ est la matrice des poids,  
   - $$b$$ est le biais et  
   - $$x$$ est l'ensemble des entrées du perceptron.  

.. slide::
1.2. Perceptron intuition
~~~~~~~~~~~~~~~~~~~~~~
.. image:: images/chap2_perceptron.png
    :alt: perceptron
    :align: center
    :width: 40%

avec $$y= \sigma(x_1*w_1 + x_2*w_2 + ...+ x_i*w_i + ... + x_n*w_n + b)$$

💡 **Intuition :**

    - Chaque poids $$w_i$$ mesure l’importance de la caractéristique $$x_i$$.  
    - Le biais $$b$$ déplace la frontière de décision.  
    - La fonction d’activation permet d’introduire de la non-linéarité, indispensable pour modéliser des relations complexes mais nous en parleront plus en détails par la suite.  






.. slide::
📖 7. Observer la loss et déterminer le nombre d’epochs
------------------------------------------------------
Lorsqu’on entraîne un modèle, il est essentiel de suivre l’évolution de la loss pour savoir si le modèle apprend correctement et converge vers une solution. Dans l’exemple précédent, nous avons comparé l’impact de la standardisation sur les prédictions finales. Nous allons maintenant observer l’évolution de la loss pendant l’entraînement pour mieux comprendre la convergence et déterminer un nombre d’epochs approprié. Nous allons continuer à utiliser les données suivantes pour entraîner le modèle :

.. code-block:: python

   # Données d'entraînement
   X = torch.tensor([[0.],[10.],[20.],[30.],[40.],[50.]])
   y = 2*X + 1



################################ STOP ICI ################################

################################ STOP ICI ################################

################################ STOP ICI ################################

################################ STOP ICI ################################

################################ STOP ICI ################################

################################ STOP ICI ################################



#####################

parler de dataset loader dans le chapitre 5 et parler de la gestion des outliers

Exemple avec scikit-learn :  À ajouter pour standardiser les données ????? 

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

#####################

############################ Il faudra penser à créer un gitlab pour le cours ##################

