
.. slide::

Chapitre 1 - Introduction à PyTorch et Optimisation de Modèles
================

🎯 Objectifs du Chapitre
----------------------


.. important::

   À la fin de ce chapitre, vous saurez : 

   - Créer et manipuler des tenseurs PyTorch sur CPU et GPU.
   - Calculer automatiquement les gradients à l’aide de ``autograd``.
   - Définir une fonction de coût.
   - Utiliser un optimiseur pour ajuster les paramètres d’un modèle.
   - Implémenter une boucle d'entraînement simple.

.. slide::

📖 1. Qu'est-ce que PyTorch ? 
----------------------
PyTorch est une bibliothèque Python de machine learning open-source développée par Facebook (FAIR). Elle est conçue pour faciliter la création et l'entraînement de modèles, en particulier dans le domaine du deep learning. 

Elle repose principalement sur deux éléments :

A) Les *tenseurs*, des structures de données similaires aux tableaux NumPy (``ndarray``), mais avec des fonctionnalités supplémentaires pour :
    
    - le calcul différentiel automatique,
    - l'accélération GPU,
    - l’entraînement de réseaux de neurones.

B) Le module ``autograd`` permet de calculer automatiquement les gradients nécessaires à l'entraînement des modèles, en suivant toutes les opérations effectuées sur les tenseurs.

.. slide::

D'autres bibliothèques Python similaires existent, comme :

- TensorFlow : développé par Google, très utilisé pour des déploiements à grande échelle.
- Keras : interface haut niveau de TensorFlow, plus simple mais moins flexible.
- JAX : plus récent, optimisé pour la recherche et les calculs scientifiques à haute performance.

.. slide::

Dans le cadre de ce cours, nous utiliserons PyTorch car :

- elle est largement adoptée par la communauté de la recherche en deep learning,
- elle est plus lisible et plus facile à déboguer que TensorFlow et JAX,
- elle offre plus de possibilités que Keras,
- elle est bien documentée et est l'une des bibliothèques les plus utilisées en science des données (Data Science en anglais) et en apprentissage machine (Machine Learning en anglais).

.. slide::
