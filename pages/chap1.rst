
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

A) Les *tenseurs*, des structures de données similaires aux tableaux NumPy (`ndarray`), mais avec des fonctionnalités supplémentaires pour :
    
    - le calcul différentiel automatique,
    - l'accélération GPU,
    - l’entraînement de réseaux de neurones.

B) Le module ``autograd``, qui permet de calculer automatiquement les gradients nécessaires à l'entraînement des modèles, en suivant toutes les opérations effectuées sur les tenseurs.

.. slide::

D'autres bibliothèques Python similaires existent, comme :

- **TensorFlow** : développé par Google, très utilisé pour des déploiements à grande échelle.
- **Keras** : interface haut niveau de TensorFlow, plus simple mais moins flexible.
- **JAX** : plus récent, optimisé pour la recherche et les calculs scientifiques à haute performance.

.. slide::

Dans le cadre de ce cours, nous utiliserons **PyTorch** car :

- il est largement adopté par la communauté de la recherche en deep learning,
- il est plus lisible et plus facile à déboguer que TensorFlow et JAX,
- il offre plus de possibilités que Keras,
- il est bien documenté et est l'une des bibliothèques les plus utilisées en science des données (Data Science en anglais) et en apprentissage machine (Machine Learning en anglais).

.. slide::

📖 2. Créer un environnement virtuel
----------------------
Pour installer proprement PyTorch et les bibliothèques nécessaires, nous allons d’abord créer un environnement virtuel. 

.. slide::

2.1 Qu'est-ce qu'un environnement virtuel ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Un environnement virtuel (ou Virtual Env en anglais) est un dossier isolé dans lequel on peut installer des bibliothèques Python sans interférer avec le reste du système.

En pratique, cela permet :

- d’avoir une version précise des bibliothèques pour un projet donné,
- d’éviter les conflits entre différentes versions de packages,
- de tester des versions spécifiques de bibliothèques sans risque,
- de partager facilement le projet avec d'autres personnes,
- de garantir que le code fonctionne de la même manière sur différentes machines,
- de ne pas polluer l’installation Python globale de votre ordinateur.

C’est une pratique essentielle pour tous les projets en Machine Learning.

.. slide::

2.2 Tester si ``venv`` est disponible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le module ``venv``, inclus normalement avec Python 3, permet de créer un environnement virtuel. Avant de l’utiliser, vous pouvez vérifier s’il est installé en tapant la commande suivante dans votre terminal : 

.. code-block:: bash

   python3 -m venv --help

- Si l’aide s’affiche, le module est disponible.
- Sinon, vous verrez une erreur indiquant que ``venv`` est introuvable. Dans ce cas, installez-le avec :

.. code-block:: bash

   sudo apt install python3-venv

.. slide::

2.3 Créer l’environnement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour créer un environnement virtuel, vous pouvez utiliser la commande suivante dans votre terminal :

.. code-block:: bash

   python -m venv nom_de_l_environnement
où ``nom_de_l_environnement`` est le nom que vous souhaitez donner à votre environnement virtuel.

Placez-vous dans le dossier de travail de votre projet (par exemple ``cours_dl/``), puis créez un environnement virtuel avec :

.. code-block:: bash

   python3 -m venv nom_de_l_environnement

Cela crée un sous-dossier nommé ``nom_de_l_environnement`` contenant une version isolée de Python. Par exemple, si vous nommez votre environnement ``env_dl``, vous aurez un dossier ``env_dl`` dans votre répertoire de travail.

.. slide::

2.4 Activer l’environnement virtuel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vous devez ensuite activer l’environnement pour l’utiliser en tapant dand votre terminal :

.. code-block:: bash
    
    source nom_de_l_environnement/bin/activate 

Vous saurez que l'environnement est activé lorsque le nom de l'environnement apparaîtra entre parenthèses au début de votre invite de commande dans le terminal.

.. slide::

2.5 Désactiver l’environnement virtuel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour désactiver l'environnement virtuel, vous pouvez utiliser la commande :

.. code-block:: bash

   deactivate


.. slide::

📖 3. Installation de PyTorch
----------------------


#####################################
A FAIRE : ILLUSTRER AVEC DES FIGURES
#####################################