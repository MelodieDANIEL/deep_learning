🏋️ Travaux Pratiques 3
=========================

.. warning::
  Les exercices de ce TP sont volontairement moins guidés. Il est temps de prendre un peu d'autonomie !

.. slide::
Exercice 0 : Mise en place
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Créer un notebook Jupyter et importer les bibliothèques nécessaires. Assurez-vous que celles-ci soient disponibles dans votre noyau jupyter.

- numpy
- matplotlib
- sklearn
- pandas
- torch
- (optionnel) wandb

Les exercices suivants sont à réaliser dans un (ou plusieurs) notebook(s) Jupyter.

.. slide::
Exercice 1 : Classification multi-classes - Iris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Charger le jeu de données Iris depuis sklearn et affichez le sous la forme d'un DataFrame *pandas*.

Ce jeu de données contient des données de fleurs, et l'objectif est de prédire l'espèce de la fleur à partir d'observations sur ses caractéristiques.

.. code-block:: python
    import sklearn
    import pandas as pd
    iris = sklearn.datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = [iris.target_names[i] for i in iris.target]
    print(df.head())

2) Répondez aux questions suivantes :
  
  - Combien y a-t-il de données ?
  - Combien y a-t-il de features (caractéristiques) initiales ?
  - Combien y a-t-il de classes, et quelles sont-elles ?
  - Combien y a-t-il de données par classe ? Est-ce équilibré ?
  - Quelle doit être la taille de l'entrée et de la sortie d'un réseau de neurones qui devrait classer ces données ?
  - Quelle fonction de coût devez-vous utiliser pour entraîner ce réseau de neurones ?

3) Créez un MLP à 3 couches (entrée, cachée, sortie) pour classer ces données.

4) Créez un jeu de données PyTorch à partir des données Iris et entraînez le MLP par batch de 8 en affichant la *Loss* à chaque fin d'époque.



Utilisez le code suivant pour créer un DataLoader PyTorch. A vous de trouver comment créer les tenseurs *train_X* et *train_y* (lisez la documentation `sklearn.datasets.load_iris() <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html>`__ de la bibliothèque sklearn). 

.. code-block:: python
    
    import torch
    train_loader = torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True
    ) 


5) Évaluez la performance du modèle sur l'ensemble d'entraînement en calculant les métriques suivantes : 

- l'exactitude (accuracy).

**Implémentez-la vous même**, puis comparez votre résultat avec la fonction de sklearn.

6) Affichez la matrice de confusion. Quelles sont les erreurs commises par votre modèle ?

Aidez-vous de `sklearn.metrics.confusion_matrix() <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html>`__ et `matplotlib.pyplot.imshow() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`__ (ou `matshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.matshow.html>`__ ).


.. slide::
Exercice 2 : Classification multi-classes - Breast Cancer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Charger le jeu de données BreastCancer_ depuis sklearn et affichez le sous la forme d'un DataFrame *pandas*.

.. _BreastCancer : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

Ce jeu de données contient les informations de patients, et l'objectif est de savoir si celui-ci est malade ou non.

2) Répondez aux questions suivantes :
  
  - Combien y a-t-il de données ?
  - Combien y a-t-il de features (caractéristiques) initiales ?
  - Combien y a-t-il de classes, et quelles sont-elles ?
  - Combien y a-t-il de données par classe ? Est-ce équilibré ?
  - Quelle doit être la taille de l'entrée et de la sortie d'un réseau de neurones qui devrait classer ces données ?
  - Quelle fonction de coût devez-vous utiliser pour entraîner ce réseau de neurones ?

3) Créez **deux** jeux de données distincts : un pour l'entraînement et un pour la validation du modèle. Utilisez 70% des données pour l'entraînement et 30% pour la validation.

4) Créez un MLP à 3 couches (entrée, cachée, sortie) pour classer ces données et entraînez le par batch de 8. 
A chaque époque de l'entraînement : 

- Évaluez le modèle sur le jeu de validation
- Affichez la *Train loss*, *Validation loss* et *Validation accuracy*
- Sauvegardez le modèle s'il est meilleur que les précédents (quel critère utilisez-vous ?)

⚠️ Il est préférable de ne pas calculer les performances (autres que la loss) sur les données du jeu d'entraînement. Cela peut avoir un coût calculatoire important et n'est pas très utile.

❓Est-ce une bonne idée de calculer la *Validation accuracy* ? Pourquoi ?

5) Rechargez la meilleure version du modèle et calculez les métriques suivantes sur le jeu de validation: 

- L'exactitude (accuracy)
- La précision (precision)
- Le rappel (recall)
- Le score F1 (F1-score)

**Implémentez-les vous même**, puis comparez vos résultats avec les fonctions de sklearn.

6) Affichez la matrice de confusion. Si votre modèle avait été un médecin : 

- Combien de personnes saines auraient été traitées inutilement ? (Rappel : le traitement d'un cancer peut comporter de lourds effets secondaires)
- Combien de personnes malades n'auraient pas été traitées ? (Rappel : un cancer peut être mortel)

7) Modifiez la fonction de coût pour pénaliser plus fortement les erreurs sur la classe "malade". Entraînez à nouveau le modèle.

.. slide::
Exercice 3 : Classification multi-classes - Handwritten Digits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Bonus: Utiliser la bibliothèque wandb pour suivre l'entraînement. L'utilisation de wandb est optionnelle.*

0) Installez et importez la bibliothèque *wandb*, et créez un compte sur https://wandb.ai/.

1) Charger le jeu de données Digits_ depuis sklearn et affichez le sous la forme d'un DataFrame *pandas*.

.. _Digits: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

Ce jeu de données contient des images de chiffres manuscrits, et l'objectif est de prédire quel chiffre est représenté dans chaque image.

2) Répondez aux questions suivantes :

  - Combien y a-t-il de données ?
  - Combien y a-t-il de features (caractéristiques) initiales ?
  - Combien y a-t-il de classes, et quelles sont-elles ?
  - Combien y a-t-il de données par classe ? Est-ce équilibré ?
  - Quelle doit être la taille de l'entrée et de la sortie d'un réseau de neurones qui devrait classer ces données ?
  - Quelle fonction de coût devez-vous utiliser pour entraîner ce réseau de neurones ?

3) Créez deux jeux de données distincts : un pour l'entraînement et un pour la validation du modèle. Utilisez 70% des données pour l'entraînement et 30% pour la validation.

4) Créez un MLP à 5 couches pour classer ces données. Faites en sorte que le réseau ait 2 sorties : une pour les logits, et une pour les caractéristiques en sortie de l'avant dernière couche. 

5) Entraînez le réseau en utilisant la bibliothèque *wandb* pour le suivi des expériences.

⚠️ Ce réseau a 2 sorties, on utilise uniquement les logits pour calculer la fonction de coût.

A chaque époque de l'entraînement :

- Évaluez le modèle sur le jeu de validation
- Affichez la *Train loss*, *Validation loss* et *Validation accuracy*
- Sauvegardez le modèle s'il est meilleur que les précédents (quel critère utilisez-vous ?)

6) Affichez la matrice de confusion. Quels sont les chiffres les plus souvent confondus ?

7) Utilisez `sklearn.manifold.TSNE() <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>`__ pour réduire les données à 2 dimensions. Affichez les points dans un nuage de points 2D en coloriant chaque point selon sa classe.

Analysez le résultat. Cela est-il cohérent avec ce que vous observez dans la matrice de confusion ?

8) Faites varier les hyperparamètres suivants et analysez leur impact sur les performances du modèle :

- Le taux d'apprentissage 
- La taille des batchs 
- Le nombre de couches cachées 
- Le nombre de neurones par couche
- Le nombre d'époques

Paramétez *wandb* pour suivre ces hyperparamètres, et modifier l'interface web pour afficher leur importance sur les performances du modèle.