🏋️ Travaux Pratiques 3
=========================

.. slide::
Exercice 0 : Mise en place
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Créer un notebook Jupyter et importer les bibliothèques nécessaires. Assurez-vous que celles-ci soient disponibles dans votre noyau jupyter.

- numpy
- matplotlib
- sklearn
- pandas
- torch

Les exercices suivants sont à réaliser dans un (ou plusieurs) notebook(s) Jupyter.

.. slide::
Exercice 1 : Classification multi-classes - Iris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Charger le jeu de données Iris depuis sklearn et affichez le sous la forme d'un DataFrame *pandas*.

2) Répondez aux questions suivantes :
  
  - Combien y a-t-il de données ?
  - Combien y a-t-il de features (caractéristiques) ?
  - Combien y a-t-il de classes, et quelles sont-elles ?
  - Combien y a-t-il de données par classe ? Est-ce équilibré ?
  - Quelle doit être la taille de l'entrée et de la sortie d'un réseau de neurones qui devrait classer ces données ?
  - Quelle fonction de coût devez-vous utiliser pour entraîner ce réseau de neurones ?

3) Créez un MLP à 3 couches (entrée, cachée, sortie) pour classer ces données.

4) Créez un jeu de données PyTorch à partir des données Iris et entraînez le MLP par batch de 8 en affichant la *Loss* à chaque fin d'époque.

5) Évaluez la performance du modèle sur l'ensemble d'entraînement en calculant les métriques suivantes : 

- l'exactitude (accuracy).

**Implémentez-la vous même**, puis comparez votre résultat avec la fonction de sklearn.

6) Affichez la matrice de confusion. Quelles sont les erreurs commises par votre modèle ?

Aidez-vous de *sklearn.metrics.confusion_matrix* et *matplotlib.pyplot.imshow* (ou *matshow*).

.. slide::
Exercice 2 : Classification multi-classes - Breast Cancer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Charger le jeu de données Breast Cancer depuis sklearn et affichez le sous la forme d'un DataFrame *pandas*.

2) Répondez aux questions suivantes :
  
  - Combien y a-t-il de données ?
  - Combien y a-t-il de features (caractéristiques) ?
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

1) Charger le jeu de données Digits depuis sklearn et affichez le sous la forme d'un DataFrame *pandas*.

2) Répondez aux questions suivantes :

  - Combien y a-t-il de données ?
  - Combien y a-t-il de features (caractéristiques) ?
  - Combien y a-t-il de classes, et quelles sont-elles ?
  - Combien y a-t-il de données par classe ? Est-ce équilibré ?
  - Quelle doit être la taille de l'entrée et de la sortie d'un réseau de neurones qui devrait classer ces données ?
  - Quelle fonction de coût devez-vous utiliser pour entraîner ce réseau de neurones ?

3) Créez deux jeux de données distincts : un pour l'entraînement et un pour la validation du modèle. Utilisez 70% des données pour l'entraînement et 30% pour la validation.

4) Créez un MLP à 5 couches pour classer ces données. Faites en sorte que le réseau ait 2 sorties : une pour les logits, et une pour les caractéristiques en sortie de l'avant dernière couche (features embedding). 

5) Entraînez le réseau. 

⚠️ Ce réseau a 2 sorties, on utilise uniquement les logits pour calculer la fonction de coût.

A chaque époque de l'entraînement :

- Évaluez le modèle sur le jeu de validation
- Affichez la *Train loss*, *Validation loss* et *Validation accuracy*
- Sauvegardez le modèle s'il est meilleur que les précédents (quel critère utilisez-vous ?)

6) Affichez la matrice de confusion. Quels sont les chiffres les plus souvent confondus ?

7) Utilisez *sklearn.manifold.TSNE* pour réduire les dimensions des features embeddings à 2D. Affichez les points dans un nuage de points 2D en coloriant chaque point selon sa classe.

Analysez le résultat. Cela est-il cohérent avec ce que vous observez dans la matrice de confusion ?