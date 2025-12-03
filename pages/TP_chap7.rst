🏋️ Travaux Pratiques 7
=========================

Dans les exercices qui suivent, vous allez entraîner un modèle de classification sur un jeu de données d'images de vêtements bien connu : FashionMNIST. Ce jeu de données contient 60 000 images d'entraînement et 10 000 images de test, chacune représentant un vêtement appartenant à l'une des 10 catégories suivantes.

.. warning::
    ⚠️ Ces exercices nécessitent un temps d'entraînement conséquent. 
    Il est important de développer une bonne pratique de base de Data Scientist en IA : lorsqu'un entraînement est en cours sur une version du code V, vous **devez** commencer à programmer la version suivante V+1. 

    Cependant, une fois l'entraînement V terminé, il faut revenir dessus pour l'analyser et potentiellement effectuer des modifications avant de lancer V+1, ou planifier des modifications pour V+2.

    Le développement de modèles d'IA est un processus itératif expérimental qui nécessite de la rigueur et de l'organisation.

Exercice 1 : Chargement du jeu de données
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le jeu de données FashionMNIST est disponible directement via la bibliothèque ``torchvision`` dans ``torchvision.datasets.FashionMNIST`` dont voici la documentation_.

.. _documentation: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html

.. step::
    1) Téléchargez les jeux ``train`` et ``validation`` en appelant 2 fois cette classe avec les bons arguments. Utilisez les transformations nécessaires pour convertir les images en tenseurs de dimensions $$H \times W = 28 \times 28$$.

.. note:: 
    🧠 **Rappel:** Les transformations au chargement de jeux de données ont été vu au Chapitre 5, voir ``torchvision.transforms``.

.. step::
    2) Affichez la liste des classes du jeu de données. Quel est le format des labels (entiers, one-hot, etc.) ?

.. step::
    3) Pour chaque classe, affichez une image. Quelles sont les dimensions $$H \times W \times C$$ de chaque image ? 


Exercice 2 : Définition d'une architecture de réseau de neurones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. step::
    1) Créez une architecture de réseau de neurone correspondant à LeNet-5:

- C1: Convolution 6 filtres 5x5, stride 1, padding 0, activation ReLU
- Pooling 2x2, kernel 2, stride 2
- C2: Convolution 16 filtres 5x5, stride 1, padding 0, activation ReLU
- Pooling 2x2, kernel 2, stride 2
- Flatten
- F3: Fully connected 120 neurones, activation ReLU
- F4: Fully connected 84 neurones, activation ReLU
- F5: Fully connected 10 neurones (une par classe)

.. warning::
    ⚠️ Quelle activation devez-vous mettre en sortie de réseau ? Pourquoi ?

.. step::
    2) Quel est le nombre de paramètres entraînable dans cette architecture ? (Indice : utilisez ``model.parameters()`` et ``numel()``)

Exercice 3 : Lancer l'entraînement 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. step::
    1) Chargez vos jeux de données en batchs avec ``torch.utils.data.DataLoader``.

.. step::
    2) Initialisez votre fonction de coût ``torch.nn.CrossEntropyLoss()`` et votre optimiseur ``torch.optim.SGD()`` avec $$momentum=0.9$$ et $$learning\_rate=0.1$$).

.. step::
    3) Implémentez une boucle d'entraînement classique avec :

- une phase d'entraînement
- une phase de validation
- affichage de la loss et de l'accuracy à chaque époque pour les deux phases.
- affichage de la courbe de loss et val_loss en fonction des époques.

.. step::
    4) Lancez l'entraînement pour 100 époques et notez les résultats.

.. warning::

    Si l'entraînement est trop long, utilisez un sous-ensemble du jeu de données.
    
    .. spoiler::
        .. discoverList::
            .. code-block:: python
                
                trainset_target_size = 10000
                valset_target_size = 2000
                
                trainset = datasets.FashionMNIST(...)
                trainset.data = trainset.data[:trainset_target_size]
                trainset.targets = trainset.targets[:trainset_target_size]

                valset = datasets.FashionMNIST(...)
                valset.data = valset.data[:valset_target_size]
                valset.targets = valset.targets[:valset_target_size]

Exercice 4 : Early Stopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. step::
    1) Implémentez une stratégie d'Early Stopping qui arrête l'entraînement si la validation loss ne s'est pas améliorée d'au moins 0.005 pendant 5 époques consécutives.

.. step::
    2) Comparez le nombre d'époques effectuées avec et sans Early Stopping. Quel est l'impact sur les performances du modèle ?

Exercice 5 : Régularisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour cet exercice, vous pouvez afficher les valeurs des paramètres de votre modèle (ou quelques valeurs descriptives telles que min, max, mean et std).
.. spoiler::
    .. discoverList::
        .. code-block:: python
            model = YourModel(...)

            param_stats = {
                "min": float("inf"),
                "max": float("-inf"),
                "mean": 0,
                "std": 0
            }
            all_params = torch.cat([p.view(-1) for p in model.parameters()])
            param_stats["min"] = all_params.min().item()
            param_stats["max"] = all_params.max().item()
            param_stats["mean"] = all_params.mean().item()
            param_stats["std"] = all_params.std().item()
            print(f"Parameter stats - Min: {param_stats['min']:.5f}, Max: {param_stats['max']:.5f}, Mean: {param_stats['mean']:.5f}, Std: {param_stats['std']:.5f}")


L2 Regularization
^^^^^^^^^^^^^^^^^^

.. step::
    1) Ajoutez une régularisation L2 (weight decay) à votre optimiseur avec un coefficient de 0.01.

.. step::
    2) Entraînez à nouveau le modèle avec cette régularisation et comparez les performances obtenues avec celles sans régularisation.

L1 Regularization
^^^^^^^^^^^^^^^^^^

.. step::
    3) Enlevez la régularisation L2 et ajoutez une régularisation L1 à votre fonction de coût avec un coefficient de 0.001.

.. step::
    4) Entraînez à nouveau le modèle avec cette régularisation et comparez les performances obtenues avec celles sans régularisation.

Dropout
^^^^^^^^

.. step::
    3) Enlevez la régularisation L1 et ajoutez une couche de Dropout à votre réseau de neurones avec un taux de 0.5 entre les convolutions (C1 et C2) et entre les deux premières couches fully connected (F3 et F4).

.. step::
    4) Entraînez à nouveau le modèle avec cette régularisation et comparez les performances obtenues avec celles sans régularisation.

Exercice 6 : Learning Rate Scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le learning rate (taux d'apprentissage) fixé à l'exercice 3 est peut être trop élevé... 0.001 serait sans doute trop faible... 0.01 serait peut être un bon compromis ?

En réalité, nous n'en savons rien et il vaut mieux éviter de faire des suppositions. L'idéal sera en fait de commencer par une valeur suffisamment élevée, puis corriger (diminuer) cette valeur au fur et à mesure de l'entraînement.

.. step::
    1) Implémentez le scheduler ``torch.optim.lr_scheduler.ReduceLROnPlateau`` dans votre boucle d'entraînement.

.. step::
    2) Relancez un entraînement en affichant la valeur du learning rate à chaque époque. Constatez l'évolution de la loss en fonction des "paliers" du learning rate.

.. warning::
    ⚠️ Attention au réglage de la "patience" de la classe ``ReduceLROnPlateau`` qui ne doit pas être en conflit avec celle de l'Early Stopping ! Voyez-vous pourquoi ?


Exercice 7 : Normalization 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. step::
    1) Ajoutez une couche de ``torch.nn.BatchNorm??`` entre les deux couches de convolution. Attention à choisir la bonne version de la couche (nombre de dimenions) !
    
.. step::
    2) Affichez les valeurs min, max, mean et std des caractéristiques (features) des données avant et après la normalisation, puis relancez l'entraînement. Que constatez-vous ?

Exercice 8 : Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jusqu'ici, nous avons utilisé plusieurs techniques qui, théoriquement, améliorent les capacités du modèle (sa performance, sa robustesse, ou son coût d'entraînement). Cependant, chaque technique nécessite un ou plusieurs choix d'algorithme (quelle normalisation ? quel scheduler ? etc.) et de paramètres (taux de dropout, coefficient de régularisation, etc.).

Vous avez sans doute pu constater qu'il est difficile de savoir quelle technique a un réel impact positif sur les performances du modèle pendant l'apprentissage et sur sa capacité à généraliser.

.. step::
    1) Cet exercice est libre et consiste à tester différentes combinaisons de techniques et d'hyperparamètres pour améliorer les performances de votre modèle sur le jeu de validation. Que vous cherchiez manuellement ou que vous utilisiez une méthode automatique, essayez désormais de maximiser les performances de votre réseau !

.. note::
    🧠 **Rappel:**

    1. Définir un espace d'hyperparamètres à explorer 
    2. Définir une fonction objectif (par exemple, maximiser l'accuracy sur le jeu de validation)
    3. Pour chaque combinaison d'hyperparamètres testée, entraîner un modèle et évaluer ses performances sur le jeu de validation
    4. Garder en mémoire la meilleure combinaison d'hyperparamètres trouvée

    **Conseil: ** Lorsque le nombre de possibilités est trop important, vous pouvez commencer par un définir un espace de recherche avec uniquement les hyperparamètres les plus influents. Cette première recherche permet donc de trouver une valeur optimale que vous pouvez fixer pour ceux-ci. Ensuite, une deuxième phase de recherche peut être lancée en incluant les autres hyperparamètres. 
