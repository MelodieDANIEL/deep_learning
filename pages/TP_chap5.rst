🏋️ Travaux Pratiques 5
=========================
.. slide::
Sur cette page se trouvent des exercices de TP sur le Chapitre 5. Ils sont classés par niveau de difficulté suivant :
.. discoverList::
    * Facile : 🍀
    * Moyen : ⚖️
    * Difficile : 🌶️

.. slide::
🍀 Exercice 1 : Premier CNN pour la classification de chiffres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez créer un réseau de neurones convolutif (CNN) pour classifier des chiffres manuscrits du dataset MNIST.

**Objectif :** Comprendre la différence entre un MLP classique et un CNN sur des images.

On va travailler avec les données MNIST :

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Charger MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                  download=True, transform=transform)

.. note::
    Pour simplifier cet exercice d'introduction, nous utilisons uniquement les datasets train et test, mais en pratique il faudrait aussi un dataset de validation pour surveiller l'overfitting pendant l'entraînement.

**Consigne :** Écrire un programme qui :

.. step:: 
    1) Crée deux modèles différents :
    
    - **MLP classique** : aplatit l'image ($$28×28$$ → 784), puis 2 couches fully-connected de 128 neurones avec ReLU, sortie 10 classes
    - **CNN simple** : 
        
        - 1 couche convolutive (1→16 filtres, kernel $$3×3$$, padding=1)
        - ReLU + MaxPooling $$2×2$$
        - 1 couche convolutive (16→32 filtres, kernel $$3×3$$, padding=1)
        - ReLU + MaxPooling $$2×2$$
        - Aplatir puis 1 couche fully-connected vers 10 classes

.. step:: 
    2) Entraîne les deux modèles pendant 5 epochs avec :
    
    - Batch size de 64
    - Optimiseur Adam avec learning rate 0.001
    - Loss CrossEntropyLoss

.. step:: 
    3) Compare le nombre de paramètres de chaque modèle (utiliser ``sum(p.numel() for p in model.parameters())``)

.. step::
    4) Évalue la précision (accuracy) sur le test set pour les deux modèles

.. step::
    5) Affiche quelques exemples de prédictions (correctes et incorrectes) pour chaque modèle


**Questions :**

.. step::
    6) Quel modèle a le moins de paramètres ?

.. step::
    7) Quel modèle obtient la meilleure accuracy ?

.. step::
    8) Pourquoi le CNN est-il plus efficace malgré moins de paramètres ?


**Astuce :**
.. spoiler::
    .. discoverList::
        1. Pour le MLP, utiliser ``x.view(x.size(0), -1)`` pour aplatir l'image
        2. Pour calculer la taille après convolution et pooling : avec 2 pooling $$2×2$$, une image $$28×28$$ devient $$7×7$$  
        3. Pour l'accuracy : ``(predicted == labels).sum().item() / len(labels)``
        4. Utiliser ``torch.no_grad()`` lors de l'évaluation pour économiser la mémoire
        5. Le CNN préserve la structure spatiale de l'image, ce qui est crucial pour la reconnaissance

**Astuce avancée :**        
.. spoiler::
    .. discoverList:: 
        **Voici le code pour visualiser les prédictions correctes et incorrectes :**
        
        .. code-block:: python
        
            import matplotlib.pyplot as plt
            
            # Fonction pour afficher des exemples de prédictions
            def visualize_predictions(images, labels, predictions, model_name, num_examples=5, correct=True):
                """
                Affiche des exemples de prédictions correctes ou incorrectes
                
                Args:
                    images: liste d'images
                    labels: vrais labels
                    predictions: prédictions du modèle
                    model_name: nom du modèle (pour le titre)
                    num_examples: nombre d'exemples à afficher
                    correct: True pour afficher les prédictions correctes, False pour les incorrectes
                """
                # Trouver les indices selon le critère
                if correct:
                    indices = [i for i in range(len(predictions)) 
                              if predictions[i] == labels[i]]
                    title_color = 'green'
                    main_title = f'{model_name} - Prédictions CORRECTES'
                else:
                    indices = [i for i in range(len(predictions)) 
                              if predictions[i] != labels[i]]
                    title_color = 'red'
                    main_title = f'{model_name} - Prédictions INCORRECTES'
                
                # Créer la figure
                fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
                fig.suptitle(main_title, fontsize=14, fontweight='bold')
                
                # Afficher les exemples
                for i in range(min(num_examples, len(indices))):
                    idx = indices[i]
                    axes[i].imshow(images[idx].squeeze(), cmap='gray')
                    axes[i].set_title(f'Vrai: {labels[idx]}\\nPrédit: {predictions[idx]}', 
                                    color=title_color, fontweight='bold')
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.show()
            
            # Utilisation après évaluation :
            # visualize_predictions(mlp_images, mlp_labels, mlp_preds, "MLP", correct=True)
            # visualize_predictions(mlp_images, mlp_labels, mlp_preds, "MLP", correct=False)


**Résultat attendu :** 

- MLP : environ 100k paramètres, accuracy ~97%
- CNN : environ 20k paramètres, accuracy ~98-99%


.. slide::
⚖️ Exercice 2 : Comprendre l'effet du padding et du stride
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez explorer comment les paramètres ``padding`` et ``stride`` affectent la taille des feature maps dans un CNN.

**Objectif :**  
Comprendre l'impact du padding et du stride sur les dimensions spatiales et visualiser les feature maps.

On va créer un mini-dataset synthétique avec des formes géométriques :

.. code-block:: python

    import torch
    import matplotlib.pyplot as plt
    
    # Créer des images synthétiques avec des formes
    def create_shape_image(shape_type='square'):
        img = torch.zeros(1, 1, 28, 28)
        if shape_type == 'square':
            img[0, 0, 10:18, 10:18] = 1.0
        elif shape_type == 'cross':
            img[0, 0, 14, :] = 1.0
            img[0, 0, :, 14] = 1.0
        elif shape_type == 'diagonal':
            for i in range(28):
                img[0, 0, i, i] = 1.0
        return img


**Consigne :** Écrire un programme qui :

.. step::
    1) Crée 3 images : un carré, une croix, une diagonale

.. step::
    2) Définit 4 configurations de convolution différentes :
    
    - Config A : kernel=3, stride=1, padding=0
    - Config B : kernel=3, stride=1, padding=1
    - Config C : kernel=3, stride=2, padding=0
    - Config D : kernel=5, stride=1, padding=2

.. step::
    3) Pour chaque configuration :
    
    - Applique la convolution avec 8 filtres sur une des images
    - Calcule et affiche la taille de sortie
    - Vérifie avec la formule : $$H_{out} = \lfloor \frac{H_{in} + 2 \times padding - kernel\_size}{stride} \rfloor + 1$$

.. step::
    4) Visualise les 8 feature maps obtenues pour chaque configuration

.. step::
    5) Applique ensuite un MaxPooling $$2×2$$ après la convolution et observe la nouvelle taille


**Questions :**

.. step::
    6) Quelle configuration préserve la taille spatiale de l'image ?

.. step::
    7) Quelle configuration réduit le plus la taille ?

.. step::
    8) Que se passe-t-il si on applique plusieurs convolutions successives sans padding ?

.. step::
    9) Pourquoi utilise-t-on souvent padding=1 avec kernel=3 ?


**Astuce :**
.. spoiler::
    .. discoverList::
        1. Utiliser ``nn.Conv2d(1, 8, kernel_size=k, stride=s, padding=p)``
        2. Pour visualiser : ``plt.imshow(feature_map[0, i].detach(), cmap='viridis')``
        3. Config B (padding=1, stride=1, kernel=3) préserve la taille : $$28×28$$ → $$28×28$$
        4. Sans padding, chaque convolution réduit la taille de (kernel_size - 1)
        5. padding=1 avec kernel=3 est un choix standard car il préserve la taille


**Résultat attendu :**

Les tailles de sortie attendues pour une image $$28×28$$ :

- Config A (k=3, s=1, p=0) : $$26×26$$
- Config B (k=3, s=1, p=1) : $$28×28$$
- Config C (k=3, s=2, p=0) : $$13×13$$
- Config D (k=5, s=1, p=2) : $$28×28$$

Après MaxPooling $$2×2$$, les tailles sont divisées par 2.


.. slide::
🌶️ Exercice 3 : CNN et Data Augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cet exercice vous guide d'un CNN et l'utilisation de data augmentation pour améliorer les performances.

**Objectif :**

    - Créer une architecture CNN profonde avec blocs convolutifs
    - Implémenter et comparer l'entraînement avec/sans data augmentation
    - Gérer les datasets avec train/validation/test splits
    - Sauvegarder le meilleur modèle basé sur la validation

**Consigne :** Écrire un programme qui :

.. step:: 
    1) Charge le dataset CIFAR-10 avec deux types de transformations :
    
    - **Sans augmentation** : seulement ToTensor et Normalize
    - **Avec augmentation** : RandomHorizontalFlip, RandomCrop(32, padding=4), ToTensor, Normalize

.. step::
    2) Divise le training set en train (80%) et validation (20%) avec ``random_split``

.. step::
    3) Crée un CNN avec l'architecture suivante :
    
    .. code-block:: python
    
        # Bloc 1
        Conv2d(3, 64, kernel=3, padding=1) + ReLU
        Conv2d(64, 64, kernel=3, padding=1) + ReLU
        MaxPool2d(2, 2)  # 32×32 → 16×16
        
        # Bloc 2
        Conv2d(64, 128, kernel=3, padding=1) + ReLU
        Conv2d(128, 128, kernel=3, padding=1) + ReLU
        MaxPool2d(2, 2)  # 16×16 → 8×8
        
        # Bloc 3
        Conv2d(128, 256, kernel=3, padding=1) + ReLU
        Conv2d(256, 256, kernel=3, padding=1) + ReLU
        MaxPool2d(2, 2)  # 8×8 → 4×4
        
        # Classification
        Flatten
        Linear(256 * 4 * 4, 512) + ReLU
        Dropout(0.5)
        Linear(512, 10)

.. step::
    4) Entraîne deux versions du modèle (10 epochs chacune) :
    
    - Modèle A : sans data augmentation
    - Modèle B : avec data augmentation

.. step::
    5) Pour chaque epoch, calcule et stocke :
    
    - Train loss et train accuracy
    - Validation loss et validation accuracy

.. step::
    6) Implémente un système de sauvegarde qui garde le meilleur modèle basé sur la validation accuracy

.. step::
    7) Trace 4 courbes sur un même graphique :
    
    - Train et validation loss (un subplot)
    - Train et validation accuracy (un autre subplot)
    - Faire cela pour les deux modèles A et B

.. step::
    8) Évalue le meilleur modèle (A et B) sur le test set et affiche :
    
    - Test accuracy finale
    - Matrice de confusion
    - Classification report


**Questions :**

.. step::
    9) Quel modèle (A ou B) généralise mieux ? Comment le voyez-vous sur les courbes ?

.. step::
    10) Observe-t-on de l'overfitting ? Sur quel modèle et comment ?

.. step::
    11) Comment la data augmentation aide-t-elle à réduire l'overfitting ?

.. step::
    12) Quelle est la différence de performance sur le test set ?


**Astuce :**
.. spoiler::
    .. discoverList::
    1. Pour CIFAR-10 : ``transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])``
    2. Pour la matrice de confusion : ``from sklearn.metrics import confusion_matrix, classification_report``
    3. Utiliser ``model.train()`` avant l'entraînement et ``model.eval()`` avant l'évaluation
    4. Sauvegarder avec : ``torch.save({'model_state_dict': model.state_dict(), 'accuracy': best_acc}, 'best_model.pth')``
    5. Le dropout aide aussi à éviter l'overfitting en désactivant aléatoirement 50% des neurones pendant l'entraînement


**Résultats attendus :**

- **Modèle A** (sans augmentation) : test accuracy ~76-77%, **mais fort overfitting** (gap train/val de ~12-13%)
- **Modèle B** (avec augmentation) : test accuracy ~76-77%, **excellente généralisation** (gap train/val de seulement ~2%)
- **Observation importante** : Avec seulement 10 epochs, les deux modèles peuvent avoir des test accuracy similaires, **mais le Modèle B généralise beaucoup mieux**
- Le Modèle A montre des signes clairs d'overfitting après l'epoch 5 (train accuracy continue à monter, val accuracy stagne)
- Le Modèle B a une meilleure loss sur le test set (~0.70 vs ~0.79), indiquant des prédictions plus confiantes
- **Avec plus d'epochs (20-30)**, le Modèle B dépasserait clairement A en test accuracy
- **Leçon importante** : L'accuracy seule ne suffit pas ! Analysez toujours le gap train/val pour évaluer la vraie qualité du modèle


.. slide::
🌶️ Exercice 4 : Créer son propre Dataset d'images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez créer un ``Dataset`` PyTorch qui charge, prétraite et étiquette automatiquement des images.

.. note::

   Cet exercice utilise les images de l'exercice 6 du chapitre 4 : `elephants.png <images/tp4/elephants.png>`_, `chien.jpg <images/tp4/chien.jpg>`_, `chat.jpg <images/tp4/chat.jpg>`_ et `cheval.jpg <images/tp4/cheval.jpg>`_. Placez-les dans le même dossier que votre notebook Jupyter.

**Objectif :** Préparer des images pour l'entraînement d'un réseau de neurones avec un ``Dataset`` et un ``DataLoader``.

**Consigne :** Écrire un programme qui :

.. step::
    1) Crée une classe ``MyDataset`` qui hérite de ``torch.utils.data.Dataset`` et qui prend en paramètres une liste de chemins d'images, une liste de labels et une transformation.

.. step::
    2) Prétraite automatiquement chaque image de la manière suivante :

    - redimensionnement à 64 × 64 pixels,
    - lissage avec un flou gaussien,
    - conversion en tenseur,
    - normalisation des valeurs de chaque canal entre −0.5 et 0.5.

.. step::
    3) Associe un label (aussi appelé étiquette ou vérité terrain) à chaque image : 0 pour l'éléphant, 1 pour le chien, 2 pour le chat et 3 pour le cheval.

.. step::
    4) Crée un ``DataLoader`` avec des batchs de 2 images mélangées, le parcourt en affichant la forme des images et les labels de chaque batch, puis affiche les images du dernier batch.

.. step::
    5) Crée une seconde version du dataset pour l'entraînement, avec de l'augmentation de données (miroir horizontal et rotation aléatoires), et vérifie que deux appels à ``dataset[0]`` renvoient des images différentes.

.. warning::

   ⚠️ Votre classe doit bien **hériter** de ``torch.utils.data.Dataset``, et il est impératif d'implémenter les méthodes ``__len__()`` et ``__getitem__()``.


**Questions :**

.. step::
    6) Quelles valeurs de ``mean`` et de ``std`` faut-il donner à ``transforms.Normalize`` pour obtenir des valeurs entre −0.5 et 0.5 ?

.. step::
    7) Pourquoi faut-il appliquer les transformations dans ``__getitem__``, et pas une seule fois au chargement des images ?

.. step::
    8) Pourquoi ne faut-il pas utiliser d'augmentation de données pour la validation et le test ?


**Astuce :**
.. spoiler::
    .. discoverList::
        1. Revoyez la classe ``ImageFolderDataset`` de la section 4.1 du cours, et ``transforms.Compose`` dans la section 5.4 du chapitre 4
        2. ``transforms.ToTensor()`` donne des valeurs dans [0, 1], et ``transforms.Normalize`` calcule ``(x - mean) / std``
        3. Pour afficher une image normalisée, il faut d'abord annuler la normalisation (section 5.3 du chapitre 4)
        4. ``torch.equal(a, b)`` vérifie si deux tenseurs sont identiques


**Résultat attendu :**

- ``len(dataset)`` vaut 4
- ``dataset[0]`` renvoie une image de forme ``torch.Size([3, 64, 64])``, avec des valeurs entre −0.5 et 0.5, et le label 0
- Chaque batch du ``DataLoader`` contient des images de forme ``torch.Size([2, 3, 64, 64])`` et 2 labels
- Avec l'augmentation, ``torch.equal(dataset_train[0][0], dataset_train[0][0])`` renvoie le plus souvent ``False``


.. slide::
🏋️ Exercices supplémentaires 5
===============================
Dans cette section, il y a des exercices supplémentaires pour vous entraîner. Ils suivent le même classement de difficulté que précédemment.


.. slide::
⚖️ Exercice supplémentaire 1 : Visualisation des filtres appris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cet exercice propose de visualiser ce que les filtres convolutifs ont appris après l'entraînement.

**Objectif :** Comprendre ce que les filtres convolutifs détectent dans les premières couches d'un CNN.

**Consignes** :

.. step::
    1) Entraîner un CNN simple sur MNIST pendant 3 epochs :

    .. code-block:: python
    
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.fc = nn.Linear(32 * 7 * 7, 10)
            
            def forward(self, x):
                x = F.max_pool2d(F.relu(self.conv1(x)), 2)
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

.. step::
    2) Après l'entraînement, extraire les poids de la première couche convolutive :

    .. code-block:: python
    
        filters = model.conv1.weight.data  # shape: [16, 1, 3, 3]

.. step::
    3) Visualiser les 16 filtres $$3×3$$ de la première couche sur une grille $$4×4$$

.. step::
    4) Prendre une image de test et visualiser les feature maps produites par la première couche convolutive :
    
    - Appliquer ``model.conv1(image)`` puis ``F.relu()``
    - Afficher les 16 feature maps obtenues

.. step::
    5) Faire de même pour la deuxième couche convolutive (afficher 32 feature maps sur une grille 4×8)


**Questions :**

.. step::
6) Que détectent les filtres de la première couche ? (contours, textures, ...)

.. step::
    7) Les feature maps de la deuxième couche sont-elles plus abstraites que celles de la première ?

.. step::
    8) Comment évoluent les patterns détectés entre les couches ?


**Astuce :**
.. spoiler::
    .. discoverList::
        1. Pour visualiser : ``plt.imshow(filters[i, 0].cpu(), cmap='gray')``
        2. Pour obtenir les feature maps : ``with torch.no_grad(): features = F.relu(model.conv1(image))``
        3. Les premiers filtres détectent souvent des contours horizontaux, verticaux, diagonaux
        4. Les couches profondes détectent des motifs plus complexes et abstraits
        5. Utiliser ``plt.subplots()`` pour créer une grille de visualisation


**Résultats attendus :**

- Une grille montrant les 16 filtres $$3×3$$ de la première couche
- Une grille montrant les 16 feature maps activées par une image
- Les filtres de la première couche devraient ressembler à des détecteurs de contours
- Les feature maps montrent quelles parties de l'image activent chaque filtre


.. slide::
🌶️ Exercice supplémentaire 2 : Transfer Learning avec un modèle pré-entraîné
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez utiliser un modèle pré-entraîné (ResNet18) et le fine-tuner sur un nouveau dataset.

.. warning::
    ⏰ **Attention : Temps d'entraînement très long !**
    
    Cet exercice nécessite d'entraîner 2 modèles ResNet18 sur des images $$224×224$$, ce qui prend **beaucoup de temps** (plusieurs heures sans GPU, et reste long même avec GPU). **Il n'est PAS possible de terminer l'entraînement pendant la séance de TP**. Il est recommandé de :
    
    - 🏠 Lancer l'entraînement à la maison ou utiliser Google Colab avec GPU
    - ⚡ Réduire drastiquement à 3-5 epochs pour tester rapidement (résultats moins concluants)
    - 💾 Sauvegarder régulièrement les modèles pour reprendre plus tard

.. note::
    **ResNet18** est une architecture CNN de 18 couches qui a gagné le concours ImageNet en 2015. **ImageNet** est une immense base de 1.2 million d'images réparties en 1000 classes (animaux, véhicules, objets du quotidien). Le **transfer learning** consiste à réutiliser les filtres appris sur ImageNet (qui détectent contours, textures, formes génériques) pour classifier CIFAR-10 (10 classes seulement) : au lieu de tout réapprendre depuis zéro, on adapte juste la dernière couche !

**Objectif** :

- Comprendre le concept de transfer learning
- Charger un modèle pré-entraîné et modifier sa dernière couche
- Comparer l'entraînement from scratch vs transfer learning

**Consignes** :

.. step::
    1) Charger le dataset CIFAR-10 avec des transformations appropriées :

    .. code-block:: python
    
        from torchvision import models
        
        transform = transforms.Compose([
            transforms.Resize(224),  # ResNet attend du 224×224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])

.. step::
    2) Créer deux modèles :
    
    - **Modèle A (from scratch)** : ResNet18 initialisé aléatoirement
    - **Modèle B (transfer learning)** : ResNet18 pré-entraîné sur ImageNet, on gèle toutes les couches sauf la dernière

    .. code-block:: python
    
        # Modèle A
        model_scratch = models.resnet18(pretrained=False)
        model_scratch.fc = nn.Linear(model_scratch.fc.in_features, 10)
        
        # Modèle B
        model_transfer = models.resnet18(pretrained=True)
        # Geler toutes les couches
        for param in model_transfer.parameters():
            param.requires_grad = False
        # Remplacer la dernière couche et la dégeler
        model_transfer.fc = nn.Linear(model_transfer.fc.in_features, 10)

.. step::
    3) Entraîner les deux modèles pendant 10 epochs avec :
    
    - Batch size 64
    - Adam optimizer, learning rate 0.001
    - Diviser train en train (80%) et validation (20%)

.. step::
    4) Pour chaque modèle, tracer :
    
    - Evolution de la train loss et validation loss
    - Evolution de la train accuracy et validation accuracy

.. step::           
    5) Comparer le temps d'entraînement par epoch pour les deux modèles

.. step::
    6) Évaluer les deux modèles sur le test set

.. step::
    7) Afficher une matrice de confusion pour chaque modèle


**Questions :**

.. step::
    8) Quel modèle converge le plus rapidement ?

.. step::
    9) Quel modèle atteint la meilleure accuracy finale ?

.. step::
    10) Pourquoi le transfer learning est-il plus efficace ?

.. step::
    11) Que se passerait-il si on dégelait aussi les couches intermédiaires ?


**Astuce :**
.. spoiler::
    .. discoverList:: 
        1. Pour mesurer le temps : ``import time; start = time.time(); ... ; elapsed = time.time() - start``
        2. Le modèle pré-entraîné a déjà appris des features génériques sur ImageNet
        3. En gelant les couches, on a moins de paramètres à entraîner → plus rapide
        4. Le transfer learning devrait donner ~80-85% accuracy en quelques epochs
        5. From scratch atteindra peut-être ~70-75% après 10 epochs
        6. Si on dégèle tout, on risque de détruire les features pré-apprises (sauf si learning rate très faible)


**Résultats attendus :**

- Modèle from scratch : convergence lente, accuracy finale ~70-75%
- Modèle transfer learning : convergence rapide (2-3 epochs), accuracy finale ~80-85%
- Temps par epoch : similaire pour les deux, mais transfer learning nécessite moins d'epochs
- Le gap train/validation devrait être plus petit pour le transfer learning


.. slide::
🌶️ Exercice supplémentaire 3 : Early Stopping et Learning Rate Scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez implémenter un système complet d'entraînement avec early stopping et ajustement dynamique du learning rate.

.. warning::
    ⏰ **Attention : Temps d'entraînement long !**
    
    Cet exercice nécessite d'entraîner **3 modèles différents** sur CIFAR-10, ce qui prend **un temps conséquent** (le temps varie beaucoup selon votre GPU). **Il n'est pas possible de terminer l'entraînement pendant la séance de TP**. Options recommandées :
    
    - 🏠 Lancer l'entraînement à la maison ou utiliser Google Colab avec GPU
    - ⚡ Réduire à 10-15 epochs max pour tester rapidement (résultats moins concluants)
    - 💾 Sauvegarder régulièrement les modèles pour reprendre plus tard

**Objectif** :

- Implémenter un early stopping robuste pour éviter l'overfitting
- Utiliser un learning rate scheduler pour améliorer la convergence
- Comparer différentes stratégies de training

**Consignes** :

.. step::
    1) Utiliser CIFAR-10 avec data augmentation et diviser en train/val/test (70%/15%/15%)

.. step::
    2) Créer un CNN de taille moyenne :

    .. code-block:: python
    
        class MediumCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(128 * 8 * 8, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

.. step::
    3) Implémenter une classe ``EarlyStopping`` avec les paramètres :
    
    - ``patience`` : nombre d'epochs sans amélioration avant d'arrêter
    - ``min_delta`` : amélioration minimale pour considérer un progrès
    - Sauvegarder le meilleur modèle automatiquement

.. step::
    4) Entraîner 3 versions du modèle (max 50 epochs) :
    
    - **Modèle A** : learning rate constant 0.001, pas d'early stopping
    - **Modèle B** : learning rate constant 0.001, early stopping (patience=5)
    - **Modèle C** : learning rate avec ``ReduceLROnPlateau`` + early stopping (patience=7)

.. step::
    5) Pour chaque modèle, tracer sur un même graphique :
    
    - Train et validation loss
    - Train et validation accuracy
    - Marquer l'epoch où l'entraînement s'arrête (si early stopping)
    - Marquer les epochs où le learning rate change (modèle C)

.. step::
    6) Comparer les résultats finaux sur le test set

.. step::
    7) Afficher pour chaque modèle :
    
    - Nombre total d'epochs entraînées
    - Meilleure validation accuracy
    - Test accuracy finale
    - Temps d'entraînement total


**Questions :**

.. step::
    8) Quel modèle évite le mieux l'overfitting ?

.. step::
    9) Quel est l'impact du learning rate scheduler ?

.. step::
    10) L'early stopping permet-il de gagner du temps d'entraînement ?

.. step::
    11) Quelle stratégie recommanderiez-vous pour un nouveau projet ?


**Astuce :**
.. spoiler::
    .. discoverList:: 
        **Implémentation de la classe EarlyStopping :**
        
        .. code-block:: python
        
            class EarlyStopping:
                def __init__(self, patience=5, min_delta=0, path='checkpoint.pth'):
                    self.patience = patience
                    self.min_delta = min_delta
                    self.path = path
                    self.counter = 0
                    self.best_loss = None
                    self.early_stop = False
                
                def __call__(self, val_loss, model):
                    if self.best_loss is None:
                        self.best_loss = val_loss
                        self.save_checkpoint(model)
                    elif val_loss > self.best_loss - self.min_delta:
                        self.counter += 1
                        if self.counter >= self.patience:
                            self.early_stop = True
                    else:
                        self.best_loss = val_loss
                        self.save_checkpoint(model)
                        self.counter = 0
                
                def save_checkpoint(self, model):
                    torch.save(model.state_dict(), self.path)
        
        **Pour le scheduler :**
        
        .. code-block:: python
        
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            
            scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                         factor=0.5, patience=3, verbose=True)
            
            # Dans la boucle d'entraînement, après validation :
            scheduler.step(val_loss)


**Résultats attendus :**

- Modèle A : overfitting après ~20 epochs, test accuracy ~75%
- Modèle B : s'arrête à ~15-20 epochs, test accuracy ~78%
- Modèle C : s'arrête à ~25-30 epochs avec LR réduit, test accuracy ~79-80%
- Modèle C devrait avoir la meilleure généralisation
- Le temps total d'entraînement est réduit pour B et C par rapport à A
