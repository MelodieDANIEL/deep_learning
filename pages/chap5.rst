.. slide::

Chapitre 5 — Techniques avancées et bonnes pratiques PyTorch
================

🎯 Objectifs du Chapitre
----------------------


.. important::

   À la fin de ce chapitre, vous saurez : 

   - Comprendre la différence entre un MLP et les réseaux convolutifs (CNN).
   - Utiliser les couches de convolution pour le traitement d'images.
   - Appliquer les techniques de pooling pour réduire la dimensionnalité.
   - Gérer les mini-batchs pour un entraînement efficace.
   - Sauvegarder et charger les poids d'un modèle entraîné.
   - Utiliser les datasets PyTorch pour organiser vos données.

.. slide::

📖 1. MLP vs Convolutions : pourquoi les CNN ?
----------------------

Dans les chapitres précédents, nous avons utilisé des perceptrons multi-couches (MLP) pour résoudre divers problèmes. Cependant, lorsqu'on travaille avec des images, les MLP présentent plusieurs limitations importantes.

1.1. Limitations des MLP pour les images
~~~~~~~~~~~~~~~~~~~

Imaginons une image en couleur de taille $$224×224$$ pixels. Si on "aplatit" (avec ``flatten`` par exemple) cette image pour la donner à un MLP :

- Chaque pixel RGB → 3 valeurs
- Total d'entrées : $$224 \times 224 \times 3 = 150528$$ valeurs

Si la première couche cachée a 512 neurones :

- Nombre de poids : $$150528 \times 512 = 77070336$$ paramètres

**Problèmes** :

1. **Trop de paramètres** : le modèle devient énorme, difficile à entraîner et très gourmand en mémoire.
2. **Perte de structure spatiale** : en aplatissant l'image, on perd l'information sur la proximité des pixels. Or, dans une image, les pixels voisins sont fortement corrélés.
3. **Pas de généralisation spatiale** : un MLP doit réapprendre le même motif s'il apparaît à des positions différentes dans l'image.

.. slide::

1.2. Solution : les réseaux convolutifs (CNN)
~~~~~~~~~~~~~~~~~~~

Les réseaux de neurones convolutifs (CNN, de Convolutional Neural Networks en anglais) résolvent ces problèmes en utilisant des **convolutions** au lieu de couches entièrement connectées.

1.2.1. Qu'est-ce qu'un filtre (ou noyau de convolution) ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Un **filtre** (aussi appelé *kernel* ou *noyau*) est une petite matrice de poids apprenables qui sert à **détecter des motifs** dans l'image.

- **Taille typique** : $$3×3$$, $$5×5$$, ou $$7×7$$ pixels
- **Fonctionnement** : le filtre "glisse" sur toute l'image (comme un tampon qu'on déplacerait)
- **Détection** : à chaque position, il calcule une somme pondérée des pixels qu'il couvre
- **Apprentissage** : les poids du filtre sont appris automatiquement pendant l'entraînement

💡 **Intuition** : imaginez que vous cherchez des visages dans une photo. Vos yeux scannent l'image en cherchant des motifs caractéristiques (deux yeux, un nez, une bouche). Les filtres font exactement la même chose, mais de manière automatique et sur des milliers de motifs différents !

.. slide::
1.2.2. À quoi servent les filtres ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chaque filtre est spécialisé dans la détection d'un type de motif :

- **Contours** : verticaux, horizontaux, diagonaux
- **Textures** : lignes, points, motifs répétés
- **Formes** : coins, courbes, angles
- **Caractéristiques complexes** : yeux, roues, fenêtres (dans les couches profondes)

Les filtres s'organisent de manière hiérarchique :

- **Premières couches** : détectent des caractéristiques simples (bords, couleurs)
- **Couches intermédiaires** : combinent ces caractéristiques pour détecter des formes
- **Couches profondes** : détectent des objets complexes (visages, voitures, animaux)

.. slide::
1.2.3. Qu'est-ce qui détermine quel filtre fait quoi ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C'est l'**entraînement** qui détermine la spécialisation de chaque filtre ! Voici comment :

1. **Initialisation aléatoire** : au départ, les poids des filtres sont initialisés aléatoirement (petites valeurs proches de 0).

2. **Apprentissage automatique** : pendant l'entraînement, l'algorithme de descente de gradient ajuste progressivement les poids de chaque filtre pour **minimiser l'erreur** du réseau.

3. **Spécialisation émergente** : chaque filtre "apprend" naturellement à détecter les motifs les plus utiles pour la tâche. Par exemple :
   
   - Si le réseau doit reconnaître des chats, certains filtres apprendront à détecter des oreilles pointues
   - Si c'est pour des voitures, d'autres détecteront des roues ou des phares

4. **Pas de programmation manuelle** : on ne dit **jamais** explicitement à un filtre "tu dois détecter les contours verticaux". C'est le réseau qui découvre lui-même quels motifs sont importants !

💡 **Analogie** : c'est comme apprendre à reconnaître des champignons comestibles. Au début, vous ne savez pas quoi regarder. Après avoir vu des centaines d'exemples, votre cerveau apprend automatiquement à repérer les indices pertinents (couleur du chapeau, forme du pied, présence d'un anneau, etc.). Les filtres font exactement pareil !

.. slide::
1.2.4. Avantages des convolutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Partage de poids** : le même filtre est appliqué sur toute l'image, réduisant drastiquement le nombre de paramètres.
2. **Invariance par translation** : un motif appris à un endroit peut être détecté ailleurs dans l'image (un visage reste un visage, qu'il soit en haut à gauche ou en bas à droite).
3. **Préservation de la structure spatiale** : les convolutions traitent des régions locales, préservant les relations entre pixels voisins.

**Exemple de gain en paramètres** :

- Un filtre $$3×3$$ sur une image RGB → $$3 \times 3 \times 3 = 27$$ poids par filtre
- Avec 64 filtres différents → $$64 \times 27 = 1728$$ paramètres au total

Comparé aux 77 millions de paramètres du MLP, c'est une réduction spectaculaire !

.. slide::

📖 2. Les couches de convolution dans PyTorch
----------------------

Comme nous l'avons vu au chapitre 4, une convolution 2D applique un filtre sur une image en le faisant glisser sur toute la surface. PyTorch fournit ``nn.Conv2d`` pour créer ces couches convolutives.

2.1. Syntaxe de base
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn

   # Créer une couche de convolution
   conv = nn.Conv2d(
       in_channels=3,      # nombre de canaux en entrée (1 pour niveaux de gris, 3 pour RGB, 4 pour RGBA)
       out_channels=64,    # nombre de filtres à apprendre (64 détecteurs de motifs différents)
       kernel_size=3,      # taille du filtre 3×3 pixels (valeurs courantes : 3, 5, 7, etc.)
       stride=1,           # pas de déplacement du filtre (un stride de 1 déplace d'1 pixel à chaque fois et un stride de 2 divise la taille spatiale par 2)
       padding=1           # ajout de zéros autour de l'image pour contrôler la taille de sortie (ajoute 1 pixel de zéros autour de l'image pour conserver la taille)
   )

   # Exemple d'utilisation
   x = torch.randn(1, 3, 224, 224)  # batch_size=1, canaux=3, Height=224, Width=224
   y = conv(x)
   print(y.shape)  # torch.Size([1, 64, 224, 224])

.. slide::

2.2. Calcul de la taille de sortie
~~~~~~~~~~~~~~~~~~~

.. math::

   H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{padding} - \text{kernel_size}}{\text{stride}} \right\rfloor + 1

Avec padding=1, kernel_size=3, stride=1 sur une image 224×224 :

.. math::

   H_{out} = \left\lfloor \frac{224 + 2 - 3}{1} \right\rfloor + 1 = 224

La taille spatiale est préservée.

.. slide::

📖 3. Pooling : réduire la dimensionnalité
----------------------

Les couches de pooling permettent de réduire progressivement la taille spatiale des représentations, ce qui :

- **Diminue le nombre de paramètres et le temps de calcul** : en réduisant la taille spatiale (par exemple de 224×224 à 112×112), on divise par 4 le nombre de valeurs à traiter dans les couches suivantes ce qui implique moins de paramètres et un entraînement plus rapide.
- **Apporte une invariance aux petites translations** : si un motif (par exemple un œil) se déplace légèrement dans l'image (de quelques pixels), le max pooling va quand même détecter la même valeur maximale dans la région. Cela rend le réseau plus robuste aux petits déplacements des objets
- **Augmente le champ réceptif** : après un pooling, chaque neurone "voit" une région plus grande de l'image d'origine, ce qui lui permet de capturer des motifs plus globaux

3.1. Max Pooling
~~~~~~~~~~~~~~~~~~~

Le max pooling prend le maximum dans chaque région. C'est le type de pooling le plus utilisé.

.. code-block:: python

   import torch.nn.functional as F

   # Exemple : matrice 4×4
   x = torch.tensor([[[[1., 2., 3., 4.],
                       [5., 6., 7., 8.],
                       [9., 10., 11., 12.],
                       [13., 14., 15., 16.]]]])  # [batch=1, canaux=1, height=4, width=4]

   # Max pooling avec kernel 2×2 et stride 2
   # kernel_size=2 : on regarde des fenêtres de 2×2 pixels
   # stride=2 : on déplace la fenêtre de 2 pixels à chaque fois (pas de chevauchement)
   y = F.max_pool2d(x, kernel_size=2, stride=2)
   print(y)
   # tensor([[[[ 6.,  8.],
   #           [14., 16.]]]])  # [1, 1, 2, 2] - taille divisée par 2

**Explication détaillée** : 

Le max pooling divise l'image en régions de $$2×2$$ pixels et garde seulement le maximum de chaque région.

**Visualisation des 4 régions** :

.. code-block:: text

   Image d'origine 4×4 :
   ┌─────────┬─────────┐
   │  1   2  │  3   4  │  → région 1 : max([1,2,5,6]) = 6
   │  5   6  │  7   8  │  → région 2 : max([3,4,7,8]) = 8
   ├─────────┼─────────┤
   │  9  10  │ 11  12  │  → région 3 : max([9,10,13,14]) = 14
   │ 13  14  │ 15  16  │  → région 4 : max([11,12,15,16]) = 16
   └─────────┴─────────┘

   Résultat après max pooling 2×2 :
   ┌─────────┐
   │  6   8  │
   │ 14  16  │
   └─────────┘

.. slide::

3.2. Average Pooling
~~~~~~~~~~~~~~~~~~~

L'average pooling calcule la moyenne de chaque région.

.. code-block:: python

   y = F.avg_pool2d(x, kernel_size=2, stride=2)
   print(y)
   # tensor([[[[ 3.5,  5.5],
   #           [11.5, 13.5]]]])

**Explication** :

- [1,2,5,6] → (1+2+5+6)/4 = 3.5
- [3,4,7,8] → 5.5
- etc.

.. slide::

3.3. Utilisation dans un réseau
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CNNWithPooling(nn.Module):
       def __init__(self):
           super(CNNWithPooling, self).__init__()
           self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
           self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
           self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
           self.fc = nn.Linear(64 * 56 * 56, 10)
       
       def forward(self, x):
           x = F.relu(self.conv1(x))  # [batch, 32, 224, 224]
           x = self.pool(x)            # [batch, 32, 112, 112]
           x = F.relu(self.conv2(x))  # [batch, 64, 112, 112]
           x = self.pool(x)            # [batch, 64, 56, 56]
           x = x.view(x.size(0), -1)
           x = self.fc(x)
           return x

**💡 Astuce** : le max pooling est généralement préféré car il préserve mieux les caractéristiques importantes (contours, textures).

.. slide::

3.4. Exemple complet : CNN avec convolution et pooling
~~~~~~~~~~~~~~~~~~~

Maintenant que nous avons vu les convolutions et le pooling, voici un exemple complet de CNN pour la classification d'images :

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class SimpleCNN(nn.Module):
       def __init__(self, num_classes=10):
           super(SimpleCNN, self).__init__()
           
           # Première couche convolutive : 3 canaux → 32 filtres
           self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
           
           # Deuxième couche convolutive : 32 canaux → 64 filtres
           self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
           
           # Couches fully-connected pour la classification
           self.fc1 = nn.Linear(64 * 56 * 56, 128)
           self.fc2 = nn.Linear(128, num_classes)
       
       def forward(self, x):
           # x: [batch_size, 3, 224, 224] - image RGB d'entrée
           
           # Bloc 1 : Convolution + ReLU + Max Pooling
           x = F.relu(self.conv1(x))      # [batch, 32, 224, 224] - applique 32 filtres
           x = F.max_pool2d(x, 2)          # [batch, 32, 112, 112] - divise la taille par 2
           
           # Bloc 2 : Convolution + ReLU + Max Pooling
           x = F.relu(self.conv2(x))      # [batch, 64, 112, 112] - applique 64 filtres
           x = F.max_pool2d(x, 2)          # [batch, 64, 56, 56] - divise encore par 2
           
           # Aplatir les features maps pour les couches fully-connected
           x = x.view(x.size(0), -1)       # [batch, 64*56*56] = [batch, 200704]
           
           # Classification avec couches fully-connected
           x = F.relu(self.fc1(x))         # [batch, 128]
           x = self.fc2(x)                 # [batch, num_classes] - scores pour chaque classe
           
           return x

   # Créer et tester le modèle
   model = SimpleCNN(num_classes=10)
   
   # Afficher l'architecture
   print(model)
   
   # Test avec un batch d'images
   x = torch.randn(4, 3, 224, 224)  # batch de 4 images RGB 224×224
   output = model(x)
   print(f"Input shape: {x.shape}")
   print(f"Output shape: {output.shape}")  # torch.Size([4, 10])

**📊 Analyse du modèle** :

- **Entrée** : images 224×224 RGB (3 canaux)
- **Après conv1 + pool** : 32 feature maps de 112×112
- **Après conv2 + pool** : 64 feature maps de 56×56
- **Après aplatissement** : vecteur de 200 704 valeurs
- **Sortie** : 10 scores (un par classe)

Ce modèle réduit progressivement la taille spatiale tout en augmentant le nombre de canaux, ce qui est le pattern typique des CNN.

.. slide::

📖 4. Mini-batchs : entraînement efficace
----------------------

L'entraînement par mini-batchs est une technique fondamentale en deep learning qui combine les avantages de deux approches extrêmes.

4.1. Trois approches d'entraînement
~~~~~~~~~~~~~~~~~~~

**1. Batch Gradient Descent (tout le dataset)** :

- Calcule le gradient sur toutes les données
- Mise à jour stable mais très lente
- Nécessite beaucoup de mémoire

**2. Stochastic Gradient Descent (SGD, un exemple à la fois)** :

- Calcule le gradient sur un seul exemple
- Très rapide mais gradient bruité
- Converge de manière erratique

**3. Mini-Batch Gradient Descent** :

- Calcule le gradient sur un petit groupe d'exemples (typiquement 32, 64, 128)
- **Compromis idéal** : rapide et gradient raisonnablement stable
- Exploite efficacement le parallélisme du GPU

.. slide::

4.2. Pourquoi les mini-batchs ?
~~~~~~~~~~~~~~~~~~~

**Avantages** :

1. **Efficacité GPU** : les GPUs sont optimisés pour traiter plusieurs données en parallèle
2. **Estimation du gradient** : le gradient calculé sur un mini-batch est une bonne approximation du gradient sur tout le dataset
3. **Régularisation** : le bruit dans les mini-batchs peut aider à éviter les minima locaux
4. **Gestion mémoire** : on ne charge qu'une partie du dataset en mémoire à la fois

**Choix de la taille** :

- Petits batchs (16-32) : gradient plus bruité, convergence plus exploratrice
- Grands batchs (128-256) : gradient plus stable, convergence plus directe
- Compromis courant : 32 ou 64

.. slide::

4.3. Mini-batchs dans PyTorch
~~~~~~~~~~~~~~~~~~~

En PyTorch, tous les tenseurs ont une dimension de batch en première position :

.. code-block:: python

   # Format attendu : [batch_size, channels, height, width]
   images = torch.randn(32, 3, 224, 224)  # batch de 32 images RGB 224×224

   # Les opérations sont automatiquement appliquées sur tout le batch
   conv = nn.Conv2d(3, 64, kernel_size=3)
   output = conv(images)  # [32, 64, 222, 222]

**Exemple d'entraînement avec mini-batchs** :

.. code-block:: python

   # Supposons qu'on a des données et un modèle
   model = SimpleCNN()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.CrossEntropyLoss()

   # Données factices
   images = torch.randn(100, 3, 224, 224)
   labels = torch.randint(0, 10, (100,))

   # Paramètres
   batch_size = 32
   num_batches = len(images) // batch_size

   # Entraînement par mini-batchs
   for epoch in range(10):
       for i in range(num_batches):
           # Extraire un mini-batch
           start_idx = i * batch_size
           end_idx = start_idx + batch_size
           
           batch_images = images[start_idx:end_idx]
           batch_labels = labels[start_idx:end_idx]
           
           # Forward pass
           outputs = model(batch_images)
           loss = criterion(outputs, batch_labels)
           
           # Backward pass et optimisation
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
       
       print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

.. slide::

📖 5. Datasets et DataLoaders PyTorch
----------------------

Gérer manuellement les mini-batchs comme ci-dessus devient rapidement fastidieux. PyTorch fournit ``Dataset`` et ``DataLoader`` pour automatiser ce processus.

5.1. La classe Dataset
~~~~~~~~~~~~~~~~~~~

``Dataset`` est une classe abstraite qui représente votre jeu de données. Vous devez implémenter trois méthodes :

.. code-block:: python

   from torch.utils.data import Dataset

   class CustomDataset(Dataset):
       def __init__(self, data, labels):
           # Initialisation du dataset
           self.data = data
           self.labels = labels
       
       def __len__(self):
           # Retourne le nombre total d'exemples
           return len(self.data)
       
       def __getitem__(self, idx):
           # Retourne un exemple à l'indice idx
           sample = self.data[idx]
           label = self.labels[idx]
           return sample, label

.. slide::

**Exemple concret** :

.. code-block:: python

   import torch
   from torch.utils.data import Dataset

   class SimpleImageDataset(Dataset):
       def __init__(self, num_samples=1000):
           # Générer des données factices
           self.images = torch.randn(num_samples, 3, 64, 64)
           self.labels = torch.randint(0, 10, (num_samples,))
       
       def __len__(self):
           return len(self.images)
       
       def __getitem__(self, idx):
           image = self.images[idx]
           label = self.labels[idx]
           return image, label

   # Créer une instance
   dataset = SimpleImageDataset(num_samples=1000)
   print(f"Nombre d'exemples : {len(dataset)}")
   
   # Accéder à un exemple
   image, label = dataset[0]
   print(f"Shape de l'image : {image.shape}")
   print(f"Label : {label}")

.. slide::

5.2. La classe DataLoader
~~~~~~~~~~~~~~~~~~~

``DataLoader`` encapsule un ``Dataset`` et fournit :

- Le découpage automatique en mini-batchs
- Le mélange des données (shuffle)
- Le chargement parallèle (multiprocessing)
- La gestion du dernier batch incomplet

.. code-block:: python

   from torch.utils.data import DataLoader

   # Créer le dataset
   dataset = SimpleImageDataset(num_samples=1000)

   # Créer le dataloader
   dataloader = DataLoader(
       dataset,
       batch_size=32,        # taille des batchs
       shuffle=True,         # mélanger les données
       num_workers=4,        # nombre de processus pour le chargement
       drop_last=False       # garder ou non le dernier batch incomplet
   )

   # Itération sur les batchs
   for batch_idx, (images, labels) in enumerate(dataloader):
       print(f"Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
       # Batch 0: images shape = torch.Size([32, 3, 64, 64]), labels shape = torch.Size([32])

.. slide::

5.3. Exemple complet d'entraînement avec Dataset et DataLoader
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import Dataset, DataLoader

   # 1. Définir le Dataset
   class MyDataset(Dataset):
       def __init__(self, num_samples=1000):
           self.data = torch.randn(num_samples, 3, 64, 64)
           self.labels = torch.randint(0, 10, (num_samples,))
       
       def __len__(self):
           return len(self.data)
       
       def __getitem__(self, idx):
           return self.data[idx], self.labels[idx]

   # 2. Créer les datasets (train et validation)
   train_dataset = MyDataset(num_samples=800)
   val_dataset = MyDataset(num_samples=200)

   # 3. Créer les dataloaders
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

   # 4. Définir le modèle
   class SimpleCNN(nn.Module):
       def __init__(self):
           super().__init__()
           self.net = nn.Sequential(
               nn.Conv2d(3, 16, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2, 2),
               nn.Conv2d(16, 32, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2, 2),
               nn.Flatten(),
               nn.Linear(32 * 16 * 16, 128),
               nn.ReLU(),
               nn.Linear(128, 10)
           )
       
       def forward(self, x):
           return self.net(x)

   model = SimpleCNN()
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # 5. Boucle d'entraînement
   num_epochs = 5

   for epoch in range(num_epochs):
       # Phase d'entraînement
       model.train()
       train_loss = 0.0
       
       for images, labels in train_loader:
           # Forward
           outputs = model(images)
           loss = criterion(outputs, labels)
           
           # Backward
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
       
       # Phase de validation
       model.eval()
       val_loss = 0.0
       correct = 0
       total = 0
       
       with torch.no_grad():
           for images, labels in val_loader:
               outputs = model(images)
               loss = criterion(outputs, labels)
               val_loss += loss.item()
               
               _, predicted = torch.max(outputs, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
       
       # Affichage
       print(f"Epoch {epoch+1}/{num_epochs}")
       print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
       print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
       print(f"  Val Accuracy: {100*correct/total:.2f}%")

.. slide::

5.4. Datasets PyTorch intégrés
~~~~~~~~~~~~~~~~~~~

PyTorch fournit de nombreux datasets prêts à l'emploi dans ``torchvision.datasets`` :

.. code-block:: python

   from torchvision import datasets, transforms

   # MNIST (chiffres manuscrits)
   mnist_train = datasets.MNIST(
       root='./data',
       train=True,
       download=True,
       transform=transforms.ToTensor()
   )

   # CIFAR-10 (images naturelles, 10 classes)
   cifar_train = datasets.CIFAR10(
       root='./data',
       train=True,
       download=True,
       transform=transforms.ToTensor()
   )

   # Créer un DataLoader
   train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

   # Utilisation
   for images, labels in train_loader:
       print(images.shape)  # torch.Size([64, 1, 28, 28]) pour MNIST
       break

.. slide::

📖 6. Sauvegarder et charger les poids d'un modèle
----------------------

Après avoir entraîné un modèle pendant des heures (voire des jours), il est essentiel de pouvoir sauvegarder son état pour le réutiliser plus tard sans avoir à tout ré-entraîner.

6.1. Sauvegarder un modèle complet
~~~~~~~~~~~~~~~~~~~

PyTorch offre deux approches pour sauvegarder un modèle :

**Méthode 1 : Sauvegarder tout le modèle**

.. code-block:: python

   import torch

   # Entraînement du modèle
   model = SimpleCNN()
   # ... entraînement ...

   # Sauvegarder le modèle complet
   torch.save(model, 'model_complet.pth')

   # Charger le modèle complet
   model_charge = torch.load('model_complet.pth')
   model_charge.eval()  # passer en mode évaluation

**⚠️ Attention** : cette méthode sauvegarde toute la structure du modèle. Si vous modifiez la définition de la classe, le chargement peut échouer.

.. slide::

6.2. Sauvegarder uniquement les poids (méthode recommandée)
~~~~~~~~~~~~~~~~~~~

**Méthode 2 : Sauvegarder uniquement les paramètres (state_dict)**

.. code-block:: python

   # Sauvegarder uniquement les poids
   torch.save(model.state_dict(), 'model_weights.pth')

   # Charger les poids
   model = SimpleCNN()  # créer d'abord une instance du modèle
   model.load_state_dict(torch.load('model_weights.pth'))
   model.eval()

**💡 Avantages** :

- Plus flexible : on peut modifier légèrement l'architecture
- Fichier plus léger
- Meilleure pratique recommandée par PyTorch

.. slide::

6.3. Sauvegarder l'état complet de l'entraînement
~~~~~~~~~~~~~~~~~~~

Pour reprendre l'entraînement exactement où vous l'aviez arrêté, sauvegardez également l'optimiseur et l'epoch :

.. code-block:: python

   # Sauvegarder tout l'état d'entraînement
   checkpoint = {
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
   }
   torch.save(checkpoint, 'checkpoint.pth')

   # Charger et reprendre l'entraînement
   model = SimpleCNN()
   optimizer = torch.optim.Adam(model.parameters())

   checkpoint = torch.load('checkpoint.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   start_epoch = checkpoint['epoch']
   loss = checkpoint['loss']

   model.train()  # reprendre l'entraînement

.. slide::

6.4. Exemple complet avec sauvegarde automatique
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader
   import os

   # Configuration
   model = SimpleCNN()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.CrossEntropyLoss()
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

   # Créer un dossier pour les checkpoints
   os.makedirs('checkpoints', exist_ok=True)

   # Variables pour sauvegarder le meilleur modèle
   best_val_loss = float('inf')

   # Entraînement avec sauvegarde
   for epoch in range(num_epochs):
       model.train()
       train_loss = 0.0
       
       for images, labels in train_loader:
           outputs = model(images)
           loss = criterion(outputs, labels)
           
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
       
       # Validation
       model.eval()
       val_loss = 0.0
       with torch.no_grad():
           for images, labels in val_loader:
               outputs = model(images)
               loss = criterion(outputs, labels)
               val_loss += loss.item()
       
       val_loss /= len(val_loader)
       
       # Sauvegarder si c'est le meilleur modèle
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'val_loss': val_loss,
           }, 'checkpoints/best_model.pth')
           print(f"✓ Nouveau meilleur modèle sauvegardé (val_loss: {val_loss:.4f})")
       
       # Sauvegarder un checkpoint régulier tous les 10 epochs
       if (epoch + 1) % 10 == 0:
           torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
           }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')

   print(f"Entraînement terminé. Meilleure val_loss: {best_val_loss:.4f}")

.. slide::

6.5. Utiliser un modèle sauvegardé pour l'inférence
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch

   # Définir la classe du modèle (doit être identique)
   class SimpleCNN(nn.Module):
       # ... définition du modèle ...
       pass

   # Charger le meilleur modèle
   model = SimpleCNN()
   checkpoint = torch.load('checkpoints/best_model.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()

   # Passer le modèle sur GPU si disponible
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)

   # Faire des prédictions
   with torch.no_grad():
       for images, labels in test_loader:
           images = images.to(device)
           outputs = model(images)
           _, predicted = torch.max(outputs, 1)
           # ... traiter les prédictions ...

.. slide::

📖 7. Récapitulatif et bonnes pratiques
----------------------

7.1. Pipeline complet d'entraînement
~~~~~~~~~~~~~~~~~~~

Voici le pipeline standard pour entraîner un CNN avec toutes les techniques vues :

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import Dataset, DataLoader
   import os

   # 1. Définir le Dataset
   class CustomDataset(Dataset):
       def __init__(self, data_path, transform=None):
           # Charger vos données
           pass
       
       def __len__(self):
           return len(self.data)
       
       def __getitem__(self, idx):
           return self.data[idx], self.labels[idx]

   # 2. Définir le modèle avec convolutions et pooling
   class CNN(nn.Module):
       def __init__(self, num_classes):
           super(CNN, self).__init__()
           self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
           self.pool = nn.MaxPool2d(2, 2)
           self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
           self.fc = nn.Linear(64 * 16 * 16, num_classes)
       
       def forward(self, x):
           x = self.pool(torch.relu(self.conv1(x)))
           x = self.pool(torch.relu(self.conv2(x)))
           x = x.view(x.size(0), -1)
           x = self.fc(x)
           return x

   # 3. Préparer les données avec DataLoader
   train_dataset = CustomDataset('train_data')
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
   
   val_dataset = CustomDataset('val_data')
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

   # 4. Initialiser le modèle, la loss et l'optimiseur
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = CNN(num_classes=10).to(device)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # 5. Créer un dossier pour les sauvegardes
   os.makedirs('checkpoints', exist_ok=True)
   best_val_loss = float('inf')

   # 6. Boucle d'entraînement
   num_epochs = 50
   
   for epoch in range(num_epochs):
       # PHASE D'ENTRAÎNEMENT
       model.train()
       train_loss = 0.0
       
       for batch_idx, (images, labels) in enumerate(train_loader):
           images, labels = images.to(device), labels.to(device)
           
           # Forward pass
           outputs = model(images)
           loss = criterion(outputs, labels)
           
           # Backward pass et optimisation
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
       
       # PHASE DE VALIDATION
       model.eval()
       val_loss = 0.0
       correct = 0
       total = 0
       
       with torch.no_grad():
           for images, labels in val_loader:
               images, labels = images.to(device), labels.to(device)
               outputs = model(images)
               loss = criterion(outputs, labels)
               val_loss += loss.item()
               
               _, predicted = torch.max(outputs, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
       
       # Calcul des moyennes
       train_loss /= len(train_loader)
       val_loss /= len(val_loader)
       val_acc = 100 * correct / total
       
       print(f"Epoch [{epoch+1}/{num_epochs}]")
       print(f"  Train Loss: {train_loss:.4f}")
       print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
       
       # Sauvegarder le meilleur modèle
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'val_loss': val_loss,
               'val_acc': val_acc,
           }, 'checkpoints/best_model.pth')
           print(f"  ✓ Meilleur modèle sauvegardé!")

   print("Entraînement terminé!")

.. slide::

7.2. Bonnes pratiques
~~~~~~~~~~~~~~~~~~~

**Organisation des données** :

1. Toujours séparer train/validation/test
2. Utiliser ``Dataset`` et ``DataLoader`` pour gérer les données
3. Appliquer les transformations (normalisation, augmentation) dans le ``Dataset``

**Architecture du modèle** :

1. Utiliser des convolutions pour les images (pas de MLP)
2. Alterner convolutions et pooling pour réduire progressivement la taille
3. Ajouter du batch normalization pour stabiliser l'entraînement
4. Utiliser ReLU comme activation dans les couches cachées

**Entraînement** :

1. Utiliser des mini-batchs (taille typique : 32-64)
2. Shuffler les données d'entraînement (``shuffle=True``)
3. Ne PAS shuffler les données de validation/test
4. Utiliser ``model.train()`` pour l'entraînement et ``model.eval()`` pour l'évaluation
5. Utiliser ``torch.no_grad()`` pendant la validation pour économiser la mémoire

**Sauvegarde** :

1. Sauvegarder le meilleur modèle basé sur la validation loss
2. Sauvegarder régulièrement des checkpoints pour pouvoir reprendre
3. Préférer ``state_dict()`` à sauvegarder le modèle entier

.. slide::

7.3. Checklist avant de lancer un entraînement
~~~~~~~~~~~~~~~~~~~

✅ **Données** :

- [ ] Dataset implémenté correctement (``__len__`` et ``__getitem__``)
- [ ] Données normalisées (mean=0, std=1)
- [ ] Train/val/test bien séparés
- [ ] DataLoader créés avec la bonne batch_size

✅ **Modèle** :

- [ ] Architecture adaptée au problème
- [ ] Modèle déplacé sur le bon device (CPU/GPU)
- [ ] Taille des tenseurs vérifiée à chaque couche

✅ **Entraînement** :

- [ ] Loss function appropriée
- [ ] Optimiseur configuré avec un bon learning rate
- [ ] ``optimizer.zero_grad()`` appelé avant chaque backward
- [ ] ``model.train()`` et ``model.eval()`` utilisés correctement

✅ **Monitoring** :

- [ ] Loss affichée régulièrement
- [ ] Métriques de validation calculées
- [ ] Meilleur modèle sauvegardé

.. slide::

🏋️ Travaux Pratiques 5
--------------------

.. toctree::

    TP_chap5





#######################################################################
########################Stop ici pour le moment########################
########################Stop ici pour le moment########################
########################Stop ici pour le moment########################
#######################################################################