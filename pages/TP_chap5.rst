.. slide::

TP Chapitre 5 - Classification d'images avec CNN
================

🎯 Objectifs du TP
----------------------

.. important::

   Dans ce TP, vous allez :

   - Implémenter un réseau de neurones convolutif (CNN)
   - Utiliser les datasets et dataloaders PyTorch
   - Entraîner un modèle sur le dataset CIFAR-10
   - Comparer les performances d'un MLP et d'un CNN
   - Sauvegarder et charger des modèles entraînés
   - Expérimenter avec différentes architectures

.. slide::

📋 Exercice 1 : Préparation des données (🍀)
----------------------

**Objectif** : Charger et préparer le dataset CIFAR-10

CIFAR-10 est un dataset célèbre contenant 60 000 images couleur 32×32 réparties en 10 classes :
avion, voiture, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion.

1.1. Charger CIFAR-10
~~~~~~~~~~~~~~~~~~~

Créez un fichier ``tp5_preparation.py`` et implémentez le code suivant :

.. code-block:: python

   import torch
   import torchvision
   import torchvision.transforms as transforms
   import matplotlib.pyplot as plt
   import numpy as np

   # Transformations pour normaliser les données
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Télécharger et charger les données d'entraînement
   trainset = torchvision.datasets.CIFAR10(
       root='./data',
       train=True,
       download=True,
       transform=transform
   )

   # Télécharger et charger les données de test
   testset = torchvision.datasets.CIFAR10(
       root='./data',
       train=False,
       download=True,
       transform=transform
   )

   # Créer les DataLoaders
   trainloader = torch.utils.data.DataLoader(
       trainset,
       batch_size=64,
       shuffle=True,
       num_workers=2
   )

   testloader = torch.utils.data.DataLoader(
       testset,
       batch_size=64,
       shuffle=False,
       num_workers=2
   )

   # Les 10 classes de CIFAR-10
   classes = ('avion', 'voiture', 'oiseau', 'chat', 'cerf',
              'chien', 'grenouille', 'cheval', 'bateau', 'camion')

   print(f"Nombre d'images d'entraînement : {len(trainset)}")
   print(f"Nombre d'images de test : {len(testset)}")

.. slide::

1.2. Visualiser les données
~~~~~~~~~~~~~~~~~~~

Ajoutez une fonction pour visualiser quelques images :

.. code-block:: python

   def imshow(img):
       """Fonction pour afficher une image"""
       img = img / 2 + 0.5     # dénormaliser
       npimg = img.numpy()
       plt.imshow(np.transpose(npimg, (1, 2, 0)))
       plt.show()

   # Obtenir un batch d'images d'entraînement
   dataiter = iter(trainloader)
   images, labels = next(dataiter)

   # Afficher les images
   imshow(torchvision.utils.make_grid(images[:8]))
   # Afficher les labels
   print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))

**Questions** :

1. Quelle est la forme d'un batch d'images ?
2. Pourquoi normalise-t-on les images avec ``(0.5, 0.5, 0.5)`` pour la moyenne et l'écart-type ?
3. Que fait la transformation ``ToTensor()`` ?

.. slide::

📋 Exercice 2 : MLP de base (🍀)
----------------------

**Objectif** : Créer un perceptron multi-couches pour servir de référence

2.1. Implémenter un MLP
~~~~~~~~~~~~~~~~~~~

Créez un fichier ``tp5_mlp.py`` avec le code suivant :

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class SimpleMLP(nn.Module):
       def __init__(self):
           super(SimpleMLP, self).__init__()
           # Une image CIFAR-10 fait 32x32x3 = 3072 pixels
           self.fc1 = nn.Linear(3 * 32 * 32, 512)
           self.fc2 = nn.Linear(512, 256)
           self.fc3 = nn.Linear(256, 10)  # 10 classes
       
       def forward(self, x):
           # Aplatir l'image
           x = x.view(-1, 3 * 32 * 32)
           
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = self.fc3(x)
           return x

   # Créer le modèle
   model_mlp = SimpleMLP()
   print(model_mlp)

   # Compter le nombre de paramètres
   total_params = sum(p.numel() for p in model_mlp.parameters())
   print(f"\nNombre total de paramètres : {total_params:,}")

**À faire** :

1. Exécutez le code et notez le nombre de paramètres
2. Calculez manuellement le nombre de paramètres de la première couche

.. slide::

2.2. Entraîner le MLP
~~~~~~~~~~~~~~~~~~~

Ajoutez le code d'entraînement :

.. code-block:: python

   import torch.optim as optim
   from tp5_preparation import trainloader, testloader

   # Configuration
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model_mlp = SimpleMLP().to(device)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model_mlp.parameters(), lr=0.001)

   # Entraînement
   num_epochs = 10

   for epoch in range(num_epochs):
       model_mlp.train()
       running_loss = 0.0
       correct = 0
       total = 0
       
       for i, (images, labels) in enumerate(trainloader):
           images, labels = images.to(device), labels.to(device)
           
           # Forward
           outputs = model_mlp(images)
           loss = criterion(outputs, labels)
           
           # Backward
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           # Statistiques
           running_loss += loss.item()
           _, predicted = torch.max(outputs, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
       
       train_acc = 100 * correct / total
       print(f"Epoch [{epoch+1}/{num_epochs}], "
             f"Loss: {running_loss/len(trainloader):.4f}, "
             f"Train Acc: {train_acc:.2f}%")

   # Sauvegarder le modèle
   torch.save(model_mlp.state_dict(), 'mlp_cifar10.pth')
   print("Modèle MLP sauvegardé!")

.. slide::

2.3. Évaluer le MLP
~~~~~~~~~~~~~~~~~~~

Ajoutez la fonction d'évaluation :

.. code-block:: python

   def evaluate_model(model, dataloader, device):
       """Évalue le modèle sur un dataset"""
       model.eval()
       correct = 0
       total = 0
       
       with torch.no_grad():
           for images, labels in dataloader:
               images, labels = images.to(device), labels.to(device)
               outputs = model(images)
               _, predicted = torch.max(outputs, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
       
       accuracy = 100 * correct / total
       return accuracy

   # Évaluer sur le test set
   test_acc = evaluate_model(model_mlp, testloader, device)
   print(f"Précision sur le test set : {test_acc:.2f}%")

**Questions** :

1. Quelle précision obtenez-vous après 10 epochs ?
2. Le modèle semble-t-il sur-apprendre (overfitting) ?
3. Combien de temps prend une epoch ?

.. slide::

📋 Exercice 3 : Premier CNN simple (⚖️)
----------------------

**Objectif** : Créer un réseau convolutif et comparer avec le MLP

3.1. Implémenter un CNN simple
~~~~~~~~~~~~~~~~~~~

Créez un fichier ``tp5_cnn.py`` :

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class SimpleCNN(nn.Module):
       def __init__(self):
           super(SimpleCNN, self).__init__()
           # Couches de convolution
           self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
           self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
           self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
           
           # Pooling
           self.pool = nn.MaxPool2d(2, 2)
           
           # Couches fully-connected
           # Après 3 poolings : 32 -> 16 -> 8 -> 4
           self.fc1 = nn.Linear(64 * 4 * 4, 512)
           self.fc2 = nn.Linear(512, 10)
           
           # Dropout pour la régularisation
           self.dropout = nn.Dropout(0.5)
       
       def forward(self, x):
           # Block 1
           x = F.relu(self.conv1(x))       # [batch, 32, 32, 32]
           x = self.pool(x)                 # [batch, 32, 16, 16]
           
           # Block 2
           x = F.relu(self.conv2(x))       # [batch, 64, 16, 16]
           x = self.pool(x)                 # [batch, 64, 8, 8]
           
           # Block 3
           x = F.relu(self.conv3(x))       # [batch, 64, 8, 8]
           x = self.pool(x)                 # [batch, 64, 4, 4]
           
           # Aplatir
           x = x.view(-1, 64 * 4 * 4)
           
           # Fully-connected
           x = F.relu(self.fc1(x))
           x = self.dropout(x)
           x = self.fc2(x)
           
           return x

   # Créer le modèle
   model_cnn = SimpleCNN()
   print(model_cnn)

   # Compter les paramètres
   total_params = sum(p.numel() for p in model_cnn.parameters())
   print(f"\nNombre total de paramètres : {total_params:,}")

**À faire** :

1. Comparez le nombre de paramètres avec le MLP
2. Ajoutez des commentaires pour indiquer la taille des tenseurs à chaque étape

.. slide::

3.2. Entraîner le CNN
~~~~~~~~~~~~~~~~~~~

Ajoutez le code d'entraînement (similaire au MLP) :

.. code-block:: python

   import torch.optim as optim
   from tp5_preparation import trainloader, testloader
   from tp5_mlp import evaluate_model

   # Configuration
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model_cnn = SimpleCNN().to(device)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

   # Entraînement
   num_epochs = 20

   for epoch in range(num_epochs):
       model_cnn.train()
       running_loss = 0.0
       correct = 0
       total = 0
       
       for images, labels in trainloader:
           images, labels = images.to(device), labels.to(device)
           
           outputs = model_cnn(images)
           loss = criterion(outputs, labels)
           
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           running_loss += loss.item()
           _, predicted = torch.max(outputs, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
       
       train_acc = 100 * correct / total
       
       # Évaluation sur le test set
       test_acc = evaluate_model(model_cnn, testloader, device)
       
       print(f"Epoch [{epoch+1}/{num_epochs}], "
             f"Loss: {running_loss/len(trainloader):.4f}, "
             f"Train Acc: {train_acc:.2f}%, "
             f"Test Acc: {test_acc:.2f}%")

   # Sauvegarder le modèle
   torch.save(model_cnn.state_dict(), 'cnn_cifar10.pth')
   print("Modèle CNN sauvegardé!")

**Questions** :

1. Comparez la précision du CNN avec celle du MLP
2. Le CNN apprend-il plus vite que le MLP ?
3. Y a-t-il du sur-apprentissage ? Comment le détecter ?

.. slide::

📋 Exercice 4 : Dataset personnalisé (⚖️)
----------------------

**Objectif** : Créer un Dataset PyTorch pour séparer train/validation

4.1. Implémenter un Dataset avec split train/val
~~~~~~~~~~~~~~~~~~~

Créez un fichier ``tp5_dataset.py`` :

.. code-block:: python

   import torch
   from torch.utils.data import Dataset, DataLoader, random_split
   import torchvision
   import torchvision.transforms as transforms

   # Charger CIFAR-10 complet
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   full_trainset = torchvision.datasets.CIFAR10(
       root='./data',
       train=True,
       download=True,
       transform=transform
   )

   testset = torchvision.datasets.CIFAR10(
       root='./data',
       train=False,
       download=True,
       transform=transform
   )

   # Séparer train et validation (80/20)
   train_size = int(0.8 * len(full_trainset))
   val_size = len(full_trainset) - train_size

   trainset, valset = random_split(
       full_trainset,
       [train_size, val_size],
       generator=torch.Generator().manual_seed(42)  # pour la reproductibilité
   )

   print(f"Train set : {len(trainset)} images")
   print(f"Validation set : {len(valset)} images")
   print(f"Test set : {len(testset)} images")

   # Créer les DataLoaders
   trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
   valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
   testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

.. slide::

4.2. Entraîner avec validation
~~~~~~~~~~~~~~~~~~~

Modifiez la boucle d'entraînement pour inclure la validation :

.. code-block:: python

   from tp5_dataset import trainloader, valloader, testloader
   from tp5_cnn import SimpleCNN
   from tp5_mlp import evaluate_model
   import torch
   import torch.nn as nn
   import torch.optim as optim

   # Configuration
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = SimpleCNN().to(device)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # Pour sauvegarder le meilleur modèle
   best_val_acc = 0.0

   # Entraînement avec validation
   num_epochs = 20

   for epoch in range(num_epochs):
       # PHASE D'ENTRAÎNEMENT
       model.train()
       train_loss = 0.0
       train_correct = 0
       train_total = 0
       
       for images, labels in trainloader:
           images, labels = images.to(device), labels.to(device)
           
           outputs = model(images)
           loss = criterion(outputs, labels)
           
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
           _, predicted = torch.max(outputs, 1)
           train_total += labels.size(0)
           train_correct += (predicted == labels).sum().item()
       
       train_acc = 100 * train_correct / train_total
       
       # PHASE DE VALIDATION
       val_acc = evaluate_model(model, valloader, device)
       
       # Sauvegarder le meilleur modèle
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'val_acc': val_acc,
           }, 'best_cnn_cifar10.pth')
           print(f"✓ Nouveau meilleur modèle sauvegardé! Val Acc: {val_acc:.2f}%")
       
       print(f"Epoch [{epoch+1}/{num_epochs}], "
             f"Loss: {train_loss/len(trainloader):.4f}, "
             f"Train Acc: {train_acc:.2f}%, "
             f"Val Acc: {val_acc:.2f}%")

   # Évaluer sur le test set avec le meilleur modèle
   checkpoint = torch.load('best_cnn_cifar10.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   test_acc = evaluate_model(model, testloader, device)
   print(f"\nPrécision finale sur le test set : {test_acc:.2f}%")

.. slide::

📋 Exercice 5 : CNN amélioré (🌶️)
----------------------

**Objectif** : Améliorer l'architecture avec des techniques avancées

5.1. CNN avec Batch Normalization
~~~~~~~~~~~~~~~~~~~

Créez un fichier ``tp5_cnn_advanced.py`` :

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class ImprovedCNN(nn.Module):
       def __init__(self):
           super(ImprovedCNN, self).__init__()
           
           # Block 1
           self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
           self.bn1 = nn.BatchNorm2d(32)
           
           # Block 2
           self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
           self.bn2 = nn.BatchNorm2d(64)
           
           # Block 3
           self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
           self.bn3 = nn.BatchNorm2d(128)
           
           # Block 4
           self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
           self.bn4 = nn.BatchNorm2d(128)
           
           # Pooling
           self.pool = nn.MaxPool2d(2, 2)
           
           # Fully-connected
           self.fc1 = nn.Linear(128 * 2 * 2, 512)
           self.bn_fc = nn.BatchNorm1d(512)
           self.fc2 = nn.Linear(512, 10)
           
           # Dropout
           self.dropout = nn.Dropout(0.5)
       
       def forward(self, x):
           # Block 1: [batch, 3, 32, 32] -> [batch, 32, 16, 16]
           x = self.conv1(x)
           x = self.bn1(x)
           x = F.relu(x)
           x = self.pool(x)
           
           # Block 2: [batch, 32, 16, 16] -> [batch, 64, 8, 8]
           x = self.conv2(x)
           x = self.bn2(x)
           x = F.relu(x)
           x = self.pool(x)
           
           # Block 3: [batch, 64, 8, 8] -> [batch, 128, 4, 4]
           x = self.conv3(x)
           x = self.bn3(x)
           x = F.relu(x)
           x = self.pool(x)
           
           # Block 4: [batch, 128, 4, 4] -> [batch, 128, 2, 2]
           x = self.conv4(x)
           x = self.bn4(x)
           x = F.relu(x)
           x = self.pool(x)
           
           # Flatten
           x = x.view(x.size(0), -1)
           
           # FC layers
           x = self.fc1(x)
           x = self.bn_fc(x)
           x = F.relu(x)
           x = self.dropout(x)
           x = self.fc2(x)
           
           return x

   model = ImprovedCNN()
   total_params = sum(p.numel() for p in model.parameters())
   print(f"Nombre total de paramètres : {total_params:,}")

**À faire** :

1. Entraînez ce modèle et comparez les performances
2. Expérimentez avec différentes valeurs de dropout (0.3, 0.5, 0.7)

.. slide::

5.2. Data Augmentation
~~~~~~~~~~~~~~~~~~~

Ajoutez de l'augmentation de données pour améliorer la généralisation :

.. code-block:: python

   import torchvision.transforms as transforms

   # Transformations pour l'entraînement (avec augmentation)
   train_transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4),
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Transformations pour validation/test (sans augmentation)
   test_transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Charger les données avec les bonnes transformations
   trainset = torchvision.datasets.CIFAR10(
       root='./data',
       train=True,
       download=True,
       transform=train_transform
   )

   testset = torchvision.datasets.CIFAR10(
       root='./data',
       train=False,
       download=True,
       transform=test_transform
   )

**Questions** :

1. Quelle amélioration apporte l'augmentation de données ?
2. Pourquoi n'applique-t-on pas l'augmentation sur le test set ?

.. slide::

📋 Exercice 6 : Visualisation et analyse (🌶️)
----------------------

**Objectif** : Analyser les performances du modèle

6.1. Matrice de confusion
~~~~~~~~~~~~~~~~~~~

Créez un fichier ``tp5_analysis.py`` :

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.metrics import confusion_matrix, classification_report
   import seaborn as sns

   def plot_confusion_matrix(model, dataloader, classes, device):
       """Affiche la matrice de confusion"""
       model.eval()
       all_preds = []
       all_labels = []
       
       with torch.no_grad():
           for images, labels in dataloader:
               images = images.to(device)
               outputs = model(images)
               _, predicted = torch.max(outputs, 1)
               
               all_preds.extend(predicted.cpu().numpy())
               all_labels.extend(labels.numpy())
       
       # Calculer la matrice de confusion
       cm = confusion_matrix(all_labels, all_preds)
       
       # Afficher
       plt.figure(figsize=(12, 10))
       sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
       plt.title('Matrice de confusion')
       plt.ylabel('Vraie classe')
       plt.xlabel('Classe prédite')
       plt.tight_layout()
       plt.savefig('confusion_matrix.png')
       plt.show()
       
       # Afficher le rapport de classification
       print("\nRapport de classification :")
       print(classification_report(all_labels, all_preds, target_names=classes))

   # Utilisation
   from tp5_dataset import testloader
   from tp5_cnn_advanced import ImprovedCNN
   from tp5_preparation import classes

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = ImprovedCNN().to(device)

   # Charger le meilleur modèle
   checkpoint = torch.load('best_cnn_cifar10.pth')
   model.load_state_dict(checkpoint['model_state_dict'])

   plot_confusion_matrix(model, testloader, classes, device)

.. slide::

6.2. Visualiser les prédictions
~~~~~~~~~~~~~~~~~~~

Ajoutez une fonction pour visualiser les prédictions :

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   import torchvision

   def visualize_predictions(model, dataloader, classes, device, num_images=20):
       """Visualise des prédictions du modèle"""
       model.eval()
       
       # Obtenir un batch
       dataiter = iter(dataloader)
       images, labels = next(dataiter)
       images, labels = images.to(device), labels.to(device)
       
       # Faire les prédictions
       with torch.no_grad():
           outputs = model(images)
           _, predicted = torch.max(outputs, 1)
       
       # Afficher les images avec leurs prédictions
       fig, axes = plt.subplots(4, 5, figsize=(15, 12))
       
       for idx, ax in enumerate(axes.flat):
           if idx >= num_images:
               break
           
           # Dénormaliser l'image
           img = images[idx].cpu() / 2 + 0.5
           img = np.transpose(img.numpy(), (1, 2, 0))
           
           # Afficher
           ax.imshow(img)
           
           true_label = classes[labels[idx]]
           pred_label = classes[predicted[idx]]
           color = 'green' if labels[idx] == predicted[idx] else 'red'
           
           ax.set_title(f'Vrai: {true_label}\nPréd: {pred_label}', 
                       color=color, fontsize=10)
           ax.axis('off')
       
       plt.tight_layout()
       plt.savefig('predictions_visualization.png')
       plt.show()

   # Utilisation
   visualize_predictions(model, testloader, classes, device)

.. slide::

📋 Exercice 7 : Expérimentations (🌶️)
----------------------

**Objectif** : Explorer différentes configurations

7.1. Comparaison de plusieurs modèles
~~~~~~~~~~~~~~~~~~~

Créez un fichier ``tp5_experiments.py`` pour comparer plusieurs configurations :

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader
   import pandas as pd
   import matplotlib.pyplot as plt

   def train_and_evaluate(model, trainloader, valloader, testloader, 
                          device, num_epochs=20, lr=0.001, model_name="Model"):
       """Entraîne et évalue un modèle"""
       criterion = nn.CrossEntropyLoss()
       optimizer = optim.Adam(model.parameters(), lr=lr)
       
       history = {
           'train_loss': [],
           'train_acc': [],
           'val_acc': []
       }
       
       best_val_acc = 0.0
       
       for epoch in range(num_epochs):
           # Phase d'entraînement
           model.train()
           train_loss = 0.0
           train_correct = 0
           train_total = 0
           
           for images, labels in trainloader:
               images, labels = images.to(device), labels.to(device)
               
               outputs = model(images)
               loss = criterion(outputs, labels)
               
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               
               train_loss += loss.item()
               _, predicted = torch.max(outputs, 1)
               train_total += labels.size(0)
               train_correct += (predicted == labels).sum().item()
           
           train_acc = 100 * train_correct / train_total
           
           # Phase de validation
           model.eval()
           val_correct = 0
           val_total = 0
           
           with torch.no_grad():
               for images, labels in valloader:
                   images, labels = images.to(device), labels.to(device)
                   outputs = model(images)
                   _, predicted = torch.max(outputs, 1)
                   val_total += labels.size(0)
                   val_correct += (predicted == labels).sum().item()
           
           val_acc = 100 * val_correct / val_total
           
           # Enregistrer l'historique
           history['train_loss'].append(train_loss / len(trainloader))
           history['train_acc'].append(train_acc)
           history['val_acc'].append(val_acc)
           
           # Sauvegarder le meilleur
           if val_acc > best_val_acc:
               best_val_acc = val_acc
               torch.save(model.state_dict(), f'{model_name}_best.pth')
           
           if (epoch + 1) % 5 == 0:
               print(f"[{model_name}] Epoch [{epoch+1}/{num_epochs}], "
                     f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
       
       # Test final
       model.load_state_dict(torch.load(f'{model_name}_best.pth'))
       model.eval()
       test_correct = 0
       test_total = 0
       
       with torch.no_grad():
           for images, labels in testloader:
               images, labels = images.to(device), labels.to(device)
               outputs = model(images)
               _, predicted = torch.max(outputs, 1)
               test_total += labels.size(0)
               test_correct += (predicted == labels).sum().item()
       
       test_acc = 100 * test_correct / test_total
       
       return history, test_acc

**À faire** :

Comparez les architectures suivantes :

1. MLP simple
2. CNN simple (3 couches conv)
3. CNN avec Batch Normalization
4. CNN plus profond (5-6 couches conv)

Pour chaque modèle, tracez l'évolution de la loss et de l'accuracy.

.. slide::

7.2. Tracer les courbes d'apprentissage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def plot_training_curves(histories, model_names):
       """Trace les courbes d'apprentissage pour plusieurs modèles"""
       fig, axes = plt.subplots(1, 3, figsize=(18, 5))
       
       for history, name in zip(histories, model_names):
           epochs = range(1, len(history['train_loss']) + 1)
           
           # Loss
           axes[0].plot(epochs, history['train_loss'], label=name)
           
           # Train accuracy
           axes[1].plot(epochs, history['train_acc'], label=name)
           
           # Val accuracy
           axes[2].plot(epochs, history['val_acc'], label=name)
       
       axes[0].set_title('Loss d\'entraînement')
       axes[0].set_xlabel('Epoch')
       axes[0].set_ylabel('Loss')
       axes[0].legend()
       axes[0].grid(True)
       
       axes[1].set_title('Précision d\'entraînement')
       axes[1].set_xlabel('Epoch')
       axes[1].set_ylabel('Accuracy (%)')
       axes[1].legend()
       axes[1].grid(True)
       
       axes[2].set_title('Précision de validation')
       axes[2].set_xlabel('Epoch')
       axes[2].set_ylabel('Accuracy (%)')
       axes[2].legend()
       axes[2].grid(True)
       
       plt.tight_layout()
       plt.savefig('training_curves_comparison.png')
       plt.show()

.. slide::

📋 Exercice Bonus : Transfer Learning (🌶️🌶️)
----------------------

**Objectif** : Utiliser un modèle pré-entraîné

8.1. Charger un modèle pré-entraîné
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torchvision.models as models
   import torch.nn as nn

   # Charger ResNet18 pré-entraîné sur ImageNet
   model = models.resnet18(pretrained=True)

   # Geler les poids des couches convolutives
   for param in model.parameters():
       param.requires_grad = False

   # Remplacer la dernière couche pour CIFAR-10 (10 classes)
   num_features = model.fc.in_features
   model.fc = nn.Linear(num_features, 10)

   # Seule la dernière couche sera entraînée
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)

   # Optimiseur uniquement pour la dernière couche
   optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

**À faire** :

1. Entraînez ce modèle et comparez avec vos CNN from scratch
2. Expérimentez avec le fine-tuning : dégelez les dernières couches convolutives

.. slide::

📊 Récapitulatif et questions
----------------------

**Résumé des concepts clés** :

- MLP vs CNN : réduction drastique des paramètres
- Convolutions : partage de poids et structure spatiale
- Pooling : réduction de dimensionnalité
- Mini-batchs : compromis efficacité/stabilité
- Dataset/DataLoader : gestion automatique des données
- Sauvegarde : state_dict pour la flexibilité

**Questions de réflexion** :

1. Pourquoi les CNN performent-ils mieux que les MLP sur les images ?
2. Quel est le rôle du pooling dans un CNN ?
3. Comment choisir la taille des mini-batchs ?
4. Pourquoi sépare-t-on train/validation/test ?
5. Quand utiliser le transfer learning ?

**Pour aller plus loin** :

- Essayez d'autres architectures : VGG, ResNet, DenseNet
- Implémentez des techniques d'augmentation avancées
- Utilisez un scheduler pour le learning rate
- Explorez la visualisation des filtres convolutifs
- Testez sur d'autres datasets : CIFAR-100, STL-10

.. note::

   💡 **Conseil** : Gardez trace de toutes vos expériences dans un tableur ou un notebook. Notez les hyperparamètres, les résultats et vos observations. C'est une pratique essentielle en deep learning !
