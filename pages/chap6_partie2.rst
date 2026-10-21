.. slide::

Chapitre 6 — Détection d'objets avec des boîtes englobantes (partie 2)
================

🎯 Objectifs du Chapitre
----------------------

.. important::

   À la fin de ce chapitre, vous saurez : 

   - Comprendre la différence entre classification et détection d'objets.
   - Extraire des images depuis une vidéo.
   - Utiliser Label Studio pour annoter des objets avec des boîtes englobantes de manière collaborative.
   - Comprendre et manipuler les formats d'annotations.
   - Créer un dataset PyTorch pour la détection d'objets.
   - Entraîner un détecteur custom.
   - Comparer avec YOLO et choisir le bon modèle selon le contexte.
   - Effectuer l'inférence sur des images en temps réel.

.. slide::

📖 6. CNN ultra-simple : régression directe de boîte
----------------------

Pour des cas simples avec **1 seul objet par image**, on peut utiliser une approche beaucoup plus simple que YOLO ou Faster R-CNN : **régression directe des coordonnées** de la boîte. Le modèle prédit directement 4 nombres : ``(x_center, y_center, width, height)`` normalisés dans [0,1].

.. note::

   💡 **Quand utiliser cette approche ?**
   
   ✅ **OUI** : 1 objet par image, objet centré, peu de variations (ex: détection de visage, logo)
   
   ❌ **NON** : plusieurs objets, abscence de l'objet, objets qui se chevauchent, etc.

.. slide::
    
6.1. Architecture ultra-simple
~~~~~~~~~~~~~~~~~~~

Comme au chapitre 5 (section 3.3), le modèle est construit avec deux blocs ``nn.Sequential`` :

- un bloc ``convolutions`` (4 couches Conv2D + ReLU + MaxPool), appelé *backbone* en anglais, qui extrait les caractéristiques de l'image ;
- un bloc ``regression`` (2 couches fully-connected), appelé *head* en anglais, qui remplace le bloc ``classification`` du chapitre 5 et prédit directement les 4 coordonnées normalisées.

Dans l'exemple, l'entrée $$224×224$$ est réduite **4 fois** par MaxPool(2) : $$224→112→56→28→14$$ ; les cartes de caractéristiques finales font donc $$14×14$$. Si vous changez la taille d'entrée ou le nombre de couches de pooling, cette taille changera, et donc aussi la taille d'entrée de la première couche ``nn.Linear``.

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from tqdm import tqdm # Pour les barres de progression

   # Vérifier si GPU disponible
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"🔧 Device utilisé : {device}")

   class SimpleBBoxRegressor(nn.Module):
       """
       CNN ultra-simple qui régresse directement UNE boîte par image.
       Sortie : [x_center, y_center, width, height] normalisés dans [0,1]
       """
       
       def __init__(self):
           super().__init__()

           # Bloc convolutions (backbone) : Conv2D + ReLU + MaxPool, comme au chapitre 5
           self.convolutions = nn.Sequential(
               nn.Conv2d(3, 16, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),                              # 224 -> 112
               nn.Conv2d(16, 32, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),                              # 112 -> 56
               nn.Conv2d(32, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),                              # 56 -> 28
               nn.Conv2d(64, 128, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),                              # 28 -> 14 : [B, 128, 14, 14]
           )

           # Bloc de régression (head) : 4 sorties (x_center, y_center, width, height)
           self.regression = nn.Sequential(
               nn.Flatten(),                                 # [B, 128*14*14]
               nn.Linear(128 * 14 * 14, 128),
               nn.ReLU(),
               nn.Linear(128, 4),
               nn.Sigmoid(),                                 # sorties dans [0, 1]
           )

       def forward(self, x):
           x = self.convolutions(x)
           x = self.regression(x)
           return x  # [B, 4] : (x_center, y_center, w, h) normalisés


   # Créer le modèle
   simple_model = SimpleBBoxRegressor().to(device)
   num_params = sum(p.numel() for p in simple_model.parameters() if p.requires_grad)
   print(f"✅ Modèle créé : {num_params:,} paramètres")
   print(f"📊 Architecture : Conv2D (16→32→64→128) + Flatten + FC(128→4)")

.. note::

   📊 **Taille du modèle**

   Ce modèle a environ **3.3 millions** de paramètres (principalement dans la première couche FC ``128*14*14 → 128``). C'est bien plus petit que Faster R-CNN (``>40M``) ou YOLO qui sont plus génériques.

.. slide::

6.2. Loss et optimiseur
~~~~~~~~~~~~~~~~~~~

**Loss MSE** pour les coordonnées normalisées (x_center, y_center, width, height) + **préparation des targets**.

.. code-block:: python

   import torch.optim as optim
   # Loss simple : MSE sur les coordonnées
   criterion = nn.MSELoss()
   optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)
   
   # Fonction de préparation des targets
   def prepare_single_box_target(targets, img_size=224):
       """
       Convertit les targets (dict) en format (x_center, y_center, w, h) normalisés.
       """
       batch_targets = []
       
       for target in targets:
           # Directement la première (et unique) boîte
           box = target['boxes'][0]  # [x1, y1, x2, y2]
           
           x1, y1, x2, y2 = box
           x_center = ((x1 + x2) / 2) / img_size
           y_center = ((y1 + y2) / 2) / img_size
           width = (x2 - x1) / img_size
           height = (y2 - y1) / img_size
           
           # Clamp dans [0, 1]
           x_center = torch.clamp(x_center, 0, 1)
           y_center = torch.clamp(y_center, 0, 1)
           width = torch.clamp(width, 0, 1)
           height = torch.clamp(height, 0, 1)
           
           batch_targets.append(torch.tensor([x_center, y_center, width, height], device=device))
       
       return torch.stack(batch_targets)

.. note::

   📐 **Normalisation des coordonnées**
   
   - Entrée : boîtes en pixels ``[x1, y1, x2, y2]`` dans ``[0, 224]``.
   - Sortie : coordonnées normalisées ``[x_c, y_c, w, h]`` dans ``[0, 1]``.
   - Le modèle prédit directement ces 4 valeurs normalisées.

.. slide::

6.3. Entraînement (boucles train/val)
~~~~~~~~~~~~~~~~~~~

Boucles simples d'entraînement et d'évaluation.

.. code-block:: python
   
   # Fonctions d'entraînement
   def train_simple_epoch(model, criterion, optimizer, loader):
       model.train()
       total_loss = 0
       
       for images, targets in tqdm(loader, desc="Training"):
           images = torch.stack([img.to(device) for img in images])
           
           # Préparer les targets
           batch_targets = prepare_single_box_target(targets)
           
           # Forward
           preds = model(images)
           loss = criterion(preds, batch_targets)
           
           # Backward
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           total_loss += loss.item() * images.size(0)
       
       return total_loss / len(loader.dataset)
   
   @torch.no_grad()
   def eval_simple_epoch(model, criterion, loader):
       model.eval()
       total_loss = 0
       
       for images, targets in loader:
           images = torch.stack([img.to(device) for img in images])
           batch_targets = prepare_single_box_target(targets)
           
           preds = model(images)
           loss = criterion(preds, batch_targets)
           
           total_loss += loss.item() * images.size(0)
       
       return total_loss / len(loader.dataset)
   
.. slide::

**Lancer l'entraînement** :

.. code-block:: python
   print("\n🚀 Entraînement du CNN simple...\n")
   
   num_epochs = 20
   best_val = float('inf')
   history = {'train_loss': [], 'val_loss': []}
   
   for epoch in range(num_epochs):
       train_loss = train_simple_epoch(simple_model, criterion, optimizer, train_loader)
       val_loss = eval_simple_epoch(simple_model, criterion, val_loader)

       history['train_loss'].append(train_loss)
       history['val_loss'].append(val_loss)
       
       print(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
       
       if val_loss < best_val:
           best_val = val_loss
           torch.save(simple_model.state_dict(), 'simple_bbox_regressor.pth')
           print("  ✅ Meilleur modèle sauvegardé!")
   
   print("\n✅ Entraînement terminé!")
   print(f"📁 Modèle sauvegardé : simple_bbox_regressor.pth")

.. note::
   
   Avec ce modèle simple, vous devriez voir la loss descendre rapidement (à partir de l'epoch 5). Si la loss ne descend pas, vérifiez que vos données sont bien normalisées.

.. slide::
**Visualiser la loss** :

.. code-block:: python

   import matplotlib.pyplot as plt

   # Courbe d'apprentissage
   plt.figure(figsize=(10, 5))
   plt.plot(history['train_loss'], label='Train Loss', marker='o')
   plt.plot(history['val_loss'], label='Val Loss', marker='s')
   plt.xlabel('Epoch')
   plt.ylabel('Loss (MSE)')
   plt.title('Courbe d\'apprentissage')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

   print(f"📊 Loss finale - Train: {history['train_loss'][-1]:.4f} | Val: {history['val_loss'][-1]:.4f}")

.. slide::

6.4. Évaluation sur tout le test data
~~~~~~~~~~~~~~~~~~~

Calcul de l'**IoU moyen** (Intersection over Union) sur le test set.

.. code-block:: python

   def calculate_iou(box1, box2):
       """
       Calcule l'IoU (Intersection over Union) entre deux boîtes.

       Args:
           box1, box2: tensors ou arrays de forme [x1, y1, x2, y2]

       Returns:
           iou: float entre 0 et 1
       """
       # Convertir en numpy si nécessaire
       if torch.is_tensor(box1):
           box1 = box1.numpy()
       if torch.is_tensor(box2):
           box2 = box2.numpy()

       # Calculer l'intersection
       x1_inter = max(box1[0], box2[0])
       y1_inter = max(box1[1], box2[1])
       x2_inter = min(box1[2], box2[2])
       y2_inter = min(box1[3], box2[3])

       # Aire de l'intersection
       inter_width = max(0, x2_inter - x1_inter)
       inter_height = max(0, y2_inter - y1_inter)
       inter_area = inter_width * inter_height

       # Aire de chaque boîte
       box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
       box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

       # Aire de l'union
       union_area = box1_area + box2_area - inter_area

       # IoU
       if union_area == 0:
           return 0.0

       iou = inter_area / union_area
       return iou


   @torch.no_grad()
   def evaluate_all_dataset(model, dataset, img_size=224, iou_threshold=0.5):
       """
       Évalue le modèle sur toutes les images du dataset.

       Args:
           model: modèle PyTorch
           dataset: dataset PyTorch
           img_size: taille des images (224x224)
           iou_threshold: seuil pour considérer une détection comme correcte

       Returns:
           dict avec les métriques (IoU moyen, précision, etc.)
       """
       model.eval()

       ious = []
       correct_detections = 0
       total_images = len(dataset)

       print(f"📊 Évaluation sur {total_images} images...\n")

       for i in tqdm(range(total_images), desc="Évaluation"):
           img, target = dataset[i]

           # Prédiction
           img_batch = img.unsqueeze(0).to(device)
           pred = model(img_batch)[0].cpu()

           # Convertir la prédiction en format [x1, y1, x2, y2]
           x_c, y_c, w, h = pred.numpy()
           x1_pred = (x_c - w/2) * img_size
           y1_pred = (y_c - h/2) * img_size
           x2_pred = (x_c + w/2) * img_size
           y2_pred = (y_c + h/2) * img_size
           pred_box = np.array([x1_pred, y1_pred, x2_pred, y2_pred])

           # Si il y a une ground truth
           if len(target['boxes']) > 0:
               gt_box = target['boxes'][0].numpy()

               # Calculer l'IoU
               iou = calculate_iou(gt_box, pred_box)
               ious.append(iou)

               # Compter comme correct si IoU > threshold
               if iou >= iou_threshold:
                   correct_detections += 1

       # Calculer les métriques
       mean_iou = np.mean(ious) if ious else 0.0
       precision = correct_detections / total_images if total_images > 0 else 0.0

       results = {
           'mean_iou': mean_iou,
           'precision': precision,
           'correct_detections': correct_detections,
           'total_images': total_images,
           'iou_threshold': iou_threshold,
           'all_ious': ious
       }

       return results


   def print_evaluation_results(results):
       """Affiche les résultats d'évaluation de manière lisible."""
       print("\n" + "="*60)
       print("📊 RÉSULTATS DE L'ÉVALUATION")
       print("="*60)
       print(f"\n📈 Métriques globales :")
       print(f"   • IoU moyen            : {results['mean_iou']:.4f} ({results['mean_iou']*100:.2f}%)")
       print(f"   • Précision            : {results['precision']:.4f} ({results['precision']*100:.2f}%)")
       print(f"   • Seuil IoU            : {results['iou_threshold']}")
       print(f"\n✅ Détections correctes  : {results['correct_detections']} / {results['total_images']}")
       print(f"❌ Détections incorrectes : {results['total_images'] - results['correct_detections']} / {results['total_images']}")

       # Distribution des IoU
       ious = results['all_ious']
       if ious:
           print(f"\n📊 Distribution des IoU :")
           print(f"   • Min  : {min(ious):.4f}")
           print(f"   • Max  : {max(ious):.4f}")
           print(f"   • Médiane : {np.median(ious):.4f}")
           print(f"   • Écart-type : {np.std(ious):.4f}")

       print("="*60 + "\n")


   # Évaluer sur le test set
   print("🎯 Évaluation complète du modèle sur le test set\n")
   test_results = evaluate_all_dataset(simple_model, test_dataset, img_size=224, iou_threshold=0.5)
   print_evaluation_results(test_results)

.. note::

   📈 **Interprétation de l'IoU**

   - IoU $$> 0.5$$ : Bonne détection
   - IoU $$> 0.75$$ : Très bonne détection
   - IoU $$> 0.9$$ : Détection quasi-parfaite

   Un modèle bien entraîné sur ce dataset simple devrait obtenir un IoU moyen $$> 0.8$$.

.. slide::

6.5. Visualisation
~~~~~~~~~~~~~~~~~~~

Affichage des prédictions sur une grille d'images avec GT (vert) et prédictions (rouge).

.. code-block:: python

   @torch.no_grad()
   def visualize_best_worst_predictions(model, dataset, results, img_size=224, num_samples=4):
       """
       Affiche les meilleures et pires prédictions du modèle.

       Args:
           model: modèle PyTorch
           dataset: dataset PyTorch
           results: résultats de l'évaluation (dict)
           img_size: taille des images
           num_samples: nombre d'exemples à afficher pour chaque catégorie
       """
       model.eval()

       # Trier les images par IoU
       ious = results['all_ious']
       sorted_indices = np.argsort(ious)

       # Indices des meilleures et pires prédictions
       best_indices = sorted_indices[-num_samples:][::-1]  # Les N meilleures
       worst_indices = sorted_indices[:num_samples]  # Les N pires

       # Créer la figure
       fig, axes = plt.subplots(2, num_samples, figsize=(20, 10))

       # Afficher les meilleures prédictions
       print("✅ MEILLEURES PRÉDICTIONS :")
       for i, idx in enumerate(best_indices):
           img, target = dataset[idx]

           # Prédiction
           img_batch = img.unsqueeze(0).to(device)
           pred = model(img_batch)[0].cpu()

           # Convertir l'image pour affichage
           img_np = img.permute(1, 2, 0).numpy()

           ax = axes[0, i]
           ax.imshow(img_np)
           ax.axis('off')

           # Dessiner la GT en vert
           if len(target['boxes']) > 0:
               box_gt = target['boxes'][0].numpy()
               x1, y1, x2, y2 = box_gt
               width_gt = x2 - x1
               height_gt = y2 - y1

               rect_gt = patches.Rectangle(
                   (x1, y1), width_gt, height_gt,
                   linewidth=2, edgecolor='green', facecolor='none',
                   label='GT'
               )
               ax.add_patch(rect_gt)

           # Dessiner la prédiction en rouge
           x_c, y_c, w, h = pred.numpy()
           x1_pred = (x_c - w/2) * img_size
           y1_pred = (y_c - h/2) * img_size
           width_pred = w * img_size
           height_pred = h * img_size

           rect_pred = patches.Rectangle(
               (x1_pred, y1_pred), width_pred, height_pred,
               linewidth=2, edgecolor='red', facecolor='none',
               linestyle='--', label='Pred'
           )
           ax.add_patch(rect_pred)

           iou_val = ious[idx]
           ax.set_title(f'IoU: {iou_val:.3f}', fontsize=12, color='green', fontweight='bold')
           ax.legend(loc='upper right', fontsize=8)

           print(f"   Image {idx}: IoU = {iou_val:.4f}")

       # Afficher les pires prédictions
       print("\n❌ PIRES PRÉDICTIONS :")
       for i, idx in enumerate(worst_indices):
           img, target = dataset[idx]

           # Prédiction
           img_batch = img.unsqueeze(0).to(device)
           pred = model(img_batch)[0].cpu()

           # Convertir l'image pour affichage
           img_np = img.permute(1, 2, 0).numpy()

           ax = axes[1, i]
           ax.imshow(img_np)
           ax.axis('off')

           # Dessiner la GT en vert
           if len(target['boxes']) > 0:
               box_gt = target['boxes'][0].numpy()
               x1, y1, x2, y2 = box_gt
               width_gt = x2 - x1
               height_gt = y2 - y1

               rect_gt = patches.Rectangle(
                   (x1, y1), width_gt, height_gt,
                   linewidth=2, edgecolor='green', facecolor='none',
                   label='GT'
               )
               ax.add_patch(rect_gt)

           # Dessiner la prédiction en rouge
           x_c, y_c, w, h = pred.numpy()
           x1_pred = (x_c - w/2) * img_size
           y1_pred = (y_c - h/2) * img_size
           width_pred = w * img_size
           height_pred = h * img_size

           rect_pred = patches.Rectangle(
               (x1_pred, y1_pred), width_pred, height_pred,
               linewidth=2, edgecolor='red', facecolor='none',
               linestyle='--', label='Pred'
           )
           ax.add_patch(rect_pred)

           iou_val = ious[idx]
           ax.set_title(f'IoU: {iou_val:.3f}', fontsize=12, color='red', fontweight='bold')
           ax.legend(loc='upper right', fontsize=8)

           print(f"   Image {idx}: IoU = {iou_val:.4f}")

       # Titres des lignes
       axes[0, 0].text(-50, img_size/2, '✅ BEST', rotation=90, 
           fontsize=16, fontweight='bold', color='green',
           va='center', ha='center')
       axes[1, 0].text(-50, img_size/2, '❌ WORST', rotation=90, 
           fontsize=16, fontweight='bold', color='red',
           va='center', ha='center')

       plt.tight_layout()
       plt.show()


   # Visualiser les meilleures et pires prédictions
   print("🎯 Visualisation des meilleures et pires prédictions\n")
   visualize_best_worst_predictions(simple_model, test_dataset, test_results, num_samples=4)


.. note::

   🎨 **Légende**
   
   - **Vert** : Ground truth (annotation réelle)
   - **Rouge** (pointillé) : Prédiction du modèle
   
   Si les boîtes se superposent bien, le modèle fonctionne correctement !

.. slide::

📖 7. Entraînement avec YOLO sur dataset existant
----------------------

Nous allons maintenant utiliser **YOLOv11** (Ultralytics) pour entraîner un détecteur sur un dataset standard. YOLO (You Only Look Once) est un modèle utilisé pour la détection d'objets rapide et efficace, parfait pour la détection en temps réel.

7.1. Introduction à YOLO
~~~~~~~~~~~~~~~~~~~

**YOLO** découpe l'image en une **grille** de cellules et, pour chaque **cellule**, prédit en une seule passe :

- **une boîte englobante candidate** : YOLO11 prédit directement les distances entre le centre de la cellule et les 4 bords de la boîte (gauche, haut, droite, bas) ;
- **un score pour chaque classe** : la probabilité que la boîte contienne un objet de cette classe. Le plus grand de ces scores sert de **score de confiance** de la boîte.

Pour détecter des objets de tailles très différentes, YOLO11 utilise **3 grilles** plus ou moins fines. Pour une image $$640×640$$ :

- une grille $$80×80$$ (cellules de 8 pixels) pour les petits objets,
- une grille $$40×40$$ (cellules de 16 pixels) pour les objets moyens,
- une grille $$20×20$$ (cellules de 32 pixels) pour les gros objets.

➡️ **Au total** : $$80×80 + 40×40 + 20×20 = 8400$$ **boîtes candidates** par image ! Avec le dataset COCO (80 classes), chaque boîte candidate est donc décrite par $$4 + 80 = 84$$ valeurs.

**Avantages de YOLO :**

- ✅ **Rapide** : 30-80 FPS (temps réel)
- ✅ **One-stage** : prédiction directe
- ✅ **Précis** : performances supérieures à Faster R-CNN
- ✅ **Facile à utiliser** : librairie Ultralytics assez simple

**YOLO11** (2024) améliore YOLOv8 (2023) : son architecture est plus optimisée, il est plus précis tout en étant plus rapide.

.. note::

   📚 **Ressources YOLO**
   
   - Documentation officielle : https://docs.ultralytics.com/
   - GitHub : https://github.com/ultralytics/ultralytics
   - Papier YOLOv11 (2024) : https://arxiv.org/abs/2410.17725
   - Papier YOLOv1 original (2015) : https://arxiv.org/abs/1506.02640

.. slide::

7.2. Concepts clés : anchors et NMS
~~~~~~~~~~~~~~~~~~~

**C'est quoi un anchor (ancre) ?**

Dans plusieurs anciennes versions de YOLO (par exemple YOLOv2, YOLOv3, YOLOv5 ou YOLOv7), chaque cellule ne prédisait pas ses boîtes « à partir de rien » : elle partait de plusieurs **anchors**, des boîtes de référence prédéfinies avec des proportions différentes, par exemple :

- un petit carré ($$0.2 × 0.2$$ de l'image) pour les petits objets,
- un rectangle vertical ($$0.1 × 0.3$$) pour les personnes debout,
- un rectangle horizontal ($$0.4 × 0.2$$) pour les voitures.

Le modèle **ajustait** ces anchors (en les décalant et en les redimensionnant) pour coller aux objets réels. Inconvénient : il fallait choisir les tailles des anchors en fonction du dataset.

Depuis YOLOv8, YOLO est **sans anchor** (*anchor-free*) : comme on l'a vu dans la section 7.1, YOLO11 prédit directement la boîte à partir du centre de chaque cellule. C'est plus simple, et le modèle s'adapte mieux à des objets de formes inhabituelles.

.. slide::

**🧹 C'est quoi le NMS (Non-Maximum Suppression) ?**

Problème : Plusieurs boîtes détectent souvent le **même objet** (ex: 5 boîtes qui se chevauchent sur une voiture).

Avant le NMS, on élimine d'abord les boîtes candidates dont le score de confiance est trop faible (paramètre ``conf``, par exemple 0.25) : la grande majorité des 8 400 boîtes candidates ne contiennent aucun objet.

**NMS élimine ensuite les doublons en 4 étapes :**

1. **Trier** les boîtes restantes par score de confiance décroissant

2. **Garder** la boîte avec le meilleur score

3. **Supprimer** toutes les boîtes de la même classe qui se chevauchent trop (IoU > seuil, ex: 0.5) avec la boîte gardée

4. Répéter pour les boîtes restantes

**Exemple :**

- Au départ : 8 400 boîtes candidates
- Après le seuil de confiance : quelques dizaines de boîtes

- Après NMS : 3-10 détections finales (les meilleures, sans doublons)

Le modèle filtre ainsi avec **NMS** pour garder les meilleures détections sans redondance.

.. slide::

7.3. Installation de YOLOv11 (Ultralytics)
~~~~~~~~~~~~~~~~~~~

Installation simple via pip :

.. code-block:: python

   # Installer Ultralytics (inclut YOLOv11)
   !pip install ultralytics
   
   # Imports
   from ultralytics import YOLO
   import torch
   
   print(f"✅ Ultralytics installé !")
   print(f"🔥 PyTorch version: {torch.__version__}")
   print(f"🎮 CUDA disponible: {torch.cuda.is_available()}")

.. note::

   💡 **Versions compatibles**
   
   - Python ≥ 3.8
   - PyTorch ≥ 1.8
   - Ultralytics maintient automatiquement les dépendances

.. slide::

7.4. Dataset COCO (Common Objects in Context)
~~~~~~~~~~~~~~~~~~~

**COCO** est le dataset de référence pour la détection d'objets :

- **80 classes** d'objets courants (personne, voiture, chien, etc.)
- **118 000 images** d'entraînement (COCO complet)
- **5 000 images** de validation
- Annotations au format JSON (boîtes + segmentation)

**Pour ce cours, nous utilisons COCO128**, une version réduite avec seulement 128 images, car :

- ✅ Téléchargement rapide (6.8 Mo au lieu de ~20 Go)
- ✅ Entraînement rapide (2-3 min au lieu de 6-10h)
- ✅ Parfait pour apprendre et tester

.. note::

   📊 **Classes COCO (extrait)**
   
   0: person, 1: bicycle, 2: car, 3: motorcycle, ... 5: bus, ... 7: truck, ... 15: bird, 16: cat, 17: dog, ... 39: bottle, ... 41: cup, ... 56: chair, ...


.. slide::

7.5. Entraînement YOLOv11 sur COCO128
~~~~~~~~~~~~~~~~~~~

**7.5.1. Choisir et charger le modèle**

YOLOv11 propose plusieurs tailles. Nous utilisons **YOLOv11n (Nano)** pour le cours car il est rapide :

.. code-block:: python
   
   # Charger YOLOv11 Nano (le plus rapide)
   model = YOLO('yolo11n.pt')
   
   print(f"✅ Modèle YOLOv11n chargé (2.6M paramètres, 80+ FPS)")

.. note::

   📦 **Autres modèles disponibles** (pour information)
   
   - ``yolo11n.pt`` : Nano (2.6M params, 80+ FPS) ← **on utilise celui-ci**
   - ``yolo11s.pt`` : Small (9M params, 60 FPS)
   - ``yolo11m.pt`` : Medium (20M params, 45 FPS)
   - ``yolo11l.pt`` : Large (26M params, 35 FPS)
   - ``yolo11x.pt`` : XLarge (57M params, 30 FPS)

.. slide::

**7.5.2. Télécharger COCO128**

Téléchargez le dataset COCO128 via Ultralytics :

.. code-block:: python

   from ultralytics.data.utils import check_det_dataset
   
   # Télécharger COCO128 (6.8 Mo, 128 images)
   print("📥 Téléchargement de COCO128 (6.8 Mo)...")
   data_dict = check_det_dataset('coco128.yaml', autodownload=True)
   print(f"✅ Dataset téléchargé : {data_dict['path']}")

.. note::

   💾 **COCO128 : 128 images, 80 classes possibles**
   
   - **128 images** : le nombre d'images dans le dataset
   - **80 classes** : les types d'objets que le modèle peut détecter (person, car, dog, etc.)
   - **20-30 classes présentes** : seulement ces classes apparaissent dans les 128 images
   - Dataset téléchargé dans : ``./datasets/coco128/``

.. slide::

**7.5.3. Lancer l'entraînement**

.. code-block:: python

   # Entraîner YOLOv11n sur COCO128
   results = model.train(
       data='coco128.yaml',        # COCO128 (128 images)
       epochs=3,                   # 3 epochs pour le cours (rapide)
       imgsz=640,                  # Taille des images
       batch=16,                   # Batch size (ajuster selon votre GPU)
       device=0,                   # GPU 0 (ou 'cpu' sans GPU)
       project='runs/detect',      # Dossier de sortie
       name='yolo11_coco128'       # Nom de l'expérience
   )
   
   print(f"✅ Entraînement terminé !")
   print(f"📁 Résultats : runs/detect/yolo11_coco128/")

.. note::

   ⏱️ **Temps d'entraînement**
   
   - **COCO128** (128 images, 3 epochs) : ~5-10 minutes sur GPU
   - **COCO complet** (118k images, 50 epochs) : ~5-10 heures sur GPU
   
   Pour ce cours, COCO128 suffit amplement pour comprendre le fonctionnement !

.. slide::

**7.5.4. Visualiser les résultats de l'entraînement**

Ultralytics génère automatiquement plusieurs fichiers de résultats dans ``runs/detect/yolo11_coco128/`` :

- **results.png** : graphiques avec toutes les courbes (loss, mAP, etc.)
- **Courbes de loss** (train/val)
- **Métriques mAP** (mean Average Precision)
- **Exemples de prédictions**

.. code-block:: python

   # Afficher les résultats de l'entraînement
   from IPython.display import Image, display
   
   # Afficher la courbe de loss
   results_path = 'runs/detect/yolo11_coco128/results.png'
   try:
       print(f"📊 Affichage des courbes d'entraînement YOLO\n")
       display(Image(filename=results_path))
       print(f"\n✅ Graphiques chargés depuis : {results_path}")
   except FileNotFoundError:
       print(f"⚠️ Fichier non trouvé : {results_path}")
       print("   Les résultats seront disponibles après l'entraînement.")

.. slide::

**7.5.5. Pour aller plus loin : COCO complet (optionnel)**

Si vous voulez entraîner sur le dataset complet après avoir testé avec COCO128 :

.. code-block:: python

   # Télécharger COCO complet (~20 Go, peut prendre 30-60 min)
   # print("📥 Téléchargement de COCO complet (~20 Go)...")
   # data_dict = check_det_dataset('coco.yaml', autodownload=True)
   
   # Entraîner sur COCO complet (plusieurs heures)
   # results = model.train(
   #     data='coco.yaml',         # COCO complet (118k images)
   #     epochs=50,                # 50 epochs minimum
   #     imgsz=640,
   #     batch=16,
   #     device=0,
   #     project='runs/detect',
   #     name='yolo11_coco_full'
   # )

.. slide::

7.6. Évaluation sur le test set
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Charger le meilleur modèle
   model = YOLO('runs/detect/yolo11_coco128/weights/best.pt')
   print("✅ Modèle chargé !")
   
   # Évaluer sur le validation set
   metrics = model.val()

   print(f"📊 mAP@0.5: {metrics.box.map50:.3f}")
   print(f"📊 mAP@0.5:0.95: {metrics.box.map:.3f}")
   print(f"📊 Precision: {metrics.box.mp:.3f}")
   print(f"📊 Recall: {metrics.box.mr:.3f}")

.. note::

   📈 **Métriques COCO**
   
   - **mAP@0.5** : Précision moyenne avec seuil IoU=0.5
   - **mAP@0.5:0.95** : Précision moyenne sur plusieurs seuils (standard COCO)
   - **Objectif** : mAP@0.5:0.95 > 0.40 pour un bon modèle


.. slide::

7.7. Inférence et visualisation
~~~~~~~~~~~~~~~~~~~

Une fois le modèle entraîné, vous pouvez l'utiliser pour détecter des objets dans de nouvelles images.

**Étape 1 : Faire une prédiction sur une image**

.. code-block:: python

   # Prédiction sur une image
   results = model.predict(
       source='path/to/image.jpg',  # Chemin vers votre image, dossier, vidéo, ou URL.
       conf=0.5,                    # Seuil de confiance minimum. Le modèle ne garde que les détections avec une confiance ≥ 50 %.
       iou=0.45,                    # Seuil NMS (élimination des doublons). Élimine les boîtes qui se chevauchent trop (IoU > 45 %) pour éviter les doublons.
       show=False,                  # Ne pas afficher automatiquement
       save=False                   # Ne pas sauvegarder automatiquement
   )

.. slide::

**Étape 2 : Extraire les résultats**

.. code-block:: python

   # Récupérer les résultats de la première image
   result = results[0]
   
   # Extraire les informations des détections
   boxes = result.boxes.xyxy.cpu().numpy()    # Coordonnées [x1, y1, x2, y2] en pixels
   confs = result.boxes.conf.cpu().numpy()    # Confiances [0-1]
   classes = result.boxes.cls.cpu().numpy()   # IDs des classes détectées
   
   print(f"🎯 {len(boxes)} objets détectés !")
   
   # Afficher les détails de chaque détection
   for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
       x1, y1, x2, y2 = box
       class_name = model.names[int(cls)]  # Nom de la classe
       print(f"  Objet {i+1}: {class_name} (confiance: {conf:.2f})")

.. slide::

**Étape 3 : Visualiser les détections**

.. code-block:: python

   from matplotlib import pyplot as plt
   
   # Ultralytics dessine automatiquement les boîtes avec labels
   # La méthode ``result.plot()`` dessine automatiquement : les boîtes englobantes avec couleurs par classe, les noms des classes et les scores de confiance.
   img_with_boxes = result.plot()  # Image numpy avec boîtes dessinées
   
   plt.figure(figsize=(12, 8))
   plt.imshow(img_with_boxes)
   plt.axis('off')
   plt.title(f'{len(boxes)} objets détectés')
   # L'image s'affiche automatiquement dans le notebook

.. slide::

**Étape 4 : Visualisation sur plusieurs images**

.. code-block:: python

   # Prédire sur un dossier
   results = model.predict(
       source='datasets/coco/images/val2017/',
       conf=0.5,
       save=True,            # Sauvegarder les images annotées
       project='runs/detect',
       name='predictions'
   )
   
   print(f"✅ Prédictions sauvegardées dans runs/detect/predictions/")


.. slide::

📖 8. Entraîner YOLO sur votre dataset personnalisé
-----------

Maintenant, entraînons **YOLO** sur le même dataset personnalisé que vous avez créé avec ``SimpleBBoxRegressor`` pour comparer les performances !

**Rappel** : vous avez déjà créé un dataset avec :

- Des images de votre objet (cube, balle, voiture, etc.)
- Annotations Label Studio au format YOLO exportées (``images/``, ``labels/``, ``classes.txt``)
- Un Dataset PyTorch ``YOLODetectionDataset``
- Un split train/val/test avec ``random_split`` (seed=42)

.. slide::

8.1. Préparer le dataset pour l'entraînement YOLO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contrairement à ``SimpleBBoxRegressor`` qui charge les données via PyTorch Dataset, **YOLO (Ultralytics)** utilise une structure de dossiers spécifique. Nous allons organiser le dataset existant pour YOLO tout en **conservant exactement le même split** que ``SimpleBBoxRegressor``.

**8.1.1. Structure requise par YOLO**

YOLO attend cette organisation :

.. code-block:: text

   data_yolo/
   ├── images/
   │   ├── train/           # Images d'entraînement
   │   ├── val/             # Images de validation
   │   └── test/            # Images de test
   ├── labels/
   │   ├── train/           # Labels d'entraînement (.txt)
   │   ├── val/             # Labels de validation (.txt)
   │   └── test/            # Labels de test (.txt)
   └── dataset.yaml         # Fichier de configuration

.. slide::

**8.1.2. Script pour réorganiser le dataset**

Créons une fonction qui réutilise **le même split** que ``SimpleBBoxRegressor`` :

.. code-block:: python

   import torch
   import shutil
   from pathlib import Path
   from torch.utils.data import random_split

   def prepare_yolo_from_existing_dataset(
       images_dir, 
       labels_dir, 
       classes_file,
       output_dir='data_yolo',
       seed=42,
       train_ratio=0.70,
       val_ratio=0.15
   ):
       """
       Réorganise un dataset YOLO existant pour l'entraînement YOLO Ultralytics.
       Utilise le MÊME split que SimpleBBoxRegressor (seed=42).
       
       Args:
           images_dir: Dossier contenant toutes les images (ex: 'dataset_yolo/images')
           labels_dir: Dossier contenant tous les labels .txt (ex: 'dataset_yolo/labels')
           classes_file: Fichier classes.txt (ex: 'dataset_yolo/classes.txt')
           output_dir: Dossier de sortie pour la structure YOLO (défaut: 'data_yolo')
           seed: Seed pour reproductibilité (défaut: 42, MÊME que SimpleBBoxRegressor)
           train_ratio: Proportion du train set (défaut: 0.70)
           val_ratio: Proportion du val set (défaut: 0.15)
       
       Returns:
           Path vers le dossier output_dir, liste des classes
       """
       images_dir = Path(images_dir)
       labels_dir = Path(labels_dir)
       output_dir = Path(output_dir)
       classes_file = Path(classes_file)
       
       print(f"🔄 Préparation du dataset YOLO depuis : {images_dir}")
       
       # 1. Créer la structure de dossiers pour YOLO
       for split in ['train', 'val', 'test']:
           (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
           (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
       
       # 2. Récupérer toutes les images
       image_files = sorted(list(images_dir.glob('*.jpg')) + 
                           list(images_dir.glob('*.jpeg')) + 
                           list(images_dir.glob('*.png')))
       
       print(f"📁 {len(image_files)} images trouvées")
       
       if len(image_files) == 0:
           print("❌ Aucune image trouvée ! Vérifiez le chemin.")
           return None, []
       
       # 3. Créer le MÊME split que SimpleBBoxRegressor
       total_size = len(image_files)
       train_size = int(train_ratio * total_size)
       val_size = int(val_ratio * total_size)
       test_size = total_size - train_size - val_size
       
       train_indices, val_indices, test_indices = random_split(
           range(len(image_files)),
           [train_size, val_size, test_size],
           generator=torch.Generator().manual_seed(seed)
       )
       
       print(f"📊 Split (seed={seed}) : {train_size} train, {val_size} val, {test_size} test")
       
       # 4. Copier les fichiers dans les bons dossiers
       splits = {
           'train': train_indices.indices,
           'val': val_indices.indices,
           'test': test_indices.indices
       }
       
       for split_name, indices in splits.items():
           print(f"\n📂 Préparation du split '{split_name}'...")
           
           copied_count = 0
           for idx in indices:
               img_file = image_files[idx]
               label_file = labels_dir / f"{img_file.stem}.txt"
               
               # Copier l'image
               dest_img = output_dir / 'images' / split_name / img_file.name
               shutil.copy(img_file, dest_img)
               
               # Copier le label correspondant (si existe)
               if label_file.exists():
                   dest_label = output_dir / 'labels' / split_name / label_file.name
                   shutil.copy(label_file, dest_label)
                   copied_count += 1
               else:
                   print(f"   ⚠️  Label manquant pour : {img_file.name}")
           
           print(f"   ✅ {copied_count} images + labels copiés")
       
       # 5. Copier le fichier classes.txt à la racine
       shutil.copy(classes_file, output_dir / 'classes.txt')
       
       # 6. Charger les classes pour le fichier YAML
       with open(classes_file, 'r') as f:
           classes = [line.strip() for line in f.readlines()]
       
       print(f"\n📋 Classes ({len(classes)}) : {classes}")
       
       print(f"\n✅ Dataset YOLO préparé dans : {output_dir}")
       print(f"   Structure : images/{{train,val,test}} + labels/{{train,val,test}}")
       
       return output_dir, classes


   # 🎯 UTILISATION
   # Adapter ces chemins selon votre export Label Studio
   output_path, classes = prepare_yolo_from_existing_dataset(
       images_dir='dataset_yolo/images',      # ADAPTEZ : Dossier des images exportées
       labels_dir='dataset_yolo/labels',      # ADAPTEZ : Dossier des labels exportés
       classes_file='dataset_yolo/classes.txt',  # ADAPTEZ : Fichier classes.txt
       output_dir='data_yolo',                # Dossier de sortie
       seed=42                                # MÊME seed que SimpleBBoxRegressor
   )

.. note::

   💡 **Pourquoi seed=42 ?**
   
   - ✅ **Même split** que SimpleBBoxRegressor 
   - ✅ **Comparaison équitable** : YOLO et SimpleBBoxRegressor testés sur **exactement les mêmes images**
   - ✅ Les images du test set sont identiques pour les deux modèles

.. slide::

8.2. Créer le fichier de configuration YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

YOLO nécessite un fichier YAML décrivant l'organisation du dataset. Créons-le automatiquement :

.. code-block:: python

   import yaml
   from pathlib import Path

   def create_yolo_yaml(output_dir, classes, yaml_filename='dataset.yaml'):
       """
       Crée le fichier YAML de configuration pour YOLO.
       
       Args:
           output_dir: Dossier racine du dataset YOLO (ex: 'data_yolo')
           classes: Liste des noms de classes (ex: ['cube', 'bouteille'])
           yaml_filename: Nom du fichier YAML (défaut: 'dataset.yaml')
       
       Returns:
           Path vers le fichier YAML créé
       """
       output_dir = Path(output_dir)
       yaml_path = output_dir / yaml_filename
       
       # Configuration YOLO avec chemins relatifs
       config = {
           'path': str(output_dir.absolute()),  # Chemin absolu vers la racine
           'train': 'images/train',             # Chemin relatif vers images train
           'val': 'images/val',                 # Chemin relatif vers images val
           'test': 'images/test',               # Chemin relatif vers images test
           
           'nc': len(classes),                  # Nombre de classes
           'names': classes                     # Noms des classes
       }
       
       # Écrire le fichier YAML
       with open(yaml_path, 'w') as f:
           yaml.dump(config, f, default_flow_style=False, sort_keys=False)
       
       print(f"\n✅ Fichier YAML créé : {yaml_path}")
       print(f"   Contenu :")
       print(f"      - path: {config['path']}")
       print(f"      - train: {config['train']}")
       print(f"      - val: {config['val']}")
       print(f"      - test: {config['test']}")
       print(f"      - nc: {config['nc']}")
       print(f"      - names: {config['names']}")
       
       return yaml_path


   # 🎯 UTILISATION (après avoir préparé le dataset)
   if output_path and classes:
       yaml_path = create_yolo_yaml(output_path, classes)
       print(f"\n🎯 Fichier de configuration prêt : {yaml_path}")
   else:
       print("❌ Erreur : dataset non préparé correctement")

**Exemple de fichier YAML généré** (``data_yolo/dataset.yaml``) :

.. code-block:: yaml

   path: /chemin/absolu/vers/data_yolo
   train: images/train
   val: images/val
   test: images/test
   nc: 1
   names:
   - cube

.. note::

   💡 **Structure du fichier YAML**
   
   - ``path`` : Chemin absolu vers la racine du dataset
   - ``train/val/test`` : Chemins **relatifs** vers les dossiers d'images
   - ``nc`` : Nombre de classes (calculé automatiquement)
   - ``names`` : Liste des noms de classes (depuis ``classes.txt``)

.. slide::

8.3. Entraîner YOLO sur votre dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maintenant que le dataset est organisé et le fichier YAML créé, lançons l'entraînement YOLO :

.. code-block:: python

   from ultralytics import YOLO
   
   # Charger YOLOv11n pré-entraîné
   model = YOLO('yolo11n.pt')
   
   # Entraîner sur votre dataset
   results = model.train(
       data=str(yaml_path),               # Chemin vers le YAML créé automatiquement
       epochs=50,                         # ADAPTEZ : en fonction du batch et de la taille de la base de données
       imgsz=224,                         # Même taille que SimpleBBoxRegressor
       batch=2,                           # ADAPTEZ : en fonction du nombre d'epoch et de la taille de la base de données
       name='yolo11_my_object',           # ADAPTEZ : avec le nom du dossier de sauvegarde du model
       patience=10,                       # Early stopping
       device=0,                          # GPU (ou 'cpu' si pas de GPU)
       project='runs/detect'              # ADAPTEZ : avec le chemin vers le dossier de sauvegarde du model         
   )
   
   print("✅ Entraînement terminé !")


.. slide::

8.4. Tester YOLO sur le test set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Testons YOLO sur les **15% d'images de test** (jamais vues pendant l'entraînement) - les **mêmes images** que celles utilisées pour tester ``SimpleBBoxRegressor`` :

.. code-block:: python

   from ultralytics import YOLO
   from pathlib import Path
   import os
   
   # Charger le meilleur modèle entraîné
   yolo_model = YOLO('runs/detect/yolo11_my_object/weights/best.pt')  # ADAPTEZ le chemin
   
   # 1. Évaluer sur le test set
   print("🎯 Évaluation de YOLO sur le TEST SET...\n")
   test_metrics = yolo_model.val(
       data=str(yaml_path),  # Utiliser le YAML créé automatiquement
       split='test'
   )
   
   print("\n📊 Métriques YOLO sur le TEST SET :")
   print(f"  mAP@0.5     : {test_metrics.box.map50:.3f}")
   print(f"  mAP@0.5:0.95: {test_metrics.box.map:.3f}")
   print(f"  Precision   : {test_metrics.box.mp:.3f}")
   print(f"  Recall      : {test_metrics.box.mr:.3f}")
   
   # 2. Prédire sur quelques images du test set pour visualisation
   test_images_dir = output_path / 'images' / 'test'
   test_images = sorted(list(test_images_dir.glob('*.jpg')))[:5]  # Prendre 5 images
   
   print(f"\n📸 Prédiction sur {len(test_images)} images de test...")
   
   for img_path in test_images:
       results = yolo_model.predict(
           source=str(img_path),
           conf=0.5,                   # ADAPTEZ : seuil de confiance
           save=True,
           project='runs/detect',
           name='yolo_test_predictions'
       )
       print(f"   ✅ {img_path.name}")
   
   print(f"\n✅ Prédictions sauvegardées dans : runs/detect/yolo_test_predictions/")

.. note::

   📊 **Interprétation des métriques YOLO**
   
   - **mAP@0.5** : Précision moyenne avec IoU ≥ 0.5 (métrique principale)
   - **mAP@0.5:0.95** : Précision moyenne sur plusieurs seuils (plus stricte)
   - **Precision** : Proportion de détections correctes parmi toutes les détections
   - **Recall** : Proportion d'objets réels détectés

.. slide::

8.5. Comparaison SimpleBBoxRegressor vs YOLO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maintenant que nous avons entraîné et testé les deux modèles sur **exactement le même dataset** (même split avec seed=42), comparons leurs performances :

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt

   def compare_models(simple_results, yolo_metrics):
       """
       Compare les performances de SimpleBBoxRegressor et YOLO.
       
       Args:
           simple_results: dict des résultats de SimpleBBoxRegressor (section 6.4)
           yolo_metrics: métriques YOLO retournées par model.val()
       """
       print("\n" + "="*70)
       print("📊 COMPARAISON SimpleBBoxRegressor vs YOLO")
       print("="*70)
       
       # Créer un tableau comparatif
       comparison = pd.DataFrame({
           'Modèle': ['SimpleBBoxRegressor', 'YOLOv11n'],
           'IoU Moyen': [
               simple_results['mean_iou'],
               yolo_metrics.box.map50  # mAP@0.5 est comparable à l'IoU
           ],
           'Précision': [
               simple_results['precision'],
               yolo_metrics.box.mp
           ],
           'Taille (paramètres)': [
               '~3.3M',
               '~2.6M'
           ],
           'Vitesse (relative)': [
               'Rapide',
               'Très rapide'
           ]
       })
       
       print("\n" + comparison.to_string(index=False))
       
       # Graphique comparatif
       fig, axes = plt.subplots(1, 2, figsize=(14, 5))
       
       # Graphique 1 : IoU / mAP
       ax1 = axes[0]
       models = ['SimpleBBox\nRegressor', 'YOLOv11n']
       scores = [simple_results['mean_iou'], yolo_metrics.box.map50]
       colors = ['#3498db', '#e74c3c']
       
       bars1 = ax1.bar(models, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
       ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
       ax1.set_title('IoU Moyen / mAP@0.5', fontsize=14, fontweight='bold')
       ax1.set_ylim(0, 1)
       ax1.grid(axis='y', alpha=0.3)
       
       # Ajouter les valeurs sur les barres
       for bar, score in zip(bars1, scores):
           height = bar.get_height()
           ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
       
       # Graphique 2 : Précision
       ax2 = axes[1]
       precisions = [simple_results['precision'], yolo_metrics.box.mp]
       
       bars2 = ax2.bar(models, precisions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
       ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
       ax2.set_title('Précision', fontsize=14, fontweight='bold')
       ax2.set_ylim(0, 1)
       ax2.grid(axis='y', alpha=0.3)
       
       # Ajouter les valeurs sur les barres
       for bar, prec in zip(bars2, precisions):
           height = bar.get_height()
           ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prec:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
       
       plt.tight_layout()
       plt.show()
       
       # Analyse
       print("\n📈 ANALYSE :")
       
       if simple_results['mean_iou'] > yolo_metrics.box.map50:
           diff = (simple_results['mean_iou'] - yolo_metrics.box.map50) * 100
           print(f"   ✅ SimpleBBoxRegressor gagne en IoU (+{diff:.1f}%)")
       else:
           diff = (yolo_metrics.box.map50 - simple_results['mean_iou']) * 100
           print(f"   ✅ YOLO gagne en mAP (+{diff:.1f}%)")
       
       print("\n💡 RECOMMANDATIONS :")
       print("   • SimpleBBoxRegressor : Parfait pour 1 seul objet, rapide, simple à comprendre")
       print("   • YOLO : Meilleur pour plusieurs objets, plus robuste, plus générique")
       print("   • Pour ce dataset (1 objet) : les deux sont comparables !")
       
       print("="*70 + "\n")


   # 🎯 UTILISATION
   # Comparer les résultats (utilisez les variables des sections précédentes)
   compare_models(test_results, test_metrics)

.. warning::

   ⚠️ **Prérequis pour la comparaison**
   
   Assurez-vous d'avoir exécuté :
   
   1. Section 6.4 : ``test_results = evaluate_all_dataset(simple_model, test_dataset)``
   2. Section 8.4 : ``test_metrics = yolo_model.val(data=str(yaml_path), split='test')``

.. note::

   📊 **Quand utiliser quel modèle ?**
   
   **SimpleBBoxRegressor** :
   
   - ✅ **1 seul objet** par image
   - ✅ Objet **centré** et toujours présent
   - ✅ **Apprentissage** : comprendre les bases de la régression de bbox
   - ✅ Dataset **simple** et contrôlé
   
   **YOLO** :
   
   - ✅ **Plusieurs objets** par image
   - ✅ Objets **multiples** de classes différentes
   - ✅ **Production** : applications réelles, temps réel
   - ✅ **Robustesse** : gère les cas complexes (occlusions, variations)
   - ✅ Dataset **réel** avec variabilité

.. slide::

🏋️ Travaux Pratiques 6
--------------------

.. toctree::

    TP_chap6

