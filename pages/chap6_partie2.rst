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

📖 7. CNN ultra-simple : régression directe de boîte
----------------------

Pour des cas simples avec **1 seul objet par image**, on peut utiliser une approche beaucoup plus simple que YOLO ou Faster R-CNN : **régression directe des coordonnées** de la boîte. Le modèle prédit directement 4 nombres : ``(x_center, y_center, width, height)`` normalisés dans [0,1].

.. note::

   💡 **Quand utiliser cette approche ?**
   
   ✅ **OUI** : 1 objet par image, objet centré, peu de variations (ex: détection de visage, logo)
   
   ❌ **NON** : plusieurs objets, positions variables, objets qui se chevauchent, etc.

.. slide::
    
7.1. Architecture ultra-simple
~~~~~~~~~~~~~~~~~~~

Le modèle est constitué d'un **backbone CNN** (4 couches Conv2D + MaxPool) suivi d'un **head de régression** (2 couches FC) qui prédit directement les 4 coordonnées normalisées. Dans l'exemple, l’entrée $$224×224$$ est réduite **4 fois** par MaxPool(2): $$224→112→56→28→14$$; la carte de features finale est donc $$14×14$$. Si vous changez la taille d’entrée ou le nombre de couches à stride 2, la taille de la grille changera.

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from tqdm import tqdm # Pour les barres de progression

   class SimpleBBoxRegressor(nn.Module):
       """
       CNN ultra-simple qui régresse directement UNE boîte par image.
       Sortie : [x_center, y_center, width, height] normalisés dans [0,1]
       """
       
       def __init__(self):
           super().__init__()
           
           # Backbone simple : Conv2D + MaxPool (comme chapitre 5)
           self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
           self.pool1 = nn.MaxPool2d(2)  # 224 -> 112
           
           self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
           self.pool2 = nn.MaxPool2d(2)  # 112 -> 56
           
           self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
           self.pool3 = nn.MaxPool2d(2)  # 56 -> 28
           
           self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
           self.pool4 = nn.MaxPool2d(2)  # 28 -> 14
           
           # Après 4 MaxPool: 224→112→56→28→14
           # Taille finale: [B, 128, 14, 14]
           
           # Head de régression : 4 sorties (x, y, w, h)
           self.fc1 = nn.Linear(128 * 14 * 14, 128)
           self.fc2 = nn.Linear(128, 4)  # x_center, y_center, width, height
       
       def forward(self, x):
           # Backbone
           x = self.pool1(F.relu(self.conv1(x)))
           x = self.pool2(F.relu(self.conv2(x)))
           x = self.pool3(F.relu(self.conv3(x)))
           x = self.pool4(F.relu(self.conv4(x)))
           
           # Flatten
           x = x.view(x.size(0), -1)  # [B, 128*14*14]
           
           # Régression
           x = F.relu(self.fc1(x))
           x = torch.sigmoid(self.fc2(x))  # Sortie dans [0, 1]
           
           return x  # [B, 4] : (x_center, y_center, w, h) normalisés


   # Créer le modèle
   simple_model = SimpleBBoxRegressor().to(device)
   num_params = sum(p.numel() for p in simple_model.parameters() if p.requires_grad)
   print(f"✅ Modèle créé : {num_params:,} paramètres")
   print(f"📊 Architecture : Conv2D (16→32→64→128) + Flatten + FC(128→4)")

.. note::

   📊 **Taille du modèle**

   Ce modèle a environ **25 millions** de paramètres (principalement dans la première couche FC ``128*14*14 → 128``). C'est bien plus petit que Faster R-CNN (``>40M``) qui est plus générique.

.. slide::

7.2. Loss et optimiseur
~~~~~~~~~~~~~~~~~~~

**Loss MSE** pour les coordonnées normalisées (x_center, y_center, width, height) + **préparation des targets**.

.. code-block:: python

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
   
   - Entrée : boîtes en pixels ``[x1, y1, x2, y2]`` dans ``[0, 224]``
   - Sortie : coordonnées normalisées ``[x_c, y_c, w, h]`` dans ``[0, 1]``
   - Le modèle prédit directement ces 4 valeurs normalisées

.. slide::

7.3. Entraînement (boucles train/val)
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
   
   # LANCER L'ENTRAÎNEMENT
   print("\n🚀 Entraînement du CNN simple...\n")
   
   num_epochs = 20
   best_val = float('inf')
   
   for epoch in range(num_epochs):
       train_loss = train_simple_epoch(simple_model, criterion, optimizer, train_loader)
       val_loss = eval_simple_epoch(simple_model, criterion, val_loader)
       
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

7.4. Évaluation sur tout le test data
~~~~~~~~~~~~~~~~~~~

Calcul de l'**IoU moyen** (Intersection over Union) sur le test set.

.. code-block:: python

   # Évaluation sur TOUT le test set
   print(f"\n📊 ÉVALUATION SUR TOUT LE TEST SET ({len(test_dataset)} images)")
   print("="*60)
   
   @torch.no_grad()
   def evaluate_on_test(model, dataset, img_size=224):
       model.eval()
       
       total_iou = 0
       num_samples = len(dataset)
       
       for idx in tqdm(range(num_samples), desc="Évaluation"):
           img, target = dataset[idx]
           img_tensor = img.unsqueeze(0).to(device)
           
           # Prédiction
           pred = model(img_tensor)[0].cpu()
           
           # Convertir en [x1, y1, x2, y2] pixels
           xc, yc, w, h = pred
           pred_x1 = (xc - w/2) * img_size
           pred_y1 = (yc - h/2) * img_size
           pred_x2 = (xc + w/2) * img_size
           pred_y2 = (yc + h/2) * img_size
           
           # GT (prendre la première boîte)
           if len(target['boxes']) > 0:
               gt_box = target['boxes'][0]
               gt_x1, gt_y1, gt_x2, gt_y2 = gt_box.tolist()
               
               # Calculer IoU
               x1_inter = max(pred_x1.item(), gt_x1)
               y1_inter = max(pred_y1.item(), gt_y1)
               x2_inter = min(pred_x2.item(), gt_x2)
               y2_inter = min(pred_y2.item(), gt_y2)
               
               if x2_inter > x1_inter and y2_inter > y1_inter:
                   inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                   pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                   gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                   union = pred_area + gt_area - inter
                   iou = inter / (union + 1e-6)
                   total_iou += iou.item()
       
       mean_iou = total_iou / num_samples
       print(f"\n📈 IoU moyen sur le test set : {mean_iou:.3f}")
       return mean_iou
   
   # Évaluer
   mean_iou = evaluate_on_test(simple_model, test_dataset)

.. note::

   📈 **Interprétation de l'IoU**

   - IoU $$> 0.5$$ : Bonne détection
   - IoU $$> 0.75$$ : Très bonne détection
   - IoU $$> 0.9$$ : Détection quasi-parfaite

   Un modèle bien entraîné sur ce dataset simple devrait obtenir un IoU moyen $$> 0.8$$.

.. slide::

7.5. Visualisation
~~~~~~~~~~~~~~~~~~~

Affichage des prédictions sur une grille d'images avec GT (vert) et prédictions (rouge).

.. code-block:: python

   # Visualisation de quelques exemples
   print(f"\n🖼️  VISUALISATION D'EXEMPLES")
   print("="*60)
   
   import matplotlib.pyplot as plt
   import matplotlib.patches as patches
   import numpy as np
   
   # Afficher 9 exemples (3x3)
   num_to_show = min(9, len(test_dataset))
   indices = np.linspace(0, len(test_dataset)-1, num_to_show, dtype=int)
   
   fig, axes = plt.subplots(3, 3, figsize=(15, 15))
   axes = axes.flatten()
   
   simple_model.eval()
   
   for plot_idx, test_idx in enumerate(indices):
       img, target = test_dataset[test_idx]
       img_tensor = img.unsqueeze(0).to(device)
       
       # Prédiction
       with torch.no_grad():
           pred = simple_model(img_tensor)[0].cpu()
       
       # Convertir en pixels
       xc, yc, w, h = pred
       pred_x1 = (xc - w/2) * 224
       pred_y1 = (yc - h/2) * 224
       pred_x2 = (xc + w/2) * 224
       pred_y2 = (yc + h/2) * 224
       
       # Affichage
       ax = axes[plot_idx]
       img_np = img.permute(1, 2, 0).cpu().numpy()
       ax.imshow(img_np)
       ax.set_title(f'Test {test_idx}', fontsize=10)
       
       # GT en vert
       for box in target['boxes']:
           x1_gt, y1_gt, x2_gt, y2_gt = box.tolist()
           rect = patches.Rectangle(
               (x1_gt, y1_gt), x2_gt-x1_gt, y2_gt-y1_gt,
               linewidth=2, edgecolor='green', facecolor='none'
           )
           ax.add_patch(rect)
       
       # Prédiction en rouge
       rect = patches.Rectangle(
           (pred_x1.item(), pred_y1.item()), 
           (pred_x2 - pred_x1).item(), 
           (pred_y2 - pred_y1).item(),
           linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
       )
       ax.add_patch(rect)
       
       ax.axis('off')
   
   plt.tight_layout()
   plt.suptitle('Prédictions CNN Simple (Vert=GT, Rouge=Pred)', y=1.002, fontsize=14, weight='bold')
   plt.show()


.. note::

   🎨 **Légende**
   
   - **Vert** : Ground truth (annotation réelle)
   - **Rouge** (pointillé) : Prédiction du modèle
   
   Si les boîtes se superposent bien, le modèle fonctionne correctement !

.. slide::

📖 8. Entraînement avec YOLO sur dataset existant
----------------------

Nous allons maintenant utiliser **YOLOv11** (Ultralytics) pour entraîner un détecteur sur un dataset standard. YOLO (You Only Look Once) est un modèle utilisé pour la détection d'objets rapide et efficace, parfait pour la détection en temps réel.

8.1. Introduction à YOLO
~~~~~~~~~~~~~~~~~~~

**YOLO** divise l'image en une **grille** (ex: $$7×7$$, $$13×13$$, etc.) et pour chaque **cellule** de la grille, prédit :

- **Plusieurs boîtes englobantes candidates** (typiquement 3-9 selon les versions) grâce aux **anchors**
- Chaque boîte est représentée par : **(x, y, w, h)** relatives au centre de la cellule
- **Objectness** : probabilité qu'un objet soit présent dans cette boîte
- **Classes** : probabilités pour chaque classe (si objet détecté)

**Avantages de YOLO :**

- ✅ **Rapide** : 30-80 FPS (temps réel)
- ✅ **One-stage** : prédiction directe
- ✅ **Précis** : performances supérieures à Faster R-CNN
- ✅ **Facile à utiliser** : librairie Ultralytics très simple

**YOLOv11** est la dernière version stable (2024) avec des améliorations significatives par rapport à YOLOv8 (2023) : l'architecture est plus optimisée et la précision est améliorée tout en étant plus rapide. 

.. note::

   📚 **Ressources YOLO**
   
   - Documentation officielle : https://docs.ultralytics.com/
   - GitHub : https://github.com/ultralytics/ultralytics
   - Papier YOLOv11 (2024) : https://arxiv.org/abs/2410.17725
   - Papier YOLOv1 original (2015) : https://arxiv.org/abs/1506.02640

.. slide::

8.2. Concepts clés : Anchors et NMS
~~~~~~~~~~~~~~~~~~~

**C'est quoi un anchor (ancre) ?**

Un anchor est une boîte de référence prédéfinie avec des proportions spécifiques (largeur/hauteur).

**Exemple d'anchors :** 

- Anchor 1 : petit carré ($$0.2 × 0.2$$ de l'image) → pour détecter petits objets

- Anchor 2 : rectangle vertical ($$0.1 × 0.3$$) → pour personnes debout

- Anchor 3 : rectangle horizontal ($$0.4 × 0.2$$) → pour voitures

Le modèle **ajuste** ces anchors (décale et redimensionne) pour coller aux objets réels. C'est plus efficace que de prédire la taille depuis zéro !

➡️ **Au total** : Si grille $$13×13$$ avec 3 anchors par cellule = $$13×13×3$$ = **507 boîtes candidates** par image !

.. slide::

**🧹 C'est quoi le NMS (Non-Maximum Suppression) ?**

Problème : Plusieurs boîtes détectent souvent le **même objet** (ex: 5 boîtes qui se chevauchent sur une voiture).

**NMS élimine les doublons en 3 étapes :**

1. **Trier** les boîtes par score de confiance (objectness) décroissant

2. **Garder** la boîte avec le meilleur score

3. **Supprimer** toutes les boîtes qui se chevauchent trop (IoU > seuil, ex: 0.5) avec la boîte gardée

4. Répéter pour les boîtes restantes

**Exemple :**

- Avant NMS : 507 boîtes candidates

- Après NMS : 3-10 détections finales (les meilleures, sans doublons)

Le modèle filtre ainsi avec **NMS** pour garder les meilleures détections sans redondance.

.. slide::

8.3. Installation de YOLOv11 (Ultralytics)
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

8.4. Dataset COCO (Common Objects in Context)
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

.. note::

   💡 **Format COCO vs Format YOLO - Clarification importante**
   
   Le mot "COCO" désigne **deux choses différentes** :
   
   1. **COCO le dataset** : 118k images avec 80 classes d'objets (person, car, dog...)
   2. **COCO le format d'annotation** : fichier JSON avec boîtes au format ``[x, y, width, height]`` en pixels
   
   **YOLO** utilise son **propre format** : fichiers TXT avec coordonnées normalisées ``[x_center, y_center, w, h]`` (section 4.3)
   
   **Quand on entraîne YOLO sur le dataset COCO :**
   
   - Ultralytics télécharge les annotations COCO (format JSON)
   - Les convertit **automatiquement** en format YOLO (TXT) en interne
   - Vous n'avez **rien à faire manuellement** !
   
   **Pour un dataset custom (section 9) :**
   
   - Vous annotez dans Label Studio
   - Vous exportez directement au **format YOLO** (section 9.1)
   - Pas besoin de passer par le format COCO


.. slide::

8.5. Entraînement YOLOv11 sur COCO128
~~~~~~~~~~~~~~~~~~~

**8.5.1. Choisir et charger le modèle**

YOLOv11 propose plusieurs tailles. Nous utilisons **YOLOv11n (Nano)** pour le cours car il est rapide :

.. code-block:: python
   
   # Charger YOLOv11 Nano (le plus rapide)
   model = YOLO('yolo11n.pt')
   
   print(f"✅ Modèle YOLOv11n chargé (3M paramètres, 80+ FPS)")

.. note::

   📦 **Autres modèles disponibles** (pour information)
   
   - ``yolo11n.pt`` : Nano (3M params, 80+ FPS) ← **on utilise celui-ci**
   - ``yolo11s.pt`` : Small (9M params, 60 FPS)
   - ``yolo11m.pt`` : Medium (20M params, 45 FPS)
   - ``yolo11l.pt`` : Large (26M params, 35 FPS)
   - ``yolo11x.pt`` : XLarge (57M params, 30 FPS)

.. slide::

**8.5.2. Télécharger COCO128**

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
   - **~20-30 classes présentes** : seulement ces classes apparaissent dans les 128 images
   - Dataset téléchargé dans : ``./datasets/coco128/``

.. slide::

**8.5.3. Lancer l'entraînement**

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

**8.5.4. Visualiser les résultats de l'entraînement**

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

**8.5.5. Pour aller plus loin : COCO complet (optionnel)**

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

8.6. Évaluation sur le test set
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

8.7. Inférence et visualisation
~~~~~~~~~~~~~~~~~~~

Une fois le modèle entraîné, vous pouvez l'utiliser pour détecter des objets dans de nouvelles images.

**Étape 1 : Faire une prédiction sur une image**

.. code-block:: python

   # Prédiction sur une image
   results = model.predict(
       source='path/to/image.jpg',  # Chemin vers votre image, dossier, vidéo, ou URL.
       conf=0.5,                    # Seuil de confiance minimum. Le modèle ne garde que les détections avec une confiance $$≥ 50%$$.
       iou=0.45,                    # Seuil NMS (élimination des doublons). Élimine les boîtes qui se chevauchent trop (IoU $$ > 45%$$) pour éviter les doublons.
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

📖 9. Entraîner YOLO sur votre dataset personnalisé
-----------

Maintenant, entraînons **YOLO** sur le même dataset personnalisé que vous avez créé dans les sections 5, 6 et 7 pour comparer avec ``SimpleBBoxRegressor`` !

**Rappel** : vous avez déjà créé un dataset avec :

- Des images de votre objet (cube, balle, voiture, etc.)
- Annotations Label Studio au format JSON
- Un Dataset PyTorch ``LabelStudioDetectionDataset`` (section 6)
- Un split train/val/test avec ``random_split`` (section 6)


.. slide::

9.1. Exporter votre dataset au format YOLO
~~~~~~~~~~~~~~~~~~~~~

Label Studio peut exporter directement au format YOLO !

**Étapes dans Label Studio :**

1. Ouvrez votre projet d'annotation
2. Cliquez sur **"Export"** en haut à droite
3. Dans la liste des formats, vous verrez plusieurs options avec "YOLO". **Choisissez :**
   
   - ✅ **"YOLO"** (tout court) → format standard YOLO
   - ❌ Pas "YOLOv5 PyTorch" (format spécifique YOLOv5)
   - ❌ Pas "YOLOv8 Detection" (format spécifique YOLOv8)

4. Cliquez sur **"Export"** → télécharge un fichier ZIP

.. note::

   💡 **Pourquoi "YOLO" tout court ?**
   
   Le format **"YOLO"** est le format texte standard compatible avec toutes les versions (YOLOv5, YOLOv8, YOLOv11, etc.). Les formats spécifiques (YOLOv5 PyTorch, YOLOv8 Detection) sont pour des structures de projet particulières.

**Contenu du ZIP :**

.. code-block:: text

   export_yolo.zip
   ├── classes.txt          # Liste des classes (ex: "cube")
   ├── notes.json           # Métadonnées (optionnel)
   └── labels/              # Fichiers .txt au format YOLO
       ├── ad2a7904-image1.txt
       ├── caed06ef-image2.txt
       └── ...

⚠️ **Problème** : Les images ne sont **pas incluses** dans l'export, il faut les ajouter manuellement.


.. slide::

9.2. Nettoyer les labels YOLO
~~~~~~~~~~~~~~~~~~~~~~~~

L'export Label Studio contient les **labels** (fichiers ``.txt``) mais pas les **images**. De plus, Label Studio ajoute des **préfixes UUID** aux noms de fichiers (ex: ``ad2a7904-frame_000001.txt``). Ce code nettoie les noms pour qu'ils correspondent à vos images :

.. code-block:: python

   import shutil
   from pathlib import Path
   
   def remove_uuid_prefix(filename):
       """Enlève le préfixe UUID de Label Studio.
       Ex: 'ad2a7904-frame_000001.jpg' → 'frame_000001.jpg'
       """
       if '-' in filename:
           return '-'.join(filename.split('-')[1:])  # Garde tout après le premier '-'
       return filename
   
   # Dossiers
   yolo_export = Path('export_yolo')           # ADAPTEZ : votre export décompressé Label Studio 
   output_dir = Path('my_dataset_yolo')        # ADAPTEZ : nom du dossier de sortie
   
   # Créer la structure
   output_dir.mkdir(exist_ok=True)
   (output_dir / 'labels').mkdir(exist_ok=True)
   
   # Copier et renommer les labels (enlever UUID)
   num_labels = 0
   for label_file in (yolo_export / 'labels').glob('*.txt'):
       clean_name = remove_uuid_prefix(label_file.name)
       shutil.copy(label_file, output_dir / 'labels' / clean_name)
       print(f"  {label_file.name} → {clean_name}")
       num_labels += 1
   
   # Copier classes.txt
   shutil.copy(yolo_export / 'classes.txt', output_dir / 'classes.txt')
   
   print(f"\n✅ {num_labels} labels nettoyés dans : {output_dir / 'labels'}")

💡 **Pas besoin de copier les images !** On va pointer vers le dossier existant dans le fichier YAML (étape suivante).

.. slide::

9.3. Organiser le dataset pour YOLO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

🎯 **Structure requise par YOLO**

YOLO a besoin d'une structure spécifique :

1. Dossier ``images/`` contenant les images
2. Dossier ``labels/`` contenant les labels (mêmes noms que les images mais en .txt)
3. Fichiers train.txt, val.txt, test.txt listant les chemins des images

.. code-block:: python

   import torch
   import shutil
   from pathlib import Path
   from torch.utils.data import random_split
   
   def create_yolo_dataset(images_dir, labels_dir, output_dir, seed=42):
       """
       Prépare le dataset pour YOLO avec la structure attendue.
       
       Args:
           images_dir: Dossier source des images (ex: 'data/cube_frames')
           labels_dir: Dossier source des labels (ex: 'my_dataset_yolo/labels')
           output_dir: Dossier de sortie (ex: 'data_yolo')
           seed: Seed pour reproductibilité (défaut: 42)
       """
       images_dir = Path(images_dir)
       labels_dir = Path(labels_dir)
       output_dir = Path(output_dir)
       
       # 1. Créer la structure YOLO
       images_out = output_dir / 'images'
       labels_out = output_dir / 'labels'
       images_out.mkdir(parents=True, exist_ok=True)
       labels_out.mkdir(parents=True, exist_ok=True)
       
       # 2. Copier les images et labels
       image_files = sorted(list(images_dir.glob('*.jpg')))
       print(f"📁 {len(image_files)} images trouvées")
       
       for img_file in image_files:
           # Copier l'image
           shutil.copy(img_file, images_out / img_file.name)
           
           # Copier le label correspondant
           lbl_file = labels_dir / f"{img_file.stem}.txt"
           if lbl_file.exists():
               shutil.copy(lbl_file, labels_out / lbl_file.name)
       
       print(f"✅ Fichiers copiés dans {output_dir}")
       
       # 3. Split 70/15/15
       train_size = int(0.7 * len(image_files))
       val_size = int(0.15 * len(image_files))
       test_size = len(image_files) - train_size - val_size
       
       train_idx, val_idx, test_idx = random_split(
           range(len(image_files)),
           [train_size, val_size, test_size],
           generator=torch.Generator().manual_seed(seed)
       )
       
       # 4. Écrire les fichiers .txt avec chemins ABSOLUS
       for indices, name in [(train_idx, 'train'), (val_idx, 'val'), (test_idx, 'test')]:
           with open(output_dir / f'{name}.txt', 'w') as f:
               for idx in indices.indices:
                   img_path = image_files[idx]
                   # Chemin absolu vers l'image dans images/
                   abs_path = (images_out / img_path.name).absolute()
                   f.write(f"{abs_path}\n")
       
       print(f"✅ Split : {train_size} train, {val_size} val, {test_size} test")
       return output_dir
   
   # Utilisation :
   output_path = create_yolo_dataset(
       images_dir='data/cube_frames',
       labels_dir='my_dataset_yolo/labels',
       output_dir='data_yolo',
       seed=42
   )

.. slide::

**Structure finale** :

.. code-block:: text

   data_yolo/
   ├── my_dataset.yaml          # Configuration YOLO
   ├── train.txt                # Chemins absolus des images train
   ├── val.txt                  # Chemins absolus des images val
   ├── test.txt                 # Chemins absolus des images test
   ├── images/                  # Toutes les images
   │   ├── frame_000001.jpg
   │   ├── frame_000002.jpg
   │   └── ...
   └── labels/                  # Tous les labels (mêmes noms que images/)
       ├── frame_000001.txt
       ├── frame_000002.txt
       └── ...

.. note::

   💡 **Pourquoi seed=42 ?**
   
   - ✅ **Même split** que SimpleBBoxRegressor (section 7)
   - ✅ **Comparaison équitable** : YOLO et SimpleBBoxRegressor testés sur les mêmes images

.. slide::

9.4. Créer le fichier de configuration YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Créez un fichier ``my_dataset.yaml`` dans le dossier ``data_yolo/`` :

.. code-block:: yaml

   # my_dataset.yaml
   
   path: /chemin/absolu/vers/data_yolo  # ADAPTEZ : avec le chemin absolu correct
   train: train.txt
   val: val.txt
   test: test.txt
   
   nc: 1
   names: ['mon_objet']  # ADAPTEZ : avec le nom de votre classe

.. warning::

   ⚠️ **Important : Utiliser des chemins ABSOLUS**
   
   YOLO fonctionne mieux avec des chemins absolus. Dans le YAML, utilisez le chemin complet vers ``data_yolo/``.
   
   Les fichiers ``train.txt``, ``val.txt``, ``test.txt`` contiennent déjà des chemins absolus vers les images.


.. slide::

9.5. Entraîner YOLO sur votre dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ultralytics import YOLO
   
   # Charger YOLOv11n pré-entraîné
   model = YOLO('yolo11n.pt')
   
   # Entraîner sur votre dataset
   results = model.train(
       data='data_yolo/my_dataset.yaml',  # ADAPTEZ : avec le chemin vers votre YAML
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

9.6. Tester YOLO sur le test set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Testons YOLO sur les **15% d'images de test** (jamais vues pendant l'entraînement) :

.. code-block:: python

   from ultralytics import YOLO
   from pathlib import Path
   
   # Charger le meilleur modèle entraîné
   yolo_model = YOLO('runs/detect/yolo11_my_object/weights/best.pt')  # ADAPTEZ le chemin
   
   # 1. Évaluer sur le test set
   test_metrics = yolo_model.val(
       data='data_yolo/my_dataset.yaml', # ADAPTEZ le chemin
       split='test'
   )
   
   print("📊 Métriques YOLO sur le TEST SET :")
   print(f"  mAP@0.5     : {test_metrics.box.map50:.3f}")
   print(f"  mAP@0.5:0.95: {test_metrics.box.map:.3f}")
   print(f"  Precision   : {test_metrics.box.mp:.3f}")
   print(f"  Recall      : {test_metrics.box.mr:.3f}")
   
   # 2. Tester sur quelques images du test set
   with open('data_yolo/test.txt', 'r') as f: # ADAPTEZ le chemin
       test_images = [line.strip() for line in f.readlines()[:5]]
   
   for img_path in test_images:
       results = yolo_model.predict(
           source=img_path,
           conf=0.5, # ADAPTEZ : en fonction de la confiance du modèle dans ses prédictions
           save=True,
           project='./predictions',  # Dossier principal  # ADAPTEZ le chemin
           name='test_results'        # Sous-dossier
       )
       print(f"✅ Prédiction pour {Path(img_path).name}")
   
   print(f"✅ Prédictions sauvegardées dans : ./predictions/test_results/") # ADAPTEZ le chemin

.. slide::

🏋️ Travaux Pratiques 6
--------------------

.. toctree::

    TP_chap6

