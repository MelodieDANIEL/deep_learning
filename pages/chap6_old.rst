
STOP ICI

STOP ICI

STOP ICI


Il y a une section qui a été ajouté donc là le 8 c'est en fait 9 etc....

.. slide::

📖 8. Inférence : détecter sur nouvelles images et vidéos
----------------------

Une fois le modèle entraîné, utilisons-le pour détecter des objets sur de nouvelles données.

8.1. Inférence sur une image unique
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import cv2
   from PIL import Image
   from torchvision.transforms import functional as F

   def detect_objects(model, image_path, device, threshold=0.5):
       """
       Détecte les objets dans une image.
       
       Args:
           model: modèle Faster R-CNN entraîné
           image_path: chemin vers l'image
           device: 'cuda' ou 'cpu'
           threshold: seuil de confiance minimum (0-1)
       
       Returns:
           boxes, labels, scores
       """
       # Charger l'image
       img = Image.open(image_path).convert('RGB')
       
       # Convertir en tensor
       img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
       
       # Passer le modèle en mode évaluation
       model.eval()
       
       # Inférence (sans calcul de gradients)
       with torch.no_grad():
           predictions = model(img_tensor)
       
       # Extraire les résultats
       pred = predictions[0]
       boxes = pred['boxes'].cpu().numpy()
       labels = pred['labels'].cpu().numpy()
       scores = pred['scores'].cpu().numpy()
       
       # Filtrer par seuil de confiance
       keep = scores >= threshold
       boxes = boxes[keep]
       labels = labels[keep]
       scores = scores[keep]
       
       return boxes, labels, scores

   # Exemple d'utilisation
   model.load_state_dict(torch.load('best_model.pth'))
   boxes, labels, scores = detect_objects(model, 'test_image.jpg', device, threshold=0.7)

   print(f"Détecté {len(boxes)} objets :")
   for box, label, score in zip(boxes, labels, scores):
       print(f"  Classe {label}, confiance {score:.2f}, bbox {box}")

.. slide::

8.2. Visualiser les détections
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   import numpy as np

   def visualize_predictions(image_path, boxes, labels, scores, class_names, output_path):
       """
       Dessine les boîtes de détection sur l'image.
       
       Args:
           image_path: chemin image source
           boxes: array numpy de shape [N, 4]
           labels: array numpy de shape [N]
           scores: array numpy de shape [N]
           class_names: liste des noms de classes
           output_path: où sauvegarder le résultat
       """
       # Charger l'image avec OpenCV
       img = cv2.imread(image_path)
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
       # Dessiner chaque détection
       for box, label, score in zip(boxes, labels, scores):
           x1, y1, x2, y2 = box.astype(int)
           
           # Couleur selon la classe
           colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
           color = colors[label % len(colors)]
           
           # Rectangle
           cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
           
           # Texte avec classe et confiance
           class_name = class_names[label - 1]  # -1 car background est 0
           text = f"{class_name}: {score:.2f}"
           
           # Fond pour le texte
           (text_width, text_height), _ = cv2.getTextSize(
               text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
           )
           cv2.rectangle(
               img, (x1, y1 - text_height - 10), 
               (x1 + text_width, y1), color, -1
           )
           
           # Texte
           cv2.putText(
               img, text, (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
           )
       
       # Sauvegarder
       img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
       cv2.imwrite(output_path, img_bgr)
       print(f"✓ Résultat sauvegardé : {output_path}")

   # Utilisation
   class_names = ['bouteille', 'gobelet']
   boxes, labels, scores = detect_objects(model, 'test.jpg', device, threshold=0.5)
   visualize_predictions('test.jpg', boxes, labels, scores, class_names, 'test_output.jpg')

.. slide::

8.3. Inférence sur une vidéo
~~~~~~~~~~~~~~~~~~~

Pour traiter une vidéo, on applique la détection frame par frame :

.. code-block:: python

   import cv2
   from PIL import Image
   import numpy as np

   def process_video(model, video_path, output_path, device, class_names, threshold=0.5):
       """
       Applique la détection sur chaque frame d'une vidéo.
       
       Args:
           model: modèle entraîné
           video_path: chemin vidéo d'entrée
           output_path: chemin vidéo de sortie
           device: 'cuda' ou 'cpu'
           class_names: liste des classes
           threshold: seuil de confiance
       """
       # Ouvrir la vidéo
       cap = cv2.VideoCapture(video_path)
       
       # Propriétés de la vidéo
       fps = int(cap.get(cv2.CAP_PROP_FPS))
       width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
       # Writer pour la vidéo de sortie
       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
       
       model.eval()
       frame_count = 0
       
       print(f"Traitement de {total_frames} frames à {fps} fps...")
       
       from tqdm import tqdm
       pbar = tqdm(total=total_frames)
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           # Convertir BGR → RGB
           frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           img_pil = Image.fromarray(frame_rgb)
           
           # Convertir en tensor
           img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(device)
           
           # Détection
           with torch.no_grad():
               predictions = model(img_tensor)
           
           pred = predictions[0]
           boxes = pred['boxes'].cpu().numpy()
           labels = pred['labels'].cpu().numpy()
           scores = pred['scores'].cpu().numpy()
           
           # Filtrer par confiance
           keep = scores >= threshold
           boxes = boxes[keep]
           labels = labels[keep]
           scores = scores[keep]
           
           # Dessiner les boîtes sur la frame
           for box, label, score in zip(boxes, labels, scores):
               x1, y1, x2, y2 = box.astype(int)
               
               color = (0, 255, 0)  # Vert
               cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
               
               class_name = class_names[label - 1]
               text = f"{class_name}: {score:.2f}"
               cv2.putText(frame, text, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
           
           # Écrire la frame
           out.write(frame)
           frame_count += 1
           pbar.update(1)
       
       # Nettoyer
       cap.release()
       out.release()
       pbar.close()
       
       print(f"✓ Vidéo traitée : {frame_count} frames")
       print(f"✓ Sauvegardée : {output_path}")

   # Utilisation
   process_video(
       model=model,
       video_path='test_video.mp4',
       output_path='test_video_detected.mp4',
       device=device,
       class_names=['bouteille', 'gobelet'],
       threshold=0.6
   )

💡 **Optimisation pour la vitesse** :

- Réduire la résolution : ``cv2.resize(frame, (640, 480))``
- Augmenter le threshold : traiter moins de détections
- Traiter 1 frame sur 2 ou 3 pour les vidéos rapides
- Utiliser un batch de frames si assez de mémoire GPU

.. slide::

8.4. Inférence en temps réel (webcam)
~~~~~~~~~~~~~~~~~~~

Pour détecter en temps réel depuis une webcam :

.. code-block:: python

   def realtime_detection(model, device, class_names, threshold=0.5):
       """Détection en temps réel depuis la webcam."""
       cap = cv2.VideoCapture(0)  # 0 = webcam par défaut
       
       model.eval()
       
       print("Appuyez sur 'q' pour quitter")
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           # Préparer l'image
           frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           img_pil = Image.fromarray(frame_rgb)
           img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(device)
           
           # Détection
           with torch.no_grad():
               predictions = model(img_tensor)
           
           pred = predictions[0]
           boxes = pred['boxes'].cpu().numpy()
           labels = pred['labels'].cpu().numpy()
           scores = pred['scores'].cpu().numpy()
           
           keep = scores >= threshold
           boxes = boxes[keep]
           labels = labels[keep]
           scores = scores[keep]
           
           # Dessiner
           for box, label, score in zip(boxes, labels, scores):
               x1, y1, x2, y2 = box.astype(int)
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               
               class_name = class_names[label - 1]
               text = f"{class_name}: {score:.2f}"
               cv2.putText(frame, text, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           
           # Afficher
           cv2.imshow('Detection en temps réel', frame)
           
           # Quitter avec 'q'
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
       
       cap.release()
       cv2.destroyAllWindows()

   # Lancer
   realtime_detection(model, device, ['bouteille', 'gobelet'], threshold=0.6)

.. warning::

   ⚠️ **FPS en temps réel**
   
   Faster R-CNN est relativement lent (~5-10 fps sur GPU, <1 fps sur CPU). Pour du vrai temps réel (>30 fps), préférez YOLO.

.. slide::

📖 9. Comparaison : Faster R-CNN vs YOLO
----------------------

Maintenant que nous maîtrisons Faster R-CNN, comparons-le avec YOLO, une alternative très populaire.

9.1. Différences d'architecture
~~~~~~~~~~~~~~~~~~~

**Faster R-CNN (Two-stage)** :

1. **Stage 1** : RPN propose ~2000 régions candidates
2. **Stage 2** : Classification et raffinement de chaque région

**YOLO (One-stage)** :

1. **Un seul passage** : prédit simultanément classes et boîtes sur une grille

.. code-block:: text

   Faster R-CNN :
   Image → CNN → RPN → Proposals → RoI Pooling → Classifier → Détections
           (ResNet50)  (2000 boxes)              (pour chaque box)
   
   YOLO :
   Image → CNN → Prédictions (grille) → Détections
           (CSPDarknet)  (directement)

.. slide::

9.2. Tableau comparatif détaillé
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Critère
     - Faster R-CNN
     - YOLO (v5/v8)
   * - **Vitesse (fps)**
     - 5-10 (GPU)
     - 30-100+ (GPU)
   * - **Précision**
     - Très bonne, surtout petits objets
     - Bonne, parfois moins sur petits objets
   * - **Entraînement**
     - Fonctionne bien avec peu de données
     - Nécessite plus de données
   * - **Complexité**
     - Plus complexe (2 stages)
     - Plus simple (1 stage)
   * - **Mémoire GPU**
     - ~4-6 GB
     - ~2-4 GB
   * - **Facilité d'utilisation**
     - torchvision (standard)
     - ultralytics (très facile)
   * - **Déploiement embarqué**
     - Difficile (lourd)
     - Facile (optimisé)
   * - **Cas d'usage**
     - Recherche, haute précision
     - Production, temps réel

.. slide::

9.3. Exemple d'utilisation de YOLO (Ultralytics)
~~~~~~~~~~~~~~~~~~~

Pour comparaison, voici comment utiliser YOLOv8 (ultralytics) :

.. code-block:: bash

   # Installation
   pip install ultralytics

.. code-block:: python

   from ultralytics import YOLO

   # 1. Charger un modèle pré-entraîné
   model = YOLO('yolov8n.pt')  # nano (le plus rapide)

   # 2. Entraîner sur vos données (format YOLO requis)
   model.train(
       data='dataset.yaml',  # Fichier de config
       epochs=100,
       imgsz=640,
       batch=16
   )

   # 3. Inférence
   results = model('test.jpg')
   
   # Afficher les résultats
   for r in results:
       print(f"Détecté {len(r.boxes)} objets")
       r.show()  # Affiche l'image avec boîtes

**Fichier** ``dataset.yaml`` :

.. code-block:: yaml

   path: /path/to/dataset
   train: images/train
   val: images/val
   
   nc: 2  # nombre de classes
   names: ['bouteille', 'gobelet']

💡 **Ultralytics** est extrêmement simple d'utilisation avec un pipeline clé-en-main complet (entraînement, validation, export, déploiement).

.. slide::

9.4. Quand choisir Faster R-CNN vs YOLO ?
~~~~~~~~~~~~~~~~~~~

**Choisir Faster R-CNN si** :

- Vous avez **peu de données** (< 500 images) → meilleur transfer learning
- Vous cherchez la **précision maximale** (recherche, médical)
- Les **petits objets** sont importants
- Vous n'êtes **pas limité par la vitesse**
- Vous voulez comprendre l'architecture en profondeur

**Choisir YOLO si** :

- Vous avez besoin de **temps réel** (>30 fps)
- Vous voulez un **pipeline simple** et rapide à mettre en place
- Vous avez **assez de données** (>1000 images)
- Vous visez un **déploiement embarqué** (mobile, edge)
- Vous voulez un **écosystème complet** (Ultralytics)

**Compromis recommandé pour débutants** :

1. Commencer avec **Faster R-CNN** pour comprendre les concepts
2. Prototyper et valider sur un petit dataset
3. Si besoin de vitesse, migrer vers **YOLO** avec plus de données

.. slide::

9.5. Benchmark pratique
~~~~~~~~~~~~~~~~~~~

Voici un exemple de benchmark sur un dataset typique (200 images, 2 classes) :

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Modèle
     - mAP@0.5
     - Vitesse (fps)
     - Temps entraînement
     - Mémoire GPU
   * - Faster R-CNN
     - 0.89
     - 8
     - 2h (10 epochs)
     - 5 GB
   * - YOLOv8n
     - 0.85
     - 95
     - 30min (50 epochs)
     - 3 GB
   * - YOLOv8m
     - 0.91
     - 45
     - 1h30 (50 epochs)
     - 6 GB

💡 **Conclusion** : Faster R-CNN donne de bons résultats avec peu d'epochs et peu de données, mais YOLO est beaucoup plus rapide en inférence.

.. slide::

📖 10. Bonnes pratiques et conseils d'expert
----------------------

10.1. Qualité des annotations
~~~~~~~~~~~~~~~~~~~

**Règles d'annotation cohérentes** :

- **Objets partiellement visibles** : annoter si >30% visible
- **Occlusion** : annoter l'objet entier (même si partiellement caché)
- **Objets très petits** : ignorer si <10×10 pixels (bruit)
- **Instances multiples** : annoter chaque instance séparément
- **Bordures** : la boîte doit contenir tout l'objet visible, sans trop d'espace

**Mesurer la qualité** :

.. code-block:: python

   def calculate_iou(box1, box2):
       """Calcule l'IoU (Intersection over Union) entre deux boîtes."""
       x1_min, y1_min, x1_max, y1_max = box1
       x2_min, y2_min, x2_max, y2_max = box2
       
       # Intersection
       inter_min_x = max(x1_min, x2_min)
       inter_min_y = max(y1_min, y2_min)
       inter_max_x = min(x1_max, x2_max)
       inter_max_y = min(y1_max, y2_max)
       
       inter_width = max(0, inter_max_x - inter_min_x)
       inter_height = max(0, inter_max_y - inter_min_y)
       inter_area = inter_width * inter_height
       
       # Union
       box1_area = (x1_max - x1_min) * (y1_max - y1_min)
       box2_area = (x2_max - x2_min) * (y2_max - y2_min)
       union_area = box1_area + box2_area - inter_area
       
       # IoU
       iou = inter_area / union_area if union_area > 0 else 0
       return iou

   # Exemple : comparer les annotations de 2 annotateurs
   box_annotator1 = [100, 50, 300, 200]
   box_annotator2 = [105, 55, 295, 205]
   iou = calculate_iou(box_annotator1, box_annotator2)
   print(f"IoU entre annotateurs : {iou:.3f}")
   # IoU > 0.7 = bon accord

💡 **Objectif** : IoU moyen > 0.7 entre annotateurs sur un échantillon de 50 images.

.. slide::

10.2. Augmentation de données
~~~~~~~~~~~~~~~~~~~

Pour améliorer la robustesse du modèle avec peu de données :

.. code-block:: python

   import torchvision.transforms as T
   import torch

   class DetectionAugmentation:
       """Augmentations pour la détection (image + boîtes)."""
       
       def __init__(self, p=0.5):
           self.p = p  # Probabilité d'application
       
       def __call__(self, img, target):
           # Flip horizontal (50% chance)
           if torch.rand(1) < self.p:
               img = F.hflip(img)
               # Adapter les coordonnées des boîtes
               w, h = img.size
               boxes = target['boxes']
               boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
               target['boxes'] = boxes
           
           # Color jitter
           if torch.rand(1) < self.p:
               img = T.ColorJitter(
                   brightness=0.2,
                   contrast=0.2,
                   saturation=0.2,
                   hue=0.1
               )(img)
           
           return img, target

⚠️ **Attention** : certaines augmentations (rotation, crop) nécessitent d'ajuster les coordonnées des boîtes de manière non triviale.

.. slide::

10.3. Débogage : le modèle ne converge pas
~~~~~~~~~~~~~~~~~~~

**Symptômes et solutions** :

**1. Loss reste élevée (~4-5) et ne baisse pas** :

- Vérifier que les coordonnées des boîtes sont correctes
- Vérifier que les labels commencent à 1 (pas 0)
- Réduire le learning rate (tester 0.001 au lieu de 0.005)
- Augmenter le nombre d'epochs

**2. Loss descend puis remonte (overfitting)** :

- Ajouter plus de données ou d'augmentation
- Réduire le nombre d'epochs
- Ajouter du weight decay

**3. Détections bizarres (toutes au même endroit)** :

- Vérifier la conversion des coordonnées (YOLO → PyTorch)
- Visualiser quelques exemples du dataset avec ``visualize_sample()``

**4. Aucune détection (boîtes vides)** :

- Le threshold est trop élevé → essayer 0.3 au lieu de 0.5
- Le modèle n'a pas convergé → entraîner plus longtemps

.. slide::

10.4. Optimisation de la vitesse
~~~~~~~~~~~~~~~~~~~

**Pour accélérer l'entraînement** :

.. code-block:: python

   # 1. Mixed precision (AMP)
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()

   for images, targets in train_loader:
       with autocast():  # Calculs en float16
           loss_dict = model(images, targets)
           losses = sum(loss for loss in loss_dict.values())
       
       scaler.scale(losses).backward()
       scaler.step(optimizer)
       scaler.update()
       optimizer.zero_grad()

   # 2. Réduire la résolution des images
   # Redimensionner à 512×512 au lieu de 1024×1024

   # 3. Utiliser DataLoader avec num_workers > 0
   train_loader = DataLoader(..., num_workers=8, pin_memory=True)

**Pour accélérer l'inférence** :

- Convertir en TorchScript : ``model = torch.jit.script(model)``
- Exporter en ONNX pour déploiement optimisé
- Utiliser un modèle plus léger (ResNet18 au lieu de ResNet50)

.. slide::

10.5. Métriques d'évaluation
~~~~~~~~~~~~~~~~~~~

**mAP (mean Average Precision)** est la métrique standard :

.. code-block:: python

   from torchvision.ops import box_iou

   def calculate_ap(pred_boxes, pred_scores, true_boxes, iou_threshold=0.5):
       """Calcule l'Average Precision pour une classe."""
       # Trier par score décroissant
       sorted_indices = torch.argsort(pred_scores, descending=True)
       pred_boxes = pred_boxes[sorted_indices]
       pred_scores = pred_scores[sorted_indices]
       
       true_positives = []
       false_positives = []
       matched_gt = set()
       
       for i, pred_box in enumerate(pred_boxes):
           # Calculer IoU avec toutes les GT boxes
           ious = box_iou(pred_box.unsqueeze(0), true_boxes)
           max_iou, max_idx = ious.max(dim=1)
           
           if max_iou >= iou_threshold and max_idx.item() not in matched_gt:
               true_positives.append(1)
               false_positives.append(0)
               matched_gt.add(max_idx.item())
           else:
               true_positives.append(0)
               false_positives.append(1)
       
       # Calculer precision et recall
       tp_cumsum = torch.cumsum(torch.tensor(true_positives), dim=0)
       fp_cumsum = torch.cumsum(torch.tensor(false_positives), dim=0)
       
       recalls = tp_cumsum / len(true_boxes)
       precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
       
       # AP = aire sous la courbe precision-recall
       ap = torch.trapz(precisions, recalls).item()
       return ap

💡 **Interprétation** :

- mAP@0.5 > 0.8 : excellent
- mAP@0.5 > 0.6 : bon
- mAP@0.5 < 0.4 : à améliorer

.. slide::

📖 11. Ressources complémentaires
-----------------------

**Documentation** :

- PyTorch Detection : https://pytorch.org/vision/stable/models.html#object-detection
- Label Studio : https://labelstud.io/guide/
- Faster R-CNN paper : https://arxiv.org/abs/1506.01497
- YOLO Ultralytics : https://docs.ultralytics.com/

**Datasets publics pour s'entraîner** :

- COCO : 330k images, 80 classes
- Pascal VOC : 20k images, 20 classes
- Open Images : 9M images, 600 classes

**Outils utiles** :

- Roboflow : conversion de formats, augmentation
- CVAT : alternative à Label Studio
- Netron : visualiser l'architecture des modèles
.. slide::

🏋️ Travaux Pratiques 6
--------------------

Maintenant que vous maîtrisez tous les concepts, passez à la pratique !

.. toctree::

    TP_chap6



