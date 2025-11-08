Nous allons maintenant entraîner un détecteur d'objets avec **Faster R-CNN**, l'un des modèles les plus populaires et performants.

8.1. Qu'est-ce que Faster R-CNN ?
~~~~~~~~~~~~~~~~~~~

**Faster R-CNN** (Region-based Convolutional Neural Network) est un modèle **two-stage** :

**Stage 1 : Region Proposal Network (RPN)**

- Scanne l'image pour proposer des régions susceptibles de contenir des objets
- Génère ~1000-2000 propositions de boîtes

**Stage 2 : Classification et raffinement**

- Pour chaque proposition, prédit la classe et affine les coordonnées
- Filtre les boîtes redondantes (Non-Maximum Suppression)

.. image:: images/faster_rcnn_architecture.png
   :width: 80%
   :align: center
   :alt: Architecture Faster R-CNN

**Avantages** :

- Très précis, surtout sur petits objets
- Entraînement possible avec peu de données (quelques centaines d'images)
- Architecture bien comprise et stable

**Inconvénients** :

- Plus lent que YOLO en inférence (~5-10 fps)
- Plus complexe qu'un modèle one-stage

.. slide::

8.2. Charger un modèle pré-entraîné
~~~~~~~~~~~~~~~~~~~

torchvision fournit Faster R-CNN pré-entraîné sur COCO (80 classes). On va le fine-tuner sur nos propres classes.

.. code-block:: python

   import torch
   import torchvision
   from torchvision.models.detection import fasterrcnn_resnet50_fpn
   from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

   def get_model(num_classes):
       """
       Charge Faster R-CNN et adapte la tête de classification.
       
       Args:
           num_classes: nombre de classes + 1 (pour le background)
                       Ex : 2 classes → num_classes = 3
       """
       # Charger le modèle pré-entraîné
       model = fasterrcnn_resnet50_fpn(pretrained=True)
       
       # Récupérer le nombre de features en entrée de la tête de classification
       in_features = model.roi_heads.box_predictor.cls_score.in_features
       
       # Remplacer la tête par une nouvelle pour nos classes
       model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
       
       return model

   # Exemple : 2 classes (bouteille, gobelet) + background
   model = get_model(num_classes=3)
   print(model)

**Explication** :

- ``pretrained=True`` : charge les poids entraînés sur COCO (80 classes)
- On **remplace uniquement la dernière couche** pour nos classes
- Les couches précédentes (backbone ResNet50, RPN) sont déjà très performantes et seront fine-tunées

💡 **Transfer learning** : on réutilise les connaissances du modèle (formes, textures, objets génériques) pour accélérer l'apprentissage sur notre tâche spécifique.

.. slide::

8.3. Configuration de l'entraînement
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader

.. code-block:: python

   # 📊 ÉVALUATION COMPLÈTE DU CNN CUSTOM SUR TOUT LE TEST SET (depuis le notebook)

   import numpy as np

   def compute_iou_boxes(box1, box2):
       """
       Calcule l'IoU entre deux boîtes.
       
       Args:
           box1, box2: [x1, y1, x2, y2]
       
       Returns:
           iou: score IoU (0-1)
       """
       x1_inter = max(box1[0], box2[0])
       y1_inter = max(box1[1], box2[1])
       x2_inter = min(box1[2], box2[2])
       y2_inter = min(box1[3], box2[3])
       
       if x2_inter < x1_inter or y2_inter < y1_inter:
           return 0.0
       
       inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
       
       box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
       box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
       
       union_area = box1_area + box2_area - inter_area
       
       iou = inter_area / (union_area + 1e-6)
       return iou

   # Charger le meilleur modèle
   custom_model.load_state_dict(torch.load('best_custom_cube_detector.pth'))
   custom_model.eval()
   
   print("🔍 ÉVALUATION SUR LE TEST SET COMPLET")
   print("="*60)
   
   # Métriques globales
   total_gt_objects = 0
   total_detected = 0
   total_true_positives = 0
   iou_threshold = 0.5
   conf_threshold = 0.75  # seuil relevé pour réduire les faux positifs
   #conf_threshold = 0.3  # exemple d'autre seuil
   
   all_ious = []
   detection_per_image = []
   
   print(f"\n📈 Test sur {len(test_dataset)} images...\n")
   
   # Tester sur chaque image
   for idx in range(len(test_dataset)):
       test_img, test_target = test_dataset[idx]
       test_img_tensor = test_img.unsqueeze(0).to(device)
       
       # Prédiction
       with torch.no_grad():
           predictions = custom_model(test_img_tensor)
           results = custom_model.decode_predictions(predictions, conf_threshold=conf_threshold, device=device)
       
       detected_boxes = results[0]['boxes'].cpu().numpy()
       detected_labels = results[0]['labels'].cpu().numpy()
       detected_scores = results[0]['scores'].cpu().numpy()
       
       gt_boxes = test_target['boxes'].cpu().numpy()
       gt_labels = test_target['labels'].cpu().numpy()
       
       # Compter les objets
       num_gt = len(gt_boxes)
       num_detected = len(detected_boxes)
       
       total_gt_objects += num_gt
       total_detected += num_detected
       
       detection_per_image.append({
           'image_idx': idx,
           'gt_count': num_gt,
           'detected_count': num_detected
       })
       
       # Calculer les True Positives (matching avec IoU)
       matched_gt = set()
       image_ious = []
       
       for det_box in detected_boxes:
           best_iou = 0
           best_gt_idx = -1
           
           for gt_idx, gt_box in enumerate(gt_boxes):
               if gt_idx in matched_gt:
                   continue
               
               iou = compute_iou_boxes(det_box, gt_box)
               
               if iou > best_iou:
                   best_iou = iou
                   best_gt_idx = gt_idx
           
           if best_iou >= iou_threshold:
               total_true_positives += 1
               matched_gt.add(best_gt_idx)
               image_ious.append(best_iou)
       
       all_ious.extend(image_ious)
       
       # Afficher les détails de cette image
       print(f"Image {idx+1}/{len(test_dataset)}:")
       print(f"   Ground Truth: {num_gt} cubes")
       print(f"   Détections:   {num_detected} cubes")
       print(f"   True Positives: {len(image_ious)}")
       if image_ious:
           print(f"   IoU moyen:    {np.mean(image_ious):.3f}")
   
   # Calculer les métriques globales
   print("\n" + "="*60)
   print("📊 RÉSULTATS GLOBAUX")
   print("="*60)
   
   precision = total_true_positives / total_detected if total_detected > 0 else 0
   recall = total_true_positives / total_gt_objects if total_gt_objects > 0 else 0
   f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
   mean_iou = np.mean(all_ious) if all_ious else 0
   
   print(f"\n🎯 Métriques de détection (IoU threshold = {iou_threshold}, conf = {conf_threshold}):")
   print(f"   Objets ground truth:  {total_gt_objects}")
   print(f"   Objets détectés:      {total_detected}")
   print(f"   True Positives:       {total_true_positives}")
   print(f"   False Positives:      {total_detected - total_true_positives}")
   print(f"   False Negatives:      {total_gt_objects - total_true_positives}")
   
   print(f"\n📈 Scores:")
   print(f"   Precision:  {precision:.3f} ({total_true_positives}/{total_detected})")
   print(f"   Recall:     {recall:.3f} ({total_true_positives}/{total_gt_objects})")
   print(f"   F1-Score:   {f1_score:.3f}")
   print(f"   IoU moyen:  {mean_iou:.3f}")
   
   # Taux de détection par image
   perfect_detections = sum(1 for d in detection_per_image if d['detected_count'] == d['gt_count'])
   print(f"\n🎨 Détections parfaites (nombre exact): {perfect_detections}/{len(test_dataset)} images ({perfect_detections/len(test_dataset)*100:.1f}%)")
   
   # Visualisation de quelques résultats
   print("\n" + "="*60)
   print("🖼️  VISUALISATION DE 3 EXEMPLES DU TEST SET")
   print("="*60)
   
   fig, axes = plt.subplots(3, 2, figsize=(16, 18))
   
   for i, test_idx in enumerate(range(min(3, len(test_dataset)))):
       test_img, test_target = test_dataset[test_idx]
       test_img_tensor = test_img.unsqueeze(0).to(device)
       
       # Prédiction
       with torch.no_grad():
           predictions = custom_model(test_img_tensor)
           results = custom_model.decode_predictions(predictions, conf_threshold=conf_threshold, device=device)
       
       detected_boxes = results[0]['boxes']
       detected_labels = results[0]['labels']
       detected_scores = results[0]['scores']
       
       # Convertir l'image pour affichage
       img_np = test_img.permute(1, 2, 0).cpu().numpy()
       
       # Ground Truth
       ax = axes[i, 0]
       ax.imshow(img_np)
       ax.set_title(f'Image {test_idx+1} - Ground Truth ({len(test_target["boxes"])} cubes)', 
                    fontsize=14, weight='bold')
       
       for box, label in zip(test_target['boxes'], test_target['labels']):
           x1, y1, x2, y2 = box.tolist()
           width = x2 - x1
           height = y2 - y1
           
           rect = patches.Rectangle(
               (x1, y1), width, height,
               linewidth=3, edgecolor='green', facecolor='none'
           )
           ax.add_patch(rect)
           
           class_name = full_dataset.get_class_name(label.item())
           ax.text(x1, y1-5, f"{class_name}",
                   bbox=dict(facecolor='green', alpha=0.7),
                   fontsize=10, color='white', weight='bold')
       
       ax.axis('off')
       
       # Prédictions
       ax = axes[i, 1]
       ax.imshow(img_np)
       ax.set_title(f'Image {test_idx+1} - Prédictions ({len(detected_boxes)} cubes)', 
                    fontsize=14, weight='bold')
       
       colors = ['red', 'blue', 'orange', 'purple', 'yellow']
       
       for box, label, score in zip(detected_boxes, detected_labels, detected_scores):
           x1, y1, x2, y2 = box.tolist()
           width = x2 - x1
           height = y2 - y1
           
           color = colors[(label.item() - 1) % len(colors)]
           
           rect = patches.Rectangle(
               (x1, y1), width, height,
               linewidth=3, edgecolor=color, facecolor='none'
           )
           ax.add_patch(rect)
           
           class_name = full_dataset.get_class_name(label.item())
           label_text = f"{class_name}: {score:.2f}"
           ax.text(x1, y1-5, label_text,
                   bbox=dict(facecolor=color, alpha=0.7),
                   fontsize=10, color='white', weight='bold')
       
       ax.axis('off')
   
   plt.tight_layout()
   plt.savefig('test_set_evaluation.png', bbox_inches='tight', dpi=150)
   print(f"\n✅ Visualisation sauvegardée : test_set_evaluation.png")
   plt.show()
   
   print("\n" + "="*60)
   print("✨ CONCLUSION")
   print("="*60)
   
   if f1_score > 0.8:
       print("🌟 Excellent ! Le CNN custom détecte très bien les cubes.")
   elif f1_score > 0.6:
       print("👍 Très bien ! Le CNN custom a de bonnes performances.")
   elif f1_score > 0.4:
       print("👌 Correct. Le CNN custom fonctionne mais pourrait être amélioré.")
   else:
       print("⚠️  Le modèle nécessite plus d'entraînement ou d'ajustements.")
   
   print(f"\nLe modèle a {custom_model.count_parameters():,} paramètres et a été")
   print("entraîné from scratch sans transfer learning sur seulement")
   print(f"{len(train_dataset)} images d'entraînement.")

7.6. Remarque — Batch Normalization et seuil de confiance
           # Forward pass
           loss_dict = model(images, targets)
Qu’est-ce que la BatchNorm (BN) ? Elle normalise les activations par la moyenne/variance du mini-batch. Effets concrets: stabilise et accélère l’entraînement, aide la profondeur et agit comme une légère régularisation. Avec de très petits batchs (1–4), les statistiques de batch peuvent être bruyantes et la calibration des scores varier.

Dans ce notebook, la BN est commentée par simplicité (lignes `#self.bn...`). Vous pouvez:

- Laisser SANS BN (comme dans le code ci-dessus): souvent plus de faux positifs → utilisez un seuil plus haut en inférence (≈ 0.75 par défaut dans l’évaluation).
- Activer AVEC BN: décommentez les lignes `self.bn*` et les passes `F.relu(self.bn*(...))` dans le forward. Scores souvent mieux calibrés → un seuil plus bas est possible (≈ 0.5–0.6).

Observation du notebook: «sans BN ça fonctionne, mais avec BN les résultats étaient meilleurs». Présentez les deux aux étudiants et faites varier le seuil: plus bas avec BN, plus haut sans BN.

Alternative petits batchs: GroupNorm (``nn.GroupNorm``) ne dépend pas de la taille de batch et peut remplacer BatchNorm2d(C) par ``GroupNorm(8, C)``.
           losses.backward()
           optimizer.step()
           
           # Tracking
           total_loss += losses.item()
           pbar.set_postfix({
               'loss': f"{losses.item():.4f}",
               'loss_classifier': f"{loss_dict['loss_classifier'].item():.3f}",
               'loss_box_reg': f"{loss_dict['loss_box_reg'].item():.3f}",
               'loss_objectness': f"{loss_dict['loss_objectness'].item():.3f}",
               'loss_rpn_box_reg': f"{loss_dict['loss_rpn_box_reg'].item():.3f}"
           })
       
       return total_loss / len(data_loader)

   @torch.no_grad()
   def evaluate(model, data_loader, device):
       """Évalue le modèle sur le set de validation."""
       model.train()  # Faster R-CNN nécessite train() même en eval !
       total_loss = 0
       
       for images, targets in tqdm(data_loader, desc="Validation"):
           images = list(image.to(device) for image in images)
           targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
           
           loss_dict = model(images, targets)
           losses = sum(loss for loss in loss_dict.values())
           total_loss += losses.item()
       
       return total_loss / len(data_loader)

   # Entraînement principal
   best_loss = float('inf')
   
   for epoch in range(num_epochs):
       # Entraînement
       train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
       
       # Validation
       val_loss = evaluate(model, val_loader, device)
       
       # Mise à jour du learning rate
       lr_scheduler.step()
       
       # Affichage
       print(f"\nEpoch {epoch}:")
       print(f"  Train Loss: {train_loss:.4f}")
       print(f"  Val Loss:   {val_loss:.4f}")
       print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
       
       # Sauvegarder le meilleur modèle
       if val_loss < best_loss:
           best_loss = val_loss
           torch.save(model.state_dict(), 'best_model.pth')
           print("  ✓ Meilleur modèle sauvegardé !")
   
   print("\n✓ Entraînement terminé !")

**Détails importants** :

- **4 losses** dans Faster R-CNN :
  
  - ``loss_classifier`` : classification des objets
  - ``loss_box_reg`` : régression des boîtes
  - ``loss_objectness`` : score objectness du RPN
  - ``loss_rpn_box_reg`` : régression des proposals du RPN

- Le modèle doit rester en mode ``train()`` même pour la validation (particularité de l'implémentation torchvision)

.. warning::

   ⚠️ **Mémoire GPU limitée ?**
   
   Si vous obtenez une erreur "CUDA out of memory" :
   
   - Réduire ``batch_size`` à 2 ou 1
   - Réduire la résolution des images
   - Utiliser des images de validation moins nombreuses

.. slide::

8.5. Surveiller l'entraînement
~~~~~~~~~~~~~~~~~~~

Pour un suivi plus détaillé, utilisez TensorBoard ou wandb :

.. code-block:: python

   from torch.utils.tensorboard import SummaryWriter

   # Créer un writer TensorBoard
   writer = SummaryWriter('runs/detection_experiment_1')

   # Dans la boucle d'entraînement, ajouter :
   for epoch in range(num_epochs):
       train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
       val_loss = evaluate(model, val_loader, device)
       
       # Logger dans TensorBoard
       writer.add_scalar('Loss/train', train_loss, epoch)
       writer.add_scalar('Loss/val', val_loss, epoch)
       writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
   
   writer.close()

   # Visualiser avec : tensorboard --logdir=runs

💡 **Quand arrêter l'entraînement ?**

- La val_loss ne diminue plus pendant 3-5 epochs → probablement convergé
- La train_loss continue de baisser mais val_loss augmente → overfitting
- Après 10-20 epochs pour un petit dataset

.. slide::

📖 9. Inférence : utiliser le modèle entraîné
----------------------

Maintenant que notre modèle est entraîné, voyons comment l'utiliser pour détecter des objets sur de nouvelles images.

9.1. Charger le modèle sauvegardé
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from PIL import Image
   from torchvision.transforms import functional as F
   import matplotlib.pyplot as plt
   import matplotlib.patches as patches

   # Charger le modèle
   num_classes = 3  # 2 classes + background
   model = get_model(num_classes)
   model.load_state_dict(torch.load('best_model.pth'))
   
   device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   model.to(device)
   model.eval()  # Mode évaluation (important !)
   
   print("✓ Modèle chargé")

.. slide::

9.2. Faire une prédiction
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @torch.no_grad()
   def predict(image_path, model, device, threshold=0.5):
       """
       Effectue une prédiction sur une image.
       
       Args:
           image_path: chemin vers l'image
           model: modèle entraîné
           device: CPU ou GPU
           threshold: seuil de confiance minimum (0-1)
       
       Returns:
           boxes: tensor [N, 4] des boîtes détectées
           labels: tensor [N] des classes
           scores: tensor [N] des scores de confiance
       """
       # Charger et préparer l'image
       img = Image.open(image_path).convert('RGB')
       img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
       
       # Prédiction
       model.eval()
       predictions = model(img_tensor)[0]
       
       # Filtrer par score de confiance
       keep = predictions['scores'] > threshold
       boxes = predictions['boxes'][keep].cpu()
       labels = predictions['labels'][keep].cpu()
       scores = predictions['scores'][keep].cpu()
       
       return img, boxes, labels, scores

   # Exemple d'utilisation
   img, boxes, labels, scores = predict(
       'data/images/test/frame_00001.jpg',
       model,
       device,
       threshold=0.5
   )

   print(f"Objets détectés : {len(boxes)}")
   for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
       x1, y1, x2, y2 = box.tolist()
       class_name = train_dataset.get_class_name(label.item())
       print(f"  {i+1}. {class_name} (conf: {score:.2f}) - bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

.. slide::

9.3. Visualiser les détections
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def visualize_predictions(img, boxes, labels, scores, class_names, threshold=0.5):
       """Affiche l'image avec les boîtes détectées."""
       fig, ax = plt.subplots(1, figsize=(12, 8))
       ax.imshow(img)
       
       # Couleurs pour chaque classe
       colors = ['red', 'blue', 'green', 'yellow', 'orange']
       
       for box, label, score in zip(boxes, labels, scores):
           if score < threshold:
               continue
           
           x1, y1, x2, y2 = box.tolist()
           width = x2 - x1
           height = y2 - y1
           
           # Dessiner la boîte
           color = colors[(label.item() - 1) % len(colors)]
           rect = patches.Rectangle(
               (x1, y1), width, height,
               linewidth=3, edgecolor=color, facecolor='none'
           )
           ax.add_patch(rect)
           
           # Ajouter le label et le score
           class_name = class_names[label.item() - 1]
           label_text = f'{class_name} {score:.2f}'
           ax.text(
               x1, y1 - 5,
               label_text,
               bbox=dict(facecolor=color, alpha=0.7),
               fontsize=12, color='white', weight='bold'
           )
       
       plt.axis('off')
       plt.tight_layout()
       return fig

   # Utilisation
   class_names = train_dataset.classes
   fig = visualize_predictions(img, boxes, labels, scores, class_names, threshold=0.5)
   plt.savefig('detection_result.jpg', bbox_inches='tight', dpi=150)
   print("✓ Résultat sauvegardé : detection_result.jpg")

.. slide::

9.4. Traiter une vidéo complète
~~~~~~~~~~~~~~~~~~~

Pour détecter des objets dans une vidéo, on traite chaque frame :

.. code-block:: python

   import cv2
   from tqdm import tqdm

   def detect_in_video(video_path, model, device, output_path='output_video.mp4', threshold=0.5):
       """Applique la détection sur chaque frame d'une vidéo."""
       cap = cv2.VideoCapture(video_path)
       
       # Propriétés de la vidéo
       width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       fps = int(cap.get(cv2.CAP_PROP_FPS))
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
       # Créer le writer pour la vidéo de sortie
       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
       
       pbar = tqdm(total=total_frames, desc="Traitement vidéo")
       
       model.eval()
       
       while cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               break
           
           # Convertir BGR → RGB
           frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           img_pil = Image.fromarray(frame_rgb)
           
           # Prédiction
           _, boxes, labels, scores = predict_from_pil(img_pil, model, device, threshold)
           
           # Dessiner les boîtes sur la frame
           for box, label, score in zip(boxes, labels, scores):
               if score < threshold:
                   continue
               
               x1, y1, x2, y2 = map(int, box.tolist())
               class_name = train_dataset.get_class_name(label.item())
               
               # Dessiner le rectangle
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               
               # Ajouter le label
               label_text = f'{class_name} {score:.2f}'
               cv2.putText(frame, label_text, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
           
           # Écrire la frame
           out.write(frame)
           pbar.update(1)
       
       cap.release()
       out.release()
       pbar.close()
       
       print(f"✓ Vidéo traitée : {output_path}")

   @torch.no_grad()
   def predict_from_pil(img_pil, model, device, threshold=0.5):
       """Version de predict qui prend directement une image PIL."""
       img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(device)
       predictions = model(img_tensor)[0]
       
       keep = predictions['scores'] > threshold
       boxes = predictions['boxes'][keep].cpu()
       labels = predictions['labels'][keep].cpu()
       scores = predictions['scores'][keep].cpu()
       
       return img_pil, boxes, labels, scores

   # Utilisation
   detect_in_video(
       'ma_video.mp4',
       model,
       device,
       output_path='video_with_detections.mp4',
       threshold=0.6
   )

.. slide::

📖 10. Alternative : utiliser YOLO pré-entraîné
----------------------

Jusqu'ici, nous avons construit notre propre détecteur CNN. Mais si vous voulez des résultats rapides avec moins de code, **YOLO** est une excellente alternative.

9.1. Pourquoi YOLO ?
~~~~~~~~~~~~~~~~~~~

**YOLO (You Only Look Once)** est une famille de modèles de détection très populaires :

**Avantages** :

- ⚡ **Très rapide** : 30-100 fps (temps réel)
- 🎯 **Facile à utiliser** : 3-4 lignes de code
- 📦 **Pré-entraînés** : excellents résultats sur COCO out-of-the-box
- 🔄 **Fine-tuning simple** : export depuis Label Studio → entraînement en 2 commandes

**Inconvénients** :

- Moins bon que Faster R-CNN sur petits objets
- Moins de contrôle sur l'architecture

💡 **Quand utiliser YOLO ?**

- Vous avez besoin de détection en temps réel
- Vous débutez et voulez des résultats rapides
- Votre dataset a >500 images

💡 **Quand utiliser Faster R-CNN (ce que nous avons fait) ?**

- Vous voulez la meilleure précision possible
- Vous avez peu de données (<500 images)
- Vous voulez comprendre et contrôler l'architecture

.. slide::

9.2. Installation et premier test
~~~~~~~~~~~~~~~~~~~

YOLO version 8 (ultralytics) est la plus récente et facile à utiliser :

.. code-block:: bash

   pip install ultralytics

Test rapide avec un modèle pré-entraîné :

.. code-block:: python

   from ultralytics import YOLO
   
   # Charger le modèle pré-entraîné
   model = YOLO('yolov8n.pt')  # n = nano (le plus petit et rapide)
   
   # Détecter sur une image
   results = model('data/images/frame_00001.jpg')
   
   # Afficher les résultats
   results[0].show()  # Affiche l'image avec les boîtes
   
   # Sauvegarder
   results[0].save('yolo_detection.jpg')

Les modèles disponibles :

- ``yolov8n.pt`` : nano (le plus rapide, ~6 MB)
- ``yolov8s.pt`` : small
- ``yolov8m.pt`` : medium
- ``yolov8l.pt`` : large
- ``yolov8x.pt`` : extra large (le plus précis, ~130 MB)

.. slide::

9.3. Fine-tuner YOLO sur vos données Label Studio
~~~~~~~~~~~~~~~~~~~

**Étape 1 : Exporter depuis Label Studio en format YOLO**

1. Dans Label Studio, cliquer sur "Export"
2. Choisir le format **"YOLO"**
3. Télécharger le fichier ZIP

Le ZIP contient :

.. code-block:: text

   yolo_export.zip
   ├── classes.txt      # Liste des classes
   ├── notes.json       # Métadonnées
   └── labels/          # Un .txt par image
       ├── frame_00001.txt
       ├── frame_00002.txt
       └── ...

**Étape 2 : Organiser les données**

.. code-block:: text

   yolo_dataset/
   ├── images/
   │   ├── train/
   │   │   ├── frame_00001.jpg
   │   │   └── ...
   │   └── val/
   │       ├── frame_00151.jpg
   │       └── ...
   ├── labels/
   │   ├── train/
   │   │   ├── frame_00001.txt
   │   │   └── ...
   │   └── val/
   │       ├── frame_00151.txt
   │       └── ...
   └── data.yaml  # Fichier de configuration

**Étape 3 : Créer le fichier** ``data.yaml``

.. code-block:: yaml

   path: /chemin/absolu/vers/yolo_dataset
   train: images/train
   val: images/val

   names:
     0: bouteille
     1: gobelet

**Étape 4 : Entraîner**

.. code-block:: python

   from ultralytics import YOLO

   # Charger le modèle pré-entraîné
   model = YOLO('yolov8n.pt')

   # Entraîner (fine-tuning)
   results = model.train(
       data='yolo_dataset/data.yaml',
       epochs=50,
       imgsz=640,
       batch=16,
       name='detection_bouteille'
   )

**Étape 5 : Utiliser le modèle entraîné**

.. code-block:: python

   # Charger le meilleur modèle
   model = YOLO('runs/detect/detection_bouteille/weights/best.pt')
   
   # Prédire
   results = model('nouvelle_image.jpg', conf=0.5)
   results[0].show()

.. note::

   💡 **Comparaison YOLO vs Faster R-CNN**
   
   **YOLO** : rapide (30-100 fps), facile, bon pour temps réel
   
   **Faster R-CNN** : plus précis, meilleur sur petits objets, plus de contrôle
   
   **Conseil** : commencez par YOLO pour prototyper rapidement, puis passez à Faster R-CNN si vous avez besoin de plus de précision.

.. slide::

📖 10. Conclusion et aller plus loin
----------------------

🎓 **Ce que vous avez appris** :

1. ✅ Différence entre classification et détection d'objets
2. ✅ Extraire des frames d'une vidéo avec OpenCV
3. ✅ Annoter des images avec Label Studio (workflow collaboratif)
4. ✅ Comprendre le format JSON de Label Studio
5. ✅ Créer un Dataset PyTorch pour la détection
6. ✅ Entraîner Faster R-CNN avec transfer learning
7. ✅ Faire de l'inférence sur images et vidéos
8. ✅ Utiliser YOLO comme alternative rapide

🚀 **Pour aller plus loin** :

- **Métriques d'évaluation** : mAP (mean Average Precision), IoU
- **Data augmentation** : rotation, flip, changement de luminosité
- **Post-processing** : NMS (Non-Maximum Suppression), filtrage par taille
- **Modèles avancés** : Mask R-CNN (segmentation), DETR (Transformers)
- **Déploiement** : ONNX, TensorRT, optimisation pour mobile













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



