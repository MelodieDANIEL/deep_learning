🏋️ Travaux Pratiques 6
=========================

.. slide::

Sur cette page se trouvent des exercices de TP sur le Chapitre 6 (Détection d'objets). Ils sont classés par niveau de difficulté :

.. discoverList::
    * Facile : 🍀
    * Moyen : ⚖️
    * Difficile : 🌶️

.. slide::

🍀 Exercice 1 : Extraction de frames et visualisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez capturer une courte vidéo et en extraire des images pour préparer l'annotation.

**Objectif :** Maîtriser ffmpeg et préparer un dataset d'images.

**Matériel nécessaire :**

- Votre smartphone ou webcam
- Un objet à détecter (bouteille, tasse, livre, etc.)

**Consigne :** Écrire un script Python qui :

1) Filme l'objet pendant 30 secondes sous différents angles

2) Sauvegarde la vidéo en ``video.mp4``

3) Utilise ``subprocess`` en Python pour appeler ffmpeg et extraire :
   
   - Toutes les frames (estimer le nombre)
   - 1 frame toutes les 10 frames
   - Exactement 100 frames équidistantes

4) Affiche quelques statistiques :
   
   - Nombre total de frames dans la vidéo
   - FPS (images par seconde)
   - Résolution (largeur × hauteur)
   - Taille du fichier vidéo

5) Crée un montage avec matplotlib affichant 16 frames extraites en grille 4×4

**Code starter :**

.. code-block:: python

   import subprocess
   import os
   from pathlib import Path
   import cv2
   import matplotlib.pyplot as plt

   def get_video_info(video_path):
       """Récupère les informations d'une vidéo avec ffprobe."""
       # À compléter : utiliser ffprobe ou cv2.VideoCapture
       pass

   def extract_frames(video_path, output_dir, mode='all'):
       """
       Extrait les frames d'une vidéo.
       
       Args:
           video_path: chemin vers la vidéo
           output_dir: dossier de sortie
           mode: 'all', 'every_10', 'exact_100'
       """
       Path(output_dir).mkdir(parents=True, exist_ok=True)
       
       if mode == 'all':
           cmd = ['ffmpeg', '-i', video_path, '-q:v', '2', 
                  f'{output_dir}/frame_%05d.jpg']
       elif mode == 'every_10':
           # À compléter : extraire 1 frame toutes les 10
           pass
       elif mode == 'exact_100':
           # À compléter : extraire exactement 100 frames
           pass
       
       subprocess.run(cmd)
       print(f"✓ Frames extraites dans {output_dir}")

   def visualize_grid(image_dir, n_images=16):
       """Affiche une grille d'images."""
       # À compléter
       pass

   # Programme principal
   if __name__ == "__main__":
       video_path = "video.mp4"
       
       # 1. Infos vidéo
       info = get_video_info(video_path)
       print(f"Vidéo : {info}")
       
       # 2. Extraction
       extract_frames(video_path, "frames_all", mode='all')
       extract_frames(video_path, "frames_10", mode='every_10')
       extract_frames(video_path, "frames_100", mode='exact_100')
       
       # 3. Visualisation
       visualize_grid("frames_100", n_images=16)

**Questions :**

6) Combien de frames au total dans une vidéo de 30s à 30 fps ?
7) Pourquoi ne pas annoter toutes les frames ?
8) Quelle commande ffmpeg utiliser pour réduire la résolution à 640×480 lors de l'extraction ?

**Astuce :**

.. spoiler::
    .. discoverList::
        1. Pour ``get_video_info``, utiliser ``cv2.VideoCapture`` :
           
           .. code-block:: python
           
               cap = cv2.VideoCapture(video_path)
               fps = cap.get(cv2.CAP_PROP_FPS)
               frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
               width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
               height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
               cap.release()
        
        2. Pour extraire 1 frame toutes les 10 :
           
           .. code-block:: bash
           
               ffmpeg -i video.mp4 -vf "select=not(mod(n\,10))" -vsync vfr frames/frame_%05d.jpg
        
        3. 30s × 30 fps = 900 frames
        
        4. Pour limiter le nombre à 100 :
           
           .. code-block:: bash
           
               ffmpeg -i video.mp4 -vf "select='not(mod(n\,N))'" -frames:v 100 frames/frame_%05d.jpg
           
           où N = total_frames // 100

**Résultat attendu :**

- 3 dossiers créés : ``frames_all/``, ``frames_10/``, ``frames_100/``
- Une grille 4×4 affichant 16 images représentatives

.. slide::

⚖️ Exercice 2 : Annotation avec Label Studio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez annoter manuellement des objets dans vos images extraites avec Label Studio.

**Objectif :** Créer un dataset annoté de qualité pour l'entraînement.

**Prérequis :**

.. code-block:: bash

   pip install label-studio

**Consigne :**

1) Lancez Label Studio :
   
   .. code-block:: bash
   
       label-studio start

2) Créez un projet "Detection_MonObjet" et importez les 100 images de l'exercice 1

3) Configurez l'annotation avec le template suivant :
   
   .. code-block:: xml
   
       <View>
         <Image name="image" value="$image"/>
         <RectangleLabels name="label" toName="image">
           <Label value="mon_objet" background="green"/>
         </RectangleLabels>
       </View>

4) Annotez **au moins 50 images** (travail en groupe recommandé) :
   
   - Dessinez une boîte englobante autour de l'objet
   - Soyez cohérent dans vos annotations
   - Si l'objet est partiellement visible (>30%), annotez-le

5) Exportez les annotations :
   
   - Cliquez sur "Export"
   - Choisissez le format "JSON"
   - Sauvegardez le fichier ``annotations.json``

6) Écrivez un script Python pour analyser les annotations :
   
   - Nombre d'images annotées
   - Nombre total d'objets annotés
   - Taille moyenne des boîtes (en pixels et en % de l'image)
   - Distribution des tailles (petit/moyen/grand)

**Code starter :**

.. code-block:: python

   import json
   import numpy as np
   from pathlib import Path
   import matplotlib.pyplot as plt

   def analyze_annotations(json_path, images_dir):
       """
       Analyse les statistiques des annotations Label Studio.
       
       Returns:
           dict avec statistiques
       """
       with open(json_path, 'r') as f:
           data = json.load(f)
       
       stats = {
           'n_images': 0,
           'n_objects': 0,
           'box_sizes': [],
           'box_ratios': []
       }
       
       for item in data:
           # À compléter : parser les annotations
           # Extraire les boîtes, calculer tailles, etc.
           pass
       
       return stats

   def plot_annotation_stats(stats):
       """Visualise les statistiques d'annotations."""
       fig, axes = plt.subplots(1, 3, figsize=(15, 4))
       
       # 1. Histogramme des tailles de boîtes
       axes[0].hist(stats['box_sizes'], bins=20, edgecolor='black')
       axes[0].set_title('Distribution des tailles de boîtes')
       axes[0].set_xlabel('Aire (pixels²)')
       axes[0].set_ylabel('Fréquence')
       
       # 2. Histogramme des ratios largeur/hauteur
       axes[1].hist(stats['box_ratios'], bins=20, edgecolor='black')
       axes[1].set_title('Distribution des ratios L/H')
       axes[1].set_xlabel('Ratio largeur/hauteur')
       
       # 3. Statistiques textuelles
       axes[2].axis('off')
       text = f"""
       Images annotées: {stats['n_images']}
       Objets totaux: {stats['n_objects']}
       Objets/image: {stats['n_objects']/stats['n_images']:.1f}
       
       Taille moyenne: {np.mean(stats['box_sizes']):.0f} px²
       Taille médiane: {np.median(stats['box_sizes']):.0f} px²
       Min: {np.min(stats['box_sizes']):.0f} px²
       Max: {np.max(stats['box_sizes']):.0f} px²
       """
       axes[2].text(0.1, 0.5, text, fontsize=12, verticalalignment='center')
       
       plt.tight_layout()
       plt.show()

   # Programme principal
   stats = analyze_annotations('annotations.json', 'frames_100/')
   plot_annotation_stats(stats)

**Questions :**

7) Combien de temps avez-vous mis pour annoter 50 images ?
8) Quelle est la taille moyenne d'une boîte ? Est-elle cohérente ?
9) Si vous avez annoté en groupe, comparez vos boîtes sur 5 images communes. Calculez l'IoU moyen. Est-il > 0.7 ?
10) Que faire si un objet est très petit (<10×10 pixels) ?

**Astuce :**

.. spoiler::
    .. discoverList::
        1. Structure typique du JSON Label Studio :
           
           .. code-block:: python
           
               item['data']['image']  # nom de l'image
               item['annotations'][0]['result']  # liste des annotations
               result[0]['value']['x']  # coordonnée x en %
               result[0]['value']['width']  # largeur en %
        
        2. Pour calculer l'IoU entre deux annotateurs :
           
           .. code-block:: python
           
               def calculate_iou(box1, box2):
                   x1_min, y1_min, x1_max, y1_max = box1
                   x2_min, y2_min, x2_max, y2_max = box2
                   
                   inter_xmin = max(x1_min, x2_min)
                   inter_ymin = max(y1_min, y2_min)
                   inter_xmax = min(x1_max, x2_max)
                   inter_ymax = min(y1_max, y2_max)
                   
                   inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
                   
                   box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                   box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                   union_area = box1_area + box2_area - inter_area
                   
                   return inter_area / union_area if union_area > 0 else 0
        
        3. Temps moyen : 30-60 secondes par image → 25-50 minutes pour 50 images

**Résultat attendu :**

- Au moins 50 images annotées
- Export JSON sauvegardé
- Graphiques des statistiques d'annotations

.. slide::

⚖️ Exercice 3 : Conversion des formats d'annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez convertir les annotations Label Studio en format YOLO pour l'entraînement.

**Objectif :** Maîtriser la conversion entre formats d'annotations et vérifier visuellement le résultat.

**Consigne :** Créer un script complet qui :

1) Charge le JSON exporté de Label Studio

2) Convertit chaque annotation en format YOLO :
   
   - Format : ``class_id x_center y_center width height`` (normalisé 0-1)
   - Un fichier ``.txt`` par image
   - Sauvegarder dans ``labels_yolo/``

3) Crée un fichier ``classes.txt`` avec les noms de classes

4) Vérifie visuellement 10 images en dessinant les boîtes converties

5) Calcule des métriques de validation :
   
   - Nombre de boîtes converties
   - Nombre de boîtes invalides (hors limites, taille nulle)
   - Coordonnées min/max pour détecter les erreurs

**Code complet :**

.. code-block:: python

   import json
   import os
   from pathlib import Path
   from PIL import Image
   import cv2
   import numpy as np

   def labelstudio_to_yolo(json_path, images_dir, output_dir, class_names):
       """
       Convertit Label Studio → YOLO.
       
       Returns:
           dict avec statistiques de conversion
       """
       Path(output_dir).mkdir(parents=True, exist_ok=True)
       
       with open(json_path, 'r', encoding='utf-8') as f:
           data = json.load(f)
       
       stats = {'converted': 0, 'invalid': 0, 'errors': []}
       
       for item in data:
           try:
               # Récupérer l'image
               image_url = item['data'].get('image', '')
               image_name = os.path.basename(image_url)
               image_path = os.path.join(images_dir, image_name)
               
               if not os.path.exists(image_path):
                   stats['errors'].append(f"Image non trouvée : {image_name}")
                   continue
               
               # Dimensions de l'image
               img = Image.open(image_path)
               img_width, img_height = img.size
               
               # Récupérer annotations
               annotations = item.get('annotations', [])
               if not annotations:
                   continue
               
               result = annotations[-1].get('result', [])
               yolo_lines = []
               
               for ann in result:
                   if ann.get('type') != 'rectanglelabels':
                       continue
                   
                   value = ann['value']
                   # Label Studio donne des pourcentages
                   x_percent = value['x']
                   y_percent = value['y']
                   w_percent = value['width']
                   h_percent = value['height']
                   
                   # Convertir en YOLO (centre, normalisé)
                   x_center = (x_percent + w_percent / 2) / 100.0
                   y_center = (y_percent + h_percent / 2) / 100.0
                   width = w_percent / 100.0
                   height = h_percent / 100.0
                   
                   # Validation
                   if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                          0 < width <= 1 and 0 < height <= 1):
                       stats['invalid'] += 1
                       stats['errors'].append(
                           f"Boîte invalide dans {image_name}: "
                           f"{x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f}"
                       )
                       continue
                   
                   # Classe
                   labels = value.get('rectanglelabels', [])
                   if not labels:
                       continue
                   
                   class_name = labels[0]
                   if class_name not in class_names:
                       stats['errors'].append(f"Classe inconnue : {class_name}")
                       continue
                   
                   class_id = class_names.index(class_name)
                   yolo_lines.append(
                       f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                   )
               
               # Sauvegarder le fichier .txt
               if yolo_lines:
                   output_name = Path(image_name).stem + '.txt'
                   output_path = os.path.join(output_dir, output_name)
                   
                   with open(output_path, 'w', encoding='utf-8') as f:
                       f.write('\n'.join(yolo_lines))
                   
                   stats['converted'] += 1
           
           except Exception as e:
               stats['errors'].append(f"Erreur : {str(e)}")
       
       return stats

   def verify_conversion(images_dir, labels_dir, class_names, n_samples=10):
       """Vérifie visuellement les conversions."""
       image_files = list(Path(images_dir).glob('*.jpg'))[:n_samples]
       
       fig, axes = plt.subplots(2, 5, figsize=(20, 8))
       axes = axes.flatten()
       
       for i, img_path in enumerate(image_files):
           img = cv2.imread(str(img_path))
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           h, w = img.shape[:2]
           
           # Charger les labels YOLO
           label_path = Path(labels_dir) / (img_path.stem + '.txt')
           if label_path.exists():
               with open(label_path, 'r') as f:
                   for line in f:
                       class_id, xc, yc, width, height = map(float, line.strip().split())
                       
                       # Convertir en pixels
                       x1 = int((xc - width/2) * w)
                       y1 = int((yc - height/2) * h)
                       x2 = int((xc + width/2) * w)
                       y2 = int((yc + height/2) * h)
                       
                       # Dessiner
                       cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                       label = class_names[int(class_id)]
                       cv2.putText(img, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           
           axes[i].imshow(img)
           axes[i].set_title(img_path.name, fontsize=8)
           axes[i].axis('off')
       
       plt.tight_layout()
       plt.savefig('conversion_verification.png')
       print("✓ Vérification sauvegardée : conversion_verification.png")

   # Programme principal
   if __name__ == "__main__":
       class_names = ['mon_objet']  # Adapter selon vos classes
       
       # Conversion
       stats = labelstudio_to_yolo(
           json_path='annotations.json',
           images_dir='frames_100/',
           output_dir='labels_yolo/',
           class_names=class_names
       )
       
       print(f"✓ Conversion terminée :")
       print(f"  - Fichiers convertis : {stats['converted']}")
       print(f"  - Boîtes invalides : {stats['invalid']}")
       
       if stats['errors']:
           print(f"  - Erreurs :")
           for err in stats['errors'][:5]:  # Afficher max 5 erreurs
               print(f"      {err}")
       
       # Créer classes.txt
       with open('classes.txt', 'w') as f:
           f.write('\n'.join(class_names))
       print(f"✓ Fichier classes.txt créé")
       
       # Vérification visuelle
       verify_conversion('frames_100/', 'labels_yolo/', class_names, n_samples=10)

**Questions :**

11) Pourquoi normaliser les coordonnées entre 0 et 1 ?
12) Que signifie une boîte avec ``x_center > 1`` ? Comment corriger ?
13) Comment gérer les annotations avec plusieurs objets sur une même image ?

**Astuce :**

.. spoiler::
    .. discoverList::
        1. Normalisation : permet d'être indépendant de la résolution de l'image
        2. Si ``x_center > 1`` → erreur dans le parsing ou l'export Label Studio
        3. Une ligne par objet dans le fichier .txt YOLO
        4. Toujours vérifier visuellement au moins 10 images après conversion !

**Résultat attendu :**

- Dossier ``labels_yolo/`` avec un `.txt` par image
- Fichier ``classes.txt``
- Image ``conversion_verification.png`` montrant 10 exemples avec boîtes dessinées
- Aucune boîte invalide (ou très peu)

.. slide::

🌶️ Exercice 4 : Entraînement d'un détecteur Faster R-CNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez entraîner un détecteur d'objets custom avec Faster R-CNN sur vos propres données.

**Objectif :** Mettre en pratique tout le pipeline d'entraînement et obtenir un modèle fonctionnel.

**Prérequis :**

- Exercices 1-3 complétés
- Au moins 50 images annotées et converties
- PyTorch + torchvision installés

**Consigne :** Créer un pipeline complet d'entraînement :

1) Organiser le dataset :
   
   - 70% train (35 images)
   - 15% val (7 images)
   - 15% test (8 images)

2) Créer une classe ``DetectionDataset`` qui charge images + annotations YOLO

3) Charger Faster R-CNN pré-entraîné et adapter pour votre nombre de classes

4) Entraîner pendant 10 epochs avec :
   
   - Batch size = 2
   - Learning rate = 0.005
   - Optimiseur SGD avec momentum

5) Sauvegarder le meilleur modèle (basé sur val_loss)

6) Évaluer sur le test set et afficher :
   
   - Test loss
   - Exemples de prédictions (avec seuil = 0.5)

**Code complet à compléter :**

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch.utils.data import Dataset, DataLoader
   from torchvision.models.detection import fasterrcnn_resnet50_fpn
   from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
   from torchvision.transforms import functional as F
   from PIL import Image
   from pathlib import Path
   import os
   from tqdm import tqdm

   class DetectionDataset(Dataset):
       """Dataset pour la détection d'objets (format YOLO)."""
       
       def __init__(self, images_dir, labels_dir, classes_file):
           self.images_dir = Path(images_dir)
           self.labels_dir = Path(labels_dir)
           
           # Charger les classes
           with open(classes_file, 'r') as f:
               self.classes = [line.strip() for line in f]
           
           # Lister les images
           self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
           
           print(f"Dataset : {len(self.image_files)} images")
       
       def __len__(self):
           return len(self.image_files)
       
       def __getitem__(self, idx):
           # Charger l'image
           img_path = self.image_files[idx]
           img = Image.open(img_path).convert('RGB')
           img_width, img_height = img.size
           
           # Charger les annotations YOLO
           label_path = self.labels_dir / (img_path.stem + '.txt')
           boxes = []
           labels = []
           
           if label_path.exists():
               with open(label_path, 'r') as f:
                   for line in f:
                       # À compléter : parser la ligne YOLO
                       # Convertir en [x1, y1, x2, y2] en pixels
                       pass
           
           # Créer le target (format torchvision)
           target = {}
           target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
           target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
           target['image_id'] = torch.tensor([idx])
           
           # Si pas de boîtes, créer des tenseurs vides
           if len(boxes) == 0:
               target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
               target['labels'] = torch.zeros((0,), dtype=torch.int64)
           
           img_tensor = F.to_tensor(img)
           return img_tensor, target

   def get_model(num_classes):
       """Charge Faster R-CNN et adapte pour num_classes."""
       model = fasterrcnn_resnet50_fpn(pretrained=True)
       
       # À compléter : remplacer la tête de classification
       in_features = model.roi_heads.box_predictor.cls_score.in_features
       # ...
       
       return model

   def collate_fn(batch):
       """Fonction pour assembler un batch avec nombre variable de boîtes."""
       return tuple(zip(*batch))

   def train_one_epoch(model, optimizer, data_loader, device):
       """Entraîne pendant une epoch."""
       model.train()
       total_loss = 0
       
       pbar = tqdm(data_loader, desc="Training")
       for images, targets in pbar:
           # À compléter : déplacer sur GPU, forward, backward
           pass
       
       return total_loss / len(data_loader)

   @torch.no_grad()
   def evaluate(model, data_loader, device):
       """Évalue le modèle."""
       model.train()  # Faster R-CNN garde train() même en eval
       total_loss = 0
       
       for images, targets in tqdm(data_loader, desc="Validation"):
           # À compléter
           pass
       
       return total_loss / len(data_loader)

   # Programme principal
   if __name__ == "__main__":
       # Configuration
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       print(f"Device : {device}")
       
       # Créer les datasets (à compléter : split train/val/test)
       train_dataset = DetectionDataset(
           images_dir='data/images/train',
           labels_dir='data/labels/train',
           classes_file='classes.txt'
       )
       
       val_dataset = DetectionDataset(
           images_dir='data/images/val',
           labels_dir='data/labels/val',
           classes_file='classes.txt'
       )
       
       # Dataloaders
       train_loader = DataLoader(
           train_dataset, batch_size=2, shuffle=True,
           collate_fn=collate_fn, num_workers=4
       )
       
       val_loader = DataLoader(
           val_dataset, batch_size=2, shuffle=False,
           collate_fn=collate_fn, num_workers=4
       )
       
       # Modèle
       num_classes = 2  # 1 classe + background
       model = get_model(num_classes)
       model.to(device)
       
       # Optimiseur
       params = [p for p in model.parameters() if p.requires_grad]
       optimizer = torch.optim.SGD(
           params, lr=0.005, momentum=0.9, weight_decay=0.0005
       )
       
       # Scheduler
       lr_scheduler = torch.optim.lr_scheduler.StepLR(
           optimizer, step_size=3, gamma=0.1
       )
       
       # Entraînement
       num_epochs = 10
       best_loss = float('inf')
       
       for epoch in range(num_epochs):
           # Train
           train_loss = train_one_epoch(model, optimizer, train_loader, device)
           
           # Validation
           val_loss = evaluate(model, val_loader, device)
           
           # Learning rate
           lr_scheduler.step()
           
           print(f"\nEpoch {epoch+1}/{num_epochs}")
           print(f"  Train Loss: {train_loss:.4f}")
           print(f"  Val Loss:   {val_loss:.4f}")
           
           # Sauvegarder le meilleur
           if val_loss < best_loss:
               best_loss = val_loss
               torch.save(model.state_dict(), 'best_detector.pth')
               print("  ✓ Meilleur modèle sauvegardé")
       
       print("\n✓ Entraînement terminé !")

**Questions :**

14) Pourquoi utilise-t-on un batch_size si petit (2) ?
15) Que se passe-t-il si on oublie le ``+1`` lors du chargement des labels ?
16) Pourquoi doit-on garder ``model.train()`` même pendant l'évaluation ?
17) Comment savoir si le modèle overfitte ?

**Astuce :**

.. spoiler::
    .. discoverList::
        1. Batch_size petit car la détection consomme beaucoup de mémoire GPU
        2. Le +1 est nécessaire car PyTorch réserve l'ID 0 pour le background
        3. C'est une particularité de l'implémentation torchvision de Faster R-CNN
        4. Overfitting : train_loss baisse mais val_loss augmente
        5. Pour parser une ligne YOLO :
           
           .. code-block:: python
           
               class_id, xc, yc, w, h = map(float, line.strip().split())
               x1 = (xc - w/2) * img_width
               y1 = (yc - h/2) * img_height
               x2 = (xc + w/2) * img_width
               y2 = (yc + h/2) * img_height
               boxes.append([x1, y1, x2, y2])
               labels.append(int(class_id) + 1)

**Résultat attendu :**

- Modèle ``best_detector.pth`` sauvegardé
- Val loss < 0.5 après 10 epochs
- Train loss qui diminue régulièrement

.. slide::

🌶️ Exercice 5 : Inférence et détection sur vidéo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez utiliser votre modèle entraîné pour détecter des objets sur de nouvelles images et vidéos.

**Objectif :** Maîtriser l'inférence et créer une vidéo avec détections.

**Consigne :** Créer un script qui :

1) Charge le modèle ``best_detector.pth``

2) Effectue l'inférence sur 10 nouvelles images de test

3) Dessine les boîtes de détection avec :
   
   - Rectangle vert si confiance > 0.7
   - Rectangle jaune si 0.5 < confiance < 0.7
   - Label + score affiché

4) Sauvegarde les images avec détections

5) Traite une nouvelle vidéo frame par frame et :
   
   - Applique la détection sur chaque frame
   - Dessine les boîtes
   - Crée une vidéo de sortie ``output_detected.mp4``
   - Affiche le FPS moyen

6) (Bonus) Compte le nombre d'objets détectés dans la vidéo et trace un graphique du nombre d'objets par frame

**Code starter :**

.. code-block:: python

   import cv2
   import torch
   from torchvision.transforms import functional as F
   from PIL import Image
   import matplotlib.pyplot as plt

   def detect_objects(model, image_path, device, threshold=0.5):
       """Détecte les objets dans une image."""
       # À compléter
       pass

   def draw_detections(img_np, boxes, labels, scores, class_names, threshold=0.5):
       """Dessine les boîtes sur l'image."""
       img = img_np.copy()
       
       for box, label, score in zip(boxes, labels, scores):
           if score < threshold:
               continue
           
           x1, y1, x2, y2 = box.astype(int)
           
           # Couleur selon confiance
           if score > 0.7:
               color = (0, 255, 0)  # Vert
           else:
               color = (255, 255, 0)  # Jaune
           
           cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
           
           class_name = class_names[label - 1]
           text = f"{class_name}: {score:.2f}"
           cv2.putText(img, text, (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
       
       return img

   def process_video(model, video_path, output_path, device, class_names, threshold=0.5):
       """Traite une vidéo avec détection."""
       cap = cv2.VideoCapture(video_path)
       fps = int(cap.get(cv2.CAP_PROP_FPS))
       width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
       
       model.eval()
       frame_count = 0
       detections_per_frame = []
       
       import time
       start_time = time.time()
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           # À compléter : détection sur la frame
           # boxes, labels, scores = ...
           
           # Compter les détections
           n_detections = len([s for s in scores if s >= threshold])
           detections_per_frame.append(n_detections)
           
           # Dessiner
           frame_with_boxes = draw_detections(
               frame, boxes, labels, scores, class_names, threshold
           )
           
           out.write(frame_with_boxes)
           frame_count += 1
       
       elapsed = time.time() - start_time
       fps_avg = frame_count / elapsed
       
       cap.release()
       out.release()
       
       print(f"✓ Vidéo traitée : {frame_count} frames en {elapsed:.1f}s ({fps_avg:.1f} fps)")
       
       return detections_per_frame

   # Programme principal
   if __name__ == "__main__":
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       class_names = ['mon_objet']
       
       # Charger le modèle
       model = get_model(num_classes=2)
       model.load_state_dict(torch.load('best_detector.pth'))
       model.to(device)
       model.eval()
       
       # 1. Test sur images
       test_images = list(Path('data/images/test').glob('*.jpg'))
       for img_path in test_images[:10]:
           boxes, labels, scores = detect_objects(model, img_path, device, threshold=0.5)
           
           img = cv2.imread(str(img_path))
           img_detected = draw_detections(img, boxes, labels, scores, class_names)
           
           output_path = f'results/test_{img_path.name}'
           cv2.imwrite(output_path, img_detected)
           print(f"✓ {img_path.name}: {len(boxes)} objets détectés")
       
       # 2. Test sur vidéo
       detections = process_video(
           model=model,
           video_path='new_video.mp4',
           output_path='output_detected.mp4',
           device=device,
           class_names=class_names,
           threshold=0.6
       )
       
       # 3. Graphique du nombre de détections
       plt.figure(figsize=(12, 4))
       plt.plot(detections, linewidth=0.5)
       plt.xlabel('Frame')
       plt.ylabel('Nombre d\'objets détectés')
       plt.title('Détections par frame')
       plt.grid(True, alpha=0.3)
       plt.savefig('detections_per_frame.png')
       print("✓ Graphique sauvegardé")

**Questions :**

18) Quel est le FPS moyen de votre détecteur ? Est-ce suffisant pour du temps réel ?
19) Comment améliorer la vitesse d'inférence ?
20) Que se passe-t-il si vous baissez le threshold à 0.3 ? Et à 0.8 ?
21) Comment gérer les faux positifs (détections incorrectes) ?

**Astuce :**

.. spoiler::
    .. discoverList::
        1. FPS temps réel : >30 fps
        2. Améliorer vitesse :
           
           - Réduire résolution des frames
           - Utiliser un modèle plus léger (YOLOv8n)
           - Traiter 1 frame sur 2
           - Utiliser TensorRT ou ONNX
        
        3. Threshold bas : plus de détections mais plus de faux positifs
        4. Threshold élevé : moins de détections mais plus précises
        5. Faux positifs : augmenter le threshold ou ré-entraîner avec plus de données

**Résultat attendu :**

- 10 images de test avec boîtes dessinées dans ``results/``
- Vidéo ``output_detected.mp4`` avec détections
- Graphique ``detections_per_frame.png``
- FPS moyen affiché

.. slide::

🌶️ Exercice 6 (Bonus) : Comparaison avec YOLO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice bonus, vous allez comparer votre Faster R-CNN avec YOLOv8.

**Objectif :** Comprendre les différences pratiques entre les deux approches.

**Consigne :**

1) Installer Ultralytics :
   
   .. code-block:: bash
   
       pip install ultralytics

2) Créer le fichier de configuration ``dataset.yaml`` :
   
   .. code-block:: yaml
   
       path: /path/to/dataset
       train: images/train
       val: images/val
       
       nc: 1
       names: ['mon_objet']

3) Entraîner YOLOv8 nano :
   
   .. code-block:: python
   
       from ultralytics import YOLO
       
       model = YOLO('yolov8n.pt')
       results = model.train(
           data='dataset.yaml',
           epochs=50,
           imgsz=640,
           batch=16
       )

4) Comparer les deux modèles sur le test set :
   
   - Temps d'inférence moyen par image
   - Précision (détections correctes vs totales)
   - Nombre de paramètres
   - Taille du fichier du modèle

5) Créer un tableau comparatif et tracer un graphique vitesse vs précision

**Questions bonus :**

22) Quel modèle est le plus rapide ? Par quel facteur ?
23) Quel modèle est le plus précis sur votre dataset ?
24) Lequel choisiriez-vous pour un déploiement sur smartphone ?
25) Lequel choisiriez-vous pour une application médicale critique ?

**Résultat attendu :**

.. list-table::
   :header-rows: 1
   
   * - Modèle
     - Temps/image
     - mAP@0.5
     - Paramètres
     - Taille fichier
   * - Faster R-CNN
     - ~100ms
     - ?
     - ?
     - ~160 MB
   * - YOLOv8n
     - ~10ms
     - ?
     - ?
     - ~6 MB

---

**Félicitations !** Vous maîtrisez maintenant le pipeline complet de détection d'objets, de la capture vidéo jusqu'au déploiement ! 🎉
