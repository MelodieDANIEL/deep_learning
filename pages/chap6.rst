.. slide::

Chapitre 6 — Détection d'objets avec des boîtes englobantes
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

📖 1. Classification vs Détection : comprendre la différence
----------------------

1.1. Classification d'images (chapitres précédents)
~~~~~~~~~~~~~~~~~~~

Dans les chapitres précédents, nous avons travaillé sur la **classification d'images** : le modèle devait répondre à la question *"Qu'est-ce qu'il y a dans cette image ?"*

**Exemple** : 

- Entrée : une image $$224×224$$ pixels
- Sortie : une classe parmi N possibles (ex : "chat", "chien", "voiture")
- Une seule prédiction par image

.. code-block:: python

   # Classification : une image → une classe
   output = model(image)  # Shape: [batch_size, num_classes]
   predicted_class = torch.argmax(output, dim=1)
   print(f"Cette image contient : {classes[predicted_class]}")

.. slide::

1.2. Détection d'objets : localiser ET classifier
~~~~~~~~~~~~~~~~~~~

La **détection d'objets** va plus loin : le modèle doit répondre à *"Qu'est-ce qu'il y a dans cette image ET où se trouve chaque objet ?"*

**Pour chaque objet détecté, le modèle doit fournir** :

1. **La classe** de l'objet (ex : "personne", "voiture", "chien")
2. **La boîte englobante** (bounding box en anglais) : 4 coordonnées définissant un rectangle autour de l'objet
3. **Un score de confiance** : probabilité que la détection soit correcte (0 à 1)

**Exemple de sortie** :

.. code-block:: python

   # Détection : une image → plusieurs objets localisés
   outputs = model(image)
   # outputs[0]['boxes']: tensor([[x1, y1, x2, y2], [x1, y1, x2, y2], ...])
   # outputs[0]['labels']: tensor([1, 3, 1, ...])  # IDs des classes
   # outputs[0]['scores']: tensor([0.95, 0.87, 0.76, ...])  # Confiances

💡 **Intuition** : imaginez que vous regardez une photo de famille. La classification dirait "photo de groupe", tandis que la détection indiquerait "3 personnes aux positions (x1,y1,x2,y2), (x3,y3,x4,y4), (x5,y5,x6,y6)".

.. slide::

1.3. Qu'est-ce qu'une boîte englobante ?
~~~~~~~~~~~~~~~~~~~

Une **boîte englobante** est un rectangle défini par 4 valeurs. Il existe plusieurs formats :

**Format 1 : (x1, y1, x2, y2)** —> utilisé par PyTorch/torchvision

- ``x1, y1`` : coordonnées du coin supérieur gauche
- ``x2, y2`` : coordonnées du coin inférieur droit

**Format 2 : (x, y, w, h)** —> utilisé par COCO

- ``x, y`` : coordonnées du coin supérieur gauche
- ``w`` : largeur de la boîte
- ``h`` : hauteur de la boîte

**Format 3 : (x_center, y_center, w, h) normalisé** —> utilisé par YOLO

- ``x_center, y_center`` : coordonnées du centre (normalisées entre 0 et 1)
- ``w, h`` : largeur et hauteur (normalisées entre 0 et 1)

.. code-block:: text

   Exemple d'une image $$640×480$$ pixels avec un objet :
   
   Format PyTorch : [100, 50, 300, 250]
   → Rectangle du pixel (100,50) au pixel (300,250)
   
   Format COCO : [100, 50, 200, 200]
   → Rectangle démarrant en (100,50) de taille $$200×200$$
   
   Format YOLO : [0.3125, 0.3125, 0.3125, 0.4167]
   → Centre à 31.25% de la largeur/hauteur, boîte de 31.25%×41.67% de l'image

.. slide::

1.4. Applications concrètes de la détection
~~~~~~~~~~~~~~~~~~~

La détection d'objets est au cœur de nombreuses applications :

- **Véhicules autonomes** : détecter piétons, voitures, panneaux
- **Surveillance vidéo** : compter les personnes, détecter des comportements suspects
- **Commerce** : compter les produits en rayon, détecter les vols
- **Médical** : localiser des tumeurs, anomalies sur des radiographies
- **Réalité augmentée** : détecter des objets pour y superposer des informations

💡 Dans ce chapitre, nous allons apprendre à créer notre propre détecteur d'objets personnalisé, de A à Z !

.. slide::

📖 2. Préparer les données : de la vidéo aux images annotées
----------------------

Le pipeline complet pour créer un dataset de détection :

1. **Capturer une vidéo** de l'objet à détecter
2. **Extraire des images** (frames) depuis la vidéo
3. **Annoter** les objets sur chaque image
4. **Exporter** les annotations dans un format standard
5. **Organiser** le dataset pour l'entraînement

Voyons chaque étape en détail.

.. slide::

2.1. Capturer une vidéo
~~~~~~~~~~~~~~~~~~~

**Objectif** : filmer l'objet que vous voulez détecter sous différents angles et conditions.

**Conseils pratiques** :

- Durée : 30 secondes à 2 minutes suffisent
- Variété : filmez l'objet sous différents angles, distances, éclairages
- Stabilité : évitez les mouvements trop brusques
- Qualité : résolution HD ($$1280×720$$ ou $$1920×1080$$) recommandée

**Exemple** : pour détecter une bouteille d'eau :

- Filmez la bouteille sur un bureau (30 sec)
- Filmez-la dans une main (20 sec)
- Filmez-la avec différents arrière-plans (30 sec)

💡 **Astuce** : plus vous capturez de variété, meilleur sera votre détecteur !

.. note::

   **💡 Vous pouvez commencer avec beaucoup moins !**
   
   - 10-20 photos de votre smartphone suffisent pour débuter
   - Pas besoin de vidéo : des images fixes fonctionnent très bien
   - Résolution modeste (640×480) acceptable pour un prototype
   - Même avec peu de variété, vous obtiendrez déjà des résultats !

.. slide::

.. slide::

2.2. Installation d'OpenCV
~~~~~~~~~~~~~~~~~~~

**OpenCV (cv2)** est une bibliothèque Python très puissante pour manipuler des vidéos. Elle s'utilise directement en Python sans installer d'outils externes.

.. code-block:: bash

   # Installer OpenCV dans votre environnement virtuel
   pip install opencv-python

.. slide::

2.3. Script d'extraction de base
~~~~~~~~~~~~~~~~~~~

Voici un script complet pour extraire toutes les frames d'une vidéo :

.. code-block:: python

   import cv2
   import os

   def extraire_frames(video_path, output_dir):
       """
       Extrait toutes les frames d'une vidéo.
       
       Args:
           video_path: chemin vers la vidéo
           output_dir: dossier où sauvegarder les images
       """
       # Créer le dossier de sortie (exist_ok=True évite l'erreur si le dossier existe déjà)
       os.makedirs(output_dir, exist_ok=True)
       
       # Ouvrir la vidéo
       cap = cv2.VideoCapture(video_path)
       
       # Vérifier que la vidéo s'ouvre correctement
       if not cap.isOpened():
           print(f"❌ Erreur : impossible d'ouvrir {video_path}")
           return

       # Obtenir les propriétés de la vidéo (fps, nombre total de frames)
       fps = cap.get(cv2.CAP_PROP_FPS)
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
       print(f"📹 Vidéo : {total_frames} frames à {fps:.2f} fps")
       
       frame_count = 0
       
       while True:
           # Lire la frame suivante
           ret, frame = cap.read()
           
           # Si plus de frames, sortir de la boucle
           if not ret:
               break
           
           # Sauvegarder la frame en jpg pour compression
           output_path = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
           cv2.imwrite(output_path, frame)
           
           frame_count += 1
       
       # Libérer les ressources
       cap.release()
       
       print(f"✓ {frame_count} frames extraites dans {output_dir}")

   # Utilisation
   extraire_frames('ma_video.mp4', 'frames/')

.. warning::

   ⚠️ **Attention à la quantité !**
   
   Une vidéo de 30 secondes à 30 fps génère **900 images**. C'est souvent trop pour annoter manuellement !

.. slide::

2.4. Extraction intelligente (sous-échantillonnage)
~~~~~~~~~~~~~~~~~~~

Pour réduire le nombre d'images à annoter, on extrait seulement certaines frames :

.. code-block:: python

   import cv2
   import os

   def extraire_frames_espacees(video_path, output_dir, intervalle=10):
       """
       Extrait 1 frame tous les N frames.
       
       Args:
           video_path: chemin vers la vidéo
           output_dir: dossier de sortie
           intervalle: extraire 1 frame tous les N frames (ex: 10)
       """
       os.makedirs(output_dir, exist_ok=True)
       
       cap = cv2.VideoCapture(video_path)
       
       if not cap.isOpened():
           print(f"❌ Erreur : impossible d'ouvrir {video_path}")
           return
       
       fps = cap.get(cv2.CAP_PROP_FPS)
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
       print(f"📹 Extraction de 1 frame tous les {intervalle} frames")
       print(f"   Total attendu : ~{total_frames // intervalle} images")
       
       frame_count = 0
       saved_count = 0
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           # Sauvegarder seulement toutes les N frames
           if frame_count % intervalle == 0:
               output_path = os.path.join(output_dir, f'frame_{saved_count:05d}.jpg')
               cv2.imwrite(output_path, frame)
               saved_count += 1
           
           frame_count += 1
       
       cap.release()
       
       print(f"✓ {saved_count} frames extraites sur {frame_count} totales")

   # Exemple : extraire 1 frame toutes les 10 frames
   extraire_frames_espacees('ma_video.mp4', 'frames/', intervalle=10)

**Recommandation pratique** : pour débuter, extraire 50-200 images est un bon compromis entre travail d'annotation et qualité du modèle.

**Règles de calcul de l'intervalle** :

- Vidéo à 30 fps, 1 frame/seconde → ``intervalle=30``
- Vidéo à 30 fps, 1 frame toutes les 10 frames → ``intervalle=10``
- Extraire ~100 images d'une vidéo de 900 frames → ``intervalle=9``

.. slide::

2.5. Redimensionner les images à l'extraction
~~~~~~~~~~~~~~~~~~~

Pour économiser l'espace disque et accélérer le traitement, on peut redimensionner directement.

.. warning::

   ⚠️ **Attention à la déformation !**
   
   Si votre vidéo n'est **pas carrée** (ex : $$1920×1080$$) et que vous redimensionnez en **carré** (ex : $$224×224$$), l'image sera **déformée** (écrasée ou étirée).
   
   **Deux solutions** :
   
   1. **Crop au centre** (RECOMMANDÉ) : découper un carré au centre avant de redimensionner
   2. **Padding** : ajouter des bordures noires pour garder le ratio

.. slide::

Voici les deux approches :

**Approche 1 : Crop au centre (recommandée - pas de déformation)**

.. code-block:: python

   import cv2
   import os

   def extraire_frames_crop_redimensionner(video_path, output_dir, intervalle=10, 
                                           target_size=224):
       """
       Extrait, crop au centre en carré, puis redimensionne.
       ÉVITE la déformation en découpant l'image.
       
       Args:
           video_path: chemin vers la vidéo
           output_dir: dossier de sortie
           intervalle: extraire 1 frame tous les N frames
           target_size: taille finale du carré (ex: 224 pour 224×224)
       """
       os.makedirs(output_dir, exist_ok=True)
       
       cap = cv2.VideoCapture(video_path)
       
       if not cap.isOpened():
           print(f"❌ Erreur : impossible d'ouvrir {video_path}")
           return
       
       # Obtenir les dimensions originales
       original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       
       print(f"📹 Résolution originale : {original_width}×{original_height}")
       print(f"📐 Nouvelle résolution : {target_size}×{target_size} (carré)")
       print(f"✂️  Méthode : Crop au centre (pas de déformation)")
       
       frame_count = 0
       saved_count = 0
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           if frame_count % intervalle == 0:
               # ÉTAPE 1 : Crop au centre pour obtenir un carré
               h, w = frame.shape[:2]
               size = min(h, w)  # Prendre la plus petite dimension
               
               # Calculer les coordonnées du crop au centre
               start_y = (h - size) // 2
               start_x = (w - size) // 2
               
               # Découper le carré au centre
               cropped = frame[start_y:start_y+size, start_x:start_x+size]
               
               # ÉTAPE 2 : Redimensionner le carré à la taille souhaitée
               resized = cv2.resize(cropped, (target_size, target_size))
               
               # Sauvegarder
               output_path = os.path.join(output_dir, f'frame_{saved_count:05d}.jpg')
               cv2.imwrite(output_path, resized)
               saved_count += 1
           
           frame_count += 1
       
       cap.release()
       
       print(f"✓ {saved_count} frames extraites, croppées et redimensionnées")

   # Exemple : extraire 1 frame/seconde en 224×224 (format standard CNN)
   extraire_frames_crop_redimensionner('ma_video.mp4', 'frames/', 
                                       intervalle=30, target_size=224)


.. slide::

**Approche 2 : Redimensionnement direct (DÉCONSEILLÉ si ratio différent)**

.. code-block:: python

   def extraire_frames_redimensionner_simple(video_path, output_dir, intervalle=10, 
                                             target_width=640, target_height=480):
       """
       Redimensionne directement sans crop.
       ⚠️ ATTENTION : déforme l'image si le ratio change !
       """
       os.makedirs(output_dir, exist_ok=True)
       
       cap = cv2.VideoCapture(video_path)
       
       if not cap.isOpened():
           print(f"❌ Erreur : impossible d'ouvrir {video_path}")
           return
       
       original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       
       print(f"📹 Résolution originale : {original_width}×{original_height}")
       print(f"📐 Nouvelle résolution : {target_width}×{target_height}")
       
       # Vérifier si le ratio va changer
       original_ratio = original_width / original_height
       target_ratio = target_width / target_height
       
       if abs(original_ratio - target_ratio) > 0.01:
           print(f"⚠️  ATTENTION : Le ratio va changer !")
           print(f"    Original : {original_ratio:.2f}")
           print(f"    Cible : {target_ratio:.2f}")
           print(f"    → L'image sera déformée !")
       
       frame_count = 0
       saved_count = 0
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           if frame_count % intervalle == 0:
               # Redimensionner directement (PEUT DÉFORMER !)
               resized = cv2.resize(frame, (target_width, target_height))
               
               output_path = os.path.join(output_dir, f'frame_{saved_count:05d}.jpg')
               cv2.imwrite(output_path, resized)
               saved_count += 1
           
           frame_count += 1
       
       cap.release()
       
       print(f"✓ {saved_count} frames extraites et redimensionnées")

   # Exemple : ⚠️ Vidéo 16:9 → carré = DÉFORMATION !
   # extraire_frames_redimensionner_simple('ma_video.mp4', 'frames/', 
   #                                        intervalle=30, 
   #                                        target_width=224, target_height=224)

💡 **Recommandations** :

- **Pour la détection d'objets** : utilisez le **crop au centre** pour éviter les déformations
- **Pour la classification** : le crop au centre est aussi préférable
- Résolutions recommandées : $$224×224$$ (standard CNN), $$640×480$$ (compromis vitesse/qualité), $$800×600$$ (bonne qualité)

.. slide::

📖 3. Annotation avec Label Studio
----------------------

**Label Studio** est un outil open-source d'annotation collaborative qui permet de créer des boîtes englobantes, de gérer plusieurs annotateurs et d'exporter dans différents formats.

3.1. Installation et premier lancement
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Installer Label Studio (dans votre environnement virtuel)
   pip install label-studio
   
   # Lancer Label Studio
   label-studio start
   
   # L'interface web s'ouvre automatiquement sur http://localhost:8080

**Premier lancement : création du compte**

Au premier lancement, Label Studio vous demande de créer un compte.

.. note::

   💡 **Travail collaboratif**
   
   Si vous souhaitez travailler en équipe, vous pourrez inviter vos collègues via **"Invite People"** (voir section 3.5). Label Studio leur enverra automatiquement un email d'invitation.

**Si vous avez déjà un compte** : entrez simplement votre email et mot de passe pour vous connecter.

**En cas de problème** : si Label Studio ne s'ouvre pas automatiquement, ouvrez manuellement votre navigateur et allez sur ``http://localhost:8080``

.. slide::

3.2. Créer un projet d'annotation
~~~~~~~~~~~~~~~~~~~

**Étapes dans l'interface web** :

1. Cliquer sur "Create Project"

2. Donner un nom au projet (ex : "Detection_Bouteille")

3. **Import des données** :
   
   - Onglet "Data Import"
   - Sélectionner tous les fichiers du dossier ``frames/`` (ou le dossier où vous avez extrait les images)
   - Cliquer sur "Import"

4. **Configuration de l'annotation** :
   
   - Cliquez sur votre projet pour l'ouvrir
   - Cliquez sur "Settings" (en haut à droite ou dans le menu du projet)
   - Allez dans l'onglet "Labeling Interface"
   - Cliquez sur "Browse Templates"
   - Sélectionnez "Computer Vision"
   - Choisissez "Object Detection with Bounding Boxes"
   - Une page s'ouvre pour définir les labels (classes d'objets)

.. slide::

3.3. Définir les classes d'objets
~~~~~~~~~~~~~~~~~~~

Après avoir choisi le template, une interface apparaît où vous pouvez définir vos labels (classes d'objets).

**Méthode simple : ajouter des labels via l'interface**

1. Dans le champ "Add Label Name", entrez le nom de votre première classe (ex : "bouteille")
2. Cliquez sur "Add" ou appuyez sur Entrée
3. Répétez pour chaque classe d'objet à détecter
4. Cliquez sur "Save" pour valider

**Si vous préférez éditer le code XML directement**, vous pouvez voir/modifier le code de configuration :

**Exemple pour détecter des bouteilles et des gobelets** :

.. code-block:: xml

   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label value="bouteille" background="green"/>
       <Label value="gobelet" background="blue"/>
     </RectangleLabels>
   </View>

💡 **Astuce** : commencez avec une seule classe pour simplifier. Vous pourrez toujours ajouter des classes plus tard.

.. slide::

3.4. Annoter les images
~~~~~~~~~~~~~~~~~~~

**Pour créer une annotation** :

1. Cliquer sur une tâche (image) dans la liste
2. Sélectionner la classe dans le panneau en bas de l'image (ex : "bouteille")
3. Dessiner un rectangle autour de l'objet :
   
   - Cliquer et maintenir le bouton de la souris
   - Déplacer pour créer le rectangle
   - Relâcher quand l'objet est bien encadré

4. Répéter pour tous les objets de l'image
5. Cliquer sur "Submit" pour valider l'annotation

**Modifier une annotation existante** :

- **Double-cliquer** sur un rectangle pour le sélectionner
- Vous pouvez alors :
  
  - **Déplacer** le rectangle en le faisant glisser
  - **Redimensionner** en tirant sur les coins ou les bords
  - **Changer la classe** dans le panneau de droite
  - **Supprimer** avec la touche ``Suppr``

**Bonnes pratiques d'annotation** :

- La boîte doit englober **tout l'objet** visible (pas trop serrée, pas trop large)
- Si un objet est **partiellement visible** (coupé par le bord), l'annoter quand même
- Si un objet est **très petit** (<10 pixels), c'est optionnel (difficiles à détecter)
- **Cohérence** : gardez le même style d'annotation d'une image à l'autre

.. slide::

3.5. Annotation collaborative : inviter des personnes
~~~~~~~~~~~~~~~~~~~

Pour travailler en équipe sur l'annotation, suivez ces étapes :

**Étape 1 : Inviter des personnes**

1. Dans Label Studio, cliquez sur l'icône **Organization** (en haut à droite, icône avec plusieurs personnes)

2. Allez dans l'onglet **"People"**

3. Cliquez sur le bouton **"Invite People"**

4. Entrez les adresses email de vos collègues (ex : ``marie.dupont@example.com``, ``paul.martin@example.com``)

5. Choisissez le rôle pour chaque personne :
   
   - **Annotator** : peut uniquement annoter les images
   - **Reviewer** : peut annoter ET valider/corriger les annotations des autres
   - **Manager** : peut gérer les projets et les paramètres

6. Cliquez sur **"Send Invitations"**

7. Vos collègues recevront un email avec un lien pour créer leur compte

.. slide::

**Étape 2 : Ajouter les membres à votre projet**

Une fois les invitations acceptées :

1. Ouvrez votre projet d'annotation
2. Allez dans **"Settings"** → **"Members"**
3. Cliquez sur **"Add Member"**
4. Sélectionnez les personnes dans la liste
5. Assignez-leur le rôle approprié pour ce projet

**Étape 3 : Répartir le travail (optionnel mais recommandé)**

Pour éviter que deux personnes annotent les mêmes images :

1. Dans le projet, onglet **"Tasks"** (liste des images)
2. Sélectionnez un groupe d'images (ex : images 1-50)
3. Menu **"Actions"** → **"Assign Annotators"**
4. Choisissez la personne
5. Répétez pour les autres groupes d'images

**Exemple de workflow collaboratif** :

- **Marie** (Annotator) : images 1-50
- **Paul** (Annotator) : images 51-100
- **Sophie** (Reviewer) : vérifie et corrige toutes les annotations
- **Vous** (Manager) : supervise et exporte les données finales

💡 **Astuce qualité** : faites annoter 10 images par deux personnes différentes et comparez. Un IoU (Intersection over Union) > 0.7 indique une bonne cohérence entre annotateurs.

.. slide::

3.6. Raccourcis clavier utiles
~~~~~~~~~~~~~~~~~~~

Pour accélérer l'annotation :

- ``1, 2, 3...`` : sélectionner la classe 1, 2, 3...
- ``Ctrl + Enter`` ou ``Cmd + Enter`` : soumettre et passer à l'image suivante
- ``Ctrl + Z`` : annuler la dernière action
- ``Suppr`` : supprimer la boîte sélectionnée
- ``Flèches`` : ajuster finement la position d'une boîte

.. slide::

📖 4. Formats d'annotations : COCO, Pascal VOC et YOLO
----------------------

Après l'annotation, il faut exporter les données dans un format exploitable par nos modèles. Il existe trois formats principaux, chacun avec ses avantages.

4.1. Format COCO (JSON)
~~~~~~~~~~~~~~~~~~~

**COCO** (Common Objects in Context) est le format le plus riche et le plus utilisé en recherche.

**Structure d'un fichier COCO** :

.. code-block:: json

   {
     "images": [
       {
         "id": 1,
         "file_name": "frame_00001.jpg",
         "width": 640,
         "height": 480
       },
       {
         "id": 2,
         "file_name": "frame_00002.jpg",
         "width": 640,
         "height": 480
       }
     ],
     "annotations": [
       {
         "id": 1,
         "image_id": 1,
         "category_id": 1,
         "bbox": [100, 50, 200, 150],
         "area": 30000,
         "iscrowd": 0
       },
       {
         "id": 2,
         "image_id": 1,
         "category_id": 2,
         "bbox": [350, 200, 100, 120],
         "area": 12000,
         "iscrowd": 0
       }
     ],
     "categories": [
       {"id": 1, "name": "bouteille"},
       {"id": 2, "name": "gobelet"}
     ]
   }

**Détails du format bbox** : ``[x, y, width, height]``

- ``x, y`` : coin supérieur gauche (en pixels)
- ``width`` : largeur de la boîte
- ``height`` : hauteur de la boîte

**Avantages COCO** :

- Format standard de l'industrie
- Supporte beaucoup de métadonnées (segmentation, keypoints, etc.)
- Compatible avec pycocotools (bibliothèque d'évaluation)

**Inconvénients** :

- Un seul fichier JSON pour tout le dataset (peut devenir lourd)
- Plus complexe à manipuler manuellement

.. slide::

4.2. Format Pascal VOC (XML)
~~~~~~~~~~~~~~~~~~~

**Pascal VOC** est un format plus ancien mais encore utilisé. Un fichier XML par image.

**Exemple de fichier** ``frame_00001.xml`` :

.. code-block:: xml

   <annotation>
     <folder>frames</folder>
     <filename>frame_00001.jpg</filename>
     <size>
       <width>640</width>
       <height>480</height>
       <depth>3</depth>
     </size>
     <object>
       <name>bouteille</name>
       <bndbox>
         <xmin>100</xmin>
         <ymin>50</ymin>
         <xmax>300</xmax>
         <ymax>200</ymax>
       </bndbox>
     </object>
     <object>
       <name>gobelet</name>
       <bndbox>
         <xmin>350</xmin>
         <ymin>200</ymin>
         <xmax>450</xmax>
         <ymax>320</ymax>
       </bndbox>
     </object>
   </annotation>

**Détails du format bbox** : ``xmin, ymin, xmax, ymax``

- ``xmin, ymin`` : coin supérieur gauche
- ``xmax, ymax`` : coin inférieur droit

**Avantages Pascal VOC** :

- Un fichier par image (facile à gérer, parallélisable)
- Format lisible et modifiable manuellement
- Simple à parser avec XML

**Inconvénients** :

- Beaucoup de fichiers à gérer
- Format verbeux (fichiers plus gros)

.. slide::

4.3. Format YOLO (TXT) 
~~~~~~~~~~~~~~~~~~~

**YOLO** utilise un format ultra-simple : un fichier texte par image.

**Exemple de fichier** ``frame_00001.txt`` :

.. code-block:: text

   0 0.3125 0.2604 0.3125 0.3125
   1 0.6250 0.5417 0.1562 0.2500

**Format d'une ligne** : ``class_id x_center y_center width height``

**Toutes les valeurs sont normalisées entre 0 et 1** :

- ``class_id`` : entier (0, 1, 2...) correspondant à l'index de la classe
- ``x_center`` : position X du centre / largeur de l'image
- ``y_center`` : position Y du centre / hauteur de l'image
- ``width`` : largeur de la boîte / largeur de l'image
- ``height`` : hauteur de la boîte / hauteur de l'image

**Exemple de calcul** (image $$640×480$$, objet de 100,50 à 300,200) :

.. code-block:: python

   # Coordonnées en pixels
   x1, y1, x2, y2 = 100, 50, 300, 200
   img_width, img_height = 640, 480
   
   # Calcul des valeurs YOLO
   x_center = ((x1 + x2) / 2) / img_width  # (100+300)/2 / 640 = 0.3125
   y_center = ((y1 + y2) / 2) / img_height  # (50+200)/2 / 480 = 0.2604
   width = (x2 - x1) / img_width  # (300-100) / 640 = 0.3125
   height = (y2 - y1) / img_height  # (200-50) / 480 = 0.3125
   
   # Ligne YOLO : "0 0.3125 0.2604 0.3125 0.3125"

**Avantages YOLO** :

- Format ultra-compact et rapide à parser
- Un fichier par image (facile à paralléliser)
- Coordonnées normalisées (insensible à la résolution)

**Inconvénients** :

- Nécessite un fichier ``classes.txt`` séparé pour les noms de classes
- Moins d'informations que COCO

.. slide::

4.5. Exporter depuis Label Studio
~~~~~~~~~~~~~~~~~~~

Label Studio peut exporter dans plusieurs formats. **Dans ce chapitre, nous allons utiliser :**

1. **Le format JSON natif de Label Studio** pour créer un **détecteur CNN custom** (sections suivantes)
2. **Le format YOLO** pour utiliser les modèles YOLO pré-entraînés (fin du chapitre)

**Étapes pour exporter** :

1. Ouvrez votre projet dans Label Studio
2. Cliquez sur "Export" en haut de la liste des tâches
3. Choisir le format :
   
   - **"JSON"** → format natif Label Studio (pour notre CNN custom)
   - **"YOLO"** → fichiers .txt au format YOLO (pour YOLO v5/v8/etc.)
   - "COCO" → fichier JSON au format COCO (autre méthode)
   - "Pascal VOC" → archive ZIP avec XMLs (autre méthode)

4. Télécharger le fichier

.. note::

   💡 **Choix du format**
   
   - **Dans la suite de ce chapitre**, nous utiliserons le **format JSON** de Label Studio pour construire notre propre détecteur avec PyTorch
   - **Plus loin dans le chapitre**, nous verrons comment utiliser le **format YOLO** avec les modèles YOLO pré-entraînés
   - Les formats **COCO** et **Pascal VOC** sont disponibles si vous souhaitez utiliser d'autres frameworks.

.. slide::

📖 5. Comprendre le format JSON de Label Studio
----------------------

Nous allons utiliser le **format JSON natif de Label Studio** pour entraîner notre détecteur custom. Voyons d'abord sa structure.

5.1. Structure du fichier JSON exporté
~~~~~~~~~~~~~~~~~~~

Après avoir cliqué sur "Export" → "JSON" dans Label Studio, vous obtenez un fichier avec cette structure :

.. code-block:: json

   [
     {
       "id": 1,
       "annotations": [
         {
           "id": 1,
           "completed_by": 1,
           "result": [
             {
               "original_width": 224,
               "original_height": 224,
               "value": {
                 "x": 24.67,
                 "y": 45.99,
                 "width": 52.41,
                 "height": 54.01,
                 "rotation": 0,
                 "rectanglelabels": ["cube"]
               },
               "type": "rectanglelabels",
               "from_name": "label",
               "to_name": "image"
             }
           ]
         }
       ],
       "file_upload": "ad2a7904-frame_000000.jpg",
       "data": {
         "image": "/data/upload/1/ad2a7904-frame_000000.jpg"
       }
     },
     {
       "id": 2,
       "annotations": [...],
       "file_upload": "caed06ef-frame_000001.jpg",
       "data": {
         "image": "/data/upload/1/caed06ef-frame_000001.jpg"
       }
     }
   ]

**Points importants** :

- Chaque élément du tableau JSON représente **une image**
- ``file_upload`` : nom original du fichier image
- ``data.image`` : chemin dans Label Studio (à ignorer, on utilise ``file_upload``)
- ``annotations[0].result`` : liste des boîtes englobantes
- ``value.x, y, width, height`` : **coordonnées en pourcentage** (0-100) de l'image
- ``value.rectanglelabels`` : liste des labels (ici un seul)
- ``original_width`` et ``original_height`` : dimensions de l'image (utile pour vérifier)

.. slide::

5.2. Extraire le nom de fichier depuis le JSON
~~~~~~~~~~~~~~~~~~~

Le champ ``file_upload`` contient le nom du fichier tel que stocké par Label Studio (avec un préfixe UUID ajouté automatiquement). Voici comment l'utiliser :

.. code-block:: python

   import json

   # Charger le JSON
   with open('project-1-annotations.json', 'r', encoding='utf-8') as f:
       data = json.load(f)

   # Examiner la première image
   first_item = data[0]
   
   # Le champ file_upload contient le nom avec le préfixe UUID
   image_name = first_item['file_upload']
   print(f"Nom du fichier : {image_name}")
   # Exemple : "ad2a7904-frame_000000.jpg"
   
   # Vous pouvez aussi l'extraire depuis data.image (même résultat)
   import os
   image_path = first_item['data']['image']
   image_name_alt = os.path.basename(image_path)
   print(f"Nom du fichier (depuis path) : {image_name_alt}")
   # Exemple : "ad2a7904-frame_000000.jpg"
   
   # Vérifier les dimensions de l'image dans les annotations
   result = first_item['annotations'][0]['result'][0]
   print(f"Dimensions : {result['original_width']}x{result['original_height']}")
   # Exemple : "Dimensions : 224x224"

.. warning::

   ⚠️ **Attention au préfixe UUID !**
   
   Label Studio ajoute automatiquement un préfixe UUID lors de l'upload (ex : ``ad2a7904-frame_000000.jpg``). 
   
   **Si vos fichiers images ont les noms originaux** (``frame_000000.jpg``), vous devrez extraire la partie originale du nom.

**Script complet : nettoyer le JSON et renommer les images** :

.. code-block:: python

   import json
   import os
   import shutil

   def clean_labelstudio_dataset(json_path, images_dir, output_json_path=None):
       """
       Nettoie complètement un dataset Label Studio :
       - Enlève les préfixes UUID du JSON
       - Renomme les fichiers images correspondants
       
       Args:
           json_path: chemin vers le JSON Label Studio
           images_dir: dossier contenant les images
           output_json_path: chemin JSON de sortie (None = écrase l'original)
       """
       
       def remove_prefix(filename):
           """Enlève le préfixe UUID (8 caractères hexa + tiret)."""
           if '-' in filename:
               parts = filename.split('-', 1)
               if len(parts[0]) == 8 and all(c in '0123456789abcdef' for c in parts[0].lower()):
                   return parts[1]
           return filename
       
       # 1. NETTOYER LE JSON
       print("📄 Nettoyage du JSON...")
       with open(json_path, 'r', encoding='utf-8') as f:
           data = json.load(f)
       
       json_changes = 0
       for item in data:
           # Nettoyer file_upload
           if 'file_upload' in item:
               original = remove_prefix(item['file_upload'])
               if original != item['file_upload']:
                   print(f"  ✓ {item['file_upload']} → {original}")
                   item['file_upload'] = original
                   json_changes += 1
           
           # Nettoyer data.image
           if 'data' in item and 'image' in item['data']:
               path = item['data']['image']
               filename = os.path.basename(path)
               cleaned = remove_prefix(filename)
               if '/' in path:
                   item['data']['image'] = path.rsplit('/', 1)[0] + '/' + cleaned
               else:
                   item['data']['image'] = cleaned
       
       # Sauvegarder le JSON nettoyé
       output_path = output_json_path or json_path
       with open(output_path, 'w', encoding='utf-8') as f:
           json.dump(data, f, indent=2, ensure_ascii=False)
       
       print(f"  ✓ {json_changes} noms nettoyés dans le JSON")
       print(f"  ✓ JSON sauvegardé : {output_path}\n")
       
       # 2. RENOMMER LES IMAGES
       print("🖼️  Renommage des images...")
       if not os.path.exists(images_dir):
           print(f"  ⚠️  Dossier introuvable : {images_dir}")
           return
       
       image_changes = 0
       for filename in os.listdir(images_dir):
           if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
               continue
           
           original = remove_prefix(filename)
           if original != filename:
               old_path = os.path.join(images_dir, filename)
               new_path = os.path.join(images_dir, original)
               
               if os.path.exists(new_path):
                   print(f"  ⚠️  {original} existe déjà, ignoré")
                   continue
               
               shutil.move(old_path, new_path)
               print(f"  ✓ {filename} → {original}")
               image_changes += 1
       
       print(f"\n✅ Terminé ! {image_changes} images renommées")

   # 🎯 UTILISATION
   clean_labelstudio_dataset(
       json_path='project-1-annotations.json',
       images_dir='data/images/',
       output_json_path='project-1-annotations-clean.json'  # Ou None pour écraser
   )

💡 **Astuce** : le format peut varier selon la configuration de Label Studio. Utilisez ``file_upload`` si disponible, sinon extrayez depuis ``data.image``.

.. slide::

5.3. Vérifier que tout fonctionne
~~~~~~~~~~~~~~~~~~~

Après avoir nettoyé le JSON, vérifiez que les données sont correctes :

.. code-block:: python

   import json
   import os
   import cv2

   def verify_labelstudio_dataset(json_path, images_dir, num_samples=5):
       """
       Vérifie que le JSON et les images correspondent.
       Affiche les statistiques et dessine quelques exemples.
       
       Args:
           json_path: JSON Label Studio (nettoyé)
           images_dir: dossier des images
           num_samples: nombre d'images à visualiser
       """
       
       # Charger le JSON
       with open(json_path, 'r', encoding='utf-8') as f:
           data = json.load(f)
       
       print(f"📊 STATISTIQUES DU DATASET")
       print(f"   Nombre d'images : {len(data)}")
       
       # Compter les objets et classes
       total_objects = 0
       classes_count = {}
       missing_images = []
       
       for item in data:
           image_name = item['file_upload']
           full_path = os.path.join(images_dir, image_name)
           
           # Vérifier que l'image existe
           if not os.path.exists(full_path):
               missing_images.append(image_name)
               continue
           
           # Compter les objets
           annotations = item.get('annotations', [])
           if annotations:
               result = annotations[-1].get('result', [])
               for ann in result:
                   if ann.get('type') == 'rectanglelabels':
                       total_objects += 1
                       label = ann['value']['rectanglelabels'][0]
                       classes_count[label] = classes_count.get(label, 0) + 1
       
       print(f"   Objets annotés : {total_objects}")
       print(f"   Classes : {list(classes_count.keys())}")
       for cls, count in classes_count.items():
           print(f"      - {cls}: {count} objets")
       
       if missing_images:
           print(f"\n⚠️  {len(missing_images)} images manquantes :")
           for img in missing_images[:5]:
               print(f"      - {img}")
       else:
           print(f"\n✅ Toutes les images sont présentes !")
       
       # Visualiser quelques exemples
       print(f"\n🖼️  VISUALISATION DE {num_samples} EXEMPLES")
       os.makedirs('verification', exist_ok=True)
       
       for idx, item in enumerate(data[:num_samples]):
           image_name = item['file_upload']
           full_path = os.path.join(images_dir, image_name)
           
           if not os.path.exists(full_path):
               continue
           
           # Charger l'image
           img = cv2.imread(full_path)
           h, w = img.shape[:2]
           
           # Dessiner les boîtes
           annotations = item.get('annotations', [])
           if annotations:
               result = annotations[-1].get('result', [])
               for ann in result:
                   if ann.get('type') != 'rectanglelabels':
                       continue
                   
                   value = ann['value']
                   label = value['rectanglelabels'][0]
                   
                   # Convertir % → pixels
                   x1 = int(value['x'] * w / 100)
                   y1 = int(value['y'] * h / 100)
                   x2 = int((value['x'] + value['width']) * w / 100)
                   y2 = int((value['y'] + value['height']) * h / 100)
                   
                   # Dessiner
                   cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                   cv2.putText(img, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
           
           # Sauvegarder
           output_path = f'verification/check_{idx:02d}_{image_name}'
           cv2.imwrite(output_path, img)
           print(f"   ✓ {output_path}")
       
       print(f"\n✅ Vérification terminée ! Consultez le dossier 'verification/'")

   # 🎯 UTILISATION
   verify_labelstudio_dataset(
       json_path='project-1-annotations-clean.json',
       images_dir='data/images/',
       num_samples=5
   )

💡 **Conseil** : vérifiez toujours vos données avant de lancer l'entraînement !

.. slide::

📖 6. Créer un Dataset PyTorch pour la détection
----------------------

Maintenant que nos annotations sont prêtes, créons un Dataset PyTorch personnalisé qui charge directement le JSON de Label Studio.

6.1. Structure de dossiers recommandée
~~~~~~~~~~~~~~~~~~~

Organisez vos fichiers ainsi :

.. code-block:: text

   mon_projet_detection/
   ├── data/
   │   ├── images/           # Toutes les images
   │   │   ├── frame_00001.jpg
   │   │   ├── frame_00002.jpg
   │   │   └── ...
   │   ├── annotations.json  # Export Label Studio
   │   └── splits.json       # Split train/val/test (optionnel)
   └── train.py              # Script d'entraînement

**Fichier** ``splits.json`` **(optionnel)** : pour séparer train/val/test

.. code-block:: json

   {
     "train": ["frame_00001.jpg", "frame_00002.jpg", ...],
     "val": ["frame_00151.jpg", "frame_00152.jpg", ...],
     "test": ["frame_00181.jpg", "frame_00182.jpg", ...]
   }

.. note::

   💡 **Split automatique avec random_split**
   
   Pas besoin de créer ``splits.json`` ! Vous pouvez séparer train/val/test directement dans le code avec ``random_split`` comme au chapitre 5.

.. slide::

6.2. Classe DetectionDataset complète
~~~~~~~~~~~~~~~~~~~

Voici une implémentation qui charge directement le JSON de Label Studio :

.. code-block:: python

   import torch
   from torch.utils.data import Dataset
   from PIL import Image
   import json
   import os
   from torchvision.transforms import functional as F

   class LabelStudioDetectionDataset(Dataset):
       """
       Dataset PyTorch qui charge directement les annotations Label Studio.
       """
       
       def __init__(self, json_path, images_dir, split_images=None, transforms=None):
           """
           Args:
               json_path: chemin vers le JSON exporté de Label Studio
               images_dir: dossier contenant les images
               split_images: liste de noms d'images à utiliser (None = toutes)
               transforms: transformations à appliquer (optionnel)
           """
           self.images_dir = images_dir
           self.transforms = transforms
           
           # Charger le JSON
           with open(json_path, 'r', encoding='utf-8') as f:
               all_data = json.load(f)
           
           # Filtrer selon split_images si fourni
           if split_images:
               split_set = set(split_images)
               self.data = [
                   item for item in all_data
                   if os.path.basename(item['data']['image']) in split_set
               ]
           else:
               self.data = all_data
           
           # Extraire les noms de classes uniques
           classes_set = set()
           for item in self.data:
               annotations = item.get('annotations', [])
               if annotations:
                   result = annotations[-1].get('result', [])
                   for ann in result:
                       if ann.get('type') == 'rectanglelabels':
                           labels = ann['value'].get('rectanglelabels', [])
                           classes_set.update(labels)
           
           self.classes = sorted(list(classes_set))
           self.class_to_idx = {cls: idx+1 for idx, cls in enumerate(self.classes)}
           
           print(f"Dataset initialisé : {len(self.data)} images, "
                 f"{len(self.classes)} classes : {self.classes}")
       
       def __len__(self):
           return len(self.data)
       
       def __getitem__(self, idx):
           """
           Charge une image et ses annotations.
           
           Returns:
               img: tensor [3, H, W]
               target: dict avec 'boxes', 'labels', 'image_id'
           """
           item = self.data[idx]
           
           # Extraire le nom de l'image et la charger
           image_path_str = item['data']['image']
           image_name = os.path.basename(image_path_str)
           full_path = os.path.join(self.images_dir, image_name)
           
           img = Image.open(full_path).convert('RGB')
           img_width, img_height = img.size
           
           # Récupérer les annotations
           boxes = []
           labels = []
           
           annotations = item.get('annotations', [])
           if annotations:
               # Prendre la dernière version (plus récente)
               result = annotations[-1].get('result', [])
               
               for ann in result:
                   if ann.get('type') != 'rectanglelabels':
                       continue
                   
                   value = ann['value']
                   
                   # Label Studio donne les coordonnées en pourcentages (0-100)
                   x_percent = value['x']
                   y_percent = value['y']
                   w_percent = value['width']
                   h_percent = value['height']
                   
                   # Convertir en pixels [x1, y1, x2, y2]
                   x1 = (x_percent / 100.0) * img_width
                   y1 = (y_percent / 100.0) * img_height
                   x2 = ((x_percent + w_percent) / 100.0) * img_width
                   y2 = ((y_percent + h_percent) / 100.0) * img_height
                   
                   boxes.append([x1, y1, x2, y2])
                   
                   # Récupérer la classe
                   class_name = value['rectanglelabels'][0]
                   class_idx = self.class_to_idx[class_name]
                   labels.append(class_idx)
           
           # Convertir en tenseurs
           boxes = torch.as_tensor(boxes, dtype=torch.float32)
           labels = torch.as_tensor(labels, dtype=torch.int64)
           
           # Créer le dictionnaire target
           target = {}
           target['boxes'] = boxes
           target['labels'] = labels
           target['image_id'] = torch.tensor([idx])
           
           # Si aucune boîte, créer des tenseurs vides
           if len(boxes) == 0:
               target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
               target['labels'] = torch.zeros((0,), dtype=torch.int64)
           
           # Appliquer les transformations
           if self.transforms:
               img = self.transforms(img)
           else:
               img = F.to_tensor(img)
           
           return img, target
       
       def get_class_name(self, class_id):
           """Retourne le nom d'une classe depuis son ID."""
           return self.classes[class_id - 1]

.. note::

   💡 **Gestion des IDs de classes**
   
   - Les classes sont automatiquement extraites du JSON
   - Les IDs commencent à **1** (0 est réservé au background dans torchvision)
   - ``class_to_idx`` : dictionnaire ``{'bouteille': 1, 'gobelet': 2}``

.. slide::

6.3. Créer les DataLoaders avec split automatique
~~~~~~~~~~~~~~~~~~~

Si vous n'avez pas de fichier ``splits.json``, utilisez ``random_split`` comme au chapitre 5 :

.. code-block:: python

   from torch.utils.data import DataLoader, random_split

   # Charger le dataset complet
   full_dataset = LabelStudioDetectionDataset(
       json_path='data/annotations.json',
       images_dir='data/images/'
   )

   # Split : 70% train, 15% val, 15% test
   total_size = len(full_dataset)
   train_size = int(0.70 * total_size)
   val_size = int(0.15 * total_size)
   test_size = total_size - train_size - val_size

   train_dataset, val_dataset, test_dataset = random_split(
       full_dataset, 
       [train_size, val_size, test_size]
   )

   print(f"Train : {len(train_dataset)} images")
   print(f"Val   : {len(val_dataset)} images")
   print(f"Test  : {len(test_dataset)} images")

   # Créer les dataloaders
   def collate_fn(batch):
       """Fonction nécessaire car chaque image a un nombre différent d'objets."""
       return tuple(zip(*batch))

   train_loader = DataLoader(
       train_dataset,
       batch_size=4,
       shuffle=True,
       num_workers=4,
       collate_fn=collate_fn
   )

   val_loader = DataLoader(
       val_dataset,
       batch_size=4,
       shuffle=False,
       num_workers=4,
       collate_fn=collate_fn
   )

💡 **Avantage** : tout en un ! Pas besoin de gérer des listes de noms de fichiers séparées.

.. slide::

6.5. Tester le chargement des données
~~~~~~~~~~~~~~~~~~~

Toujours vérifier que le Dataset charge correctement :

.. code-block:: python

   # Charger un exemple
   img, target = train_dataset[0]

   print(f"Image shape: {img.shape}")
   print(f"Nombre d'objets: {len(target['boxes'])}")
   print(f"Boxes:\n{target['boxes']}")
   print(f"Labels: {target['labels']}")

   # Visualiser quelques exemples
   import matplotlib.pyplot as plt
   import matplotlib.patches as patches

   def visualize_sample(dataset, idx):
       img, target = dataset[idx]
       
       # Convertir le tensor en numpy pour l'affichage
       img_np = img.permute(1, 2, 0).numpy()
       
       fig, ax = plt.subplots(1, figsize=(12, 8))
       ax.imshow(img_np)
       
       # Dessiner chaque boîte
       for box, label in zip(target['boxes'], target['labels']):
           x1, y1, x2, y2 = box.tolist()
           width = x2 - x1
           height = y2 - y1
           
           rect = patches.Rectangle(
               (x1, y1), width, height,
               linewidth=2, edgecolor='r', facecolor='none'
           )
           ax.add_patch(rect)
           
           # Ajouter le label
           class_name = dataset.get_class_name(label.item())
           ax.text(x1, y1-5, class_name, 
                  bbox=dict(facecolor='red', alpha=0.5),
                  fontsize=12, color='white')
       
       plt.axis('off')
       plt.tight_layout()
       plt.savefig(f'check_sample_{idx}.png')
       print(f"✓ Visualisation sauvegardée : check_sample_{idx}.png")

   # Vérifier les 5 premiers exemples
   for i in range(5):
       visualize_sample(train_dataset, i)


STOP ICI

STOP ICI

STOP ICI

.. slide::

📖 7. CNN ultra-simple : régression directe de boîte
----------------------

Pour des cas simples avec **1 seul objet par image**, on peut utiliser une approche beaucoup plus simple que YOLO ou Faster R-CNN : **régression directe des coordonnées** de la boîte. Le modèle prédit directement 4 nombres : `(x_center, y_center, width, height)` normalisés dans [0,1].

.. note::

   💡 **Quand utiliser cette approche ?**
   
   ✅ **OUI** : 1 objet par image, objet centré, peu de variations (ex: détection de visage, logo)
   
   ❌ **NON** : plusieurs objets, positions variables, objets qui se chevauchent

7.1. Architecture ultra-simple
~~~~~~~~~~~~~~~~~~~

Le modèle est constitué d'un **backbone CNN** (4 couches Conv2D + MaxPool) suivi d'un **head de régression** (2 couches FC) qui prédit directement les 4 coordonnées normalisées. Dans l'exemple, l’entrée $$224×224$$ est réduite 5 fois par MaxPool(2): 224→112→56→28→14→7; la carte de features finale est donc $$7×7$$. Si vous changez la taille d’entrée ou le nombre de couches à stride 2, la taille de la grille changera.

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from tqdm import tqdm

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
           
           # Après 4 MaxPool: 224->112->56->28->14
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
   
   Ce modèle a environ **25 millions** de paramètres (principalement dans la première couche FC `128*14*14 → 128`). C'est bien plus petit que Faster R-CNN (>40M) mais simple et efficace pour 1 objet par image.

.. warning::

   Ce modèle suppose **1 seul objet par image**. Si votre dataset contient plusieurs objets par image, utilisez plutôt un modèle avec anchors (section 7.2 de la version complète) ou Faster R-CNN (section 8).

7.2. Loss et optimiseur
~~~~~~~~~~~~~~~~~~~

**Loss MSE** pour les coordonnées normalisées (x_center, y_center, width, height) + **préparation des targets**.

.. code-block:: python

   # Loss simple : MSE sur les coordonnées
   criterion = nn.MSELoss()
   optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)
   
   # Fonction de préparation des targets
   def prepare_single_box_target(target):
       """
       Convertit les boxes du format [x1,y1,x2,y2] (pixels)
       vers [x_center, y_center, width, height] normalisés dans [0, 1].
       
       Suppose 1 seule boîte par image.
       """
       boxes = target['boxes']  # [N, 4] en pixels
       box = boxes[0]  # Prendre la première (et unique) boîte
       
       x1, y1, x2, y2 = box
       x_center = (x1 + x2) / 2 / 224  # Normaliser par la taille de l'image
       y_center = (y1 + y2) / 2 / 224
       width = (x2 - x1) / 224
       height = (y2 - y1) / 224
       
       return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

.. note::

   📐 **Normalisation des coordonnées**
   
   - Entrée : boîtes en pixels `[x1, y1, x2, y2]` dans `[0, 224]`
   - Sortie : coordonnées normalisées `[x_c, y_c, w, h]` dans `[0, 1]`
   - Le modèle prédit directement ces 4 valeurs normalisées
   - Pas d'anchors, pas de grille, pas d'objectness !


7.3. Entraînement (boucles train/val)
~~~~~~~~~~~~~~~~~~~

Boucles simples d'entraînement et d'évaluation.

.. code-block:: python

   from tqdm import tqdm
   
   # Fonctions d'entraînement
   def train_simple_epoch(model, criterion, optimizer, data_loader, device, epoch):
       """Entraîne le modèle pendant une epoch."""
       model.train()
       total_loss = 0
       
       pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
       
       for images, targets in pbar:
           images = torch.stack([img.to(device) for img in images])
           
           # Préparer les targets (batch de vecteurs [x_c, y_c, w, h])
           target_boxes = torch.stack([
               prepare_single_box_target(t).to(device) for t in targets
           ])  # [B, 4]
           
           # Forward
           predictions = model(images)  # [B, 4]
           
           # Loss MSE
           loss = criterion(predictions, target_boxes)
           
           # Backward
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           total_loss += loss.item()
           pbar.set_postfix({'loss': f"{loss.item():.4f}"})
       
       return total_loss / len(data_loader)

   @torch.no_grad()
   def eval_simple_epoch(model, criterion, data_loader, device):
       """Évalue le modèle."""
       model.eval()
       total_loss = 0
       
       for images, targets in tqdm(data_loader, desc="Validation"):
           images = torch.stack([img.to(device) for img in images])
           target_boxes = torch.stack([
               prepare_single_box_target(t).to(device) for t in targets
           ])
           
           predictions = model(images)
           loss = criterion(predictions, target_boxes)
           total_loss += loss.item()
       
       return total_loss / len(data_loader)

   # LANCER L'ENTRAÎNEMENT
   print("\n🚀 Entraînement du modèle simple...\n")

   num_epochs = 20
   best_val_loss = float('inf')

   for epoch in range(num_epochs):
       train_loss = train_simple_epoch(
           simple_model, criterion, optimizer, train_loader, device, epoch
       )
       
       val_loss = eval_simple_epoch(simple_model, criterion, val_loader, device)
       
       print(f"\n📊 Epoch {epoch}:")
       print(f"  Train Loss: {train_loss:.4f}")
       print(f"  Val Loss:   {val_loss:.4f}")
       
       # Sauvegarder le meilleur
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           torch.save(simple_model.state_dict(), 'simple_bbox_regressor.pth')
           print("  ✅ Meilleur modèle sauvegardé !")

   print("\n🎉 Entraînement terminé !")
   print(f"📁 Modèle sauvegardé : simple_bbox_regressor.pth")

.. note::

   🏋️ **Convergence**
   
   Avec ce modèle simple, vous devriez voir la loss descendre rapidement :
   
   - Epoch 0 : ~0.02-0.03 (loss MSE élevée)
   - Epoch 5 : ~0.005-0.01 (convergence)
   - Epoch 15+ : ~0.001-0.003 (modèle bien entraîné)
   
   Si la loss ne descend pas, vérifiez que vos données sont bien normalisées !

7.4. Évaluation avec IoU
~~~~~~~~~~~~~~~~~~~

Calcul de l'**IoU moyen** (Intersection over Union) sur le test set.

.. code-block:: python

   def compute_iou(box1, box2):
       """Calcule l'IoU entre deux boîtes [x1, y1, x2, y2]."""
       x1 = max(box1[0], box2[0])
       y1 = max(box1[1], box2[1])
       x2 = min(box1[2], box2[2])
       y2 = min(box1[3], box2[3])
       
       if x2 < x1 or y2 < y1:
           return 0.0
       
       inter = (x2 - x1) * (y2 - y1)
       area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
       area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
       union = area1 + area2 - inter
       
       return inter / (union + 1e-6)
   
   @torch.no_grad()
   def evaluate_on_test(model, test_dataset, device):
       """Évalue le modèle sur le test set et calcule l'IoU moyen."""
       model.eval()
       ious = []
       
       for idx in range(len(test_dataset)):
           img, target = test_dataset[idx]
           
           # Prédiction
           pred = model(img.unsqueeze(0).to(device))[0].cpu()  # [4]
           
           # GT
           gt_boxes = target['boxes']  # [N, 4] en pixels
           gt_box = gt_boxes[0]  # Prendre la première boîte
           
           # Convertir prédiction vers pixels [x1, y1, x2, y2]
           x_c, y_c, w, h = pred
           x1 = (x_c - w/2) * 224
           y1 = (y_c - h/2) * 224
           x2 = (x_c + w/2) * 224
           y2 = (y_c + h/2) * 224
           pred_box = [x1.item(), y1.item(), x2.item(), y2.item()]
           
           # Calculer IoU
           iou = compute_iou(pred_box, gt_box.tolist())
           ious.append(iou)
       
       mean_iou = torch.tensor(ious).mean().item()
       print(f"\n📊 IoU moyen sur le test set : {mean_iou:.3f}")
       print(f"  - IoU > 0.5 : {sum(1 for x in ious if x > 0.5)}/{len(ious)} images")
       print(f"  - IoU > 0.75 : {sum(1 for x in ious if x > 0.75)}/{len(ious)} images")
       
       return mean_iou
   
   # Charger le meilleur modèle et évaluer
   simple_model.load_state_dict(torch.load('simple_bbox_regressor.pth'))
   evaluate_on_test(simple_model, test_dataset, device)

.. note::

   📈 **Interprétation de l'IoU**
   
   - IoU > **0.5** : Bonne détection (standard COCO)
   - IoU > **0.75** : Très bonne détection
   - IoU > **0.9** : Détection quasi-parfaite
   
   Un modèle bien entraîné sur ce dataset simple devrait obtenir un IoU moyen > 0.8.

7.5. Visualisation
~~~~~~~~~~~~~~~~~~~


Affichage des prédictions sur une grille d'images avec GT (vert) et prédictions (rouge).

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.patches as patches
   import numpy as np
   
   @torch.no_grad()
   def visualize_predictions(model, dataset, device, n=9):
       """Affiche n prédictions avec les GT en vert et prédictions en rouge."""
       model.eval()
       
       fig, axes = plt.subplots(3, 3, figsize=(12, 12))
       axes = axes.flatten()
       
       for i in range(n):
           img, target = dataset[i]
           
           # Prédiction
           pred = model(img.unsqueeze(0).to(device))[0].cpu()
           
           # Affichage image
           img_np = img.permute(1, 2, 0).cpu().numpy()
           # Dénormaliser (ImageNet)
           mean = torch.tensor([0.485, 0.456, 0.406])
           std = torch.tensor([0.229, 0.224, 0.225])
           img_np = img_np * std.numpy() + mean.numpy()
           img_np = np.clip(img_np, 0, 1)
           
           axes[i].imshow(img_np)
           axes[i].axis('off')
           
           # GT box (vert)
           gt_box = target['boxes'][0]
           x1, y1, x2, y2 = gt_box
           w_gt, h_gt = x2 - x1, y2 - y1
           rect_gt = patches.Rectangle((x1, y1), w_gt, h_gt,
                                       linewidth=2, edgecolor='green',
                                       facecolor='none', label='GT')
           axes[i].add_patch(rect_gt)
           
           # Predicted box (rouge)
           x_c, y_c, w_pred, h_pred = pred
           x1_pred = (x_c - w_pred/2) * 224
           y1_pred = (y_c - h_pred/2) * 224
           w_pred_pix = w_pred * 224
           h_pred_pix = h_pred * 224
           rect_pred = patches.Rectangle((x1_pred, y1_pred), w_pred_pix, h_pred_pix,
                                         linewidth=2, edgecolor='red',
                                         facecolor='none', linestyle='--', label='Pred')
           axes[i].add_patch(rect_pred)
       
       # Légende
       handles, labels = axes[0].get_legend_handles_labels()
       fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
       plt.tight_layout()
       plt.show()
   
   # Visualiser
   visualize_predictions(simple_model, test_dataset, device, n=9)

.. note::

   🎨 **Légende**
   
   - **Vert** : Ground truth (annotation réelle)
   - **Rouge** (pointillé) : Prédiction du modèle
   
   Si les boîtes se superposent bien, le modèle fonctionne correctement !

.. slide::

📖 8. Entraînement avec Faster R-CNN (torchvision)
----------------------

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



