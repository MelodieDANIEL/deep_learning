.. slide::

Chapitre 6 — Détection d'objets avec des boîtes englobantes (partie 1)
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

Une **boîte englobante** est un rectangle défini par 4 valeurs. Il existe plusieurs façons de représenter ces coordonnées :

**Format 1 : (x1, y1, x2, y2)** —> coins de la boîte (PyTorch/torchvision)

- ``x1, y1`` : coordonnées du coin supérieur gauche
- ``x2, y2`` : coordonnées du coin inférieur droit

**Format 2 : (x, y, w, h)** —> coin + dimensions (Label Studio format standard)

- ``x, y`` : coordonnées du coin supérieur gauche (en % )
- ``w`` : largeur de la boîte
- ``h`` : hauteur de la boîte

**Format 3 : (x_center, y_center, w, h) normalisé** (Label Studio format utilisé pour YOLO)

- ``x_center, y_center`` : coordonnées du centre (normalisées entre 0 et 1)
- ``w, h`` : largeur et hauteur (normalisées entre 0 et 1)

.. code-block:: text

   Exemple d'une image $$640×480$$ pixels avec un objet :
   
   Format PyTorch : [100, 50, 300, 250]
   → Rectangle du pixel (100,50) au pixel (300,250)
   
   Format Label Studio : [15.625, 10.417, 31.25, 41.67]
   → Coin en (15.625%, 10.417%), taille 31.25%×41.67% de l'image
   
   Format YOLO : [0.3125, 0.3125, 0.3125, 0.4167]
   → Centre à 31.25% de la largeur/hauteur, boîte de 31.25%×41.67% de l'image

.. warning::

   ⚠️ **Format utilisé dans la suite du TP**
   
   Dans la suite de ce chapitre, nous utiliserons le **Format 3 (YOLO)** avec des coordonnées normalisées. C'est le format standard pour la détection d'objets, compatible avec YOLO et la plupart des frameworks modernes.

.. slide:

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

💡 **Astuce** : plus vous capturez de variété, meilleur sera votre détecteur !

.. note::

   **💡 Vous pouvez commencer avec beaucoup moins !**
   
   - ~50-100 photos extraites d'une vidéo faite avec un smartphone suffisent pour débuter
   - Résolution modeste ($$640×480$$ ou $$224×224$$) acceptable pour un prototype
   - Même avec peu de variété, vous obtiendrez déjà des résultats !

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

.. warning::

   ⚠️ **Code utilisé dans la suite du TP**
   
   C'est **cette fonction** (``extraire_frames_crop_redimensionner``) que nous utiliserons dans tous les exercices du chapitre. Elle évite les déformations en découpant un carré au centre de l'image avant de redimensionner.

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

- **Pour la détection d'objets** : utilisez le crop au centre pour éviter les déformations.
- **Pour la classification** : le crop au centre est aussi préférable.
- **Résolutions recommandées** : $$224×224$$ (standard CNN), $$640×480$$ (compromis vitesse/qualité), $$800×600$$ (bonne qualité).

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

4. Entrez les adresses email de vos collègues (ex : ``marie.dubois@exemple.com``, ``paul.martin@exemple.com``)

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

📖 4. Exporter depuis Label Studio au format YOLO
----------------------

Après avoir terminé l'annotation de vos images dans Label Studio, vous devez exporter les données pour l'entraînement. Dans ce chapitre, nous allons utiliser le format **"YOLO with Images"**.

.. slide::

4.1. Qu'est-ce que le format "YOLO with Images" ?
~~~~~~~~~~~~~~~~~~~

Le format **"YOLO with Images"** est un export complet proposé par Label Studio qui contient :

1. **Un dossier ``images/``** : toutes vos images annotées
2. **Un dossier ``labels/``** : un fichier texte ``.txt`` par image contenant les annotations
3. **Un fichier ``classes.txt``** : la liste des noms de classes (un par ligne)
4. **Un fichier ``notes.json``** : métadonnées sur l'export

**C'est le format idéal** car il regroupe tout ce dont vous avez besoin pour l'entraînement dans une seule archive ZIP.

.. slide::

4.2. Structure du format YOLO
~~~~~~~~~~~~~~~~~~~

**Format d'annotation YOLO** : un fichier texte par image avec des coordonnées normalisées.

**Exemple de fichier** ``labels/frame_00001.txt`` :

.. code-block:: text

   0 0.3125 0.3125 0.3125 0.4167
   1 0.6250 0.5417 0.1562 0.2500

**Format d'une ligne** : ``class_id x_center y_center width height``

**Toutes les valeurs sont normalisées entre 0 et 1** :

- ``class_id`` : entier (0, 1, 2...) correspondant à l'index de la classe
- ``x_center`` : position X du centre de la boîte / largeur de l'image
- ``y_center`` : position Y du centre de la boîte / hauteur de l'image
- ``width`` : largeur de la boîte / largeur de l'image
- ``height`` : hauteur de la boîte / hauteur de l'image

.. slide::

**Exemple concret** (image $$640×480$$, objet de 100,50 à 300,200) :

.. code-block:: python

   # Coordonnées en pixels (format classique)
   x1, y1, x2, y2 = 100, 50, 300, 200
   img_width, img_height = 640, 480
   
   # Conversion en format YOLO
   x_center = ((x1 + x2) / 2) / img_width   # Centre X : (100+300)/2 / 640 = 0.3125
   y_center = ((y1 + y2) / 2) / img_height  # Centre Y : (50+200)/2 / 480 = 0.2604
   width = (x2 - x1) / img_width            # Largeur : (300-100) / 640 = 0.3125
   height = (y2 - y1) / img_height          # Hauteur : (200-50) / 480 = 0.3125
   
   # Résultat : "0 0.3125 0.2604 0.3125 0.3125"

**Structure complète après export** :

.. code-block:: text

   dataset_yolo/
   ├── images/              # Toutes vos images annotées
   │   ├── frame_00001.jpg
   │   ├── frame_00002.jpg
   │   └── ...
   ├── labels/              # Fichiers .txt (même nom que l'image)
   │   ├── frame_00001.txt
   │   ├── frame_00002.txt
   │   └── ...
   ├── classes.txt          # Liste des classes : une par ligne
   └── notes.json           # Métadonnées (optionnel)

**Fichier** ``classes.txt`` **exemple** :

.. code-block:: text

   cube
   bouteille
   gobelet

💡 **L'ordre des classes dans** ``classes.txt`` **définit les IDs** : cube=0, bouteille=1, gobelet=2.

.. slide::

4.3. Étapes pour exporter depuis Label Studio
~~~~~~~~~~~~~~~~~~~

**1. Accéder à l'export**

- Ouvrez votre projet dans Label Studio
- En haut de la page, cliquez sur le bouton **"Export"**

**2. Choisir le format**

- Dans la liste des formats d'export, sélectionnez : **"YOLO"**
- Label Studio génère automatiquement l'archive

**3. Télécharger l'archive**

- Cliquez sur **"Export"** pour télécharger le fichier ZIP
- Le fichier se nomme généralement ``project-X-at-YYYY-MM-DD-HH-MM-XX.zip``

**4. Extraire l'archive**

.. code-block:: bash

   # Décompresser l'archive
   unzip project-1-at-2024-01-15-14-30-00.zip -d dataset_yolo/
   
   # Vérifier le contenu
   ls -R dataset_yolo/
   
   # Vous devriez voir :
   # dataset_yolo/
   # ├── images/
   # ├── labels/
   # ├── classes.txt
   # └── notes.json

.. slide::

4.4. Vérifier l'export
~~~~~~~~~~~~~~~~~~~

Avant de commencer l'entraînement, vérifiez toujours que l'export est correct :

.. code-block:: python

   import os

   def verify_yolo_export(dataset_path):
       """Vérifie la structure d'un export YOLO."""
       
       images_dir = os.path.join(dataset_path, 'images')
       labels_dir = os.path.join(dataset_path, 'labels')
       classes_file = os.path.join(dataset_path, 'classes.txt')
       
       # Vérifier que les dossiers existent
       assert os.path.exists(images_dir), "❌ Dossier 'images/' manquant"
       assert os.path.exists(labels_dir), "❌ Dossier 'labels/' manquant"
       assert os.path.exists(classes_file), "❌ Fichier 'classes.txt' manquant"
       
       # Compter les fichiers
       images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
       labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
       
       print(f"✅ Structure YOLO valide")
       print(f"   📁 Images : {len(images)}")
       print(f"   📁 Labels : {len(labels)}")
       
       # Charger les classes
       with open(classes_file, 'r') as f:
           classes = [line.strip() for line in f.readlines()]
       print(f"   📋 Classes ({len(classes)}) : {classes}")
       
       # Vérifier la correspondance images/labels
       missing_labels = []
       for img in images:
           label_name = os.path.splitext(img)[0] + '.txt'
           if label_name not in labels:
               missing_labels.append(img)
       
       if missing_labels:
           print(f"\n⚠️  {len(missing_labels)} images sans annotation :")
           for img in missing_labels[:5]:
               print(f"      - {img}")
       else:
           print(f"\n✅ Toutes les images ont leurs annotations")
       
       # Vérifier un fichier d'annotation
       if labels:
           sample_label = os.path.join(labels_dir, labels[0])
           with open(sample_label, 'r') as f:
               lines = f.readlines()
           print(f"\n📄 Exemple d'annotation ({labels[0]}) :")
           for line in lines[:3]:
               print(f"      {line.strip()}")

   # 🎯 UTILISATION
   verify_yolo_export('dataset_yolo/')

.. slide::

📖 5. Créer un Dataset PyTorch pour le format YOLO
----------------------

Maintenant que vous avez exporté votre dataset au format YOLO, créons un Dataset PyTorch personnalisé pour le charger.

5.1. Structure de dossiers YOLO
~~~~~~~~~~~~~~~~~~~

Après extraction de l'archive ZIP, votre dataset doit avoir cette structure :

.. code-block:: text

   dataset_yolo/
   ├── images/              # Toutes vos images annotées
   │   ├── frame_00001.jpg
   │   ├── frame_00002.jpg
   │   └── ...
   ├── labels/              # Fichiers .txt (même nom que l'image)
   │   ├── frame_00001.txt
   │   ├── frame_00002.txt
   │   └── ...
   └── classes.txt          # Liste des classes (une par ligne)

.. slide::

5.2. Classe YOLODetectionDataset
~~~~~~~~~~~~~~~~~~~

Voici une implémentation complète qui charge le format YOLO et gère intelligemment le redimensionnement :

.. code-block:: python

   import torch
   from torch.utils.data import Dataset
   from PIL import Image
   import os
   from torchvision import transforms

   class YOLODetectionDataset(Dataset):
       """
       Dataset PyTorch pour le format YOLO (images + labels .txt).
       Compatible avec l'export YOLO de Label Studio.
       """
       
       def __init__(self, images_dir, labels_dir, classes_file, img_size=224, custom_transforms=None):
           """
           Args:
               images_dir: dossier contenant les images
               labels_dir: dossier contenant les labels .txt au format YOLO
               classes_file: chemin vers classes.txt
               img_size: taille de redimensionnement (défaut: 224x224)
               custom_transforms: transformations à appliquer (optionnel)
           """
           self.images_dir = images_dir
           self.labels_dir = labels_dir
           self.img_size = img_size
           self.custom_transforms = custom_transforms
           
           # Pas de transformation automatique - on gérera le resize manuellement
           # pour ajuster les coordonnées des bounding boxes en conséquence
           self.to_tensor = transforms.ToTensor()
           
           # Charger les noms de classes
           with open(classes_file, 'r') as f:
               self.classes = [line.strip() for line in f.readlines()]
           
           # Liste des images (on suppose que chaque image a son label correspondant)
           self.image_files = sorted([f for f in os.listdir(images_dir) 
                                      if f.endswith(('.jpg', '.jpeg', '.png'))])
           
           print(f"✅ Dataset YOLO initialisé:")
           print(f"   - {len(self.image_files)} images")
           print(f"   - {len(self.classes)} classes : {self.classes}")
       
       def __len__(self):
           return len(self.image_files)
       
       def __getitem__(self, idx):
           """
           Charge une image et ses annotations au format YOLO.
           
           Returns:
               img: tensor [3, H, W]
               target: dict avec 'boxes' (format [x1, y1, x2, y2] en pixels), 
                       'labels', 'image_id'
           """
           # Charger l'image
           img_filename = self.image_files[idx]
           img_path = os.path.join(self.images_dir, img_filename)
           img = Image.open(img_path).convert('RGB')
           orig_width, orig_height = img.size
           
           # Calculer le ratio de resize pour garder l'aspect ratio
           scale = self.img_size / max(orig_width, orig_height)
           new_width = int(orig_width * scale)
           new_height = int(orig_height * scale)
           
           # Resize en gardant l'aspect ratio
           img = img.resize((new_width, new_height), Image.BILINEAR)
           
           # Créer une image carrée avec padding noir
           padded_img = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
           # Centrer l'image resizée
           paste_x = (self.img_size - new_width) // 2
           paste_y = (self.img_size - new_height) // 2
           padded_img.paste(img, (paste_x, paste_y))
           
           # Charger le label correspondant
           label_filename = os.path.splitext(img_filename)[0] + '.txt'
           label_path = os.path.join(self.labels_dir, label_filename)
           
           boxes = []
           labels = []
           
           # Lire le fichier de labels si il existe
           if os.path.exists(label_path):
               with open(label_path, 'r') as f:
                   for line in f:
                       parts = line.strip().split()
                       if len(parts) == 5:
                           class_id = int(parts[0])
                           x_center = float(parts[1])
                           y_center = float(parts[2])
                           width = float(parts[3])
                           height = float(parts[4])
                           
                           # Convertir du format YOLO normalisé vers pixels dans l'image originale
                           x_center_orig = x_center * orig_width
                           y_center_orig = y_center * orig_height
                           width_orig = width * orig_width
                           height_orig = height * orig_height
                           
                           # Appliquer le scale et le padding
                           x_center_scaled = x_center_orig * scale + paste_x
                           y_center_scaled = y_center_orig * scale + paste_y
                           width_scaled = width_orig * scale
                           height_scaled = height_orig * scale
                           
                           # Convertir en format [x1, y1, x2, y2]
                           x1 = x_center_scaled - width_scaled / 2
                           y1 = y_center_scaled - height_scaled / 2
                           x2 = x_center_scaled + width_scaled / 2
                           y2 = y_center_scaled + height_scaled / 2
                           
                           boxes.append([x1, y1, x2, y2])
                           labels.append(class_id + 1)  # +1 car background=0 dans certains modèles
           
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
           
           # Convertir l'image en tensor
           img = self.to_tensor(padded_img)
           
           # Appliquer les transformations personnalisées si fournies
           if self.custom_transforms:
               img = self.custom_transforms(img)
           
           return img, target
       
       def get_class_name(self, class_id):
           """Retourne le nom d'une classe depuis son ID (class_id - 1 car on a ajouté 1)."""
           return self.classes[class_id - 1]

.. note::

   💡 **Gestion intelligente du redimensionnement**
   
   - L'image est redimensionnée **proportionnellement** pour éviter toute déformation
   - Un **padding noir** est ajouté pour créer une image carrée
   - Les **coordonnées des bounding boxes** sont automatiquement ajustées
   - Les IDs de classes commencent à **1** (0 réservé au background)

.. slide::

5.3. Créer les DataLoaders avec split automatique
~~~~~~~~~~~~~~~~~~~

Chargez le dataset et créez les splits train/val/test :

.. code-block:: python

   from torch.utils.data import DataLoader, random_split

   # Charger le dataset YOLO
   full_dataset = YOLODetectionDataset(
       images_dir='dataset_yolo/images',
       labels_dir='dataset_yolo/labels',
       classes_file='dataset_yolo/classes.txt'
   )

   # Split : 70% train, 15% val, 15% test
   total_size = len(full_dataset)
   train_size = int(0.70 * total_size)
   val_size = int(0.15 * total_size)
   test_size = total_size - train_size - val_size

   train_dataset, val_dataset, test_dataset = random_split(
       full_dataset, 
       [train_size, val_size, test_size],
       generator=torch.Generator().manual_seed(42)
   )

   print(f"\n📂 Split du dataset :")
   print(f"   Train : {len(train_dataset)} images")
   print(f"   Val   : {len(val_dataset)} images")
   print(f"   Test  : {len(test_dataset)} images")

   # Créer les dataloaders
   def collate_fn(batch):
       """Fonction nécessaire car chaque image a un nombre différent d'objets."""
       return tuple(zip(*batch))

   train_loader = DataLoader(
       train_dataset,
       batch_size=4,
       shuffle=True,
       num_workers=2,
       collate_fn=collate_fn
   )

   val_loader = DataLoader(
       val_dataset,
       batch_size=4,
       shuffle=False,
       num_workers=2,
       collate_fn=collate_fn
   )

   test_loader = DataLoader(
       test_dataset,
       batch_size=4,
       shuffle=False,
       num_workers=2,
       collate_fn=collate_fn
   )

   print(f"\n✅ DataLoaders créés avec batch_size=4")

💡 **Avantage de** ``random_split`` : pas besoin de créer manuellement les listes d'images pour chaque split !

.. slide::

5.4. Visualiser les données chargées
~~~~~~~~~~~~~~~~~~~

Toujours vérifier visuellement que le Dataset charge correctement les images et bounding boxes :

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.patches as patches
   import numpy as np

   def visualize_yolo_batch(dataset, num_samples=4):
       """Affiche quelques exemples du dataset avec leurs bounding boxes."""
       fig, axes = plt.subplots(2, 2, figsize=(12, 12))
       axes = axes.flatten()
       
       for i in range(num_samples):
           img, target = dataset[i]
           
           # Convertir le tensor en numpy pour l'affichage
           img_np = img.permute(1, 2, 0).numpy()
           
           ax = axes[i]
           ax.imshow(img_np)
           ax.axis('off')
           
           # Dessiner les bounding boxes
           boxes = target['boxes'].numpy()
           labels = target['labels'].numpy()
           
           for box, label in zip(boxes, labels):
               x1, y1, x2, y2 = box
               width = x2 - x1
               height = y2 - y1
               
               # Créer le rectangle
               rect = patches.Rectangle(
                   (x1, y1), width, height,
                   linewidth=2, edgecolor='red', facecolor='none'
               )
               ax.add_patch(rect)
               
               # Ajouter le label
               class_name = dataset.dataset.get_class_name(label) if hasattr(dataset, 'dataset') else f"Classe {label}"
               ax.text(x1, y1-5, class_name, 
                      bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                      fontsize=10, color='white')
           
           ax.set_title(f'Image {i}')
       
       plt.tight_layout()
       plt.show()

   # Visualiser quelques exemples du train set
   print("📸 Visualisation d'exemples du training set:\n")
   visualize_yolo_batch(train_dataset, num_samples=4)

**Points à vérifier** :

- ✅ Les images sont bien carrées ($$224×224$$ par défaut)
- ✅ Les bounding boxes englobent correctement les objets
- ✅ Les labels affichés correspondent aux objets
- ✅ Pas de déformation visible des objets

💡 **Astuce** : si les bounding boxes ne correspondent pas aux objets, vérifiez que les fichiers ``.txt`` dans ``labels/`` ont les mêmes noms que les images.

