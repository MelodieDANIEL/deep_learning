🏋️ Travaux Pratiques 6
=========================

.. slide::

Sur cette page se trouvent des exercices de TP sur le Chapitre 6 (Détection d'objets). Ils sont classés par niveau de difficulté :

.. discoverList::
    * Facile : 🍀
    * Moyen : ⚖️
    * Difficile : 🌶️



############################

.. slide::

🍀 Exercice 1 : Dataset avec présence/absence d'objet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez créer un dataset particulier où certaines images contiennent votre objet et d'autres **ne le contiennent pas**. Le modèle devra apprendre à ne rien détecter quand l'objet est absent.

**Objectif :** Entraîner un détecteur robuste qui ne produit pas de faux positifs sur des images sans l'objet cible.

**Matériel nécessaire :**

- Votre smartphone ou webcam
- Un objet à détecter (cube, balle, tasse, etc.)
- Un environnement varié

**Partie A : Création du dataset**

**Consigne :** Créer un dataset de 150 images réparties ainsi :

1) **100 images AVEC l'objet** :
   
   - Filmez une vidéo de 30 secondes avec l'objet visible
   - Variez les angles, distances et positions
   - Extrayez 100 frames équidistantes avec OpenCV

2) **50 images SANS l'objet** :
   
   - Filmez le même environnement sans l'objet (arrière-plans variés)
   - OU : prenez des photos de scènes aléatoires (bureau, table, étagère...)
   - Extrayez/sauvegardez 50 images

3) **Annotation dans Label Studio** :
   
   - Créez un projet et importez les 150 images
   - Pour les images AVEC l'objet : dessinez la boîte englobante
   - Pour les images SANS l'objet : soumettez l'image sans annotation (important !)
   - Exportez au format JSON

4) **Vérification du dataset** :

.. code-block:: python

   import json
   import os
   from pathlib import Path

   def verify_dataset(json_path, images_dir):
       """
       Vérifie le dataset et affiche les statistiques.
       
       Returns:
           dict avec statistiques
       """
       # TODO: Charger le JSON
       # TODO: Parcourir les annotations
       # TODO: Compter images avec/sans objet
       # TODO: Afficher les statistiques
       pass
   
   # Vérification
   stats = verify_dataset('project-annotations.json', 'images/')

**Questions Partie A :**

1) Pourquoi est-il important d'avoir des images sans l'objet dans le dataset ?
2) Que se passerait-il si on entraînait uniquement sur des images avec l'objet ?
3) Quel ratio présence/absence recommandez-vous ? (ex: 70/30, 50/50, 80/20 ?)

**Astuce Partie A :**

.. spoiler::
    .. discoverList::
        1. Pensez à l'importance des exemples négatifs pour éviter les faux positifs
        2. Réfléchissez au ratio optimal entre images avec et sans objet
        3. Dans Label Studio, une image vide doit être soumise sans annotation

.. slide::

**Partie B : CNN Custom avec gestion de l'absence**

**Consigne :** Adapter le CNN simple du chapitre pour gérer l'absence d'objet.

**Approche :** Le modèle prédit maintenant 5 valeurs :

- ``objectness`` : probabilité qu'un objet soit présent (0-1)
- ``x_center, y_center, width, height`` : coordonnées si objet présent

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class SimpleBBoxRegressorWithObjectness(nn.Module):
       """
       CNN qui prédit la présence d'un objet + sa boîte.
       Sortie : [objectness, x_center, y_center, width, height]
       """
       
       def __init__(self):
           super().__init__()
           
           # TODO: Définir les couches de convolution (backbone)
           # TODO: Définir les couches fully connected (head)
           # Rappel: 5 sorties [objectness, x, y, w, h]
           pass
       
       def forward(self, x):
           # TODO: Implémenter le forward pass
           # TODO: Séparer objectness (sigmoid) et bbox (sigmoid)
           pass

**Fonction de préparation des targets :**

.. code-block:: python

   def prepare_targets_with_objectness(targets, img_size=224):
       """
       Convertit les targets en format [objectness, x_c, y_c, w, h].
       Si aucune boîte : objectness=0, bbox=[0, 0, 0, 0]
       """
       # TODO: Parcourir les targets
       # TODO: Si boxes vide : objectness=0
       # TODO: Sinon : objectness=1 + normaliser bbox
       pass

**Loss combinée :**

.. code-block:: python

   class DetectionLoss(nn.Module):
       """Loss pour détection avec objectness."""
       
       def __init__(self):
           super().__init__()
           # TODO: Définir les losses (BCE pour objectness, MSE pour bbox)
           pass
       
       def forward(self, predictions, targets):
           """
           predictions: [B, 5] = [obj, x, y, w, h]
           targets: [B, 5] = [obj_gt, x_gt, y_gt, w_gt, h_gt]
           """
           # TODO: Calculer loss_obj (BCE sur objectness)
           # TODO: Calculer loss_bbox (MSE uniquement si objet présent)
           # TODO: Pondérer et combiner les losses
           pass

**Entraînement :**

.. code-block:: python

   # TODO: Créer le modèle, criterion, optimizer
   # TODO: Implémenter la boucle d'entraînement
   # TODO: Sauvegarder le meilleur modèle
   pass

**Évaluation et visualisation :**

.. code-block:: python

   @torch.no_grad()
   def evaluate_with_objectness(model, dataset, threshold=0.5, img_size=224):
       """Évalue avec détection de présence."""
       # TODO: Parcourir le dataset
       # TODO: Compter TP, FP, TN, FN
       # TODO: Calculer precision, recall, accuracy
       pass

**Questions Partie B :**

4) Pourquoi utilise-t-on une loss BCE pour objectness et MSE pour bbox ?
5) Pourquoi pondérer la loss_bbox par un facteur 5.0 ?
6) Que se passe-t-il si on met threshold=0.3 ? Et 0.8 ?
7) Comment interpréter un modèle avec haute précision mais faible recall ?

**Astuce Partie B :**

.. spoiler::
    .. discoverList::
        1. Réfléchissez aux types de losses appropriées pour classification vs régression
        2. Pensez à l'équilibrage entre les deux composantes de la loss
        3. Analysez l'impact du threshold sur les métriques (TP/FP/TN/FN)
        4. Interprétez le trade-off entre précision et recall

.. slide::

**Partie C : YOLO avec images négatives**

**Consigne :** Entraîner YOLOv11 sur le même dataset et comparer.

1) **Exporter au format YOLO** depuis Label Studio :
   
   - Cliquez sur "Export" → "YOLO"
   - Téléchargez le ZIP

2) **Organiser le dataset** :

.. code-block:: python

   from pathlib import Path
   import shutil
   import random
   
   def organize_yolo_dataset(images_dir, labels_dir, output_dir, split=(0.7, 0.15, 0.15)):
       """
       Organise le dataset pour YOLO avec split train/val/test.
       Gère automatiquement les images sans labels (négatives).
       """
       # TODO: Créer la structure de dossiers YOLO
       # TODO: Lister et mélanger les images
       # TODO: Faire le split train/val/test
       # TODO: Copier images et labels (si existent)
       pass

3) **Créer le fichier YAML** :

.. code-block:: yaml

   # data.yaml
   # TODO: Compléter avec vos chemins
   path: /chemin/absolu/vers/data_yolo
   train: images/train
   val: images/val
   test: images/test
   
   nc: 1
   names: ['mon_objet']

4) **Entraîner YOLO** :

.. code-block:: python

   from ultralytics import YOLO
   
   # TODO: Charger YOLOv11n et entraîner
   # TODO: Choisir epochs, imgsz, batch appropriés
   pass

5) **Évaluer et comparer** :

.. code-block:: python

   # TODO: Charger le meilleur modèle YOLO
   # TODO: Évaluer sur le test set
   # TODO: Comparer avec les métriques du CNN custom
   pass

**Questions Partie C :**

8) Comment YOLO gère-t-il les images sans objet ?
9) Quel modèle est le plus rapide ? Le plus précis ?
10) Lequel utiliseriez-vous en production ? Pourquoi ?

**Astuce Partie C :**

.. spoiler::
    .. discoverList::
        1. Analysez comment YOLO traite les images sans annotation
        2. Comparez la vitesse d'inférence entre CNN custom et YOLO
        3. Réfléchissez aux avantages/inconvénients pour la production
        4. Considérez le pré-entraînement et son impact sur les performances

**Résultat attendu :**

- Dataset de 150 images (100 avec objet, 50 sans)
- Modèle CNN avec objectness entraîné
- Modèle YOLO entraîné
- Comparaison des métriques (accuracy, precision, recall)

.. slide::

⚖️ Exercice 2 : Détection de deux objets différents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez créer un détecteur capable de distinguer deux objets différents sur la même image.

**Objectif :** Maîtriser la détection multi-classe et gérer plusieurs objets simultanés.

**Matériel nécessaire :**

- Deux objets distincts visuellement (ex: cube rouge + balle bleue, tasse + bouteille)
- Smartphone ou webcam
- Environnement varié

**Partie A : Dataset multi-objets**

**Consigne :** Créer un dataset de 200 images avec la répartition suivante :

1) **60 images avec l'objet 1 uniquement**
2) **60 images avec l'objet 2 uniquement**
3) **50 images avec les DEUX objets simultanément**
4) **30 images sans aucun objet**

**Script de capture automatisé :**

.. code-block:: python

   import cv2
   import os
   from pathlib import Path
   
   def capture_scenario(output_dir, scenario_name, num_images=60):
       """
       Capture des images depuis la webcam avec indicateur visuel.
       
       Args:
           output_dir: dossier de sortie
           scenario_name: nom du scénario (obj1, obj2, both, none)
           num_images: nombre d'images à capturer
       """
       # TODO: Créer le dossier de sortie
       # TODO: Ouvrir la webcam avec cv2.VideoCapture(0)
       # TODO: Boucle de capture:
       #   - Afficher la frame avec compteur
       #   - ESPACE = capturer et sauvegarder
       #   - ESC = quitter
       # TODO: Libérer la webcam
       pass
   
   # Programme de capture complet
   if __name__ == "__main__":
       output_base = "images_multi_objects"
       
       print("📷 CAPTURE MULTI-OBJETS")
       
       # TODO: Capturer les 4 scénarios:
       # - Scénario 1: obj1 (60 images)
       # - Scénario 2: obj2 (60 images)
       # - Scénario 3: both (50 images)
       # - Scénario 4: none (30 images)

**Annotation dans Label Studio :**

Configuration avec 2 classes :

.. code-block:: xml

   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label value="objet_1" background="red"/>
       <Label value="objet_2" background="blue"/>
     </RectangleLabels>
   </View>

**Consignes d'annotation :**

- Images avec objet_1 : dessinez une boîte rouge autour de objet_1
- Images avec objet_2 : dessinez une boîte bleue autour de objet_2
- Images avec les deux : dessinez les deux boîtes (rouge + bleue)
- Images sans objet : soumettez sans annotation

**Questions Partie A :**

11) Pourquoi capturer des images avec les deux objets ensemble ?
12) Quelle est la difficulté principale de ce dataset comparé à l'exercice 1 ?
13) Comment équilibrer le dataset si un objet est plus difficile à détecter ?

**Astuce Partie A :**

.. spoiler::
    .. discoverList::
        1. Pensez à l'importance d'avoir des images avec les deux objets simultanément
        2. Identifiez les difficultés spécifiques d'un dataset multi-classe
        3. Réfléchissez aux stratégies d'équilibrage si une classe est plus difficile
        4. Variez les configurations spatiales des objets

.. slide::

**Partie B : YOLO multi-classe**

**Consigne :** Entraîner YOLOv11 pour détecter les 2 objets.

1) **Organiser le dataset YOLO** :

.. code-block:: python

   import shutil
   from pathlib import Path
   import random
   
   def organize_multiclass_yolo(images_dir, labels_dir, output_dir):
       """Organise le dataset multi-classe pour YOLO."""
       # TODO: Créer la structure des dossiers
       # TODO: Lister et mélanger les images
       # TODO: Split 70/15/15 pour train/val/test
       # TODO: Copier images et labels
       # TODO: Analyser et afficher les statistiques par scénario
       pass

2) **Créer le fichier YAML** :

.. code-block:: yaml

   # data_multiclass.yaml
   # TODO: Compléter avec vos chemins
   path: /chemin/absolu/vers/data_yolo_multiclass
   train: images/train
   val: images/val
   test: images/test
   
   nc: 2
   names: ['objet_1', 'objet_2']

3) **Entraîner YOLO** :

.. code-block:: python

   from ultralytics import YOLO
   
   # TODO: Charger yolo11n.pt
   # TODO: Entraîner avec data_multiclass.yaml
   # TODO: Choisir epochs, imgsz, batch appropriés
   pass

4) **Évaluer par classe** :

.. code-block:: python

   # TODO: Charger le meilleur modèle
   # TODO: Évaluer avec model.val()
   # TODO: Afficher mAP global
   # TODO: Afficher métriques par classe (precision, recall, mAP)
   pass

5) **Tester sur images avec les 2 objets** :

.. code-block:: python

   import cv2
   from pathlib import Path
   import matplotlib.pyplot as plt
   
   # TODO: Récupérer les images 'both_*.jpg' du test set
   # TODO: Faire les prédictions
   # TODO: Visualiser en grille 2x3
   # TODO: Sauvegarder le résultat
   pass

**Questions Partie B :**

14) Comment YOLO gère-t-il plusieurs objets de classes différentes sur une même image ?
15) Que se passe-t-il si les deux objets se chevauchent beaucoup ?
16) Comment améliorer la détection si un objet est systématiquement mieux détecté que l'autre ?
17) Quelle est la différence entre mAP@0.5 et mAP@0.5:0.95 ?

**Astuce Partie B :**

.. spoiler::
    .. discoverList::
        1. Analysez le mécanisme de détection multi-classe de YOLO (grille + NMS)
        2. Réfléchissez aux problèmes de chevauchement d'objets
        3. Pensez aux stratégies pour gérer le déséquilibre entre classes
        4. Comprenez la différence entre mAP@0.5 et mAP@0.5:0.95
        5. Utilisez la visualisation pour déboguer

**Résultat attendu :**

- Dataset de 200 images organisé (train/val/test)
- Modèle YOLO entraîné avec 2 classes
- mAP@0.5 > 0.7 pour chaque classe
- Visualisation des prédictions sur images avec 2 objets

.. slide::

🌶️ Exercice 3 : Tracking vidéo en temps réel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez créer un système de tracking qui détecte et suit vos objets dans une vidéo, en temps réel si possible.

**Objectif :** Implémenter un système complet de tracking avec détection, suivi d'identité et comptage des apparitions/disparitions.

**Matériel nécessaire :**

- Vidéo de 30-60 secondes avec vos objets qui entrent/sortent du champ
- Modèle YOLO entraîné (exercice 2)
- (Optionnel) Webcam pour test en temps réel

**Partie A : Tracking simple avec détection frame par frame**

**Consigne :** Créer un script de base qui détecte les objets sur chaque frame.

1) **Créer une vidéo de test** :

.. code-block:: python

   """
   SCÉNARIO DE LA VIDÉO (30-60 secondes) :
   
   - 0-10s  : Aucun objet visible
   - 10-20s : Objet 1 entre dans le champ, se déplace
   - 20-30s : Objet 2 entre aussi (les 2 sont visibles)
   - 30-40s : Objet 1 sort du champ (seul objet 2 reste)
   - 40-50s : Objet 2 sort aussi
   - 50-60s : Aucun objet visible
   
   Filmez avec votre smartphone !
   """

2) **Détection frame par frame** :

.. code-block:: python

   import cv2
   from ultralytics import YOLO
   import numpy as np
   from collections import defaultdict
   import time
   
   def detect_on_video(model_path, video_path, output_path, conf_threshold=0.5):
       """
       Détecte les objets sur chaque frame et sauvegarde la vidéo.
       
       Returns:
           dict avec statistiques de détection
       """
       # TODO: Charger le modèle YOLO
       # TODO: Ouvrir la vidéo et récupérer fps, dimensions, nombre de frames
       # TODO: Créer VideoWriter pour la sortie
       # TODO: Initialiser un dict de statistiques
       # TODO: Boucle sur les frames:
       #   - Lire frame
       #   - Faire la prédiction YOLO
       #   - Mesurer le temps de traitement
       #   - Dessiner les détections + info (frame, objets, FPS)
       #   - Sauvegarder dans la vidéo de sortie
       #   - Accumuler les statistiques
       # TODO: Libérer les ressources
       # TODO: Afficher les statistiques finales
       pass

3) **Analyser les détections** :

.. code-block:: python

   import matplotlib.pyplot as plt
   
   def plot_detection_stats(stats, class_names):
       """Visualise les statistiques de détection."""
       # TODO: Créer figure 2x2
       # TODO: Graphique 1: Détections par frame (ligne)
       # TODO: Graphique 2: Détections par classe (barres)
       # TODO: Graphique 3: Distribution du nombre d'objets (histogramme)
       # TODO: Graphique 4: Temps de traitement (ligne + moyenne)
       # TODO: Sauvegarder la figure
       pass

**Questions Partie A :**

18) Quel est le FPS moyen de votre système ? Est-ce suffisant pour du temps réel (>30 fps) ?
19) Pourquoi le temps de traitement varie-t-il d'une frame à l'autre ?
20) Comment pourriez-vous améliorer la vitesse si elle est trop lente ?

**Astuce Partie A :**

.. spoiler::
    .. discoverList::
        1. Évaluez si votre FPS est suffisant pour le temps réel (seuil ~30 fps)
        2. Analysez les causes de variation du temps de traitement
        3. Réfléchissez aux optimisations possibles (résolution, modèle, fréquence)
        4. Pensez à adapter pour une webcam en direct

.. slide::

**Partie B : Tracking avec identité (Object ID)**

**Consigne :** Ajouter un système de suivi qui assigne un ID unique à chaque objet.

1) **Implémenter un tracker simple** :

.. code-block:: python

   from scipy.spatial import distance
   import numpy as np
   
   class SimpleTracker:
       """
       Tracker simple basé sur la distance entre détections.
       """
       
       def __init__(self, max_distance=50, max_disappeared=30):
           """
           Args:
               max_distance: distance max (pixels) pour associer détection à objet existant
               max_disappeared: nombre de frames max avant de supprimer un objet
           """
           # TODO: Initialiser next_object_id, objects dict, paramètres
           pass
       
       def update(self, detections):
           """
           Met à jour le tracker avec nouvelles détections.
           
           Args:
               detections: list of dict {'bbox': [x1, y1, x2, y2], 'class': int, 'conf': float}
           
           Returns:
               dict {object_id: {'centroid': (x, y), 'class': int, 'bbox': [x1, y1, x2, y2]}}
           """
           # TODO: Si pas de détections, incrémenter disappeared et supprimer si > max
           # TODO: Calculer centroids des nouvelles détections
           # TODO: Si pas d'objets existants, créer tous les nouveaux
           # TODO: Sinon:
           #   - Calculer matrice de distances (distance.cdist)
           #   - Associer par plus proche voisin
           #   - Vérifier distance max et même classe
           #   - Mettre à jour les objets associés
           #   - Marquer les non-associés comme disparus
           #   - Créer les nouveaux objets
           # TODO: Retourner objects dict
           pass
       
       def _register(self, centroid, class_id, bbox):
           """Enregistre un nouvel objet."""
           # TODO: Créer nouvel objet avec next_object_id
           # TODO: Incrémenter next_object_id
           pass

2) **Appliquer le tracker sur la vidéo** :

.. code-block:: python

   def track_video(model_path, video_path, output_path, class_names, conf_threshold=0.6):
       """Tracking avec identités sur vidéo."""
       # TODO: Charger YOLO et créer SimpleTracker
       # TODO: Ouvrir vidéo et créer VideoWriter
       # TODO: Initialiser dict events (appeared/disappeared) et previous_ids
       # TODO: Boucle sur frames:
       #   - Faire la détection YOLO
       #   - Convertir en format tracker
       #   - Mettre à jour tracker
       #   - Détecter apparitions/disparitions
       #   - Dessiner boîtes avec IDs et couleurs uniques
       #   - Dessiner centroids
       #   - Afficher info frame
       #   - Sauvegarder frame
       # TODO: Afficher statistiques finales
       # TODO: Retourner events
       pass

3) **Analyser les événements** :

.. code-block:: python

   from collections import defaultdict
   
   def analyze_events(events, class_names):
       """Analyse les événements d'apparition/disparition."""
       # TODO: Afficher tableau des apparitions (frame, objet, ID)
       # TODO: Afficher tableau des disparitions (frame, ID)
       # TODO: Calculer et afficher statistiques par classe
       pass

**Questions Partie B :**

21) Comment le tracker gère-t-il deux objets de la même classe proches l'un de l'autre ?
22) Que se passe-t-il si un objet est temporairement occulté (caché) ?
23) Comment améliorer le tracker pour gérer les occlusions ?
24) Pourquoi utiliser la distance euclidienne entre centroids plutôt que l'IoU entre boîtes ?

**Astuce Partie B :**

.. spoiler::
    .. discoverList::
        1. Analysez les risques de confusion d'ID entre objets proches
        2. Réfléchissez au rôle du paramètre `max_disappeared` pour les occlusions
        3. Pensez aux améliorations possibles (IoU, features visuelles, prédiction)
        4. Comparez distance euclidienne vs IoU pour l'association
        5. Explorez les trackers avancés (DeepSORT, ByteTrack)

**Résultat attendu :**

- Vidéo `output_tracking.mp4` avec IDs affichés et couleurs uniques
- Liste des événements d'apparition/disparition
- Statistiques du tracking (nombre d'objets uniques, durées de vie)

.. slide::

**Partie C : Tracking en temps réel sur webcam (Bonus)**

**Consigne :** Adapter le système pour fonctionner sur webcam en temps réel.

.. code-block:: python

   def track_webcam_realtime(model_path, class_names, conf_threshold=0.6):
       """Tracking en temps réel sur webcam."""
       # TODO: Charger YOLO et créer SimpleTracker
       # TODO: Ouvrir webcam avec cv2.VideoCapture(0)
       # TODO: Initialiser couleurs dict et fps_list
       # TODO: Boucle infinie:
       #   - Lire frame webcam
       #   - Mesurer temps de traitement
       #   - Faire détection YOLO
       #   - Mettre à jour tracker
       #   - Dessiner boîtes avec IDs, centroids
       #   - Calculer et afficher FPS moyen (sur 30 frames)
       #   - Afficher nombre d'objets
       #   - Gérer touches: 'q' = quitter, 's' = screenshot
       # TODO: Libérer webcam et afficher statistiques
       pass

**Questions Partie C :**

25) Quelle est la latence (délai) entre le mouvement réel et l'affichage ?
26) Comment optimiser pour atteindre 60 FPS sur webcam ?
27) Quelles sont les applications pratiques d'un tel système ?

**Astuce Partie C :**

.. spoiler::
    .. discoverList::
        1. Mesurez la latence entre mouvement réel et affichage
        2. Explorez les optimisations pour atteindre 60 FPS
        3. Identifiez des applications pratiques d'un tel système
        4. Pensez aux stratégies de réduction de latence

**Résultat attendu :**

- Système de tracking temps réel sur webcam
- FPS > 20 (minimum pour fluidité)
- Détection et suivi corrects des objets qui entrent/sortent
- Screenshots possibles pendant l'exécution

.. slide::

🎯 Conclusion du TP
~~~~~~~~~~~~~~~~~~~

**Bilan des compétences acquises :**

1. **Détection avec gestion de l'absence** :
   - Annotation de cas négatifs (images sans objet)
   - Ajout d'un score d'objectness (CNN custom)
   - Utilisation d'images négatives avec YOLO
   - Métriques : TP, FP, TN, FN pour évaluer les faux positifs

2. **Détection multi-classe** :
   - Dataset équilibré avec plusieurs objets
   - Annotation de multiple classes dans Label Studio
   - Entraînement YOLO multi-classe
   - Évaluation par classe (mAP, Precision, Recall)
   - Gestion des cas avec plusieurs objets simultanés

3. **Tracking vidéo en temps réel** :
   - Détection frame par frame sur vidéo
   - Association d'identités aux objets (tracking)
   - Comptage des apparitions/disparitions
   - Performance temps réel sur webcam
   - Gestion des occlusions temporaires

**Comparaison CNN custom vs YOLO :**


   +------------------------+--------------------------------+--------------------------------+
   | **Critère**            | **CNN Custom**                 | **YOLO**                       |
   +========================+================================+================================+
   | **Facilité**           | Nécessite implémentation       | Prêt à l'emploi                |
   |                        | complète (loss, training loop) | (``model.train()``)            |
   +------------------------+--------------------------------+--------------------------------+
   | **Performance**        | Plus lent (pas optimisé)       | Très rapide (optimisé C++)     |
   +------------------------+--------------------------------+--------------------------------+
   | **Flexibilité**        | Total contrôle sur             | Architecture fixée             |
   |                        | architecture et loss           |                                |
   +------------------------+--------------------------------+--------------------------------+
   | **Multi-objets**       | Difficile (NMS manuel)         | Natif (détections multiples)   |
   +------------------------+--------------------------------+--------------------------------+
   | **Dataset requis**     | Petit dataset suffit           | Préfère grands datasets        |
   |                        | (100-200 images)               | (500+ images)                  |
   +------------------------+--------------------------------+--------------------------------+
   | **Cas d'usage**        | - Apprentissage                | - Production                   |
   |                        | - Preuve de concept            | - Applications réelles         |
   |                        | - Recherche                    | - Temps réel                   |
   +------------------------+--------------------------------+--------------------------------+

**Pour aller plus loin :**

1. **Augmentation de données avancée** :
   - Mixup, Cutout, Mosaic
   - Augmentation spécifique au domaine (ex: conditions d'éclairage)

2. **Tracking avancé** :
   - DeepSORT (avec features d'apparence)
   - ByteTrack (gestion d'occlusions)
   - Multi-object tracking avec réidentification

3. **Optimisation pour production** :
   - Export ONNX pour déploiement
   - Quantization (INT8) pour edge devices
   - TensorRT pour GPU NVIDIA

4. **Applications avancées** :
   - Détection d'anomalies (objets inhabituels)
   - Estimation de pose (keypoints)
   - Segmentation d'instances (masques précis)

