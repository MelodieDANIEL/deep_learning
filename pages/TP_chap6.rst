🏋️ Travaux Pratiques 6 : Détection d'Objets
==============================================

.. slide::

Sur cette page se trouvent des exercices de TP sur le Chapitre 6. Ils sont classés par niveau de difficulté suivant :

.. discoverList::
    * Facile : 🍀
    * Moyen : ⚖️
    * Difficile : 🌶️

.. slide::

🍀 Exercice 1 : Détection avec présence/absence - CNN Custom vs YOLO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez créer un dataset où certaines images contiennent votre objet et d'autres non. Vous comparerez ensuite un modèle CNN custom avec YOLO, avec et sans augmentation de données.

**Objectif :** Maîtriser la détection d'objets avec gestion des cas négatifs (images sans objet) et comparer les approches custom vs YOLO.

**Matériel nécessaire :**

- Votre smartphone ou webcam
- Un objet à détecter (cube, cylindre, balle, tasse, etc.)
- Environnement varié pour les prises de vue

**Partie A : Création du dataset (100 images avec objet, 40 sans objet)**

**Consigne :** Écrire un programme qui :

.. step::
    1) Prend une vidéo :
    
    - Avec l'objet visible, variez les angles, distances et orientations,
    - et avec le même environnement SANS l'objet.

.. step::
    2) Extrait les frames de la vidéo.

.. step::
    3) Annote les images au format YOLO :
    
    - Créer les dossiers ``dataset/images/`` et ``dataset/labels/``
    - Pour les images AVEC l'objet : créer un fichier .txt avec le format ``class_id x_center y_center width height`` (normalisé 0-1)
    - Pour les images SANS l'objet : ne pas créer de fichier .txt correspondant

.. step::
    4) (optionnel) Vérifie le dataset.


**Questions Partie A :**

.. step::
    5) Pourquoi est-il crucial d'avoir des images sans l'objet dans le dataset ?

.. step::
    6) Que se passerait-il si on entraînait uniquement sur des images positives ?

.. step::
    7) Quel ratio présence/absence est optimal ? (70/30, 80/20, 50/50 ?)


**Astuce Partie A :**

.. spoiler::
    .. discoverList::
        1. Les exemples négatifs évitent les faux positifs (détections fantômes sur fond vide).
        2. Sans images négatives, le modèle détecte toujours quelque chose même sur fond vide.
        3. Un ratio 70/30 ou 60/40 (avec objet / sans objet) est généralement optimal.
        4. Images sans label = pas de fichier .txt = image négative pour YOLO (il apprend à ne rien détecter).

**Résultat attendu Partie A :**

- Dataset de ~140 images : 100 avec objet, 40 sans objet
- Structure correcte : ``dataset/images/`` et ``dataset/labels/``
- Fichiers .txt au format YOLO pour les images positives uniquement

.. slide::

**Partie B : CNN Custom avec gestion de l'objectness**

**Consigne :** Écrire un programme qui :

.. step::
    1) Implémente une architecture CNN simple pour la détection que vous devez proposer.

.. step::
    2) Crée un Dataset PyTorch pour charger les images et labels YOLO.

.. step::
    3) Implémente la loss personnalisée pour la détection que vous devez proposer.      

.. step::
    4) Entraîne le modèle SANS augmentation pendant 100 epochs avec un early stopping d'une patience = 15.

.. step::
    5) Sauvegarde le meilleur modèle dans ``best_model_no_aug.pth``.

.. step::
    6) Implémente une fonction d'évaluation avec métriques (TP, FP, TN, FN, IoU).

**Astuce Partie B :**

.. spoiler::
    .. discoverList::
        1. **Architecture CNN** : Votre CNN doit prédire 5 valeurs : ``[has_object, x_center, y_center, width, height]``. Sortie finale avec ``nn.Sigmoid()`` pour borner entre 0 et 1. Proposez une architecture simple, construite comme au chapitre 5 avec deux blocs ``nn.Sequential`` : un bloc ``convolutions`` (3-4 couches de convolution + pooling) puis un bloc de couches ``nn.Linear`` dont la dernière est ``nn.Linear(nb_entrees, 5)``.
        2. **Dataset PyTorch** : Pour les images SANS objet (pas de .txt), retournez ``target = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])``. Pour les images AVEC objet, parsez le fichier .txt et retournez ``target = torch.tensor([1.0, x_c, y_c, w, h])``.
        3. **Loss combinée** : BCE (Binary Cross Entropy) pour has_object (classification binaire), MSE (Mean Squared Error) pour bbox (régression). Utilisez un masque ``mask = target_obj > 0.5`` pour ne calculer loss_bbox QUE sur les images avec objet présent.
        4. **Entraînement** : Adam optimizer avec lr=1e-4, batch_size=4 ou 8. Implémentez early stopping avec patience=15 (arrêter si validation loss ne diminue pas pendant 15 epochs). Sauvegardez le meilleur modèle avec ``torch.save(model.state_dict(), 'best_model_no_aug.pth')``.
        5. **Évaluation (IoU)** : Pour calculer l'IoU, convertissez YOLO (x_c, y_c, w, h) en coordonnées (x_min, y_min, x_max, y_max) avec ``x_min = x_c - w/2``, puis calculez l'intersection et l'union des rectangles.
        6. **Métriques** : TP = GT a objet ET pred > threshold ET IoU > 0.5, FP = GT sans objet ET pred > threshold, TN = GT sans objet ET pred ≤ threshold, FN = GT a objet ET pred ≤ threshold.
        7. **Threshold de détection** : Threshold bas (0.3) → plus de détections → +recall, -precision. Threshold haut (0.8) → moins de détections → +precision, -recall. Optimal généralement autour de 0.5-0.6.
        8. **Split du dataset** : 70% train, 15% validation, 15% test. Utilisez ``torch.utils.data.random_split()`` pour diviser le dataset de manière reproductible.

**Résultat attendu Partie B :**

- Modèle CNN custom entraîné (sauvegardé dans ``best_model_no_aug.pth``)
- Accuracy sur le test set : très variable d'un entraînement à l'autre avec si peu d'images (entre 40 % et 100 % dans nos essais, le plus souvent autour de 75 %)
- Mean IoU : ~0.5-0.75 pour les détections correctes
- Les erreurs viennent surtout de la localisation (boîtes avec un IoU < 0.5), rarement de la présence ou de l'absence du cube

.. slide::

**Partie C : YOLO11 sans augmentation - Comparaison**

**Consigne :** Écrire un programme qui :

.. step::
    1) Organise le dataset pour YOLO avec **le même split train/val/test que le CNN custom** (fonction ``prepare_yolo_from_existing_dataset`` de la section 8.1 du cours, avec la même graine), puis crée le fichier ``dataset.yaml`` (section 8.2 du cours) :

    .. code-block:: yaml

       # data_cube/dataset.yaml
       path: /chemin/absolu/vers/data_cube
       train: images/train
       val: images/val
       test: images/test
       
       nc: 1
       names: ['nom_label']

    ⚠️ Les images de test ne doivent jamais servir à l'entraînement : sinon, les métriques sur le test set sont faussement bonnes.

.. step::
    2) Entraîne YOLO11 sur le dataset.

.. step::
    3) Évalue YOLO11 sur le test set avec les mêmes métriques que le CNN custom.

.. step::
    4) Compare les résultats dans un tableau comme par exemple :

    .. code-block:: python

       # TODO: Afficher tableau comparatif
       # +------------+---------------+-------------+
       # | Métrique   | CNN Custom    | YOLO11      |
       # +------------+---------------+-------------+
       # | Accuracy   | XX.XX%        | XX.XX%      |
       # | Precision  | XX.XX%        | XX.XX%      |
       # | Recall     | XX.XX%        | XX.XX%      |
       # | Mean IoU   | 0.XXXX        | 0.XXXX      |
       # | FPS        | X.X           | XX.X        |
       # | Params     | 3.3M          | 2.6M        |
       # +------------+---------------+-------------+

**Questions Partie C :**

.. step::
    5) Quel modèle est le plus rapide en inférence ? Le plus précis ?

.. step::
    6) Pourquoi YOLO est-il meilleur malgré moins de paramètres ?


**Astuce Partie C :**

.. spoiler::
    .. discoverList::
        1. YOLO apprend automatiquement à ne rien détecter sur les images sans .txt
        2. Mesurez le temps d'inférence des deux modèles sur les mêmes images (avec ``time.time()``) : un petit CNN peut être plus rapide que YOLO, qui doit en plus trier des centaines de boîtes candidates (NMS)
        3. YOLO bénéficie du pré-entraînement sur COCO (80 classes, 118 000 images d'entraînement, voir la section 7.4 du cours)
        4. Transfer learning : YOLO a déjà appris des filtres qui détectent des motifs génériques (contours, textures)
        5. YOLO est préférable (précision, robustesse, communauté)
        6. Ultralytics applique des augmentations **par défaut** : pour un entraînement vraiment sans augmentation, passez ``mosaic=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, translate=0, scale=0`` à ``model.train``
        7. Avec très peu d'images de validation, ``best.pt`` peut correspondre à une epoch du tout début, où les boîtes sont bien placées mais avec une confiance très faible : vérifiez les détections, et utilisez ``last.pt`` (dernière epoch) si besoin

**Résultat attendu Partie C :**

Exemple obtenu sur notre dataset (132 images, dont 21 images de test, les mêmes pour les deux modèles) :

- YOLO11 : aucune erreur sur le test set, IoU moyen ≈ 0.93 (localisation très précise)
- CNN custom : IoU moyen ≈ 0.6-0.75, accuracy très variable d'un entraînement à l'autre
- Vitesse : le CNN custom est le plus rapide (≈ 250 images/s contre ≈ 95 pour YOLO sur CPU, chargement des images compris)
- YOLO est le plus précis, surtout pour la localisation


.. slide::

**Partie D : Augmentation de données - Comparaison finale**

**Consigne :** Écrire un programme qui :

.. step::
    1) Crée un dataset augmenté avec rééquilibrage :

    .. code-block:: python

       import torchvision.transforms as T
       import random
       
       class AugmentedYOLODataset(Dataset):
           """Dataset avec augmentation adaptée à la détection."""
           
           def __init__(self, image_dir, label_dir, augment=False):
               # TODO: Initialiser comme YOLODetectionDataset
               
               # TODO: Ajouter transformations d'augmentation
               # self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3)
               pass
           
           def horizontal_flip(self, image, bbox):
               """Flip horizontal avec ajustement bbox."""
               image = T.functional.hflip(image)
               if bbox is not None:
                   bbox = bbox.copy()  # IMPORTANT: copier !
                   bbox[0] = 1.0 - bbox[0]  # x_center inversé
               return image, bbox
           
           def __getitem__(self, idx):
               # TODO: Charger image et label
               # TODO: Si augment=True:
               #   - 50% flip horizontal
               #   - 80% color jitter
               # TODO: Appliquer normalisation
               pass

.. step::
    2) Rééquilibre le dataset (dupliquer les images sans objet pour avoir 50/50).

.. step::
    3) Entraîne le CNN custom AVEC augmentation :

    .. code-block:: python

       # TODO: Créer AugmentedYOLODataset avec augment=True pour train
       # TODO: Entraîner avec les mêmes hyperparamètres que dans la partie B
       # TODO: Comparer train loss et val loss (vérifier overfitting réduit)
       # TODO: Sauvegarder dans 'best_model_with_aug.pth'

.. step::
    4) Entraîne YOLO AVEC augmentation.

.. step::
    5) Compare les 4 modèles finaux :

    .. code-block:: python

       # TODO: Charger les 4 modèles
       # TODO: Évaluer sur le MÊME test set
       # TODO: Afficher tableau comparatif complet
       # +---------------------+---------------+---------------+-------------+-------------+
       # | Métrique            | CNN No Aug    | CNN + Aug     | YOLO11      |YOLO11 + Aug |
       # +---------------------+---------------+---------------+-------------+-------------+
       # | Accuracy            | XX.XX%        | XX.XX%        | XX.XX%      |XX.XX%       |
       # | Precision           | XX.XX%        | XX.XX%        | XX.XX%      |XX.XX%       |
       # | Recall              | XX.XX%        | XX.XX%        | XX.XX%      |XX.XX%       |
       # | Mean IoU            | 0.XXXX        | 0.XXXX        | 0.XXXX      |0.XXXX       |
       # | Gap Train/Val (%)   | XX            | XX            | XX          | XX          |
       # +---------------------+---------------+---------------+-------------+-------------+


**Questions Partie D :**

.. step::
    6) L'augmentation améliore-t-elle les performances du CNN custom ?

.. step::
    7) Le gap train/val est-il réduit avec l'augmentation ?

.. step::
    8) Pourquoi l'augmentation seule ne suffit pas à rattraper YOLO ?

.. step::
    9) Quelle est l'importance du rééquilibrage (50/50) ?


**Astuce Partie D :**

.. spoiler::
    .. discoverList::
        1. L'augmentation réduit souvent l'overfitting (gap train/val plus petit), mais pas toujours : avec très peu d'images, elle peut aussi rendre l'apprentissage plus difficile pour un petit CNN
        2. Le rééquilibrage évite le biais : sans lui, le modèle détecte trop souvent
        3. YOLO reste supérieur car il combine pré-entraînement + architecture optimisée
        4. Flip horizontal : x_center → 1 - x_center, y_center inchangé, w et h inchangés
        5. Augmentation adaptée détection : éviter rotations fortes (change orientation objet)
        6. Color jitter OK car n'affecte pas les coordonnées spatiales
        7. La validation et le test ne doivent pas être augmentés : créez un dataset avec ``augment=True`` pour l'entraînement et un avec ``augment=False``, puis appliquez ``random_split`` aux deux avec la même graine (comme dans l'exercice 3 du TP 5), ou réutilisez les indices du découpage de la partie B (``train_dataset.indices``) avec ``torch.utils.data.Subset``. Rééquilibrez **après** le découpage, et seulement le train

**Résultat attendu Partie D :**

Exemple obtenu sur notre dataset (même test set de 21 images pour les 4 modèles, résultats moyennés sur une dizaine d'entraînements pour le CNN) :

- CNN sans augmentation : résultats très variables d'un entraînement à l'autre (accuracy moyenne ≈ 70 %, IoU moyen ≈ 0.55)
- CNN avec augmentation : **moins bon** dans presque tous nos essais (accuracy moyenne ≈ 50 %, IoU moyen ≈ 0.35) : avec seulement 64 images de cube, ce petit CNN n'arrive pas à profiter de l'augmentation
- YOLO11, avec ou sans augmentation : aucune erreur sur le test set, IoU moyen ≈ 0.93, gap train/val nul
- Conclusion : l'augmentation n'est pas magique ; sur un petit dataset, c'est le pré-entraînement de YOLO qui fait la différence


.. slide::

⚖️ Exercice 2 : Détection multi-objets (2 classes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez créer un détecteur capable de distinguer deux objets différents sur la même image.

**Objectif :** Maîtriser la détection multi-classe et gérer plusieurs objets simultanés.

**Matériel nécessaire :**

- Deux objets distincts visuellement (ex: cube + cylindre)
- Smartphone ou webcam

**Partie A : Dataset multi-objets**

**Consigne :** Écrire un programme qui :

.. step::
    1) Capture ~200 images réparties ainsi par exemple :
    
    - **60 images** : objet 1 uniquement
    - **60 images** : objet 2 uniquement
    - **50 images** : les DEUX objets simultanément
    - **30 images** : aucun objet

.. step::
    2) Annote le dataset :
    
    - Classe 0 : objet_1 (ex: cube)
    - Classe 1 : objet_2 (ex: cylindre)
    - Format YOLO : ``class_id x_center y_center width height``

.. step::
    3) Organise et importe le dataset pour YOLO.


**Questions Partie A :**

.. step::
    4) Quelle est la difficulté principale de ce dataset comparé à l'exercice 1 ?

.. step::
    5) Comment équilibrer le dataset si un objet est plus difficile à détecter ?

.. step::
    6) Que se passe-t-il si les deux objets se chevauchent beaucoup ?


**Astuce Partie A :**

.. spoiler::
    .. discoverList::
        1. Les images avec les deux objets apprennent au modèle à les distinguer simultanément
        2. Difficulté : confusion entre classes si objets similaires, gestion multi-détections
        3. Équilibrage : augmenter les données de la classe sous-représentée
        4. Chevauchement : YOLO gère bien grâce au NMS (Non-Maximum Suppression)

**Résultat attendu Partie A :**

- Dataset de 200 images organisé en train/val/test
- Annotations multi-classe correctes (classe 0 et 1)
- Distribution équilibrée entre les 4 scénarios

.. slide::

**Partie B : YOLO multi-classe**

**Consigne :** Écrire un programme qui :

.. step::
    1) Crée le fichier ``dataset.yaml`` pour 2 classes :

    .. code-block:: yaml

       # data_multiclass/dataset.yaml
       path: /chemin/absolu/vers/data_multiclass
       train: images/train
       val: images/val
       test: images/test
       
       nc: 2
       names: ['label1', 'label2']

.. step::
    2) Entraîne YOLO11 multi-classe.

.. step::
    3) Évalue par classe (mAP, precision, recall, etc.).


**Questions Partie B :**

.. step::
    4) Comment YOLO gère-t-il plusieurs objets de classes différentes sur une même image ?

.. step::
    5) Que se passe-t-il si les deux objets se chevauchent fortement ?

.. step::
    6) Comment améliorer la détection si objet_1 est systématiquement mieux détecté qu'objet_2 ?

.. step::
    7) Pourquoi utiliser NMS (Non-Maximum Suppression) ?


**Astuce Partie B :**

.. spoiler::
    .. discoverList::
        1. YOLO prédit plusieurs boîtes par grille, chacune avec une classe
        2. NMS élimine les détections redondantes (IoU > seuil avec même classe)
        3. Si chevauchement fort : risque de supprimer une détection valide avec NMS agressif
        4. Améliorer détection déséquilibrée : augmenter weight de la classe faible, plus de données
        5. mAP@0.5 : moyenne precision avec IoU≥0.5 (localisation tolérante)
        6. mAP@0.5:0.95 : moyenne sur plusieurs seuils IoU (plus strict, meilleure métrique)
        7. NMS garde la boîte avec le plus haut score de confiance parmi les overlaps
        8. Analyser les confusions avec une matrice de confusion (classe prédite vs GT)

**Résultat attendu Partie B :**

- YOLO multi-classe entraîné
- Exemple obtenu sur notre dataset (125 images simples, fond uni) : mAP@0.5 ≈ 0.99 et mAP@0.5:0.95 ≈ 0.89 (≈ 0.90 pour le cube, ≈ 0.88 pour le cylindre). Les valeurs dépendent beaucoup de votre dataset. pour chaque objet


.. slide::

🏋️ Exercices supplémentaires 6
===============================
Dans cette section, il y a des exercices supplémentaires pour vous entraîner. Ils suivent le même classement de difficulté que précédemment.


.. slide::

⚖️ Exercice supplémentaire 1 : Visualisation des résultats de détection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cet exercice propose de visualiser et analyser les résultats de détection de vos modèles.

**Objectif :** Comprendre les forces et faiblesses de vos modèles de détection à travers la visualisation.

**Consignes** :

.. step::
    1) Utiliser le modèle CNN custom ou YOLO entraîné dans l'exercice 1

.. step::
    2) Créer une fonction qui affiche les meilleurs résultats de détection :

    .. code-block:: python
    
        def visualize_best_detections(model, test_images, test_labels, num_examples=6):
            """
            Affiche les meilleures détections (IoU > 0.8)
            """
            # TODO: Prédire sur test set
            # TODO: Filtrer les détections avec has_object > 0.5 ET IoU > 0.8
            # TODO: Afficher en grille 2x3:
            #   - Image originale avec bbox GT (vert)
            #   - Image avec bbox prédite (bleu)
            #   - Titre: IoU = X.XX, Conf = X.XX
            pass

.. step::
    3) Créer une fonction pour visualiser les images sans objet (vrais négatifs) :

    .. code-block:: python
    
        def visualize_empty_images(model, test_images, test_labels, num_examples=6):
            """
            Affiche les images correctement identifiées comme vides
            """
            # TODO: Filtrer images où GT n'a pas d'objet (target[0] == 0)
            # TODO: Filtrer où prédiction < 0.5 (correct)
            # TODO: Afficher avec titre: "Conf vide: X.XX"
            pass

.. step::
    4) Créer une fonction pour visualiser les erreurs :

    .. code-block:: python
    
        def visualize_errors(model, test_images, test_labels, num_examples=6):
            """
            Affiche les cas problématiques
            """
            # TODO: Cas 1: Faux positifs (détection sur fond vide)
            # TODO: Cas 2: Faux négatifs (objet non détecté)
            # TODO: Cas 3: Mauvaise localisation (IoU < 0.5 mais objet détecté)
            # TODO: Afficher en 3 lignes
            pass

.. step::
    5) Comparer les visualisations entre CNN custom et YOLO11


**Questions :**

.. step::
    6) Quel type d'erreur est le plus fréquent pour chaque modèle ?

.. step::
    7) Dans quelles conditions les modèles ont-ils le plus de difficultés ?

.. step::
    8) YOLO fait-il des erreurs différentes du CNN custom ?


**Astuce :**
.. spoiler::
    .. discoverList::
        1. Pour dessiner les bbox : ``cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)``
        2. Convertir YOLO → xyxy : ``x_min = int((x_c - w/2) * img_width)``
        3. Couleur GT : vert ``(0, 255, 0)``, prédiction : bleu ``(255, 0, 0)``
        4. Pour les faux positifs : ``(GT sans objet) AND (pred > 0.5)``
        5. Pour les faux négatifs : ``(GT avec objet) AND (pred < 0.5)``
        6. Utilisez ``plt.subplots()`` pour créer des grilles de visualisation


**Résultats attendus :**

- 3 grilles de visualisation : meilleures détections, images vides, erreurs
- Analyse comparative CNN vs YOLO : types d'erreurs, fréquence
- Compréhension des cas difficiles (occlusion partielle, faible luminosité, etc.)


.. slide::

🌶️ Exercice supplémentaire 2 : Augmentation de données et comparaison finale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez implémenter l'augmentation de données adaptée à la détection et comparer les résultats.

**Objectif :**  
Comprendre l'impact de l'augmentation sur la généralisation et réduire l'overfitting.

**Consignes** :

.. step::
    1) Créer un dataset augmenté avec rééquilibrage :

    .. code-block:: python

       import torchvision.transforms as T
       import random
       
       class AugmentedYOLODataset(Dataset):
           """Dataset avec augmentation adaptée à la détection."""
           
           def __init__(self, image_dir, label_dir, augment=False):
               # TODO: Initialiser comme YOLODetectionDataset
               
               # TODO: Ajouter transformations d'augmentation
               # self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3)
               pass
           
           def horizontal_flip(self, image, bbox):
               """Flip horizontal avec ajustement bbox."""
               image = T.functional.hflip(image)
               if bbox is not None:
                   bbox = bbox.copy()  # IMPORTANT: copier !
                   bbox[0] = 1.0 - bbox[0]  # x_center inversé
               return image, bbox
           
           def __getitem__(self, idx):
               # TODO: Charger image et label
               # TODO: Si augment=True:
               #   - 50% flip horizontal
               #   - 80% color jitter
               # TODO: Appliquer normalisation
               pass

.. step::
    2) Rééquilibrer le dataset (dupliquer les images sans objet pour avoir 50/50)

.. step::
    3) Entraîner le CNN custom AVEC augmentation :

    .. code-block:: python

       # TODO: Créer AugmentedYOLODataset avec augment=True pour train
       # TODO: Entraîner avec les mêmes hyperparamètres que dans l'exercice 1 (partie B)
       # TODO: Comparer train loss et val loss (vérifier overfitting réduit)
       # TODO: Sauvegarder dans 'best_model_with_aug.pth'
    
.. step::
    4) Entraîner YOLO AVEC augmentation

.. step::
    5) Comparer les 4 modèles finaux :

    .. code-block:: python

       # TODO: Charger les 4 modèles
       # TODO: Évaluer sur le MÊME test set
       # TODO: Afficher tableau comparatif complet
       # +---------------------+---------------+---------------+-------------+-------------+
       # | Métrique            | CNN No Aug    | CNN + Aug     | YOLO11      |YOLO11 + Aug |
       # +---------------------+---------------+---------------+-------------+-------------+
       # | Accuracy            | XX.XX%        | XX.XX%        | XX.XX%      |XX.XX%       |
       # | Precision           | XX.XX%        | XX.XX%        | XX.XX%      |XX.XX%       |
       # | Recall              | XX.XX%        | XX.XX%        | XX.XX%      |XX.XX%       |
       # | Mean IoU            | 0.XXXX        | 0.XXXX        | 0.XXXX      |0.XXXX       |
       # | Gap Train/Val (%)   | XX            | XX            | XX          | XX          |
       # +---------------------+---------------+---------------+-------------+-------------+


**Questions :**

.. step::
    6) L'augmentation améliore-t-elle les performances du CNN custom ?

.. step::
    7) Le gap train/val est-il réduit avec l'augmentation ?

.. step::
    8) Pourquoi l'augmentation seule ne suffit pas à rattraper YOLO ?

.. step::
    9) Quelle est l'importance du rééquilibrage (50/50) ?


**Astuce :**

.. spoiler::
    .. discoverList::
        1. L'augmentation réduit souvent l'overfitting (gap train/val plus petit), mais pas toujours : avec très peu d'images, elle peut aussi rendre l'apprentissage plus difficile pour un petit CNN
        2. Le rééquilibrage évite le biais : sans lui, le modèle détecte trop souvent
        3. YOLO reste supérieur car il combine pré-entraînement + architecture optimisée
        4. Flip horizontal : x_center → 1 - x_center, y_center inchangé, w et h inchangés
        5. Augmentation adaptée détection : éviter rotations fortes (change orientation objet)
        6. Color jitter OK car n'affecte pas les coordonnées spatiales
        7. La validation et le test ne doivent pas être augmentés : créez un dataset avec ``augment=True`` pour l'entraînement et un avec ``augment=False``, puis appliquez ``random_split`` aux deux avec la même graine (comme dans l'exercice 3 du TP 5), ou réutilisez les indices du découpage de la partie B (``train_dataset.indices``) avec ``torch.utils.data.Subset``. Rééquilibrez **après** le découpage, et seulement le train

**Résultat attendu :**

Exemple obtenu sur notre dataset (même test set de 21 images pour les 4 modèles, résultats moyennés sur une dizaine d'entraînements pour le CNN) :

- CNN sans augmentation : résultats très variables d'un entraînement à l'autre (accuracy moyenne ≈ 70 %, IoU moyen ≈ 0.55)
- CNN avec augmentation : **moins bon** dans presque tous nos essais (accuracy moyenne ≈ 50 %, IoU moyen ≈ 0.35) : avec seulement 64 images de cube, ce petit CNN n'arrive pas à profiter de l'augmentation
- YOLO11, avec ou sans augmentation : aucune erreur sur le test set, IoU moyen ≈ 0.93, gap train/val nul
- Conclusion : l'augmentation n'est pas magique ; sur un petit dataset, c'est le pré-entraînement de YOLO qui fait la différence


.. slide::

🌶️ Exercice supplémentaire 3 : Introduction au tracking basique (détection frame par frame)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cet exercice est une introduction au tracking d'objets en détectant frame par frame dans une vidéo.

**Objectif :**  
Comprendre les bases de la détection vidéo et mesurer les performances en temps réel.

.. note::
    **Qu'est-ce que le tracking ?**
    
    Le tracking (suivi d'objets) consiste à détecter et identifier le **même objet** à travers les frames d'une vidéo. Dans cet exercice simplifié, vous allez uniquement **détecter** l'objet sur chaque frame indépendamment, **sans associer d'identité** entre les frames.
    
    **Limitation de cette approche :**
    
    - Si l'objet sort puis revient, vous ne savez pas que c'est le même
    - Impossible de compter combien d'objets distincts sont apparus
    - Pas de trajectoire ou d'historique de mouvement
    
    Cette approche est utile pour :
    
    - Détecter la présence/absence d'un objet en temps réel
    - Mesurer les performances (FPS) de votre système
    - Préparer le terrain pour un vrai tracking avec identités

**Matériel nécessaire :**

- Vidéo MP4 de test (30-60 secondes)
- Modèle YOLO entraîné (exercice 1 ou 2)

**Consignes** :

.. step::
    1) Créer ou utiliser une vidéo de test (30-60 secondes) :
    
    **Scénario recommandé** :

    - 0-10s : Aucun objet visible
    - 10-20s : Objet entre dans le champ, se déplace
    - 20-30s : Objet sort du champ
    - 30-40s : Objet réapparaît

.. step::
    2) Implémenter la détection frame par frame sur vidéo :

    .. code-block:: python

       import cv2
       from ultralytics import YOLO
       import time
       
       def detect_on_video(model_path, video_path, output_path, conf_threshold=0.5):
           """Détecte les objets sur chaque frame et sauvegarde la vidéo."""
           # TODO: Charger le modèle YOLO
           # TODO: Ouvrir la vidéo (fps, dimensions, nb_frames)
           # TODO: Créer VideoWriter pour la sortie
           # TODO: Boucle sur les frames:
           #   - Lire frame
           #   - Mesurer temps de traitement
           #   - Prédiction YOLO
           #   - Dessiner détections + info (frame, FPS)
           #   - Sauvegarder frame
           # TODO: Libérer ressources et afficher stats
           pass

.. step::
    3) Analyser les statistiques de détection :

    .. code-block:: python

       import matplotlib.pyplot as plt
       
       def plot_detection_stats(stats):
           """Visualise les statistiques."""
           # TODO: Graphique 1: Détections par frame (ligne)
           # TODO: Graphique 2: Distribution nb objets (histogramme)
           # TODO: Graphique 3: Temps de traitement (ligne + moyenne)
           # TODO: Sauvegarder la figure
           pass


**Questions :**

.. step::
    4) Quel est le FPS moyen de votre système ? Est-ce suffisant pour du temps réel (>30 fps) ?

.. step::
    5) Pourquoi le temps de traitement varie-t-il d'une frame à l'autre ?

.. step::
    6) Comment pourriez-vous améliorer la vitesse si elle est trop lente ?

.. step::
    7) Quelles sont les limitations de cette approche sans identité d'objets ?


**Astuce :**
.. spoiler::
    .. discoverList::
        1. FPS réel = 1 / temps_traitement_moyen
        2. Temps réel nécessite généralement >25-30 FPS pour fluidité
        3. Variation temps : complexité variable de l'image, nombre d'objets
        4. Pour accélérer l'inférence : réduire le nombre de pixels, par exemple 640→320 (paramètre ``imgsz`` de ``model.predict``)
        5. VideoWriter : même fps que la vidéo source pour synchronisation
        6. Utiliser ``model.predict(frame, verbose=False)`` pour éviter logs
        7. Pour mesurer FPS : ``fps = 1.0 / (time.time() - start_time)``


**Pour aller plus loin : Tracking avec identités**

.. note::
    **Limitations du tracking frame par frame :**
    
    Cette approche simple ne permet pas de :
    
    - Savoir si c'est le **même objet** d'une frame à l'autre
    - **Compter** combien d'objets distincts sont apparus dans la vidéo
    - Suivre les **trajectoires** et analyser les mouvements
    - Gérer les **occlusions** temporaires (objet caché puis réapparaît)
    
    **Solution : Tracking avec identités (Object ID)**
    
    Pour un vrai système de tracking, il faut :
    
    1. **Assigner un ID unique** à chaque objet détecté
    2. **Associer les détections** entre frames successives :
       - Comparer les positions (distance euclidienne)
       - Si deux détections sont proches → même objet
       - Si détection loin de tous les objets connus → nouvel objet
    3. **Gérer les disparitions** :
       - Garder l'ID en mémoire pendant N frames
       - Si l'objet réapparaît → retrouver son ID
       - Sinon → supprimer l'ID après N frames

    
    **Utilisation de YOLO pour le tracking avancé : https://docs.ultralytics.com/modes/track/**
    
    .. code-block:: python
    
        from ultralytics import YOLO
        model = YOLO('yolo11n.pt')
        results = model.track(source='video.mp4', persist=True)  # Track avec IDs !
        