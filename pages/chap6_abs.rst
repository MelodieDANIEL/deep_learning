.. slide::
Résumé des concepts clés du chapitre 6
================
Ce cours est interactif vous devez faire les étapes mais adaptés pour détecter des objets dans des images en utilisant des CNNs avec PyTorch.
1) prendre une vidéo de l'objet à détecter (fourni par l'enseignant)
2) extraire des frames carrées de la vidéo
3) annoter les objets dans les frames avec Label Studio
4) créer un dataset pour l'entraînement
5) entraîner un modèle de détection d'objets

.. slide::

📖 1. Comparaison des approches
-------------------------

+---------------------------+--------------------------------+------------------------------------+----------------------------------+
| **Approche**              | **Cas d'usage**                | **Avantages**                      | **Inconvénients**                |
+===========================+================================+====================================+==================================+
| **SimpleBBoxRegressor**   | 1 objet par image,             | - Très simple                      | - Limité à 1 objet               |
| (§7)                      | cas simple                     | - Rapide à entraîner               | - Pas de classification          |
|                           |                                | - Peu de paramètres                |                                  |
+---------------------------+--------------------------------+------------------------------------+----------------------------------+
| **YOLOv11**               | Plusieurs objets,              | - Rapide (30-80 FPS)               | - Besoin de GPU                  |
| (§8)                      | temps réel                     | - Très précis                      | - Dataset plus complexe          |
|                           |                                | - Facile à utiliser                |                                  |
+---------------------------+--------------------------------+------------------------------------+----------------------------------+


.. note::

   💡 **Recommandations**
   
   - **Prototypage/simple** : SimpleBBoxRegressor (§7)
   - **Production/temps réel** : YOLOv11 (§8)
   - **Recherche/précision** : Faster R-CNN