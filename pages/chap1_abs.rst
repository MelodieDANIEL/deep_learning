<<<<<<< HEAD
.. slide::
Résumé des concepts clés du chapitre 1
======================================

.. slide::
📖 1. Objectif de l’apprentissage
----------------------

- Le but de l'apprentissage automatique est **d’apprendre à approximer une fonction** à partir de données. On cherche à trouver des **paramètres** qui permettent au modèle de prédire correctement les sorties à partir des entrées.

- Exemple simple : fonction linéaire $$f(x) = a x$$. On veut trouver $$a$$ pour que $$f(x)$$ prédise les sorties observées.

- Types de problèmes :
  - **Régression** : prédire une valeur continue (ex. prix d’une maison, température).  
  - **Classification** : prédire une catégorie (ex. spam/non-spam, chat/chien/oiseau).

**Pourquoi on a besoin d’apprentissage ?**  
Les données seules ne donnent pas directement le modèle. On calcule **l’erreur de prédiction** et on ajuste les paramètres pour la réduire. Pour représenter ces données et paramètres dans PyTorch, on utilise des **tenseurs**, des tableaux multidimensionnels.

.. slide::
📖 2. Tenseurs
----------------------

- Tableaux multidimensionnels représentant **données et paramètres**.
- Exemples : ``zeros``, ``ones``, ``arange``, ``linspace``, aléatoire.
- Manipulation : changer la forme, ajouter/supprimer dimensions, permuter, concaténer.
- Base de tous les calculs dans un modèle.


Pour savoir comment ajuster les paramètres du modèle, on calcule **la perte**, puis ses **gradients**.

.. slide::
📖 3. Fonction de perte
----------------------

- Mesure combien le modèle se trompe : **plus la perte est grande → plus l’erreur est importante**.
- Guide l’apprentissage : le modèle ajuste ses paramètres pour **minimiser la perte**.
- Exemples : **MSE** pour la régression, **Cross-Entropy** pour la classification.

Pour minimiser la perte, on calcule le **gradient de la perte par rapport aux paramètres**, grâce à la rétropropagation.

.. slide::
📖 4. Rétropropagation et gradients
----------------------

- **Rétropropagation** : parcours du graphe computationnel du modèle pour calculer les gradients.  
- Exemple simple : $$f(x) = x^2$$  
  - $$x > 0$$ → gradient positif → diminuer x pour réduire la perte.  
  - $$x < 0$$ → gradient négatif → augmenter x.  
  - Au minimum : gradient = 0.

- Ces gradients indiquent **la direction dans laquelle ajuster les paramètres** pour réduire l’erreur.

.. slide::
📖 5. Autograd
----------------------

- **Autograd** : module de PyTorch qui gère la différentiation automatique.
- Suit les opérations sur les tenseurs et construit un graphe computationnel.
- Permet de calculer les gradients de manière efficace avec ``.backward()``.

.. slide::
📖 6. Optimisation
----------------------

- On utilise les gradients pour **modifier les paramètres dans la direction qui réduit la perte**.
- Algorithme classique : **descente de gradient**, ou adaptatif comme **Adam**.


.. slide::
📖 7. Boucle d’entraînement
----------------------

1) Initialisation : Initialiser les paramètres du modèle.

2) Prédiction : Calculer la sortie du modèle pour les données d’entrée.

3) Perte : Calculer la perte en comparant la sortie estimée avec la valeur attendue.

4) Gradients : Calculer les gradients de la perte via ``.backward()``.

5) Mise à Jour : Mettre à jour les paramètres pour réduire la perte.

6) Répéter les étapes 2 à 5 jusqu’à la convergence du modèle.

.. slide::
📖 8. Utilisation de la carte GPU avec les tenseurs
---------------------------------------------------

- En utilisant un GPU, on peut accélérer considérablement les calculs, surtout pour les grands modèles et ensembles de données.

- La commande ``set_default_device(device)`` permet de définir le GPU comme appareil par défaut pour les tenseurs, si ``device = "cuda"``.

- Pour transformer un Numpy array situé sur le CPU en tenseur sur le GPU, utilisez ``torch.from_numpy(nom_array).to(device)``.

- Si vous souhaitez utiliser Matplotlib pour afficher les données contenues dans un tenseur sur le GPU, vous devez d’abord transférer ce tenseur vers le CPU et le convertir en Numpy array avec ``tensor.cpu().numpy()`` avant de le passer à Matplotlib.

- Dans le cas où le tenseur a été créé avec ``requires_grad=True``, vous devez utiliser ``tensor.detach().cpu().numpy()`` pour éviter les erreurs liées au suivi des gradients.

👉 Retenez : **Numpy = CPU, Torch = CPU/GPU, Matplotlib = Numpy**.  


