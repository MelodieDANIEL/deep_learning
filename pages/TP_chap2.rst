🏋️ Travaux Pratiques 2
=========================
.. slide::
Sur cette page se trouvent des exercices de TP sur le Chapitre 2. Ils sont classés par niveau de difficulté :
.. discoverList::
    * Facile : 🍀
    * Moyen : ⚖️
    * Difficile : 🌶️




.. slide::
🍀 Exercice 1 : Approximations d’une fonction non linéaire
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez implémenter une boucle d'entraînement simple pour ajuster les paramètres d’un modèle polynômial comme dans le chapitre 1, puis comparer les résultats avec ceux d'un modèle MLP.

On vous donne les données suivantes :

.. code-block:: python

    torch.manual_seed(0)

    X = torch.linspace(-3, 3, 100).unsqueeze(1)
    y_true = torch.sin(X) + 0.1 * torch.randn(X.size())  # fonction sinusoïdale bruitée

**Objectif :** Comparer deux modèles pour approximer la fonction :

1. Polynôme cubique : $$y = f(x) = a x^3 + b x^2 + c x + d$$, où $$a, b, c, d$$ sont des paramètres appris automatiquement en minimisant l’erreur entre les prédictions et les données réelles comme dans le chapitre 1.

2. MLP simple :  

    - Implémenté sous forme de classe ``nn.Module``  
    - 2 couches cachées de 10 neurones chacune avec `ReLU`` pour l'activation
    - Entrée : 1 feature, sortie : 1 prédiction

**Consigne :** Écrire un programme qui :

1) Ajuste les paramètres du polynôme cubique aux données en utilisant PyTorch.  
2) Affiche les paramètres appris $$a, b, c, d$$.  
3) Implémente ensuite un MLP et entraîne-le sur les mêmes données pendant 5000 epochs avec un learning rate de 0.01.  
4) Compare visuellement les deux modèles avec les données réelles sur un même graphique. 
5) Que remarquez-vous sur les performances des deux modèles ?
6) Que se passe-t-il si vous augmentez le nombre du polynôme ?


**Astuce :**
.. spoiler::
    .. discoverList::
        1. Initialiser les paramètres du polynôme avec ``torch.randn(1, requires_grad=True)``.  
        2. Utiliser ``nn.MSELoss()`` comme fonction de perte pour les deux modèles.  
        3. Pour le MLP, créer une classe héritant de ``nn.Module`` et définir ``forward``.  
        4. Utiliser ``optimizer.zero_grad()``, ``loss.backward()``, ``optimizer.step()`` à chaque itération.  
        5. On voit que le MLP parvient à mieux s'adapter aux données, car il peut capturer des relations non linéaires plus complexes.

**Résultat attendu :** Vous devez obtenir un graphique similaire à celui ci-dessous où :  

- les points bleus correspondent aux données réelles (``y_true``)  
- la courbe rouge correspond au polynôme cubique  
- la courbe verte correspond au MLP  

.. image:: images/chap2_exo_1_resultat.png
    :alt: Résultat Exercice 1
    :align: center


.. slide::
⚖️ Exercice 2 : Comparaison de l'entraînement d'un MLP sur données brutes et standardisées
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cet exercice, vous allez entraîner un MLP simple sur un jeu de données synthétiques avec deux features ayant des échelles différentes. Vous comparerez les performances lorsque les données sont brutes ou standardisées.

On vous donne les données suivantes :

.. code-block:: python

    # Données synthétiques
    N = 500
    X1 = torch.linspace(0, 1, N).unsqueeze(1)      # petite amplitude
    X2 = torch.linspace(0, 100, N).unsqueeze(1)    # grande amplitude
    X = torch.cat([X1, X2], dim=1)
    y = 3*X1 + 0.05*X2**2 + torch.randn(N,1) * 0.5


**Objectif :**  
Comprendre l’importance de la standardisation des données pour l’entraînement d’un réseau de neurones et observer l’évolution de la loss.


**Consigne :** Écrire un programme qui :  

1) Définit une classe MLP simple sans couches cachées avec : 

   - une couche linéaire d’entrée (2 features) vers 20 neurones  
   - une fonction d’activation ``ReLU``  
   - une couche de sortie avec 1 prédiction 

2) Crée deux modèles : un pour les données brutes, un pour les données standardisées.  

3) Entraîne les deux modèles avec Adam et une fonction de perte MSE pendant 1000 epochs avec un learning rate de 0.01.

4) Stocke et trace l’évolution de la loss pour les deux modèles.  

5) Trace les prédictions finales des deux modèles sur le même graphique que les données réelles.  

6) Comparez les performances des deux modèles et notez lequel converge plus vite et donne de meilleures prédictions.

7) A quelle epoch peut-on considérer que le modèle sur données standardisées a convergé et comment on peut faire pour le déterminer ?


**Astuce :**
.. spoiler::
    .. discoverList::
        1. N’oubliez pas d initialiser les poids du modèle avec ``torch.randn()`` pour un démarrage aléatoire et de  mettre ``optimizer.zero_grad()`` avant ``loss.backward()``.  
        2. Pour standardiser, utilisez ``(X - X_mean)/X_std``.  
        3. Pour visualiser la loss : stockez ``loss.item()`` à chaque epoch et utilisez ``matplotlib.pyplot.plot()``.  
        4. Pour visualiser les prédictions, utilisez un scatter plot avec les données réelles et les prédictions des deux modèles.
        5. Pour savoir quand stopper l'entraînement, vous pouvez faire du Early Stopping.
        6. Pour que l’early stopping fonctionne correctement avec ce type de données, il est conseillé de :

            - Mettre le paramètre ``patience`` à 20.  
            - Comparer la perte actuelle avec la meilleure perte précédente en utilisant un seuil de tolérance. Par exemple, arrondir la perte à 5 pour considérer une amélioration significative (``if loss.item() < best_loss - 5``)

**Résultat attendu :**  
Le graphique montre les prédictions du MLP sur les données brutes (rouge) et standardisées (bleu) par rapport aux données réelles (noir).  Vous devez obtenir un résultat similaire à celui-ci avant de réduire le nombre d'epochs :

.. image:: images/chap2_exo_2_resultat.png
    :alt: Comparaison MLP brutes vs standardisées
    :align: center


.. slide::
🏋️ Exercices supplémentaires 2
===============================
Dans cette section, il y a des exercices supplémentaires pour vous entraîner. Ils suivent le même classement de difficulté que précédemment.

.. slide::
⚖️ Exercice supplémentaire 1 : Approximation d'une fonction 2D avec un MLP 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cet exercise propose l'entraînement d'un MLP avec des données en 2D.

**Objectif :** Entraîner un MLP pour approximer la fonction suivante :

.. math::

    y = \sin(X_1) + \cos(X_2)

où $(X_1, X_2) \in [-2,2]^2$, et visualiser la prédiction du modèle par rapport à la fonction réelle.

**Consigne :**  

1. Générer ``N = 800`` points aléatoires $$(X_1, X_2)$$ dans $$[-2,2]$$ et calculer $$y$$ en suivant la fonction.

2. Standardiser les entrées pour le MLP.

3. Créer un MLP simple :

   - Entrée : 2 features  
   - 2 couches cachées de 64 neurones avec activation ``Tanh``  
   - Sortie : 1 prédiction

4. Entraîner le modèle avec Adam et MSE loss pendant 1000 epochs.

5. Ajouter early stopping avec ``patience = 20`` et ``tolerance = 0.1``.

6. Préparer une grille 2D pour visualiser la fonction réelle et la prédiction du modèle.

7. Afficher sur une seule figure 3D* :

   - Surface réelle en vert transparent  
   - Surface prédite par le MLP en orange semi-transparent  
   - Ajouter une légende pour distinguer les surfaces

8. Tracer l'évolution de la loss pendant l'entraînement pour vérifier la convergence.

9. Refaire un test avec des données bruitées (ajouter un bruit gaussien de moyenne 0 et écart-type 0.6 à y) et observer l'impact sur la prédiction du MLP.

**Questions :**  

- Que remarquez-vous sur la capacité du MLP à approximer la fonction sous-jacente malgré le bruit ?  
- Que se passe-t-il si vous augmentez ou diminuez le niveau de bruit ?  
- Comment l’early stopping impacte-t-il l’apprentissage ?

**Astuce :**
.. spoiler::
    .. discoverList::
        1. Pour l'early stopping, stocker la meilleure loss et un compteur d'epochs sans amélioration.  
        2. Pour la visualisation, utiliser ``ax.plot_surface`` pour les surfaces et ``Patch`` pour la légende.  
        3. La standardisation permet au MLP de mieux converger.  
        4. Vérifier la loss finale pour s'assurer que le modèle a appris correctement la fonction.
        5. Pour générer le bruit, utilisez ``0.6 * torch.randn_like(y_clean)``.

**Astuce avancée :**        
.. spoiler::
    .. discoverList:: 
        **Voici le code pour la visualisation 3D avec Matplotlib :**
        .. code-block:: python
            x1g, x2g = torch.meshgrid(
            torch.linspace(-2, 2, 80),
            torch.linspace(-2, 2, 80),
            indexing="ij"
            )

            Xg = torch.cat([x1g.reshape(-1,1), x2g.reshape(-1,1)], dim=1)
            Xg_std = (Xg - X_mean) / X_std

            with torch.no_grad():
                y_true_grid = (torch.sin(x1g) + torch.cos(x2g))
                y_pred_grid = model(Xg_std).reshape(80, 80)
                y_pred_train = model(X_stdized)

            fig = plt.figure(figsize=(9,7))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("MLP 2D avec Early Stopping")
            ax.set_xlabel("X1"); ax.set_ylabel("X2"); ax.set_zlabel("y")
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(-2, 2)
            try:
                ax.set_box_aspect((1,1,1))
            except Exception:
                pass
            ax.view_init(elev=25, azim=35)

            ax.plot_surface(x1g.numpy(), x2g.numpy(), y_true_grid.numpy(),
                            cmap="Greens", alpha=0.45, linewidth=0)
            ax.plot_surface(x1g.numpy(), x2g.numpy(), y_pred_grid.numpy(),
                            cmap="Oranges", alpha=0.70, linewidth=0)
            legend_elements = [
                Patch(facecolor="tab:green", alpha=0.45, label="Surface vraie"),
                Patch(facecolor="tab:orange", alpha=0.70, label="Surface MLP")
            ]
            ax.legend(handles=legend_elements, loc="upper left")


            plt.tight_layout()
            plt.show()


**Résultat attendu :**

- Voici un exemple de la figure 3D attendues pour les points 1 à 8 de la consigne avec la surface réelle (verte) et la surface prédite par le MLP (orange) :

.. image:: images/chap2_exo_sup_1_resultat.png
    :alt: Résultat attendu MLP 2D
    :align: center

- Voici un exemple de la figure 3D attendues pour le point 9 de la consigne avec la surface réelle (verte) et la surface prédite par le MLP (orange) :

.. image:: images/chap2_exo_sup_1_suite_resultat.png
    :alt: Résultat attendu MLP 2D
    :align: center