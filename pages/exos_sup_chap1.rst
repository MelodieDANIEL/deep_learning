.. slide::
🏋️ Exercices supplémentaires
===============================
Dans cette section, il y a des exercices supplémentaires pour vous entraîner. Ils suivent le même classement de difficulté que précédemment.


.. slide::
🍀 Exercice supplémentaire 1 : Gradient d’une fonction polynomiale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Considérons la fonction suivante $$f(a) = 3a^3 - 2a^2 + a$$ avec $$a = 2.0$$.

**Consigne :** Utiliser les deux approches suivantes pour calculer le gradient de cette fonction par rapport à $$a$$ :

1) Calculez à la main la dérivée de $$f$$ par rapport à $$a$$. Puis évaluez ce gradient pour $$a = 2.0$$.

2) Faites l'implémentation de la même fonction avec PyTorch, calculez et évaluez son gradient.

3) Comparez le résultat obtenu par PyTorch avec le calcul manuel.

**Astuce :**
.. spoiler::
    .. discoverList::
        La dérivée de $$f(a)$$ par rapport à $$a$$ est égale à $$9a² - 4a + 1$$

**Résultat attendu :**  
Le gradient est égal à 29 dans les deux cas. 


.. slide::
🍀 Exercice supplémentaire 2 : Gradient de deux variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Considérons la fonction suivante $$f(a, b) = a \cdot b + a^2$$ avec $$a = 2.0$$ et $$b = 3.0$$.


**Consigne :** Utiliser les deux approches suivantes pour calculer les dérivées partielles de cette fonction par rapport à $$a$$ et $$b$$ :

1) Calculez à la main la dérivée partielle de $$f$$ par rapport à $$a$$ et par rapport à $$b$$. Puis évaluez ces dérivées pour $$a = 2.0$$ et $$b = 3.0$$.

2) Faites l'implémentation de la même fonction avec PyTorch, calculez et évaluez le gradient de cette fonction.

3) Comparez le résultat obtenu par PyTorch avec le calcul manuel.


**Astuce :**  
.. spoiler::
    .. discoverList::
        - La dérivée de $$f$$ par rapport à $$a$$ est $$∂f/∂a = b + 2a$$ et par rapport à $$b$$ est $$∂f/∂b = a$$.

**Résultat attendu :**  
Les dérivées partielles sont, dans les deux cas, égales à : $$∂f/∂a = 7$$ et $$∂f/∂b = 2$$.


⚖️ Exercice supplémentaire 3 : Comparaison des fonctions de perte MSE et MAE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On vous donne les données suivantes :

.. code-block:: python

    # Données bruitées suivantes
    torch.manual_seed(0)
    x = torch.linspace(-3, 3, 100)
    y_true = 2 * x**2 + 3 * x + 1 + 0.5 * torch.randn(x.size())  # avec bruit
    y_true[::10] += 15  # tous les 10 points, on ajoute une grosse valeur
    

**Objectif :** Trouver une courbe 2D de la forme :

.. math::

    y = f(x) =a x^2 + b x + c

où : $$a$$, $$b$$ et $$c$$ sont des paramètres appris automatiquement en minimisant l'erreur entre les prédictions du modèle et les données réelles.


**Consignes** : Implémenter une boucle d'entraînement pour ajuster les paramètres d'une courbe d'ordre 2 aux données fournies en utilisant une fonction de perte MAE et MSE.

1) Réutilisez la boucle d'entraînement de l’exercice 3 qui s'arrête au bout de 1000 itérations et qui utilise un learning rate de 0.01.  

2) Testez la fonction de perte MSE et MAE.

3) Pour chaque fonction de perte, affichez les paramètres appris $$a$$, $$b$$ et $$c$$.

4) Pour chaque fonction de perte, tracez les données réelles et les données prédites et comparez visuellement les résultats.

6) Quelle différence observez-vous dans la convergence et les paramètres appris ?

7) Pourquoi la MSE et la MAE ne donnent-elles pas exactement le même résultat ?

8) Dans quel cas préfèreriez-vous utiliser MSE ? Dans quel cas préfèreriez-vous utiliser MAE ?

**Astuce :**
.. spoiler::
    .. discoverList::
        - La MSE pénalise davantage les grandes erreurs.  
        - La MAE est plus robuste aux valeurs aberrantes (outliers).


**Résultat attendu :**
Vous devez obtenir des valeurs pour les paramètres proches de :

    - MSE -> a = 2.002, b = 2.866, c = 2.464
    - MAE -> a = 1.984, b = 2.997, c = 1.132

et un graphique similaire à celui ci-dessous :

.. image:: images/chap1_exo_sup_3_resultat.png
    :alt: droite ajustée aux points
    :align: center


.. slide::
🌶️ Exercice supplémentaire 4 : Visualiser une surface de perte en 3D & descente de gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On considère la fonction suivante :  

.. math::

    f(a, b) = a^2 + b^2


**Objectif :** Comprendre la descente de gradient en visualisant la surface de la fonction et la trajectoire de convergence.  


**Consignes :**

1) Calculez à la main le gradient de $$f(a,b)$$ et ses dérivées partielles .

2) Implémentez une boucle de descente de gradient avec un point de départ choisi (par exemple $$a=2.5$$, $$b=-2.0$$) et un learning rate de 0.1.

3) Stockez les points de la trajectoire au cours des itérations.  

4) Tracez la surface 3D de $$f(a, b)$$ avec Matplotlib.  

5) Ajoutez sur la surface des flèches représentant les étapes de la descente de gradient.  

6) Expliquez ce que représente la trajectoire observée et pourquoi elle converge vers $$(a, b) = (0,0)$$.

7) Testez plusieurs learning rate (ex: 0.02, 0.1, 0.5, 2.0) pour observer convergence lente, rapide, ou divergence.


**Astuce :**
.. spoiler::
    .. discoverList::
        - Utilisez ``ax.plot_surface`` pour la surface 3D.  
        - Utilisez ``ax.quiver`` pour tracer les flèches en 3D.  
        - Le minimum de la fonction est atteint en $$(0,0,0)$$. 

**Astuce avancée :**        
.. spoiler::
    .. discoverList:: 
        **Squelette de code :**
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour la 3D

            # 1) Créer une grille pour la surface
            A = np.linspace(-3, 3, 100)
            B = np.linspace(-3, 3, 100)
            AA, BB = np.meshgrid(A, B)

            # À compléter : calculer Z = f(a,b) = a^2 + b^2
            Z = ...

            # 2) Préparer une figure 3D
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(AA, BB, Z, alpha=0.5)

            # 3) Descente de gradient depuis un point de départ
            lr = 0.1   # learning rate
            a, b = 2.5, -2.0
            n_iter = 15

            traj = [(a, b, a**2 + b**2)]

            for _ in range(n_iter):
                # À compléter : calculer le gradient ga, gb
                ga, gb = ...
                
                # À compléter : mettre à jour a et b avec le learning rate
                a, b = ...
                
                traj.append((a, b, a**2 + b**2))

            # 4) Représenter la trajectoire (quiver pour flèches)
            for (a1, b1, z1), (a2, b2, z2) in zip(traj[:-1], traj[1:]):
                # À compléter : dessiner une flèche de (a1,b1,z1) vers (a2,b2,z2)
                ax.quiver(...)

            ax.set_xlabel('a')
            ax.set_ylabel('b')
            ax.set_zlabel('f(a,b)')
            ax.set_title('Surface de perte et descente de gradient')
            plt.tight_layout()
            plt.show()


**Résultat attendu :**  
Un graphique 3D montrant la surface convexe de la fonction et la descente du point de départ vers le minimum global en $$(0,0)$$ avec ``lr=0.1`` :  

.. image:: images/chap1_exo_sup_4_resultat.png
    :alt: droite ajustée aux points
    :align: center








