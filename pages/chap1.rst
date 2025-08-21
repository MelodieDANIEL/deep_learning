
.. slide::

Chapitre 1 - Introduction à PyTorch et Optimisation de Modèles
================

🎯 Objectifs du Chapitre
----------------------


.. important::

   À la fin de ce chapitre, vous saurez : 

   - Créer et manipuler des tenseurs PyTorch sur CPU et GPU.
   - Calculer automatiquement les gradients à l’aide de ``autograd``.
   - Définir une fonction de perte.
   - Utiliser un optimiseur pour ajuster les paramètres d’un modèle.
   - Implémenter une boucle d'entraînement simple.

.. slide::

📖 1. Qu'est-ce que PyTorch ? 
----------------------
PyTorch est une bibliothèque Python de machine learning open-source développée par Facebook (FAIR). Elle est conçue pour faciliter la création et l'entraînement de modèles, en particulier dans le domaine du deep learning. 

Elle repose principalement sur deux éléments :

A) Les *tenseurs*, des structures de données similaires aux tableaux NumPy (``ndarray``), mais avec des fonctionnalités supplémentaires pour :
    
    - le calcul différentiel automatique,
    - l'accélération GPU,
    - l’entraînement de réseaux de neurones.

B) Le module ``autograd`` permet de calculer automatiquement les gradients nécessaires à l'entraînement des modèles, en suivant toutes les opérations effectuées sur les tenseurs.

.. slide::

D'autres bibliothèques Python similaires existent, comme :

- TensorFlow : développé par Google, très utilisé pour des déploiements à grande échelle.
- Keras : interface haut niveau de TensorFlow, plus simple mais moins flexible.
- JAX : plus récent, optimisé pour la recherche et les calculs scientifiques à haute performance.

.. slide::

Dans le cadre de ce cours, nous utiliserons PyTorch car :

- elle est largement adoptée par la communauté de la recherche en deep learning,
- elle est plus lisible et plus facile à déboguer que TensorFlow et JAX,
- elle offre plus de possibilités que Keras,
- elle est bien documentée et est l'une des bibliothèques les plus utilisées en science des données (Data Science en anglais) et en apprentissage machine (Machine Learning en anglais).

.. slide::

📖 2. Qu'est-ce qu'un tenseur ?
----------------------

Les **tenseurs** sont la structure de base de PyTorch. Ce sont des tableaux multidimensionnels similaires aux ``ndarray`` de NumPy, mais avec des fonctionnalités supplémentaires pour le GPU et le calcul automatique des gradients. Un tenseur est une structure de données qui généralise les matrices à un nombre quelconque de dimensions:

- Un scalaire est un tenseur 0D.  
- Un vecteur est un tenseur 1D.  
- Une matrice est un tenseur 2D.  
- On peut avoir des tenseurs 3D, 4D, etc.   

Les tenseurs à haute dimensions sont très utilisés en deep learning (par exemple pour les images ou les vidéos). Nous allons voir comment créer et manipuler des tenseurs dans PyTorch. Vous pouvez copier-coller les exemples de code ci-dessous dans un notebook Jupyter pour les tester et voir les affichages. Pour utiliserles fonctions de PyTorch, il faut d'abord l'importer :
.. code-block:: python

   import torch

.. slide::
    
📖 3. Création de tenseurs
----------------------

Il existe plusieurs manières de créer un tenseur en PyTorch.

3.1. À partir de données Python (listes ou tuples)
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Depuis une liste
   a = torch.tensor([1, 2, 3])
   print(a)

   # Depuis une liste de listes (matrice)
   b = torch.tensor([[1, 2, 3], [4, 5, 6]])
   print(b)

   # On peut aussi spécifier le type de données
   c = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
   print(c, c.dtype)

.. slide::
3.2. Avec des fonctions de construction
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   # Tenseur rempli de zéros
   z = torch.zeros(2, 3)
   print(z)

   # Tenseur rempli de uns
   o = torch.ones(2, 3)
   print(o)

   # Tenseur vide (valeurs non initialisées)
   e = torch.empty(2, 3)
   print(e)

   # Identité (matrice diagonale)
   eye = torch.eye(3)
   print(eye)

.. slide::
3.3. Avec des suites régulières
~~~~~~~~~~~~~~~~~~~
PyTorch permet de générer facilement des suites de nombres avec des pas réguliers. Deux fonctions sont particulièrement utiles :

1. **torch.arange(debut, fin, pas)**  

   - Crée une suite en commençant à ``debut``  
   - S’arrête *avant* ``fin`` (attention, la borne supérieure est exclue !)  
   - Utilise le ``pas`` indiqué  

.. code-block:: python

   # De 0 à 8 inclus, avec un pas de 2
   r = torch.arange(0, 10, 2)
   print("torch.arange(0, 10, 2) :", r)

   # De 5 à 20 exclu, avec un pas de 3
   r2 = torch.arange(5, 20, 3)
   print("torch.arange(5, 20, 3) :", r2)

   # ⚠️ Remarque : la borne supérieure (ici 10 ou 20) n'est jamais incluse

.. slide::
2. **torch.linspace(debut, fin, steps)**  

   - Crée une suite de ``steps`` valeurs régulièrement espacées  
   - Inclut **à la fois** ``debut`` et ``fin``  

.. code-block:: python

   # 5 valeurs entre 0 et 1 inclus
   l = torch.linspace(0, 1, steps=5)
   print("torch.linspace(0, 1, steps=5) :", l)

   # 4 valeurs entre -1 et 1 inclus
   l2 = torch.linspace(-1, 1, steps=4)
   print("torch.linspace(-1, 1, steps=4) :", l2)

**Résumé des différences**

- ``arange`` → on fixe le **pas** entre les valeurs, la fin est exclue.  
- ``linspace`` → on fixe le **nombre de valeurs**, la fin est incluse.  

Exemple comparatif :

.. code-block:: python

   print(torch.arange(0, 1, 0.25))   # [0.00, 0.25, 0.50, 0.75]
   print(torch.linspace(0, 1, 5))    # [0.00, 0.25, 0.50, 0.75, 1.00]


.. slide::
3.4. Avec des nombres aléatoires
~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   # Attention dans les exemples suivants, les crochets [] veulent dire que la valeur de la borne est incluse, contrairement à aux parenthèses () qui signifient que la borne est exclue.
   # Uniforme entre [0, 1)
   u = torch.rand(2, 2)
   print("Uniforme [0,1) :\n", u)

   # Distribution normale (moyenne=0, écart-type=1)
   n = torch.randn(2, 2)
   print("Normale standard (0,1) :\n", n)

   # Distribution normale avec moyenne (mean) et écart-type (std) choisis
   custom = torch.normal(mean=2.0, std=3.0, size=(2,2))
   print("Normale (moyenne=10, écart-type=2) :\n", custom)

   # Fixer la graine pour la reproductibilité
   torch.manual_seed(42)
   print("Reproductibilité :\n", torch.rand(2, 2))  # toujours le même résultat


.. slide::
📖 4. Connaître la forme d'un tenseur
------------------------

Un tenseur peut avoir n’importe quelle dimension. La méthode ``.shape`` permet de connaître sa taille.

.. code-block:: python

   # Scalaire (0D)
   s = torch.tensor(5)
   print("Scalaire :", s, "shape =", s.shape)

   # Vecteur (1D)
   v = torch.tensor([1, 2, 3, 4])
   print("Vecteur :", v, "shape =", v.shape)

   # Matrice (2D)
   m = torch.tensor([[1, 2, 3], [4, 5, 6]])
   print("Matrice :\n", m, "shape =", m.shape)

   # Tenseur 3D (par exemple, 2 matrices de taille 3x3)
   t3 = torch.zeros(2, 3, 3)
   print("Tenseur 3D shape =", t3.shape)

   # Tenseur 4D (par exemple, un mini-batch de 10 images RGB de 32x32)
   t4 = torch.zeros(10, 3, 32, 32)
   print("Tenseur 4D shape =", t4.shape)


.. slide::
📖 5. Types de tenseurs et conversion
------------------------

- Vous pouvez spécifier le type de données (``dtype``) lors de la création :

.. code-block:: python

   x = torch.tensor([1.2, 3.4, 5.6])
   print(x.dtype)     # float32 par défaut

   x_int = x.to(torch.int32)
   print(x_int, x_int.dtype)

   x_float64 = x.double()
   print(x_float64, x_float64.dtype)

- Conversion d’un tenseur existant :

.. code-block:: python

   x_int = x.to(torch.int32)
   print(x_int.dtype)

.. slide::
📖 6. Opérations de base
------------------------

PyTorch supporte de nombreuses opérations sur les tenseurs :

.. code-block:: python

   a = torch.tensor([1, 2, 3])
   b = torch.tensor([4, 5, 6])

   # Addition
   print(a + b)

   # Multiplication élément par élément
   print(a * b)

   # Produit matriciel
   mat1 = torch.rand(2, 3)
   mat2 = torch.rand(3, 4)
   print(torch.mm(mat1, mat2))

.. slide::
📖 7. Tenseurs sur GPU
------------------------

Pour profiter de l’accélération GPU, il suffit de déplacer un tenseur sur le device CUDA :

.. code-block:: python

   if torch.cuda.is_available():
       device = torch.device("cuda")
       x_gpu = x.to(device)
       print("Tenseur sur GPU :", x_gpu)
   else:
       print("Pas de GPU disponible, utilisation du CPU.")

.. slide::
📖 8.  Manipulation avancée des tenseurs
--------------------

Une fois créés, les tenseurs peuvent être transformés et réarrangés. PyTorch fournit de nombreuses fonctions pour modifier leur forme, leurs dimensions ou leur ordre.

8.1. Changer la forme avec ``view`` et ``reshape``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``view`` : retourne un nouveau tenseur qui partage la même mémoire que l’original. Cela implique que le tenseur soit contigu. Un tenseur est dit contigu lorsque ses données sont stockées de manière consécutive en mémoire, c’est-à-dire que PyTorch peut lire tous les éléments dans l’ordre sans sauts.  
Certaines opérations, comme la transposition (``t()``), rendent le tenseur non contigu, et dans ce cas ``view`` échoue.
- ``reshape`` : similaire à ``view``, mais plus flexible car il tente d’utiliser la mémoire existante, mais crée une copie si nécessaire. ``reshape`` fonctionne dans tous les cas de figures.

.. code-block:: python

   x = torch.arange(12)   # tenseur 1D [0, 1, ..., 11]
   print("x :", x)

   # Transformer en matrice 3x4
   x_view = x.view(3, 4)
   print("view en 3x4 :\n", x_view)

   # Transformer en matrice 2x6
   x_reshape = x.reshape(2, 6)
   print("reshape en 2x6 :\n", x_reshape)

.. slide::
Autre exemple pour illustrer la différence entre ``view`` et ``reshape`` :

.. code-block:: python

   # Création d'un tenseur 2x3
   x = torch.arange(6).view(2, 3)
   print("x :\n", x)
   print("Contigu :", x.is_contiguous())

   # Transposition → rend le tenseur non contigu
   y = x.t()
   print("\ny (transposé) :\n", y)
   print("Contigu :", y.is_contiguous())

   # view échoue sur un tenseur non contigu
   try:
       z = y.view(6)
   except Exception as e:
       print("\nErreur avec view :", e)

   # reshape fonctionne toujours
   z2 = y.reshape(6)
   print("\nreshape fonctionne :", z2)

.. slide::
8.2. Changer l’ordre des dimensions : ``permute``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``permute`` réarrange les dimensions dans un nouvel ordre.  
- Très utile pour manipuler les données d’images ou de séquences.

.. code-block:: python

   # Exemple avec un tenseur 3D (batch, hauteur, largeur)
   t = torch.randn(2, 3, 4)  # forme (2, 3, 4)
   print("Tenseur original :", t.shape)

   # Inverser l'ordre (largeur, hauteur, batch)
   p = t.permute(2, 1, 0)
   print("Après permute :", p.shape)

.. slide::
8.3. Ajouter ou supprimer des dimensions : ``unsqueeze`` et ``squeeze``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``unsqueeze(dim)`` : ajoute une dimension de taille 1 à la position ``dim``.  
- ``squeeze()`` : supprime toutes les dimensions de taille 1.  

.. code-block:: python

   v = torch.tensor([1, 2, 3])
   print("Forme initiale :", v.shape)

   v_unsq = v.unsqueeze(0)  # ajoute une dimension au début
   print("Après unsqueeze(0) :", v_unsq.shape)

   v_sq = v_unsq.squeeze()  # supprime les dimensions de taille 1
   print("Après squeeze() :", v_sq.shape)

.. slide::
8.4. Concaténer ou empiler des tenseurs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``torch.cat`` : concatène le long d’une dimension existante.  
- ``torch.stack`` : empile en ajoutant une nouvelle dimension.  

.. code-block:: python

   a = torch.tensor([1, 2, 3])
   b = torch.tensor([4, 5, 6])

   cat = torch.cat((a, b), dim=0)
   print("torch.cat :", cat)

   stack = torch.stack((a, b), dim=0)
   print("torch.stack :", stack)
   print("Forme de stack :", stack.shape)

.. slide::
📖 9. Autograd avec PyTorch
-----------------------

En Deep Learning, nous travaillons souvent avec des fonctions compliquées dépendant de plusieurs variables. Pour entraîner un modèle, nous avons besoin de calculer automatiquement les dérivées de ces fonctions. C'est là qu'intervient Autograd qui est le moteur de différentiation automatique de PyTorch. 

9.1. Création d'un tenseur suivi
~~~~~~~~~~~~~~~~~~~

Pour qu'un tenseur suive les opérations et calcule les gradients automatiquement, il faut définir ``requires_grad=True`` :

.. code-block:: python

    x = torch.tensor([2.0, 3.0], requires_grad=True)
    print(x)

Ici, ``x`` est maintenant un tenseur avec suivi des gradients. Toutes les opérations futures sur ce tenseur seront enregistrées pour pouvoir calculer les dérivées automatiquement.


.. slide::
9.2. Opérations sur les tenseurs
~~~~~~~~~~~~~~~~~~~

Toutes les opérations effectuées sur ce tenseur sont automatiquement enregistrées dans un graphe computationnel dynamique.

.. code-block:: python

    y = x ** 2 + 3 * x # y = [y1, y2]
    print(y)

Dans ce cas :

- ``x`` est la variable d'entrée.
- ``y`` est calculé à partir de ``x`` avec les opérations ``x**2`` et ``3*x``.

Chaque opération devient un nœud du graphe et PyTorch garde la trace des dépendances pour pouvoir calculer les gradients.

.. slide::
📖 10. Graphe computationnel
-----------------------------

Un graphe computationnel est une structure qui représente toutes les opérations effectuées sur les tenseurs. Chaque nœud du graphe correspond à un tenseur ou à une opération mathématique, et les arêtes indiquent les dépendances entre eux.

10.1. ``torchviz``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pour visualiser le graphe dans PyTorch, on peut utiliser ``torchviz`` (qu'il faudra installer avec ``pip install torchviz``)  :

.. code-block:: python

    from torchviz import make_dot

    z = y.sum()
    make_dot(z, params={'x': x})

Cela produira une image avec des nœuds pour chaque opération et des flèches indiquant les dépendances :

- Les nœuds $$x^2$$ et $$3x$$ représentent les opérations effectuées sur $$x$$.
- Le nœud $$y$$ combine ces deux résultats.
- Le graphe permet à PyTorch de savoir quelles dérivées calculer et dans quel ordre.

.. slide::
10.2. Note sur le graphe généré par PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quand on visualise le graphe interne avec un outil comme ``torchviz`` :

- Le **bloc jaune avec ()** correspond au tenseur final (ici ``z``).
- Les **blocs intermédiaires** (``PowBackward0``, ``AddBackward0``, etc.) représentent
  les opérations qui seront différentiées telles que ``PowBackward0`` est l'opération opération inverse associée à ``x**2``, ``MulBackward0`` celle associée à ``3*x``, 
  ``AddBackward0`` combine les deux résultats et représente ``y`` et enfin ``SumBackward0`` correspond au ``y.sum()`` qui est égal à ``z``.
- Le **bloc ``AccumulateGrad``** correspond à l’endroit où le gradient est stocké
  dans la variable d’entrée (ici ``x.grad``).

.. slide::
📖 11. Calcul des gradients et rétropropagation 
-----------------------

Autograd utilise ce graphe pour calculer automatiquement les dérivées par rapport à ``x``, en utilisant la méthode ``backward()`` :

.. code-block:: python
    z = y.sum()  # z = y1 + y2
    z.backward()
    print(x.grad)

- ``backward()`` calcule les dérivées de ``z`` par rapport à chaque élément de ``x``.
- ``x.grad`` contient maintenant les gradients.

11.1. Principe de la rétropropagation
~~~~~~~~~~~~~~~~~

Le principe de la rétropropagation signifie PyTorch parcourt le graphe **en sens inverse** pour faire le calcul des dérivées.


1. Commence par la sortie ``z``.
2. Recule vers les nœuds précédents (``y`` puis ``x``) en appliquant la règle de dérivation.
3. Stocke le gradient dans ``x.grad``.

.. slide::
11.2. Calcul des gradients dans notre exemple
~~~~~~~~~~~~~~~~~

- $$\frac{dz}{dy} = 1$$ car $$z = y.sum()$$ 
- $$\frac{dy}{dx} =$$ dérivée de $$(x^2 + 3*x) = 2*x + 3$$
- $$\frac{dz}{dx} = \frac{dz}{dy} * \frac{dy}{dx} = 2*x + 3$$

On obtient donc :

.. code-block:: python

    print(x.grad)  # tensor([7., 9.])

.. slide::
11.3. Détail du calcul des gradients
~~~~~~~~~~~~~~~~~

On a $$y = [y_1, y_2] = [x_1² + 3x_1,  x_2² + 3x_2]$$ et $$z = y_1 + y_2$$.


**Étape 1 : dérivée de z par rapport à y**

Comme $$z = y_1 + y_2$$, on a $$\frac{dz}{dy_1} = 1$$ et $$\frac{dz}{dy_2} = 1$$.

On peut regrouper sous forme vectorielle, telle que $$\frac{dz}{dy} = [\frac{dz}{dy_1}, \frac{dz}{dy_2}] = [1, 1]$$.

**Étape 2 : dérivée de y par rapport à x**

On a $$\frac{dy_1}{dx_1} = 2x_1 + 3$$ et $$\frac{dy_2}{dx_2} = 2x_2 + 3$$.
On peut aussi regrouper sous forme vectorielle, telle que $$\frac{dy}{dx} = [\frac{dy_1}{dx_1}, \frac{dy_2}{dx_2}] = [2x_1 + 3, 2x_2 + 3]$$.

**Étape 3 : application de la règle de la chaîne**

Pour obtenir les dérivées de z par rapport à x, on applique la règle de la chaîne :

$$\frac{dz}{dx} = [\frac{dz}{dx_1}, \frac{dz}{dx_2}] = \frac{dz}{dy} * \frac{dy}{dx}$$ et $$\frac{dz}{dx} = [\frac{dz}{dy_1}*\frac{dy_1}{dx_1}, \frac{dz}{dy_2}*\frac{dy_2}{dx_2}] = [1 * (2x_1 + 3), 1 * (2x_2 + 3)]$$ 

et donc $$\frac{dz}{dx} = [2x_1 + 3, 2x_2 + 3]$$. 

.. slide::
11.4. Résultat numérique pour notre exemple 
~~~~~~~~~~~~~~~~~

.. code-block:: python

    print(x)       # tensor([2., 3.], requires_grad=True)
    print(x.grad)  # tensor([7., 9.])

Car :

- Pour $$x_1 = 2 → \frac{dz}{dx_1} = 2*2 + 3 = 7$$
- Pour $$x_2 = 3 → \frac{dz}{dx_2} = 2*3 + 3 = 9$$

Ainsi, Autograd reproduit automatiquement ce calcul grâce au graphe computationnel et à la règle de la chaîne.


.. slide::
📖 12. Désactivation du suivi des gradients
---------------------

Pour certaines opérations, par exemple lors de l'évaluation d'un modèle, il est inutile
de calculer les gradients. On peut alors désactiver le suivi avec ``torch.no_grad()`` :

.. code-block:: python

    with torch.no_grad():
        z = x * 2
    print(z)

Cela permet d'économiser de la mémoire et d'accélérer les calculs.

.. slide::
📖 13. Les fonctions de perte (Loss Functions)
-------------------------------

Lorsqu’on entraîne un réseau de neurones, l’objectif est de minimiser l’erreur entre les prédictions du modèle et les valeurs attendues. Cette erreur est mesurée par une fonction de perte (loss function en anglais).

Une fonction de perte prend en entrée :

    - la sortie du modèle (la prédiction),
    - la valeur cible (la réponse attendue, donnée par les données d’apprentissage),

et retourne un nombre réel qui indique "à quel point le modèle s'est trompé".

Par conséquent, plus la perte est grande → plus le modèle se trompe et plus la perte est petite → plus le modèle est proche de la bonne réponse.

.. slide::
📖 14. Pourquoi la fonction de perte est essentielle ?
----------------------------------------------------
La fonction de perte est essentielle pour plusieurs raisons :

    - Elle quantifie l'erreur du modèle : elle donne une mesure numérique de la performance du modèle.
    - Elle permet de guider l'apprentissage : le modèle apprend en essayant de réduire cette valeur.
    - Elle est le point de départ de la rétropropagation : les gradients sont calculés à partir de la fonction de perte.
    - Elle est utilisée par les algorithmes d'optimisation pour ajuster les paramètres du modèle.
    - Elle permet de comparer différents modèles : en utilisant la même fonction de perte, on peut évaluer quel modèle est le meilleur.
    - Elle est essentielle pour le processus d'entraînement : sans fonction de perte, le modèle n'aurait aucun signal pour savoir comment s’améliorer.

.. slide::
📖 15. Régression & Erreur quadratique moyenne (MSE)
----------------------------------------------------

15.1. Définitions
~~~~~~~~~~~~~~~~~
On appelle régression le cas où le modèle doit prédire une valeur numérique par exemple : la température demain, la taille d’une personne, etc.

Dans ce cas, la fonction de perte la plus utilisée est l’erreur quadratique moyenne (MSE de l'anglais Mean Squared Error) :

.. math::

   L(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2,

où :

    - $$L$$ est la fonction de perte,
    - $$n$$ est le nombre de données,
    - $$y_i$$ est la valeur attendue (target) et
    - $$\hat{y}_i$$ est la prédiction du modèle.

La fonction MSE calcule la moyenne des erreurs au carrées de toutes les données.

.. slide::
15.2. Exemple d'une régression avec MSE dans PyTorch
~~~~~~~~~~~~~~~~~~~~~
Pour utiliser la fonction MSE dans PyTorch, on peut utiliser la classe ``nn.MSELoss()``. Pour cela, il faut d'abord importer le module ``torch.nn`` qui contient les fonctions de perte :
.. code-block:: python

    import torch.nn as nn

**Exemple** : 

.. code-block:: python

    # Valeurs réelles et prédictions
    y_true = torch.tensor([2.0, 3.0, 4.0])
    y_pred = torch.tensor([2.5, 2.7, 4.2])

    # Définition de la fonction de perte MSE
    loss_fn = nn.MSELoss()

    # Calcul de la perte
    loss = loss_fn(y_pred, y_true)
    print(loss)

.. slide::
📖 16. Classification & Entropie croisée
------------------------------------------------------------

16.1. Définitions
~~~~~~~~~~~~~~~~~~~

On appelle classification le cas où le modèle doit prédire à quelle catégorie appartient la donnée parmi plusieurs possibles par exemple : "chat" ou "chien", ou bien "spam" ou "non spam", etc.

Dans ce cas, la fonction de perte la plus courante est l'entropie croisée (Cross-Entropy Loss en anglais). Elle compare la probabilité prédite par le modèle et la vraie catégorie (donnée par les données d’apprentissage) :

.. math::
   L(y, \hat{y}) = -\sum_{i=1}^n y_i \log(\hat{y}_i),
où :

    - $$L$$ est la fonction de perte,
    - $$n$$ est le nombre de classes,
    - $$y_i$$ est la valeur attendue (target) pour la classe $$i$$ ((souvent codée en *one-hot encoding*, c'est-à-dire un vecteur avec un 1 pour la bonne classe et 0 pour les autres),
    - $$\hat{y}_i$$ est la probabilité prédite par le modèle pour la classe $$i$$.

La fonction enropie croisée mesure la distance entre la distribution de probabilité prédite par le modèle et la distribution de probabilité réelle (la vraie classe).
La présence de la somme permet de prendre en compte toutes les classes.   Mais, dans le cas du *one-hot encoding*, seul le terme correspondant à la vraie classe reste (puisque tous les autres $$y_i$$ valent 0).

.. slide::
16.2. Pourquoi l'entropie croisée ?
~~~~~~~~~~~~~~~~~~~
L'entropie croisée est utilisée car :

    - Elle est adaptée aux problèmes de classification multi-classes.
    - Elle pénalise fortement les erreurs de classification, surtout lorsque la probabilité prédite pour la classe correcte est faible.
    - Elle est différentiable, ce qui permet de l'utiliser avec les algorithmes d'optimisation basés sur la rétropropagation.

.. slide::
16.3. Exemple d'une classification avec Cross-Entropy Loss 
~~~~~~~~~~~~~~~~~~~~
Prenons un exemple où on a 3 classes possibles : "Chat", "Chien", "Oiseau". Nous avons : 

- La sortie du modèle suivante : $$\hat{y} = [0.7, 0.2, 0.1]$$ et
- imaginons que la vraie classe est "Chat", donc $$y = [1, 0, 0]$$.

Alors :

.. math::

    L = - \big( 1 \cdot \log(0.7) + 0 \cdot \log(0.2) + 0 \cdot \log(0.1) \big)

Les termes multipliés par 0 disparaissent :

.. math::

    L = -\log(0.7)

👉 La perte est faible car le modèle a donné une forte probabilité à la bonne classe.

Si au contraire le modèle avait prédit : $$\hat{y} = [0.2, 0.7, 0.1]$$ :

.. math::

    L = -\log(0.2)

👉 La perte serait plus grande, car la probabilité attribuée à la bonne classe ("Chat") est faible.


.. slide::
16.4. Le même exemple dans PyTorch 
~~~~~~~~~~~~~~~~~~~~

Pour utiliser la fonction Cross-Entropy Loss dans PyTorch, on peut utiliser la classe ``nn.CrossEntropyLoss()`` du module ``torch.nn``.

.. code-block:: python

    # Définition de la fonction de perte
    loss_fn = nn.CrossEntropyLoss()

    # Cas 1 : le modèle prédit correctement (forte valeur pour "Chat")
    logits1 = torch.tensor([[2.0, 1.0, 0.1]])  # sortie brute du modèle qui sera convertie à l'aide d'une fonction de PyTorch en probabilités
    y_true = torch.tensor([0])  # la vraie classe est "Chat" (indice 0)

    loss1 = loss_fn(logits1, y_true)
    print("Perte (bonne prédiction) :", loss1.item())

    # Cas 2 : le modèle se trompe (forte valeur pour "Chien")
    logits2 = torch.tensor([[0.2, 2.0, 0.1]])  # sortie brute du modèle qui sera convertie à l'aide d'une fonction de PyTorch en probabilités
    loss2 = loss_fn(logits2, y_true)
    print("Perte (mauvaise prédiction) :", loss2.item())

.. slide::
📖 17. Optimisation
-----------------------

L’optimisation est l’étape qui permet d’ajuster les paramètres du modèle pour qu’il réalise mieux la tâche demandée.  

L’idée est simple :  

1. On calcule la perte (loss en anglais) qui indique l’erreur du modèle.  
2. On calcule le gradient de la perte par rapport aux paramètres (grâce à Autograd).  
3. On met à jour les paramètres dans la bonne direction (celle qui diminue la perte).  

C’est un processus itératif qui se répète jusqu’à ce que le modèle apprenne correctement.


.. slide::
📖 18. Descente de gradient
-----------------------

L’algorithme d’optimisation le plus courant est la descente de gradient (ou Gradient Descent en anglais). 

18.1. Principe et formule de la descente de gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Imaginons une montagne :  
- La hauteur correspond à la valeur de la fonction de perte.  
- Le but est de descendre la montagne pour atteindre la vallée (la perte minimale).  
- Le gradient indique la pente : on suit la pente descendante pour réduire la perte.

Formule de mise à jour des paramètres :

.. math::

   \theta_{new} = \theta_{old} - \eta \cdot \nabla_\theta L(\theta)

où :  

- $$\theta$$ représente l’ensemble des paramètres du modèle,  
- $$L$$ est la fonction de perte,  
- $$\eta$$ est le taux d’apprentissage (*learning rate* en anglais) : il contrôle la taille des pas et  
- $$\nabla_\theta L(\theta)$$ désigne le vecteur des dérivées partielles de $$L$$ par rapport à chacun des paramètres.  


.. slide::
📖 18.2. Exemple simple de la descente de gradient
~~~~~~~~~~~~~~~~~~~~~~~~
Prenons un exemple très simple : nous voulons ajuster un seul paramètre $$a$$ pour approximer une fonction.

Supposons que le modèle soit une droite passant par l’origine :

.. math::

   f(x) = a x

Nous avons une donnée d’apprentissage :  

- Entrée : $$x = 2$$  
- Sortie attendue : $$y = 4$$  

On part du paramètre initial : $$a = 0$$.

.. slide::
**1. Fonction de perte**

On utilise l’erreur quadratique (MSE) pour mesurer l’écart entre la prédiction et la vraie valeur :

.. math::

   L(a) = (f(x) - y)^2 = (a * 2 - 4)^2


**2. Calcul du gradient**

On dérive la perte par rapport à $$a$$ :

.. math::

   \frac{\partial L}{\partial a} = 2 * (a * 2 - 4) * 2 = 8a - 16

.. slide::

**3. Mise à jour avec descente de gradient**

On choisit un taux d’apprentissage $$\eta = 0.1$$ et on applique la formule :

.. math::

   a_{new} = a_{old} - \eta \cdot \frac{\partial L}{\partial a}


**4. Exemple numérique**

- Point de départ : $$a = 0$$  
- Gradient : $$\frac{\partial L}{\partial a} = 8 * 0 - 16 = -16$$  
- Mise à jour :  

.. math::

   a_{new} = 0 - 0.1 * (-16) = 1.6

👉 Après une étape, $$a$$ se rapproche déjà de la bonne valeur (qui devrait être $$a = 2$$ pour que $$f(x) = 2 * 2 = 4$$).  

En répétant plusieurs mises à jour, $$a$$ converge vers 2, et la perte devient de plus en plus faible.


.. slide::
📖 19. Descente de gradient avec PyTorch
----------------------------------------

PyTorch fournit le module ``torch.optim`` qui implémente plusieurs algorithmes d’optimisation. Dans PyTorch, l’algorithme de descente de gradient est appelé SGD (Stochastic Gradient Descent) et peut-être importé via ``torch.optim.SGD`` :

.. code-block:: python
   import torch.optim as optim

On reprend le modèle simple :

- Modèle : f(x) = a * x
- Objectif : trouver a tel que f(x) ≈ y
- Jeu de données : x = [1, 2, 3, 4], y = [2, 4, 6, 8]
- Paramètre initial : a = 0
- Taux d'apprentissage : lr = 0.1

.. slide::
.. code-block:: python
    # Données
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0])
    a = torch.tensor([0.0], requires_grad=True)

    # Optimiseur : descente de gradient
    optimizer = optim.SGD([a], lr=0.1)

    # Fonction de perte : MSE
    loss_fn = nn.MSELoss()

    for i in range(10):
        # 1. Remettre les gradients à zéro avant de recalculer
        optimizer.zero_grad()
        
        # 2. Calcul de la prédiction
        y_pred = a * x
        
        # 3. Calcul de la perte avec MSE
        loss = loss_fn(y_pred, y)
        
        # 4. Calcul automatique des gradients
        loss.backward()
        
        # 5. Mise à jour du paramètre a
        optimizer.step()
        
        print(f"Iter {i+1}: a = {a.item()}, loss = {loss.item()}")

.. note::

      Explications des nouvelles lignes de code :

         - ``optimizer.zero_grad()`` : remet à zéro les gradients calculés lors de la dernière itération.  
         Sinon, PyTorch additionne les gradients à chaque ``backward()``, ce qui fausserait les calculs.
         
         - ``optimizer.step()`` : applique la mise à jour des paramètres selon la règle de la descente de gradient :  
         $$\theta_new = \theta_old - lr * gradient$$.
         

Dans cet exemple, SGD converge très vite car le problème est simple.
 
.. slide::
📖 20. Optimiseur Adam
--------------------------------------

20.1. Définition
~~~~~~~~~~~~~~~~~~
Adam est un autre algorithme d'optimisation qui adapte le pas pour chaque paramètre grâce à une moyenne mobile des gradients ($$m_t$$ ) et une moyenne mobile des carrés des gradients ($$v_t$$).  

On définit :

- $$g_t = \nabla_\theta L(\theta)$$ : le gradient à l'itération t  
- $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$ : moyenne mobile des gradients (1er moment)  
- $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$ : moyenne mobile des carrés des gradients (2e moment)  
- $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$ : correction de biais pour le 1er moment  
- $$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$ : correction de biais pour le 2e moment  
- $$\epsilon$$ : petite constante pour éviter la division par zéro  

La mise à jour des paramètres est alors :

.. math::
  \theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

💡 Interprétation :

- $$m_t$$ capture la direction moyenne des gradients (évite les oscillations),  
- $$v_t$$ ajuste le pas selon la variance des gradients (pas plus grand si le gradient est bruité),  
- $$\epsilon$$ empêche la division par zéro et
- La correction de biais $$\hat{m}_t, \hat{v}_t$$ est importante surtout au début pour ne pas sous-estimer les moments.

.. slide::
20.2. Adam vs. SGD
~~~~~~~~~~~~~~~~~~~~~
 Différences entre Adam et la descente de gradient classique (SGD) :

    1. **SGD** applique la même règle de mise à jour pour tous les paramètres à chaque itération :  
       \theta_new = \theta_old - lr * gradient
       
    2. **Adam** adapte le taux d'apprentissage pour chaque paramètre individuellement,  
       en utilisant des moyennes mobiles des gradients et des carrés des gradients.  
       Cela permet souvent une convergence plus rapide et plus stable.
    
    3. La syntaxe PyTorch reste très similaire : on utilise toujours ``optimizer.zero_grad()``, ``loss.backward()`` et ``optimizer.step()``. On peut reprendre le même modèle simple que précédemment à titre d'exemple.

.. note::
   ⚠️ Remarque : Dans le cadre de ce cours, nous utiliserons principalement Adam pour sa robustesse et sa facilité d'utilisation. Nous allons surtout utiliser l'implémentation de ADAM dans Pytorch sans avoir à recoder les équations. Elles sont énoncées à titre informatif.

.. slide::
20.3. Implémentation d'Adam avec PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans PyTorch, Adam est implémenté via ``torch.optim.Adam`` :

.. code-block:: python
    # Données
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0])
    a = torch.tensor([0.0], requires_grad=True)

    # Optimiseur : Adam
    optimizer = torch.optim.Adam([a], lr=0.1)

    # Fonction de perte : MSE
    loss_fn = nn.MSELoss()

    for i in range(50):
        optimizer.zero_grad()  # remise à zéro des gradients
        y_pred = a * x
        loss = loss_fn(y_pred, y)  # perte MSE
        loss.backward()  # calcul automatique des gradients
        optimizer.step()  # mise à jour du paramètre
        
        print(f"Iter {i+1}: a = {a.item()}, loss = {loss.item()}")

💡 Remarques :

   - Pour des problèmes **simples** comme $$f(x)=ax$$, SGD converge très vite et Adam peut sembler plus lent sur peu d’itérations.  
   - Pour des **modèles complexes** avec beaucoup de paramètres et des gradients bruités, Adam est souvent plus efficace grâce à ses ajustements adaptatifs.

.. slide::
🏋️ Travaux Pratiques 1
--------------------

.. toctree::

    TP_chap1