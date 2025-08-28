.. slide::
Chapitre 2 — Perceptron multi-couches 
===========================================

🎯 Objectifs du Chapitre
----------------------


.. important::

    À la fin de cette section,  vous saurez :  

    - Le fonctionnement du perceptron simple.
    - Utiliser une fonction d'activation adaptée.  
    - L’importance de la normalisation / standardisation des données et l'usage des epochs.  
    - Construire un réseau de neurones avec ``torch.nn``. 
    - Faire un entraînement simple d’un MLP pour un problème de régression.   
    - Suivre l’évolution de la loss et interpréter les résultats.  
    - Utiliser ``torch-summary`` pour inspecter l’architecture du réseau.  


.. slide::

📖 1. Rappels sur les perceptrons
----------------------

Le perceptron multi-couches (MLP de Multi-Layers Perceptron en anglais) est la brique de base des réseaux de neurones modernes. Dans ce chapitre, nous allons l’appliquer à des problèmes de régression simple. Avant de commencer, voici quelques rappels.

1.1. Perceptron simple
~~~~~~~~~~~~~~~~~~~~~~

Le perceptron est le bloc de base d’un réseau de neurones. Il réalise une transformation linéaire suivie (ou pas) d’une fonction d’activation telle que :  

  .. math::

     y = \sigma(Wx + b)

  où :  
   - $$y$$ est la sortie du perceptron, 
   - $$\sigma$$ est une fonction d’activation, 
   - $$W$$ est la matrice des poids,  
   - $$b$$ est le biais et  
   - $$x$$ est l'ensemble des entrées du perceptron.  

.. slide::
1.2. Perceptron intuition
~~~~~~~~~~~~~~~~~~~~~~
.. image:: images/chap2_perceptron.png
    :alt: perceptron
    :align: center
    :width: 40%

avec $$y= \sigma(x_1*w_1 + x_2*w_2 + ...+ x_i*w_i + ... + x_n*w_n + b)$$

💡 **Intuition :**

    - Chaque poids $$w_i$$ mesure l’importance de la caractéristique $$x_i$$.  
    - Le biais $$b$$ déplace la frontière de décision.  
    - La fonction d’activation permet d’introduire de la non-linéarité, indispensable pour modéliser des relations complexes mais nous en parleront plus en détails par la suite.  


.. slide::
1.3. Mise à jour des paramètres
~~~~~~~~~~~~~~~~~~~~~~

Un perceptron possède deux types de **paramètres** : les **poids** et le **biais**.  

Lors de l’entraînement, on souhaite ajuster ces paramètres pour améliorer les prédictions du modèle.  Pour cela, il faut mettre à jour les poids après avoir calculé la loss grâce à la fonction de perte et le gradient grâce à l'optimiseur comme expliqué dans le chapitre précédent.  

Pour rappel, on met à jours les paramètres du modèle grâce à l'équation introduite dans le chapitre précédent. 

.. math::

    \theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}(\theta)

où :  
- $$\theta$$ représente l’ensemble des paramètres du modèle (ici $$W$$ et $$b$$),  
- $$\mathcal{L}$$ est la fonction de perte,  
- $$\nabla_\theta \mathcal{L}$$ est le gradient de la perte par rapport aux paramètres,  
- $$\eta$$ est le taux d’apprentissage (learning rate en anglais).


.. slide::
1.4. Exemples d'applications du perceptron simple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Un perceptron simple ne peut résoudre que les problèmes linéairement séparables puisque en trouvant les paramètres du modèle, le perceptron trace une droite dans le plan des entrées et sépare les points selon qu’ils sont au-dessus ou en dessous de cette droite.

**Exemple 1 : porte logique ET**

+-----+-----+-------+
| x₁  | x₂  | y=ET  |
+=====+=====+=======+
|  0  |  0  |   0   |
+-----+-----+-------+
|  0  |  1  |   0   |
+-----+-----+-------+
|  1  |  0  |   0   |
+-----+-----+-------+
|  1  |  1  |   1   |
+-----+-----+-------+

Dans ce cas, une droite sépare bien les deux classes :  
- la classe $$0$$ (points en bas à gauche, en haut à gauche, en bas à droite),  
- la classe $$1$$ (point en haut à droite).  

Un perceptron simple peut donc apprendre cette fonction.

.. slide::
**Exemple 2 : porte logique XOR**

+-----+-----+--------+
| x₁  | x₂  | y=XOR  |
+=====+=====+========+
|  0  |  0  |   0    |
+-----+-----+--------+
|  0  |  1  |   1    |
+-----+-----+--------+
|  1  |  0  |   1    |
+-----+-----+--------+
|  1  |  1  |   0    |
+-----+-----+--------+

Ici, il est impossible de tracer une seule droite qui sépare correctement les classes. Autrement dit, XOR n’est pas linéairement séparable.  

.. image:: images/chap2_et_vs_xor.png
   :alt: Représentation du XOR dans le plan (non-séparable linéairement)
   :align: center
   :width: 300%

**Conclusion :** 

- Le perceptron simple suffit pour des tâches linéaires (comme ET, OU).  
- Pour résoudre des problèmes plus complexes comme XOR, il faut introduire plusieurs couches de neurones et des fonctions d’activation non-linéaires : c’est le principe du **perceptron multi-couches (MLP)**. 

.. slide::
1.5. Faire un perceptron dans PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour créer un perceptron simple dans PyTorch, on peut utiliser la fonction ``Linear`` de ``torch.nn``, qui implémente une couche linéaire (ou affine) : $$y = Wx + b$$. La fonction ``Linear`` prend en entrée le nombre d'entrée $$x$$ et le nombre de sortie $$y$$.

.. code-block:: python

    import torch
    import torch.nn as nn

    # Données ET
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
    y = torch.tensor([[0],[0],[0],[1]], dtype=torch.float32)

    # Modèle linéaire (perceptron)
    model = nn.Linear(2,1)

    # Loss function et optimiseur
    loss_fc = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Entraînement
    for _ in range(500):
        optimizer.zero_grad()
        loss = loss_fc(model(X), y)
        loss.backward()
        optimizer.step()

    # Résultat
    with torch.no_grad():
        print((model(X)).round())
        print(model.weight, model.bias)

**Remarque** : si maintenant on change les entrées et sorties pour le XOR, le modèle ne pourra pas apprendre correctement la fonction (les $$W$$ restent à 0 comme à l'initialisation). Vous pouvez faire le test pour vérifier.

.. slide::

################################ STOP ICI ################################

################################ STOP ICI ################################

################################ STOP ICI ################################


.. slide::

📖 2. Fonction d'activation
-----------

Les fonctions d’activation introduisent de la non-linéarité dans le modèle, ce qui permet de mieux capturer des relations complexes dans les données. Voici quelques fonctions d’activation couramment utilisées :

1. **Sigmoïde** : $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
   - Sortie entre 0 et 1.
   - Utilisée pour les problèmes de classification binaire.

2. **Tanh** : $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
   - Sortie entre -1 et 1.
   - Souvent utilisée dans les réseaux de neurones cachés.

3. **ReLU (Rectified Linear Unit)** : $$\text{ReLU}(x) = \max(0, x)$$
   - Sortie nulle pour les entrées négatives.
   - Très utilisée dans les réseaux de neurones profonds en raison de sa simplicité et de son efficacité.

4. **Softmax** : $$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
   - Transforme un vecteur en une distribution de probabilité.
   - Utilisée en sortie des modèles de classification multi-classes.



.. slide::

📖 3. Epoch
-----------

Lorsqu’on entraîne un modèle, on doit présenter plusieurs fois l’ensemble des données d’apprentissage $$x$$.

3.1 Définitions importantes
~~~~~~~~~~~~

- Itération : mise à jour des paramètres après avoir traité un seul exemple ou un mini-batch.

- Batch / mini-batch : sous-ensemble d’exemples utilisés pour calculer la mise à jour.

-  Epoch : passage complet sur toutes les données d’apprentissage.

**Exemple** :

Si vous avez 1000 exemples et que vous utilisez des mini-batchs de 100 : Une epoch correspond à 10 itérations (1000 ÷ 100).

Après chaque epoch, chaque exemple a été utilisé exactement une fois pour mettre à jour les paramètres.

3.2 Pourquoi plusieurs epochs ?
~~~~~~~~~~~~
Au début de l’entraînement, le modèle fait souvent de grandes erreurs.

Chaque epoch permet aux poids et biais de s’ajuster progressivement pour mieux prédire les sorties.

En général, plusieurs dizaines ou centaines d’epochs sont nécessaires pour que la loss se stabilise.

💡 Intuition : imaginez un perceptron comme un élève qui apprend : il ne retient pas tout parfaitement du premier coup ; il faut plusieurs passages sur le même exercice pour maîtriser.





.. slide::
4. Normaliser / standardiser les données
-----------------------------

Pourquoi normaliser ?  

- Les entrées de grande amplitude ralentissent l’apprentissage.  
- Normaliser permet de mettre toutes les features à la même échelle.  

Deux approches classiques :  

- **Normalisation** : valeurs entre 0 et 1.  
- **Standardisation** : moyenne 0, variance 1.  

Exemple avec scikit-learn :  

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)


5. Utiliser ``torch.nn`` pour construire un MLP
------------------------------------------------

- ``Sequential`` : permet d’empiler les couches facilement.  
- ``Linear`` : couche affine (Wx+b).  
- Fonctions d’activation : donnent la non-linéarité (ex. ``nn.ReLU()``).  

Exemple minimal d’un réseau :  

.. code-block:: python

   import torch.nn as nn

   model = nn.Sequential(
       nn.Linear(1, 10),   # entrée 1D -> couche cachée 10 neurones
       nn.ReLU(),          
       nn.Linear(10, 1)    # sortie 1D (régression)
   )


6. Suivi de la loss et visualisation
-------------------------------------

- Pendant l’entraînement, enregistrer la loss à chaque epoch pour voir si elle diminue.  
- Comparer ``y_pred`` et ``y_true`` avec Matplotlib.  

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.plot(losses)              # courbe de la loss
   plt.scatter(x, y_true)        # données réelles
   plt.scatter(x, y_pred)        # prédictions


7. Inspecter le modèle avec ``torch-summary``
----------------------------------------------

Permet de voir le nombre de paramètres par couche et la structure du réseau.  

.. code-block:: python

   from torchsummary import summary
   summary(model, input_size=(1,))








################################ Activation fonction ######################################
Parler de la softmax, relu , etc.
##########################################################################################



################################ MLP ######################################

faire une classe avec fonction pour les couches et une pour le forward comme : 
..code-block:: python   
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


parler de dataset loader et parler de broadcasting ?

- parler de .detach() et .clone() ?

- parler de autograd profiler.profile

- parler de la gestion des outliers
##########################################################################################











