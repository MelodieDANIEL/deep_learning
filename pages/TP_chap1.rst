🏋️ Exercices supplémentaires
=========================
.. slide::
Sur cette page se trouvent des exercices supplémentaires pour vous entraîner. Ils sont classés par niveau de difficulté :
.. discoverList::
    * Facile : 🍀
    * Moyen : ⚖️
    * Difficile : 🌶️

.. slide::
🍀 Exercice Sup. 1 : Déterminer si une année est bissextile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. step::
    Avant de commencer cet exercise vous devez faire quelques recherches sur internet pour savoir :
.. discoverList::
    * Qu’est-ce qu’une année bissextile ?
    * Comment peut-on tester si un nombre entier est divisible par un autre en Python ?

.. step::
**Consigne** : Ecrire un programme permettant de déterminer si l’année saisie par l'utilisateur est bissextile ou non et lui afficher le resultat.

**Résultat attendu** :

.. code-block:: python
    >> Saisir une année : 2020
    >> L année 2020 est bissextile

    #ou

    >> Saisir une année : 2021
    >> L année 2021 n est pas bissextile

.. slide::
🍀 Exercice Sup. 2 : Calculs mathématiques de base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objectif** : Ecrire un programme qui permet de réaliser des opérations arithmétiques simples (addition, soustraction, multiplication, division).

**Consignes** : Demander à l’utilisateur de donner deux nombres réels et l’opération à réaliser avec une précision de deux chiffres après la virgule. 

**Astuce**:
.. spoiler::
    .. discoverList::
        * Cherchez sur internet comment tronquer un réel à deux chiffres après la virgule dans un affichage

**Résultat attendu** :

.. code-block:: python
    >> Entrer deux nombres réels : 2.2
    >> 3.5
    >> le premier nombre réel = 2.2 et le second = 3.5
    >> Veuillez choisir :
    >>    1 pour réaliser la somme de 2.20 et 3.50
    >>    2 pour réaliser la soustraction de 2.20 et 3.50
    >>    3 pour réaliser la multiplication de 2.20 et 3.50
    >>    4 pour réaliser la division de 2.20 et 3.50
    >> 1
    >> Résultat de l addition : 2.2 + 3.5 = 5.70

.. slide::
🍀 Exercice Sup. 3 : Résolution d’un polynome
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objectif** : Écrire un programme qui permet de retourner le résultat de  $$f(X) = 1 + X + aX^2 + bX^3 + cX^4 + \dots + zX^n$$

**Consigne** : Demander à l’utilisateur d’entrer la valeur entière $$X$$, le degré $$n$$ du polynome et les valeurs entières des coefficients ($$a, b, c, \dots, z$$). Puis retourner le résultat de la fonction $$f(X)$$.

**Astuce**:
.. spoiler::
    Utiliser la récurcivité ensuivant la méthode de Hörner  
    C'est à dire : $$2X^4 + 3X^3+ 2X^2 + X + 1 = ((( (2)*X+3) *X+2) *X+1) * X+1)$$.

**Résultat attendu** :
.. code-block:: python
    >> Entrer la valeur de X : 2
    >> Entrer le degré du polynome : 4
    >> Entrer la valeur de coefficient : 2
    >> Entrer la valeur de coefficient : 3
    >> Entrer la valeur de coefficient : 2
    >> Entrer la valeur de coefficient : 1
    >> le résultat = 67


.. slide::
⚖️ Exercice Sup. 4 : Facturation de télécommunication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans un tarif d’abonnement de portable, les communications sont facturées 0.15€ la minute.

**Consignes** : Connaissant en heures et minutes le début et la fin de la communication, écrire un programme calculant et affichant :
.. discoverList::
	* la durée de la communication en heures et minutes
	* le coût de la communication

**Contraintes** :
.. discoverList::
	* Une communication peut commencer le jour J et se terminer le jour J+1. 
	* Une communication ne peut durer plus de 8 heures.
	* Le prix de la communication est demi-tarif soit 0.075 euros entre 21h30 et 8h.
	* Une communication peut se dérouler en plage tarif normal et tarif réduit.

**Tests** : Afin de vérifier la validité de votre programme, testez les cas suivants :
.. discoverList::
	* *Cas 1* : la communication commence avant 8h et se termine avant 8h
		- début : 6h et fin : 7h30
		- durée : 1h30 soit 90 minutes en demi-tarif
		- coût : 90×0.075 = 6.75€

	* *Cas 2* : la communication commence avant 8h et après 8h (mais avant 21h30) 
		- début : 7h30 et fin : 9h30
		- durée : 2h00 soit 30 minutes en demi-tarif et 90 minutes en plain tarif
		- coût : 30×0.075+90×0.15 = 15.75€

	* *Cas 3* : la communication commence après 8h et se termine avant 21h30
		- début : 9h30 et fin : 10h30
		- durée : 1h00 soit 60 minutes en plein tarif
		- coût : 60×0.15 = 9.00€

	* *Cas 4* : la communication commence après 8 h et se termine après 21h30
		- début : 20h30 et fin : 22h30
		- durée : 2h00 soit 60 minutes en plein tarif et 60 minutes en demi-tarif
		- coût : 60×0.15+60×0.075 = 13.50€

	* *Cas 5* : la communication commence après 21h30 h et se termine avant 8h
		- début : 23h30 et fin : 1h30
		- durée : 2h00 soit 120 minutes en demi-tarif
		- coût : 120×0.075 = 9.00€

**Resultat attendu** :

.. code-block:: python
    >> Saisir heure de début : 23
    >> Saisir minute de début : 30
    >> Saisir heure de fin : 1
    >> Saisir minute de fin : 30
    >> la durée de la communication = 2.0 h 0
    >> la durée de la communication en demi tarif est de 120 et elle a coûté 9.0 euros
    >> la durée de la communication en plein tarif est de 0 et elle a coûté 0.0 euros
    >> le coût total de la communication = 9.0 euros


.. slide::
⚖️ Exercice Sup. 5 : Triangle de Pascal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. step::
    Ecrire un programme qui construit et affiche le triangle de Pascal de degré n (Limiter le degré à 13)
    .. warning::
        Exemple pour n = 6

        .. code-block:: text

            1  
            1  1  
            1  2  1  
            1  3  3  1  
            1  4  6  4  1  
            1  5 10 10  5  1  
            1  6 15 20 15  6  1  

**Résultat attendu** : 

.. code-block:: python
    Entrez le degré du triangle de Pascal (maximum 13): 6
    1  
    1  1  
    1  2  1  
    1  3  3  1  
    1  4  6  4  1  
    1  5 10 10  5  1  
    1  6 15 20 15  6  1

**Astuces** : 
.. spoiler::
    .. discoverList::
        * Calculer et afficher seulement les valeurs jusqu’à la diagonale principale incluse (Limiter le degré à 13)  
        * Construire le triangle ligne par ligne
            - Initialiser le premier élément et l’élément de la diagonale à 1.
            - Calculer les valeurs entre les éléments initialisés de gauche à droite en utilisant la relation suivante : $$P_{i,j} = P_{i-1,j} + P_{i-1,j-1}$$
