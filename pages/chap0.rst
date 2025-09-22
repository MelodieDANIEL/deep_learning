
.. slide::

Chapitre 0 - Installation des paquets et bibliothques nécessaires pour le cours
================

🎯 Objectifs du Chapitre
----------------------


.. important::

   À la fin de ce chapitre, vous saurez : 
   
   - Créer un environnement virtuel Python.
   - Installer PyTorch et les bibliothèques associées.
   - Vérifier l'installation de PyTorch.
   - Installer Jupyter Notebook (optionnel mais recommandé).   

.. slide::

📖 1. Créer un environnement virtuel
----------------------
Pour installer proprement PyTorch et les bibliothèques nécessaires, nous allons d’abord créer un environnement virtuel. 


1.1. Qu'est-ce qu'un environnement virtuel ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Un environnement virtuel (ou Virtual Environment en anglais) est un dossier isolé dans lequel on peut installer des bibliothèques Python sans interférer avec le reste du système.

En pratique, cela permet :

- d’avoir une version précise des bibliothèques pour un projet donné,
- d’éviter les conflits entre différentes versions de packages,
- de tester des versions spécifiques de bibliothèques sans risque,
- de partager facilement le projet avec d'autres personnes,
- de garantir que le code fonctionne de la même manière sur différentes machines,
- de ne pas polluer l’installation Python globale de votre ordinateur.

C’est une pratique essentielle pour tous les projets en Machine Learning.

.. slide::

1.2. Tester si ``venv`` est disponible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le module ``venv``, inclus normalement avec Python 3, permet de créer un environnement virtuel. Avant de l’utiliser, vous pouvez vérifier s’il est installé en tapant la commande suivante dans votre terminal : 

.. code-block:: bash

   python3 -m venv --help

- Si l’aide s’affiche, le module est disponible.
- Sinon, vous verrez une erreur indiquant que ``venv`` est introuvable. Dans ce cas, installez-le avec :

.. code-block:: bash

   sudo apt install python3-venv

.. slide::

1.3. Créer l’environnement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour créer un environnement virtuel, vous pouvez utiliser la commande suivante dans votre terminal :

.. code-block:: bash

   python -m venv nom_de_l_environnement --system-site-packages
où ``nom_de_l_environnement`` est le nom que vous souhaitez donner à votre environnement virtuel et ``--system-site-packages`` permet d'accéder aux paquets installés sur votre système global (utile pour réutiliser des bibliothèques déjà installées comme ``numpy`` ou ``matplotlib``).

Placez-vous dans le dossier de travail de votre projet (par exemple ``cours_dl/``), puis créez un environnement virtuel avec :

.. code-block:: bash

   python3 -m venv nom_de_l_environnement

Cela crée un sous-dossier nommé ``nom_de_l_environnement`` contenant une version isolée de Python. Par exemple, si vous nommez votre environnement ``env_dl``, vous aurez un dossier ``env_dl`` dans votre répertoire de travail.

.. slide::

1.4. Activer l’environnement virtuel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vous devez ensuite activer l’environnement pour l’utiliser en tapant dans votre terminal :

.. code-block:: bash
    
    source nom_de_l_environnement/bin/activate 

Vous saurez que l'environnement est activé lorsque le nom de l'environnement apparaîtra entre parenthèses au début de votre invite de commande dans le terminal.

.. slide::

1.5. Désactiver l’environnement virtuel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour désactiver l'environnement virtuel, vous pouvez utiliser la commande :

.. code-block:: bash

   deactivate


.. slide::

📖 2. Installation de PyTorch
----------------------
Une fois l’environnement virtuel activé, vous pouvez installer PyTorch et les bibliothèques associées.  Mais avant d’installer PyTorch, faisons un petit point sur ce que la bibliothèque apporte. PyTorch est une bibliothèque Python très utilisée en **deep learning**.  Elle permet de :  

- créer et entraîner facilement des réseaux de neurones,  
- utiliser le GPU (quand il est disponible) pour accélérer les calculs.  

👉 Dans ce cours, PyTorch sera notre outil principal pour manipuler des données et entraîner des modèles.

.. note::

   💡 **CPU, GPU et CUDA en deux mots**

   - Un **CPU** (processeur classique) exécute bien des calculs généraux, mais il est limité pour des calculs massifs.  
   - Un **GPU** (Graphics Processing Unit), initialement conçu pour l’affichage graphique, est capable de réaliser **des milliers de calculs en parallèle** → idéal pour l’entraînement des réseaux de neurones.  
   - **CUDA** est une bibliothèque développée par NVIDIA qui permet à PyTorch de communiquer avec le GPU pour accélérer les calculs.  

   👉 Pas d’inquiétude si vous n’avez pas de GPU : PyTorch fonctionne aussi très bien sur CPU, simplement plus lentement.


.. slide::

2.1. Choisir la version de PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch propose différentes versions adaptées à divers systèmes d'exploitation et configurations matérielles (CPU, GPU). Dans ce cours, nous utiliserons la version de PyTorch compatible par défaut avec GPU. Cependant, cette version fonctionnera sur toutes les machines (avec ou sans GPU).

Dans le terminal (dans lequel l'environnement virtuel est activé), tapez :

.. code-block:: bash

   pip install torch torchvision torchaudio

Cela installera :

- **torch** : la bibliothèque principale de PyTorch,
- **torchvision** : des outils pour manipuler des images, modèles pré-entraînés, etc.,
- **torchaudio** : pour les données audios (utile pour d'autres projets).

.. slide::
2.2. Lister les paquets installés
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vous pouvez afficher la liste des bibliothèques installées dans l’environnement virtuel avec :

.. code-block:: bash

   pip freeze

Cela vous permettra de voir les versions exactes de ``torch``, ``torchvision``, etc.

.. slide::
2.3. Vérifier l’installation de PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vous pouvez maintenant tester l’installation de PyTorch avec ce petit script Python :

.. code-block:: python

   import torch

   print("Version de PyTorch :", torch.__version__)
   print("CUDA disponible ?  :", torch.cuda.is_available())

- Si l'import fonctionne sans erreur, PyTorch est installé correctement.
- Si ``torch.cuda.is_available()`` renvoie ``False``, cela signifie que votre machine n’a pas de GPU compatible CUDA ou qu'elle n'a probablement pas accès au GPU car les pilotes CUDA/cuDNN ne sont pas correctement installés.

Vous pouvez toujours utiliser PyTorch sur CPU, mais le temps d'entraînement sera plus long notamment pour les modèles complexes.

.. slide::
2.4. Installer les pilotes NVIDIA et CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour utiliser PyTorch avec un GPU, il ne suffit pas d’installer la bibliothèque ``torch``. Votre système doit aussi disposer des pilotes NVIDIA et de CUDA/cuDNN qui permettent à PyTorch de dialoguer avec la carte graphique.

.. note::

   ⚠️ **Remarque importante pour les PC de l’IUT**  

   - Sur les ordinateurs de l’IUT, **cette étape n’est pas à faire** : les pilotes NVIDIA et CUDA sont déjà installés.  
   - Cette partie est uniquement utile si vous voulez installer PyTorch avec GPU **sur votre propre ordinateur personnel** équipé d’une carte graphique NVIDIA compatible.  

.. slide::
2.4.1. Vérifier si les pilotes sont installés
~~~~~~~~~~~~~~~~~~~~~~~
Avant d'installer quoi que ce soit, vérifiez si les pilotes NVIDIA sont déjà installés sur votre système. Vous pouvez utiliser la commande suivante dans un terminal :

.. code-block:: bash

   nvidia-smi

- Si vous voyez un tableau avec des informations sur votre GPU (nom, mémoire, utilisation, version du pilote, version CUDA), cela signifie que les pilotes sont installés et fonctionnent.
- Si la commande est inconnue ou échoue, vous devez installer les pilotes.

.. slide::
2.4.2. Installer les pilotes NVIDIA 
~~~~~~~~~~~~~~~~~~~~~~~

Mettez d’abord à jour la liste des paquets, puis installez les pilotes recommandés :

.. code-block:: bash

   sudo apt update
   sudo apt install nvidia-driver-Numero_de_version
   sudo apt install nvidia-cuda-toolkit

(Le numéro de version peut varier selon votre GPU. Vous pouvez vérifier la version conseillée en tapant ``ubuntu-drivers devices`` dans un terminal. Il sera marqué "recommended" devant le pilote recommandé.)

**Redémarrez votre ordinateur après l’installation des pilotes.**

.. slide::
2.4.3. Installer CUDA et cuDNN
~~~~~~~~~~~~~~~~~~~~~~~

Dans la plupart des cas, PyTorch télécharge automatiquement les bons binaires CUDA/cuDNN avec la commande  ``pip install torch ...``.  
Il n’est donc **pas obligatoire** d’installer CUDA séparément.

Cependant, si vous souhaitez installer CUDA manuellement (option avancée qu'il vaut mieux éviter), vous pouvez télécharger l’installateur depuis : `https://developer.nvidia.com/cuda-downloads <https://developer.nvidia.com/cuda-downloads>`_

.. slide::
2.4.4. Vérifier l’installation après redémarrage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Relancez la commande :

.. code-block:: bash

   nvidia-smi

Vous devez voir apparaître les informations sur votre GPU et la version du pilote installée.  
À ce stade, PyTorch pourra utiliser le GPU si installé avec la bonne version CUDA. Pour vous en assurer, vous pouvez relancer le script Python de vérification :

.. code-block:: python

   import torch

   print("Version de PyTorch :", torch.__version__)
   print("CUDA disponible ?  :", torch.cuda.is_available())
Si ``torch.cuda.is_available()`` renvoie ``True``, PyTorch est prêt à utiliser le GPU.

Sinon, supprimez PyTorch et réinstallez-le en vous assurant de choisir la bonne version CUDA. 


.. slide::
2.4.5. Supprimer et réinstaller PyTorch avec la bonne version CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour cela, vous pouvez taper dans un terminal :
.. code-block:: bash

   pip uninstall torch torchvision torchaudio

Avant de le réinstaller, il est important de vérifier la version de CUDA supportée par votre GPU.  
Pour cela, utilisez la commande suivante :

.. code-block:: bash

   nvidia-smi

- Dans le tableau affiché, repérez la colonne **CUDA Version**.  
- Par exemple, si elle indique par exemple ``11.8``, vous devrez installer PyTorch avec ``cu118`` :

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Une fois l’installation terminée, relancez Python et vérifiez :

.. code-block:: python

   import torch

   print("Version de PyTorch :", torch.__version__)
   print("CUDA disponible ?  :", torch.cuda.is_available())
Si ``torch.cuda.is_available()`` renvoie ``True``, PyTorch est prêt à utiliser le GPU.

.. slide::
2.4.6. Erreur ``CUDA_VISIBLE_DEVICES``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Si vous obtenez l'erreur suivante ``"CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero., etc."`` après une mise en veille, il faut taper dans un terminal les commandes suivantes pour résoudre le problème : 

.. code-block:: bash
   sudo rmmod nvidia_uvm
   sudo modprobe nvidia_uvm

.. slide::
2.5. Surveiller l’utilisation du GPU avec ``nvtop``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lorsque l’on entraîne un modèle de deep learning sur GPU, il est souvent utile de **visualiser en temps réel** l’utilisation de la carte graphique (mémoire, charge de calcul, processus en cours).

Pour cela, vous pouvez installer l’outil ``nvtop`` :

.. code-block:: bash

   sudo apt install nvtop

Ensuite, lancez la commande :

.. code-block:: bash

   nvtop

Vous verrez une interface en temps réel indiquant :  

- l’occupation de la mémoire GPU,  
- l’utilisation du GPU par processus,  
- la charge globale.  

👉 C’est l’équivalent de la commande ``top`` mais pour le GPU. Cette commande est très utile pour vérifier que **PyTorch utilise bien votre carte graphique** lors des entraînements.


.. slide::
📖 3. Installer Jupyter Notebook (optionnel mais recommandé)
----------------------

Pour coder les TPs, vous pouvez utiliser VSCode ou Jupyter Notebook.  Jupyter Notebook est un environnement interactif très utilisé en Python, idéal pour le deep learning. Il permet d’exécuter du code par blocs, de visualiser les résultats immédiatement et de documenter le travail dans le même fichier.

Pour installer Jupyter, vous devrez d'abord vous assurer que l'environnement virtuel est activé, puis exécuter la commande suivante :

.. code-block:: bash

   pip install notebook

L’installation inclut ``notebook`` ainsi que tous les outils nécessaires pour exécuter des blocs Python.

.. slide::
3.1. Lancer Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Créer un dossier pour les notebooks, par exemple ``cours_dl/notebooks/``.
Ensuite, placez-vous dans ce dossier avec la commande ``cd`` :
.. code-block:: bash

   cd cours_dl/notebooks/  

Pour démarrer Jupyter Notebook dans le dossier courant, assurez-vous d'avoir activé l'environnement virtuel, puis tapez :

.. code-block:: bash

   jupyter notebook

- Un navigateur web s’ouvrira automatiquement avec l’interface de Jupyter illustrée par la figure ci-dessous.  
- Si le navigateur ne s’ouvre pas, copiez-collez l’URL affichée dans le terminal dans votre navigateur préféré.


.. center::
    .. image:: images/jupyter_workspace.png
        :alt: Espace de travail Jupyter

.. slide::
3.2 Créer un notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Cliquez sur **New → Python 3** pour créer un nouveau notebook comme illustré par la figure ci-dessous.  

.. center::
    .. image:: images/jupyter_new_file.png
        :alt: Créer un nouveau notebook Jupyter

- Chaque cellule de la figure ci-dessous peut contenir du code Python que vous pouvez exécuter avec ``Shift + Enter``.  

.. center::
    .. image:: images/jupyter_home_page.png
        :alt: Un Jupyter Notebook vide

- Vous pouvez aussi ajouter des cellules Markdown pour documenter vos explications.

.. slide::
3.3 Vérification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour vérifier que Jupyter utilise bien votre virtualenv avec PyTorch installé, créez une cellule et tapez :

.. code-block:: python

   import torch
   print("Version de PyTorch :", torch.__version__)
   print("CUDA disponible ? :", torch.cuda.is_available())

Si tout est correct, vous devriez voir la version de PyTorch et l'état de CUDA s'afficher sans erreur.

.. center::
    .. image:: images/jupyter_test_ve.png
        :alt: Vérification de l'installation de PyTorch dans Jupyter 


.. slide::
3.4 Renommer le notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pour renommer le notebook, cliquez sur le nom par défaut (généralement ``Untitled``) en haut à gauche, puis entrez un nouveau nom, par exemple ``test_installation.ipynb``.
Cela vous permettra de garder une trace de vos notebooks et de les organiser facilement.
.. center::
    .. image:: images/jupyter_rename.png
        :alt: Renommer le notebook Jupyter

.. center::
    .. image:: images/jupyter_after_rename.png
        :alt: Après renommage du notebook Jupyter

.. slide::
📖 4. Documentation utile
----------------------

Pour approfondir vos connaissances et trouver des réponses rapides, voici quelques ressources fiables et pertinentes pour ce cours :

4.1 PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Documentation officielle : `https://pytorch.org/docs/stable/index.html <https://pytorch.org/docs/stable/index.html>`_
- Tutoriels PyTorch : `https://pytorch.org/tutorials/ <https://pytorch.org/tutorials/>`_
- Guide “Get Started” avec CUDA : `https://pytorch.org/get-started/locally/ <https://pytorch.org/get-started/locally/>`_ 

4.2 Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- AOP Python : `https://cgaspard3333.github.io/intro-python/ <https://cgaspard3333.github.io/intro-python/>`_
- Documentation Python : `https://docs.python.org/3/ <https://docs.python.org/3/>`_


4.3 Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Documentation Jupyter : `https://jupyter.org/documentation <https://jupyter.org/documentation>`_
- Tutoriel Jupyter Notebook : 

`https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html <https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html>`_

- Commandes utiles et raccourcis : 

`https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html#keyboard-shortcuts <https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html#keyboard-shortcuts>`_

4.4 CUDA / NVIDIA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Vérification des pilotes et CUDA : `https://developer.nvidia.com/cuda-toolkit <https://developer.nvidia.com/cuda-toolkit>`_
- Informations sur les GPU NVIDIA : `https://developer.nvidia.com/cuda-gpus <https://developer.nvidia.com/cuda-gpus>`_
- Guide d'installation de CUDA : `https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_

4.5 Communauté et forums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Stack Overflow (tag ``python``) : `https://stackoverflow.com/questions/tagged/python <https://stackoverflow.com/questions/tagged/python>`_
- PyTorch Forums : `https://discuss.pytorch.org/ <https://discuss.pytorch.org/>`_

Ces liens vous permettront de consulter des exemples, de comprendre les erreurs et de rester à jour sur les dernières fonctionnalités.
