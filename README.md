####################
### Projet audio ###
####################
@auteur : Olivier VU THANH & Loïc ARGENTIER, élève ingénieurs en 3ème année à ENSE3/PHELMA option SICOM

SUJET : Deep learning appliqué au débruitage de signaux de parole

Avant toute chose, vous devez installer :
- le framework pytorch : https://pytorch.org/get-started/locally/
- l'api cuda de NVIDIA : https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

Le modèle du réseau se trouve dans la classe : codes/twoLayerNet.py
La mise en forme des données se fait via la classe : codes/speechdataset.py

Pour train le modèle : éxécuter le script main.py
Pour tester le modèle : éxécuter le script load_model.py
Pour repdrendre l'entrainement : éxécuter le script reprise_train.py
Pour les deux derniers sript il faut bien évidemment fournir le chemin des modèles et optizer sauvegarder dans le dossier saved

/!\ ATTENTION /!\

remarque : Pendant l'entrainement, et les tests une figure s'ouvre et affiche en temps réel l'évolution du MSE et PSNR en fonction du nombre d'iétration. Il ne faut pas la fermer ! sinon crash du code.

/!\ ATTENTION /!\
