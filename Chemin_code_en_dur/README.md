# Mission electromob : Inscription du mouvement en dur

## Description
Ce projet a pour objectif de permettre aux élèves de la structure IMT Nord europe de manipuler un bras robotique **UR5e** a travers la réalisation d'un **travaux pratique** (TP). 
L'objectif du TP est d'utiliser le bras robotique UR5e ainsi qu'une pince **2GF7** pour transporter une batterie à différents endroits d'un parcours. Pour réaliser le parcours, on inscrit en dur les différentes commandes de mouvement pour définir les positions cibles. Des **capteurs interrupteurs à levier** permettent de détecter la position de la cellule sur le parcours.

Le projet est composé de **3 fichiers Python** pour le mouvement:
 - `robot.py` : gère les mouvement du robot UR5e,
 - `pince.py` : gère les commandes d'ouverture et de fermeture de la pince 2GF7,
 - `procedure_finale.py` : écoute les informations des capteurs et permet de lancer le script de mouvement.

## Installation

Pour faire tourner les scripts, veillez à avoir les modules et biblothèques suivant d'installés :
 - rtde_receive & rtde_control 
    [télécharger le SDK] (https://www.universal-robots.com/download/) et lire le README associé
 - numpy
 ```bash
 pip install numpy 
 ```


## Utilisation
Avant de lancer le script permettant le déplacement, assurez vous d'avoir un script de calibration correct. Pour lancer une nouvelle calibration, il faut lancer le script `calibration.py` qui se situe dans le dossier Calibration.
Pour lancer le TP il faut lancer le script `procedure_finale.py`, le mouvement commencera lorsque le capteur positioné sur la barière du convoyeur sera actionné.

## Contribuer
Guide pour ceux qui souhaitent améliorer le projet.

## Auteurs et remerciements
Réalisé au sein du CERI SN d'IMT Nord Europe par Alban CASELLA et Suzanne-Léonore GIRARD-JOLLET.

