# Mission electromob : Utilisation de la technologie des marqueurs ArUco

## Description
Ce projet a pour objectif de permettre aux élèves de la structure IMT Nord europe de manipuler un bras robotique **UR5e** a travers la réalisation d'un **travaux pratique** (TP). 
L'objectif du TP est d'utiliser le bras robotique UR5e ainsi qu'une pince **2GF7** pour transporter une batterie à différents endroits d'un parcours. Pour réaliser le parcours, on utilise la technologie des **marqueurs ArUco** pour définir les positions cibles. Des **capteurs interrupteurs à levier** permettent de détecter la position de la cellule sur le parcours.

Le projet est composé de **4 fichiers Python** :
 - `robot.py` : gère les mouvement du robot UR5e,
 - `pince.py` : gère les commandes d'ouverture et de fermeture de la pince 2GF7,
 - `marqueurArUco.py` : gère la détection des marqueurs ArUco et l'utilisation de leurs informations,
 - `procedure_finale.py` : écoute les informations des capteurs et permet de lancer le script de mouvement.

Pour reconnaître les marqueurs, on utilise une **caméra Intel RealSense** fixée sur le robot.

## Installation

Pour faire tourner les scripts, veillez à avoir les modules et biblothèques suivant d'installés :
 - rtde_receive & rtde_control 
    [télécharger le SDK] (https://www.universal-robots.com/download/) et lire le README associé
 - numpy
 - cv2
 - pyrealsense2
 - yaml
 ```bash
 pip install numpy opencv-python pyrealsense2 pyyaml
 ```

Pour utiliser la caméra il faut la calibrer. Pour ça on utilise une grille d'ArUco et les scripts de calibration `calibration_hand_eye_data.py` et `do_hand_eye_calib.py`. Avec le premier script prenez une vingtaine de photos  de la grille sous différents angles et on obtient des paramétres nécessaire au fonctionnement du second fichier. Le second script, fournit des attributs au code de détection des ArUco.

## Utilisation

Pour lancer le TP il faut lancer le script `procedure_finale.py`, le mouvement commencera lorsque le capteur positioné sur la barière du convoyeur sera actionné.

## Contribuer
Guide pour ceux qui souhaitent améliorer le projet.

## Auteurs et remerciements
Réalisé au sein du CERI SN d'IMT Nord Europe par Alban CASELLA et Suzanne-Léonore GIRARD-JOLLET.

