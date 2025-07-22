"""
do_hand_eye_calib.py - Traite les données de hand_eye_data.yaml.
Renvoie deux fichiers yaml : hand_eye_data.yaml et intrinsics.yaml

Auteur : Alban CASELLA et Suzanne-Léonore GIRARD-JOLLET
Date : Juin 2025
Description : Ce script traite les données du fichier hand_eye_data.yaml pour créer un fichier hand_eye.yaml utiliser dans la lecteure des ArUco.
"""
import yaml
import numpy as np
from pick_and_place_system import calibrate_hand_eye

# S'assurer que numpy est importé dans ce script aussi

# 1) Charge les données brutes
data = yaml.safe_load(open('/home/robot/Bureau/Stage/electromob/testCamera/testn°4/hand_eye_data.yaml'))
R_list  = data['R_gripper2base_list']
t_list  = data['t_gripper2base_list']
R2_list = data['R_target2cam_list']
t2_list = data['t_target2cam_list']

print(f"Nombre de poses collectées: {len(R_list)}")

# 2) Calibre en mode Tsai–Lenz et écrit hand_eye.yaml
T_cam2flange = calibrate_hand_eye(
    R_list, t_list,
    R2_list, t2_list,
    'testn°4/hand_eye.yaml'
)

# Affichage des résultats
print("\nTransformation caméra → flange:")
print(f"Translation: X={T_cam2flange[0,3]:.4f}, Y={T_cam2flange[1,3]:.4f}, Z={T_cam2flange[2,3]:.4f}")
print("Matrice de rotation:")
print(T_cam2flange[:3,:3])