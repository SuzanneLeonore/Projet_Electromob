"""
calibration_auto.py - Capture automatique des poses robot/caméra pour calibration Hand-Eye

Description :
Ce script utilise un robot UR et une caméra RealSense pour capturer automatiquement des positions
et orientations relatives à une cible ArUco placée devant l'objet. Les mouvements sont générés
automatiquement le long de deux arcs (plans XZ et XY) pour permettre une collecte variée des poses.
Les données sont sauvegardées dans un fichier YAML pour une future calibration de type Eye-to-Hand.

Auteur : Alban CASELLA & Suzanne-Léonore GIRARD-JOLLET
Date : Juin 2025
"""

import math
import math3d as m3d
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import cv2
import yaml
import pyrealsense2 as rs
from pick_and_place_system import ArucoDetector

# --- Configuration ---
ROBOT_IP = "10.2.30.60"
rtde_c = RTDEControl(ROBOT_IP)
rtde_r = RTDEReceive(ROBOT_IP)

detector = ArucoDetector('intrinsics.yaml')

# --- Initialisation de la caméra RealSense ---
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)

# Prend une première image pour valider la détection ArUco
frames = pipeline.wait_for_frames()
color = frames.get_color_frame()
img = np.asanyarray(color.get_data())
ok, rvec, tvec = detector.detect(img)
display_img = detector.draw_detection(img)

# --- Listes de collecte de données ---
R_gripper2base_list = []
t_gripper2base_list = []
R_target2cam_list = []
t_target2cam_list = []

# --- Paramètres de prise de vue ---
pose_object = rtde_r.getActualTCPPose()[:3]
center = m3d.Vector(pose_object)  # Position cible centrée sur l'objet
radius = 0.3                      # Rayon des arcs de prise de vue
nb_photos = 5                     # Nombre de points par arc
angle_passage = [2*math.pi/3, math.pi/3, 0, -math.pi/3, -2*math.pi/3]  # Angles équidistants

def look_at(pose_cam, target, up=[0, 1, 0]):
    """Calcule l'orientation pour que la caméra regarde un point cible donné."""
    pose_cam = np.array(pose_cam, dtype=float).flatten()
    target = np.array(target, dtype=float).flatten()
    up = np.array(up, dtype=float).flatten()

    forward = target - pose_cam
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)
    rot_matrix = np.column_stack((right, true_up, forward))

    return m3d.Orientation(rot_matrix)

# --- Première boucle : arc dans le plan XZ ---
for i in range(nb_photos):
    theta = angle_passage[i]
    x = center[0] + radius * math.cos(theta)
    y = center[1]  # constant (plan XZ)
    z = center[2] + radius * math.sin(theta)

    position = m3d.Vector([x, y, z])
    rotation = look_at(list(position), list(center))
    pose = m3d.Transform(rotation, position)

    # Conversion rotation -> vecteur d'axe-angle pour RTDE
    r = R.from_matrix(np.array(pose.orient.matrix))
    angle = r.magnitude()
    axis = r.as_rotvec() / angle if angle != 0 else np.array([1.0, 0.0, 0.0])

    pose_rtde = [pose.pos.x, pose.pos.y, pose.pos.z,
                 axis[0]*angle, axis[1]*angle, axis[2]*angle]

    # Déplacement du robot
    rtde_c.moveL(pose_rtde, speed=0.1, acceleration=0.1)

    # Collecte des poses robot et caméra
    tcp = rtde_r.getActualTCPPose()
    R_fb, _ = cv2.Rodrigues(np.array(tcp[3:6]))
    t_fb = np.array(tcp[:3])
    R_tc, _ = cv2.Rodrigues(rvec)
    t_tc = tvec.flatten()

    R_gripper2base_list.append(R_fb.tolist())
    t_gripper2base_list.append(t_fb.tolist())
    R_target2cam_list.append(R_tc.tolist())
    t_target2cam_list.append(t_tc.tolist())

    time.sleep(1)

# --- Deuxième boucle : arc dans le plan XY ---
for i in range(nb_photos):
    theta = angle_passage[i]
    x = center[0] + radius * math.cos(theta)
    y = center[1] + radius * math.sin(theta)
    z = center[2]  # constant (plan XY)

    position = m3d.Vector([x, y, z])
    rotation = look_at(list(position), list(center), [0, 0, 1])
    pose = m3d.Transform(rotation, position)

    r = R.from_matrix(np.array(pose.orient.matrix))
    angle = r.magnitude()
    axis = r.as_rotvec() / angle if angle != 0 else np.array([1.0, 0.0, 0.0])

    pose_rtde = [pose.pos.x, pose.pos.y, pose.pos.z,
                 axis[0]*angle, axis[1]*angle, axis[2]*angle]

    rtde_c.moveL(pose_rtde, speed=0.1, acceleration=0.1)

    tcp = rtde_r.getActualTCPPose()
    R_fb, _ = cv2.Rodrigues(np.array(tcp[3:6]))
    t_fb = np.array(tcp[:3])
    R_tc, _ = cv2.Rodrigues(rvec)
    t_tc = tvec.flatten()

    R_gripper2base_list.append(R_fb.tolist())
    t_gripper2base_list.append(t_fb.tolist())
    R_target2cam_list.append(R_tc.tolist())
    t_target2cam_list.append(t_tc.tolist())

    time.sleep(1)

# --- Sauvegarde des données dans un fichier YAML ---
data = {
    'R_gripper2base_list': R_gripper2base_list,
    't_gripper2base_list': t_gripper2base_list,
    'R_target2cam_list': R_target2cam_list,
    't_target2cam_list': t_target2cam_list
}

with open('hand_eye_data.yaml', 'w') as f:
    yaml.dump(data, f)

# --- Nettoyage final ---
pipeline.stop()
cv2.destroyAllWindows()
rtde_c.close()
