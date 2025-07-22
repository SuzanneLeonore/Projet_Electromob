"""
calibration_hand_eye_data.py - Collecte manuelle de données de calibration hand-eye

Description :
Ce script permet de capturer manuellement les poses du robot et les poses de la planche Charuco
(détectée via caméra RealSense) pour une future calibration hand-eye. L'utilisateur peut
interactivement capturer, supprimer ou sauvegarder les échantillons collectés.

Les données sont enregistrées dans un fichier YAML.

Auteur : Alban CASELLA et Suzanne-Léonore GIRARD-JOLLET
Date : Juin 2025
"""

import cv2
import numpy as np
import yaml
import pyrealsense2 as rs
import rtde_receive
import time
from pick_and_place_system import ArucoDetector

# --- Configuration ---
ROBOT_IP = "10.2.30.60"
detector = ArucoDetector('/home/robot/Bureau/Stage/electromob/testCamera/intrinsics_v2.yaml')
rtde_recv = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# --- Initialisation RealSense ---
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)

# --- Listes pour stocker les poses ---
R_gripper2base_list = []
t_gripper2base_list = []
R_target2cam_list = []
t_target2cam_list = []

# --- Instructions utilisateur ---
print("\n=== Hand–eye calibration data collection ===\n")
print("CONSEILS POUR UNE BONNE CALIBRATION:")
print("1. Déplacez le robot pour voir le CharucoBoard sous différents angles")
print("2. Collectez au moins 20 positions différentes")
print("3. Variez les rotations ET translations du robot")
print("4. Assurez-vous que la planche est bien détectée à chaque capture\n")

print("COMMANDES:")
print("  'c' - capturer un échantillon")
print("  'd' - supprimer le dernier échantillon")
print("  's' - sauvegarder et quitter")
print("  'q' - quitter sans sauvegarder\n")

cv2.namedWindow("Calibration Capture", cv2.WINDOW_NORMAL)

try:
    sample_count = 0
    last_capture_time = 0

    while True:
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())

        # Détection de la planche ArUco
        ok, rvec, tvec = detector.detect(img)
        display_img = detector.draw_detection(img)

        # Affichage nombre d'échantillons et messages
        cv2.putText(display_img, f"Echantillons: {sample_count}/20+", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if ok:
            cv2.drawFrameAxes(display_img, detector.camera_matrix, detector.dist_coeffs,
                              rvec, tvec, 0.05)
            cv2.putText(display_img, "DETECTION OK", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_img, "DETECTION INCOMPLETE", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Instructions de commande clavier
        y_offset = 90
        for cmd, desc in zip(['c', 'd', 's', 'q'],
                             ["capturer", "supprimer dernier", "sauvegarder & quitter", "quitter sans sauvegarder"]):
            cv2.putText(display_img, f"{cmd} - {desc}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        # Affichage position actuelle du TCP robot
        tcp = rtde_recv.getActualTCPPose()
        cv2.putText(display_img, f"TCP: X={tcp[0]:.3f} Y={tcp[1]:.3f} Z={tcp[2]:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25
        cv2.putText(display_img, f"Rot: Rx={tcp[3]:.3f} Ry={tcp[4]:.3f} Rz={tcp[5]:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Affichage de l’image
        cv2.imshow("Calibration Capture", display_img)
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()

        # --- Capture échantillon ---
        if key == ord('c') and ok and (current_time - last_capture_time > 1.0):
            R_fb, _ = cv2.Rodrigues(np.array(tcp[3:6]))
            t_fb = np.array(tcp[:3])
            R_tc, _ = cv2.Rodrigues(rvec)
            t_tc = tvec.flatten()

            R_gripper2base_list.append(R_fb.tolist())
            t_gripper2base_list.append(t_fb.tolist())
            R_target2cam_list.append(R_tc.tolist())
            t_target2cam_list.append(t_tc.tolist())

            sample_count += 1
            last_capture_time = current_time
            print(f"[+] Échantillon {sample_count} capturé")

        # --- Supprimer dernier échantillon ---
        elif key == ord('d') and sample_count > 0:
            R_gripper2base_list.pop()
            t_gripper2base_list.pop()
            R_target2cam_list.pop()
            t_target2cam_list.pop()
            sample_count -= 1
            print(f"[-] Échantillon supprimé, reste {sample_count}")

        # --- Sauvegarder et quitter ---
        elif key == ord('s'):
            if sample_count >= 5:
                break
            else:
                print("Collectez au moins 5 échantillons avant de sauvegarder")

        # --- Quitter sans sauvegarder ---
        elif key == ord('q'):
            if sample_count > 0:
                confirm = input("Quitter sans sauvegarder? (o/N): ").lower()
                if confirm == 'o':
                    print("Sortie sans sauvegarde")
                    exit(0)
            else:
                print("Sortie sans sauvegarde")
                exit(0)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# --- Vérification et sauvegarde YAML ---
if sample_count < 5:
    print("\nPas assez d'échantillons collectés (minimum 5). Données non sauvegardées.")
    exit(1)

data = {
    'R_gripper2base_list': R_gripper2base_list,
    't_gripper2base_list': t_gripper2base_list,
    'R_target2cam_list': R_target2cam_list,
    't_target2cam_list': t_target2cam_list
}

with open('testn\u00b04/hand_eye_data.yaml', 'w') as f:
    yaml.dump(data, f)

print(f"\nCalibration data saved to testn\u00b04/hand_eye_data.yaml ({sample_count} échantillons)")
print("\nPour effectuer la calibration, exécutez maintenant:")
print("from pick_and_place_system import calibrate_hand_eye")
print("import yaml\n")
print("with open('testn\u00b04/hand_eye_data.yaml', 'r') as f:")
print("    data = yaml.safe_load(f)\n")
print("calibrate_hand_eye(")
print("    data['R_gripper2base_list'],")
print("    data['t_gripper2base_list'],")
print("    data['R_target2cam_list'],")
print("    data['t_target2cam_list'],")
print("    'testn\u00b04/hand_eye.yaml'")
print(")")