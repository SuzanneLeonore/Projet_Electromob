"""
prisePhoto.py - Capture d'images avec détection de CharucoBoard via caméra RealSense

Description :
Ce script affiche en direct le flux vidéo d'une caméra RealSense, détecte la planche Charuco
à l'aide d'un ArucoDetector, et permet de capturer et sauvegarder les images en appuyant sur 'c'.
Utilisé pour créer un jeu d’images pour calibration ou analyse visuelle.

Auteur : Alban CASELLA & Suzanne-Léonore GIRARD-JOLLET
Date : Juin 2025
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import yaml
from pick_and_place_system import ArucoDetector, load_charuco

# --- Initialisation du pipeline RealSense ---
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)

# --- Chargement ou génération d'intrinsèques caméra ---
intrinsics_file = 'intrinsics.yaml'
detector = None

if os.path.exists(intrinsics_file):
    detector = ArucoDetector(intrinsics_file)
    print("Intrinsics chargés. La détection du CharucoBoard sera affichée.")
else:
    # Création d'intrinsèques estimés si aucun fichier trouvé
    temp_intrinsics = {
        'camera_matrix': [
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ],
        'dist_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],
        'charuco_params': {
            'board_squares_x': 7,
            'board_squares_y': 5,
            'square_length': 0.04,
            'marker_length': 0.02,
            'aruco_dict': 'DICT_4X4_50'
        }
    }
    with open('temp_intrinsics.yaml', 'w') as f:
        yaml.dump(temp_intrinsics, f)
    detector = ArucoDetector('temp_intrinsics.yaml')
    print("Utilisation d'intrinsèques estimés. La détection pourrait être imprécise.")

# --- Interface utilisateur ---
idx = 0
print("Appuie sur 'c' pour capturer une image, 'Esc' pour quitter.")

try:
    while True:
        # Lecture de l'image depuis la caméra
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())

        # Détection du CharucoBoard
        ok, rvec, tvec = detector.detect(img)
        display_img = detector.draw_detection(img)

        if ok:
            # Si la pose est détectée, afficher les axes
            cv2.drawFrameAxes(display_img, detector.camera_matrix, detector.dist_coeffs,
                              rvec, tvec, 0.05)
            cv2.putText(display_img, "DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Afficher le nombre de marqueurs détectés
            detected_markers = len(detector.corners) if detector.corners is not None else 0
            cv2.putText(display_img, f"Markers: {detected_markers}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Affichage du nombre d'images sauvegardées
        cv2.putText(display_img, f"Images: {idx}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Affichage dans une fenêtre OpenCV
        cv2.imshow("Calibration Capture", display_img)
        key = cv2.waitKey(1) & 0xFF

        # Touche 'c' pour capturer une image
        if key == ord('c'):
            path = f"/home/robot/Bureau/Stage/electromob/testCamera/image/{idx:03d}.jpg"
            cv2.imwrite(path, img)
            print(f"[+] Sauvegardé {path}")
            idx += 1

        # Touche 'Esc' pour quitter
        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    # Nettoyage du fichier temporaire si nécessaire
    if not os.path.exists(intrinsics_file) and os.path.exists('temp_intrinsics.yaml'):
        os.remove('temp_intrinsics.yaml')
