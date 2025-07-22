"""
calibration.py - Script de calibration pour le robot UR5e.

Ce script permet d'enregistrer les positions TCP du robot pour définir :
- P0, P1, P2 : points de repère pour construire le repère local
- injecteur : position de l’injecteur
- box : position de la boîte

Les positions sont enregistrées dans un fichier JSON utilisé par robot.py.
Auteur : Alban CASELLA et Suzanne-Léonore GIRARD-JOLLET
Date : Mai 2025
"""

import json
from robot import Robot

def enregistrer_pose(nom):
    """Demande à l'utilisateur de positionner le robot, puis lit la position TCP actuelle."""
    input(f"\nPlace le robot à la position '{nom}' puis appuie sur Entrée...")
    robot = Robot()
    pose = robot.get_position()[:3]  # On récupère seulement les coordonnées x, y, z
    print(f"{nom} enregistré : {pose}")
    return pose

def main():
    print("=== Calibration des points de repère ===")
    points = {}
    for point in ["P0", "P1", "P2", "injecteur", "box"]:
        points[point] = enregistrer_pose(point)

    # Sauvegarder les données dans un fichier JSON
    with open("calibration_data.json", "w") as f:
        json.dump(points, f, indent=4)
    
    print("\n Calibration terminée.")
    print("Les données ont été enregistrées dans le fichier 'calibration_data.json'.")

if __name__ == "__main__":
    main()
