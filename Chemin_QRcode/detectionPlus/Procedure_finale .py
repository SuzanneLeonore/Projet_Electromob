"""
Procedure_final.py - Programme principal du TP robotique UR5e + vision

Ce script pilote un système robotisé comprenant :
- Un bras UR5e commandé via RTDE
- Une pince pour la manipulation
- Une caméra détectant des marqueurs ArUco
- Un système de communication MQTT avec des capteurs (convoyeur et bacs)

Le système fonctionne à l'aide de plusieurs threads :
1. Réception des capteurs via MQTT
2. Contrôle du robot et déclenchement automatique de séquences
3. (Optionnel) Détection ArUco en boucle

Auteur : Alban CASELLA et Suzanne-Léonore GIRARD-JOLLET
Date : Juillet 2025
"""

import json
import yaml
import paho.mqtt.client as mqtt
import time
import threading
import numpy as np 
from marqueurAruco import ArucoURController

# Données partagées entre threads
convoyeur_data = [None] * 2  # État des capteurs convoyeur
bac_data = [None] * 5        # État des capteurs bacs

# Variable globale contenant toutes les données capteurs + verrou
global_out = None
out_lock = threading.Lock()

# Index du bac disponible (calculé dynamiquement)
emplacement_bac_libre = None
position = None

##########################################
# Partie 1 : MQTT - Communication capteurs
##########################################

# Paramètres de connexion au broker MQTT
MQTT_BROKER = "10.2.30.162"
MQTT_PORT = 1883
TOPIC_B = "capteurs_bac/etat"
TOPIC_C = "capteurs_convoyeur/etat"

def on_connect(client, userdata, flags, rc):
    """Callback exécuté lors de la connexion au broker MQTT."""
    print("Connecté au broker MQTT avec le code de retour", rc)
    client.subscribe(TOPIC_B)
    client.subscribe(TOPIC_C)

def on_message(client, userdata, msg):
    """Callback exécuté à la réception d'un message MQTT.
    Met à jour les variables convoyeur_data, bac_data et global_out.
    """
    global convoyeur_data, bac_data, global_out
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        if msg.topic == TOPIC_C:
            convoyeur_data = [data["pin1"], data["pin2"]]
        elif msg.topic == TOPIC_B:
            bac_data = [data[f"pin{i+1}"] for i in range(5)]

        with out_lock:
            global_out = [convoyeur_data, bac_data]
        print("Données MQTT reçues :", global_out)

    except Exception as e:
        print("Erreur lors du traitement du message :", e)

def mqtt_client_thread():
    """Thread principal MQTT : établit la connexion et reste en écoute permanente."""
    client = mqtt.Client("PythonClient")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

##########################################
# Partie 2 : Contrôle robot via RTDE
##########################################

def robot_control_thread():
    """Thread de contrôle du robot UR5e.
    Lit les capteurs, détecte un déclenchement, et exécute les mouvements via ArUcoURController.
    """
    aruco = ArucoURController()

    try:
        with open('/home/alban/Stage-Electromob/config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
            positions = np.array(config_data['position'], dtype=np.float32)
            rotations = np.array(config_data['rotation'], dtype=np.float32)
    except Exception as e:
        raise

    pos_bac = np.hstack((positions[0], rotations[0]))

    dist_slots = 0.05  # Distance entre emplacements dans le bac

    try:
        while True:
            with out_lock:
                current_data = global_out

            if global_out[0][0] is not None and global_out[1][0] is not None:
                if current_data[0][0] == 1 and not aruco.DoAction:
                    emplacement_bac_libre, nb_emplacements = disponible(global_out[1])

                    if emplacement_bac_libre != -1:
                        pos_bac[0] += (emplacement_bac_libre - 2) * dist_slots

                    aruco.DoAction = True
                    print("Déclenchement de la séquence de mouvements.")
                    aruco.run()
                    print("Séquence de mouvements terminée.")

                elif current_data[0][0] == 0 and aruco.DoAction:
                    aruco.DoAction = False
                    print("Réinitialisation de l'état de déclenchement.")

            time.sleep(0.1)

    except Exception as e:
        print("Erreur lors de la communication avec le robot via RTDE :", e)

def disponible(l):
    """Renvoie le premier emplacement bac disponible.

    Args:
        l (list): liste d'états des 5 capteurs bac
    Returns:
        (int, int): index du bac libre, nombre total
    """
    out = [i for i, val in enumerate(l) if not val]
    return (out[0], len(l)) if out else (-1, 0)

##########################################
# Partie 3 : Détection ArUco (optionnelle)
##########################################

def detect_aruco_thread():
    """Thread de détection ArUco (non activé ici)."""
    aruco_controller = ArucoURController()
    aruco_controller.run_demo()

##########################################
# Démarrage du programme principal
##########################################

if __name__ == "__main__":
    """Point d'entrée principal : lance les threads MQTT et robot."""
    mqtt_thread = threading.Thread(target=mqtt_client_thread, daemon=True)
    mqtt_thread.start()

    time.sleep(3)  # Attente connexion MQTT

    robot_thread = threading.Thread(target=robot_control_thread, daemon=True)
    robot_thread.start()
    robot_thread.join()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Arrêt du programme.")