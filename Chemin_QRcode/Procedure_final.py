import json
import paho.mqtt.client as mqtt
import threading
import time
from pince import Pince
from robot import Robot
from marqueurAruco_V3 import ArucoURController

# Variables globales pour partager les données
convoyeur_data = [None] * 2  # Pour capteurs_convoyeur
bac_data = [None] * 5        # Pour capteurs_bac

# Variable globale "global_out" accessible de partout dans le programme,
# ainsi qu'un verrou pour protéger l'accès concurrent
global_out = None
out_lock = threading.Lock()
emplacement_bac_libre = None

##########################################
# Partie 1 : MQTT
##########################################

# Paramètres MQTT
MQTT_BROKER = "10.2.30.162"
MQTT_PORT = 1883
TOPIC_B = "capteurs_bac/etat"
TOPIC_C = "capteurs_convoyeur/etat"

def on_connect(client, userdata, flags, rc):
    print("Connecté au broker MQTT avec le code de retour", rc)
    client.subscribe(TOPIC_B)
    client.subscribe(TOPIC_C)

def on_message(client, userdata, msg):
    global convoyeur_data, bac_data, global_out
    try:
        # Décoder le message JSON
        data = json.loads(msg.payload.decode('utf-8'))
        
        # Mettre à jour les données en fonction du topic
        if msg.topic == TOPIC_C:  # capteurs_convoyeur/etat (2 pins)
            convoyeur_data = [data["pin1"], data["pin2"]]
        elif msg.topic == TOPIC_B:  # capteurs_bac/etat (5 pins)
            bac_data = [data[f"pin{i+1}"] for i in range(5)]
        
        # Mettre à jour la variable globale "global_out" de façon sécurisée
        with out_lock:  # Début de la section critique
            global_out = [convoyeur_data, bac_data]
        
        print("Données MQTT reçues :", global_out)
    
    except Exception as e:
        print("Erreur lors du traitement du message :", e)

def mqtt_client_thread():
    client = mqtt.Client("PythonClient")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    # Boucle bloquante dans un thread séparé
    client.loop_forever()

##########################################
# Partie 2 : Contrôle du robot via rtde
##########################################

#def detect_aruco_thread():
    
    #aruco_controller.run_demo()

def robot_control_thread():

    robot = Robot()
    pince = Pince()
    aruco_controller = ArucoURController()

    triggered = False

    actions = [
        "robot.bougerJ(robot.pose_init_Q)",
        "pince.lacher()",
        "aruco_controller.balisation([3,2], 0, -0.0872665)",
        "aruco_controller.rotation([3,2])",
        #"aruco_controller.rotation_yaw([3,2])",
        "aruco_controller.update_cam_to_base_transform()",
        "aruco_controller.deplacementPose([3,2],[0.005,0.04,-0.006])",
        "pince.prise()",
        "robot.bougerL(robot.move_actual_pose(1, -0.2))",
        "robot.bougerL(robot.move_actual_pose(2, 0.2))",
        "robot.bougerJ(robot.joints[3])",
        "aruco_controller.balisation([0], 1, -0.05,0)",
        "aruco_controller.deplacementPose([0],[0.0,0.0,0.4])",
        "aruco_controller.balisation([20,21],0, 0.174533 )",
        "aruco_controller.deplacementPose([20,21],[0.0,0.0,0.4])",
        "aruco_controller.rotation_yaw([20,21])",
        "aruco_controller.rotation([20,21])",
        "aruco_controller.rotation_yaw([20,21])",
        "aruco_controller.rotation_yaw([20,21])",
        "aruco_controller.deplacementPose([20,21],[0.16,0,0.4])",
        "aruco_controller.deplacementPose([20,21],[0.16,0,0.063])",
        "robot.bougerL(robot.move_actual_pose(2, 0.3))",
        "aruco_controller.deplacementPose([20,21],[0.09,0.0,0.35])",
        "aruco_controller.deplacementPose([20,21],[0.09,0.005,0.20])",
        "pince.lacher()",
        "robot.bougerL(robot.move_actual_pose(2, 0.2))",
        "aruco_controller.deplacementPose([20,21],[0.00,0,0.06])",
        "pince.prise()",
        "robot.bougerL(robot.move_actual_pose(2, 0.35))",
        "aruco_controller.deplacementPose([20,21],[0.09,0.005,0.35])",
        "aruco_controller.deplacementPose([20,21],[0.09,0.005,0.165])",
        "robot.bougerL(robot.move_actual_pose(2, 0.25))",
        "aruco_controller.deplacementPose([20,21],[0.09,0.02,0.35])",
        "aruco_controller.deplacementPose([20,21],[0.012,0.015,0.35])",
        "aruco_controller.deplacementPose([20,21],[0.012,0.015,0.09])",
        "pince.lacher()",
        "robot.bougerL(robot.move_actual_pose(2, 0.35))",
        "aruco_controller.deplacementPose([20,21],[0.09,0,0.35])",
        "aruco_controller.deplacementPose([20,21],[0.085,0.013,0.085])",
        "pince.prise()",
        "robot.bougerL(robot.move_actual_pose(2, 0.3))",
        "aruco_controller.balisation([22,23], 0, -0.0872665)",
        "aruco_controller.deplacementPose([22,23],[0.0,0.0,0.2])",
        "aruco_controller.rotation_yaw([22,23])",
        "aruco_controller.rotation([22,23])",
        "aruco_controller.rotation_yaw([22,23])",
        "aruco_controller.deplacementPose([22,23],[0.0,0.0,0.2])",
        "aruco_controller.rotation_yaw([22,23])",
        "aruco_controller.rotation([22,23])",
        "aruco_controller.rotation_yaw([22,23])",
        ]
    
    actions_1 = [
        "aruco_controller.deplacementPose([20,21],[0.09,0.005,0.35])",
        "aruco_controller.deplacementPose([20,21],[0.09,0.005,0.165])",
        "robot.bougerL(robot.move_actual_pose(2, 0.25))",
        "aruco_controller.deplacementPose([20,21],[0.09,0.02,0.35])",
        "aruco_controller.deplacementPose([20,21],[0.012,0.015,0.35])",
        "aruco_controller.deplacementPose([20,21],[0.012,0.015,0.09])",
        "pince.lacher()",
        "robot.bougerL(robot.move_actual_pose(2, 0.35))",
        "aruco_controller.deplacementPose([20,21],[0.09,0,0.35])",
        "aruco_controller.deplacementPose([20,21],[0.085,0.013,0.085])",
        "pince.prise()",
        "robot.bougerL(robot.move_actual_pose(2, 0.3))",
        "aruco_controller.balisation([22,23], 0, -0.0872665)",
        "aruco_controller.deplacementPose([22,23],[0.0,0.0,0.2])",
        "aruco_controller.rotation_yaw([22,23])",
        "aruco_controller.rotation([22,23])",
        "aruco_controller.rotation_yaw([22,23])",
    ]
    actions_2 =[
        "aruco_controller.deplacementPose([3,2],[0.005,0.05,-0.003])",
        "pince.prise()",
        "robot.bougerL(robot.move_actual_pose(1, -0.2))",
        "robot.bougerL(robot.move_actual_pose(2, 0.2))",
        "robot.bougerJ(robot.joints[3])",
        "aruco_controller.balisation([0], 1, -0.05,0)",
        "aruco_controller.deplacementPose([0],[0.0,0.0,0.4])",
    ]
    marker_ids = []

    try:
        while True:
            # Lecture sécurisée de la variable globale
            with out_lock:
                current_data = global_out
            # On vérifie que les données sont disponibles
            if global_out[0][0] is not None and global_out[1][0] is not None :

                # Si le capteur (par exemple global_out[0][0]) est à 1 et qu'on n'a pas encore déclenché la séquence
                if current_data[0][0] == 1 and not triggered:

                    emplacement_bac_libre = disponible(global_out[1])

                    #if emplacement_bac_libre != -1 :
                        #actions.insert(-1,f"robot.bougerL(pointsBoite[{emplacement_bac_libre}])")

                    triggered = True
                    print("Déclenchement de la séquence de mouvements.")
                    
                    # Exécuter la séquence des mouvements
                    for i in range(len(actions)):
                        print(f"Envoi de l'action {i+1} au robot via RTDE.")
                        eval(actions[i])
                        _, markers_data = aruco_controller.detect_aruco_markers()
                        if markers_data :
                            aruco_frames = aruco_controller.compute_weighted_aruco_reference_frames(markers_data)
                            markers_all_data = aruco_frames['all_markers']
                            marker_ids = list(markers_all_data.keys())
                            print(marker_ids) 
                        # Attendre un délai pour permettre l'exécution du mouvement (pour la démonstration)
                        #time.sleep(2)
                    
                    if emplacement_bac_libre != -1 :
                        actions.pop()

                    print("Séquence de mouvements terminée.")
                
                # Réinitialiser l'état dès que la valeur revient à 0 afin de pouvoir déclencher de nouveau
                elif current_data[0][0] == 0 and triggered:
                    triggered = False
                    print("Réinitialisation de l'état de déclenchement.")
            
            # Petite pause pour éviter une boucle trop gourmande en CPU
            time.sleep(0.05)
    
    except Exception as e:
        print("Erreur lors de la communication avec le robot via RTDE :", e)
    

def disponible(l):
    out = []
    for i in range(len(l)):
        if not l[i]:
            out.append(i)

    if out != [] :
        return out[0]
    
    else :
        return -1

##########################################
# Démarrage des threads
##########################################

if __name__ == "__main__":
    # Lancer le thread du client MQTT
    mqtt_thread = threading.Thread(target=mqtt_client_thread, daemon=True)
    mqtt_thread.start()
    
    # Laisser un peu de temps pour la connexion MQTT
    time.sleep(1)

    #aruco_thread = threading.Thread(target=detect_aruco_thread, daemon=True)
    #aruco_thread.start()
    
    # Lancer le thread de contrôle du robot via RTDE
    robot_thread = threading.Thread(target=robot_control_thread, daemon=True)
    robot_thread.start()

    robot_thread.join()  # Attendre la fin du thread robot si non-infini
    
    # Lance le thread de détection des arucos
    

    # Garder le programme actif pour le thread MQTT
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Arrêt du programme.")

