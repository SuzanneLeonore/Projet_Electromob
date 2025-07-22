"""
Procedure_final.py - Contrôle automatisé d’un robot UR5e à l'aide de capteurs MQTT et de la vision par marqueurs ArUco.

Ce script :
- se connecte à un broker MQTT pour lire l'état des capteurs (convoyeur et bac),
- déclenche une séquence de mouvements du robot UR5e en fonction des capteurs,
- utilise un système de verrouillage pour le partage de données entre threads.

Fichiers dépendants :
- pince.py : contrôle de la pince 2GF7.
- robot.py : contrôle du bras UR5e.

Auteur : Alban CASELLA et Suzanne-Léonore GIRARD-JOLLET
Date : Mai 2025
"""

import json
import paho.mqtt.client as mqtt
import time
import threading
import time
from pince import Pince
from robot import Robot

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

def on_connect(client, rc):
    """
    Callback appelée lors de la connexion au broker MQTT.

    Abonne le client aux topics des capteurs convoyeur et bac.

    Args:
        client: instance du client MQTT.
        rc: code de retour de la connexion.
    """
    print("Connecté au broker MQTT avec le code de retour", rc)
    client.subscribe(TOPIC_B)
    client.subscribe(TOPIC_C)

def on_message(msg):
    """
    Callback appelée à chaque réception d’un message MQTT.

    Met à jour les variables globales `convoyeur_data` et `bac_data` 
    en fonction des topics, et stocke le tout dans `global_out`.

    Args:
        msg: message MQTT reçu.
    """
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
    """
    Initialise et lance un client MQTT dans un thread séparé.

    Ce thread reste actif grâce à `loop_forever()` pour recevoir
    les messages de capteurs en continu.
    """
    client = mqtt.Client("PythonClient")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    # Boucle bloquante dans un thread séparé
    client.loop_forever()

##########################################
# Partie 2 : Contrôle du robot via rtde
##########################################

def robot_control_thread():
    """
    Thread principal de contrôle du robot UR5e.

    Attend la détection d’un objet sur le convoyeur pour déclencher une
    séquence de mouvements prédéfinie. Utilise aussi les capteurs du bac 
    pour décider de l’emplacement de dépôt.

    Les instructions sont stockées dans une liste `actions`, exécutées
    dynamiquement avec `eval()`. La fonction vérifie et évite les déclenchements
    multiples grâce à une variable `triggered`.
    """

    robot = Robot()
    pince = Pince()

    # Définition des points de destination (vecteurs de joints)
    # Vous pouvez définir ici la séquence complète des mouvements à exécuter
    points = robot.points
    joints = robot.joints
    # Variable d'état pour s'assurer qu'on ne lance la séquence qu'une fois par activation.
    pointsBoite = robot.pointsBoite

    triggered = False

    actions = [
        "robot.bougerJ(robot.pose_init)",
        "pince.lacher()",
        "robot.bougerL(robot.points[0])",
        "robot.bougerL(robot.points[1])",
        "pince.prise()",
        "robot.bougerL(robot.points[2])",
        "robot.bougerL(robot.points[3])",
        "robot.bougerJ(robot.joints[0])",
        "robot.bougerL(robot.points[4], 0.05, 0.05)",
        "robot.bougerL(robot.points[5], 0.2, 0.2)",
        "robot.bougerL(robot.points[6], 0.2, 0.2)",
        "robot.bougerJ(robot.joints[1])",
        "robot.bougerJ(robot.joints[2])",
        "robot.bougerL(robot.points[7])",
        "robot.bougerL(robot.points[8], 0.05, 0.05)",
        "pince.lacher()",
        "robot.bougerL(robot.points[9], 0.1, 0.1)",
        "robot.bougerL(robot.points[10], 0.1, 0.1)",
        "robot.bougerL(robot.points[11], 0.1, 0.1)",
        "pince.prise()",
        "robot.bougerL(robot.points[12], 0.1, 0.1)",
        "robot.bougerL(robot.points[13], 0.1, 0.1)",
        "robot.bougerL(robot.points[14], 0.1, 0.1)",
        "robot.bougerL(robot.points[15], 0.1, 0.1)",
        "robot.bougerL(robot.points[16], 0.1, 0.1)",
        "robot.bougerL(robot.points[17], 0.1, 0.1)",
        "pince.lacher()",
        "robot.bougerL(robot.points[18], 0.1, 0.1)",
        "robot.bougerL(robot.points[19], 0.1, 0.1)",
        "robot.bougerL(robot.points[20], 0.1, 0.1)",
        "pince.prise()",
        "robot.bougerL(robot.points[21],0.1, 0.1)",
        "robot.bougerJ(robot.joints[3], 0.1, 0.1)",
        "robot.bougerL(robot.points[22], 0.05, 0.05)",
        "robot.bougerL(robot.points[5], 0.05, 0.05)",
        ### On insère ici l'instruction pour faire bouger le robot au bon emplacement du bac
        "pince.lacher()",
        ]
    
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
                    if emplacement_bac_libre != -1 :
                        actions.append(f"robot.bougerL(pointsBoite[{emplacement_bac_libre}])")
                        if emplacement_bac_libre <= 2:
                            actions.append(f"robot.bougerJ(robot.deplacement_joint( 5, 1.738790512084961))")
                            actions.append(f"robot.bougerL(robot.deplacement_point(pointsBoite[{emplacement_bac_libre}], 2, -0.05))")
                        else :
                            actions.append(f"robot.bougerJ(robot.deplacement_joint( 5, 1.7103421688079834))")
                            actions.append(f"robot.bougerL(robot.deplacement_point(pointsBoite[{emplacement_bac_libre}], 2, -0.05))")

                    triggered = True
                    print("Déclenchement de la séquence de mouvements.")
                    # Exécuter la séquence des mouvements
                    for i in range(len(actions)):
                        if i == 13 or i == 36:
                            input ("continuer l'action ...")
                        print(f"Envoi de l'action {i+1} au robot via RTDE.")
                        eval(actions[i])
                        
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
            time.sleep(0.1)
    
    except Exception as e:
        print("Erreur lors de la communication avec le robot via RTDE :", e)
    

def disponible(l):
    """
    Cherche le premier emplacement libre dans la liste d’état du bac.

    Args:
        l (list[bool]): liste contenant l'état de 5 capteurs (True = occupé, False = libre)

    Returns:
        int: index du premier emplacement libre, ou -1 si aucun n’est disponible.
    """
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
    
    # Lancer le thread de contrôle du robot via RTDE
    robot_thread = threading.Thread(target=robot_control_thread, daemon=True)
    robot_thread.start()

    robot_thread.join()  # Attendre la fin du thread robot si non-infini
    
    # Garder le programme actif pour le thread MQTT
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Arrêt du programme.")

