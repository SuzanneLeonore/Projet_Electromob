import json
import paho.mqtt.client as mqtt
import time
import threading
import numpy as np 
import rtde_receive
import rtde_control
from pince import Pince
from robot import Robot

class RobotController:
    def __init__(self, mqtt_broker="10.2.30.162", mqtt_port=1883):
        self.convoyeur_data = [None] * 2
        self.bac_data = [None] * 5
        self.global_out = None
        self.out_lock = threading.Lock()
        self.emplacement_bac_libre = None
        self.MQTT_BROKER = mqtt_broker
        self.MQTT_PORT = mqtt_port
        self.TOPIC_B = "capteurs_bac/etat"
        self.TOPIC_C = "capteurs_convoyeur/etat"
        self.triggered = False

    def on_connect(self, client, userdata, flags, rc):
        print("Connecté au broker MQTT avec le code de retour", rc)
        client.subscribe(self.TOPIC_B)
        client.subscribe(self.TOPIC_C)

    def on_message(self, client, userdata, msg):
        print(f"[MQTT] Message reçu sur {msg.topic} : {msg.payload}")
        try:
            data = json.loads(msg.payload.decode('utf-8'))
            if msg.topic == self.TOPIC_C:
                self.convoyeur_data = [data["pin1"], data["pin2"]]
            elif msg.topic == self.TOPIC_B:
                self.bac_data = [data[f"pin{i+1}"] for i in range(5)]
            with self.out_lock:
                self.global_out = [self.convoyeur_data, self.bac_data]
            print("Données MQTT reçues :", self.global_out)
        except Exception as e:
            print("Erreur lors du traitement du message :", e)

    def mqtt_client_thread(self):
        client = mqtt.Client("PythonClient")
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
        client.loop_forever()

    def disponible(self, l):
        out = [i for i, v in enumerate(l) if not v]
        return out[0] if out else -1

    def robot_control_thread(self):
        robot = Robot()
        pince = Pince()
        points = robot.points
        joints = robot.joints
        pointsBoite = robot.pointsBoite

        actions = [
            "robot.bougerJ(robot.pose_init_Q)",
            "pince.lacher()",
            "robot.balisation([6,7,8])",
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
        ]

        while True:
            with self.out_lock:
                current_data = self.global_out

            if current_data and current_data[0][0] is not None and current_data[1][0] is not None:
                if current_data[0][0] == 1 and not self.triggered:
                    self.emplacement_bac_libre = self.disponible(current_data[1])
                    if self.emplacement_bac_libre != -1:
                        actions.append(f"robot.bougerL(pointsBoite[{self.emplacement_bac_libre}])")
                        joint_value = 1.738790512084961 if self.emplacement_bac_libre <= 2 else 1.7103421688079834
                        actions.append(f"robot.bougerJ(robot.deplacement_joint(5, {joint_value}))")

                    self.triggered = True
                    print("Déclenchement de la séquence de mouvements.")
                    for i, action in enumerate(actions):
                        if i == 13 or i == 36:
                            input("continuer l'action ...")
                        print(f"Envoi de l'action {i+1} au robot via RTDE.")
                        eval(action)

                    if self.emplacement_bac_libre != -1:
                        actions.pop()

                    print("Séquence de mouvements terminée.")
                elif current_data[0][0] == 0 and self.triggered:
                    self.triggered = False
                    print("Réinitialisation de l'état de déclenchement.")
            time.sleep(0.1)

    def start(self):
        print("🟢 Démarrage du thread MQTT")
        mqtt_thread = threading.Thread(target=self.mqtt_client_thread, daemon=True)
        mqtt_thread.start()
        print("✅ Thread MQTT lancé")

        time.sleep(1)

        print("🟢 Démarrage du thread robot")
        robot_thread = threading.Thread(target=self.robot_control_thread, daemon=True)
        robot_thread.start()
        print("✅ Thread robot lancé")
        print (self.bac_data)
        robot_thread.join()    
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Arrêt du programme.")

if __name__ == "__main__":
    controller = RobotController()
    controller.start()
    print (controller.bac_data)

