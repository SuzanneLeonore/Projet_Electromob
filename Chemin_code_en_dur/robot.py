"""
robot.py - Gère les mouvements du robot UR5e

Auteur : Alban CASELLA et Suzanne-Léonore GIRARD-JOLLET
Date : Avril 2025
Description : Ce script utilise l'interface RTDE pour piloter les joints et positions TCP du robot UR5e.
"""

import rtde_receive
import rtde_control
import numpy as np
import json
import os

if os.path.exists("Calibration/calibration_data.json"):
    with open("Calibration/calibration_data.json", "r") as f:
        calib = json.load(f)
    P0 = np.array(calib["P0"])
    P1 = np.array(calib["P1"])
    P2 = np.array(calib["P2"])
    injecteur = calib["injecteur"]
    box = calib["box"]
    injecteur_local = np.array(list(injecteur) + [1])
    box_local= np.array(list(box) + [1])
else:
    raise FileNotFoundError("Fichier de calibration non trouvé. Lance calibration.py d'abord.")

class Robot :

    ROBOT_IP = "10.2.30.60"
    
    # Calcul du repère local (rotation + origine)
    x_axis = P1 - P0
    x_axis /= np.linalg.norm(x_axis)
    temp_y = P2 - P0
    z_axis = np.cross(x_axis, temp_y)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Matrice de rotation (repère local → global)
    R = np.column_stack((x_axis, y_axis, z_axis))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = P0
    T_inv = np.linalg.inv(T)
    repere_box = T_inv @ box_local
    repere_box_1 =np.array(list(repere_box[:3])+[1])
    repere_injecteur = T_inv @ injecteur_local
    repere_injecteur_1=np.array(list(repere_injecteur[:3]) + [1])


    def connexion(self):
        """Connecte le robot aux interfaces de communication au robot"""
        self.robot_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
        self.robot_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)

    def deconnexion(self): 
        """Deconnecte le robot à l'interface de controle"""
        self.robot_c.disconnect()

    def bougerJ(self, position, vitesse =0.2, acceleration=0.2):
        """Permet de bouger le Robot avec les Joints
        
        Paramètres :
            position (list): Position joint cible.
            speed (int | float): Vitesse de déplacement.
            acceleratio (int | float): Accéleration du mouvement.
        """
        self.connexion()
        self.robot_c.moveJ(position, vitesse, acceleration)
        self.deconnexion()

    def bougerL(self, position, vitesse = 0.2, acceleration =0.2) :
        """Permet de bouger le Robot avec la position TCP
        
        Paramètres :
            position (list): Position TCP cible.
            speed (int | float): Vitesse de déplacement.
            acceleration (int | float): Accéleration du mouvement.
        """
        self.connexion()
        #global_point = Robot.T @ position
        position_target = [float(x) for x in position[:3]] + self.robot_r.getActualTCPPose()[3:]
        self.robot_c.moveL(position_target, vitesse, acceleration)
        self.deconnexion()

    def deplacement_point(self, point, indice, distance):
        """Définie une nouvelle position joints à partir de la position actuelle
        
        Paramètres :
            point (list) : Position TCP.
            indice (int): Indice du joint que l'on veut modifier.
            angle (float) : Nouvelle valeur que le joint doit prendre.
        
        Retourne :
            list : Nouvelle position au format TCP.
        """
        arrivee=point.copy()
        arrivee[indice]+=distance
        return arrivee
    
    def get_position(self):
        """ Permet d'obtenir la position TCP actuelle du robot.

        Returne :
            list : Position TCP actuelle.
        """
        self.connexion()
        position_current = self.robot_r.getActualTCPPose()
        self.deconnexion()
        return position_current
    
    def deplacement_joint(self, indice, angle) :
        """Définie une nouvelle position joints à partir de la position actuelle
        
        Paramètres :
            indice (int): Indice du joint que l'on veut modifier.
            angle (float) : Nouvelle valeur que le joint doit prendre.
        
        Retourne :
            list : Nouvelle position de joint.
        """
        self.connexion()
        q_current=self.robot_r.getActualQ()
        q_current[indice]=angle
        self.deconnexion()
        return q_current
    
    def move_actual_pose(self, indice, distance ):
        """Définie une nouvelle position TCP à partir de la position actuelle ou d'une position donnée en argument
        
        Paramètres :
            indice (int): Indice de la coordonnées TCP que l'on veut modifier.
            distance (float) : Distance que l'on veut ajouter à la coordonnées
        
        Retourne :
            list : Nouvelle position de TCP.
        """
        self.connexion()
        pose_current = self.robot_r.getActualTCPPose()[:4]
        pose_current[indice] += distance
        pose_current[3]=1
        self.deconnexion()
        return pose_current

    def __init__(self):

        self.pose_init = [-1.6491854826556605, -1.6341984907733362, 1.8493223190307617,
                 -3.355762783681051, -1.4974659124957483, -1.5762279669391077]
        #déplacement / injecteur 1cm sur X, 15cm sur Y, 10cm sur Z
        self.point1 = self.deplacement_point(self.repere_injecteur_1, 0, -0.005)
        self.point1 = self.deplacement_point(self.point1, 1, 0.13)
        self.point1 = self.deplacement_point(self.point1, 2, 0.15)
        #déplacement -15cm sur Z
        self.point2 = self.deplacement_point(self.point1, 2, -0.20)
        #déplacement -9.5cm sur X
        self.point3 = self.deplacement_point(self.point1, 0, -0.102)
        #déplacement -20 cm sur Z
        self.point4 = self.deplacement_point(self.point3, 2, -0.25)
        #déplacement -10 cm sur Z
        self.point5 = self.deplacement_point(self.point1, 2, -0.112)

        # déplacement / boite
        # -14.5cm sur X, 1,85 cm, 20 cm sur Z
        self.pointBoite1 = self.deplacement_point(self.repere_box_1, 0, -0.145)
        self.pointBoite1 = self.deplacement_point(self.pointBoite1, 1, 0.0185)
        self.pointBoite1 = self.deplacement_point(self.pointBoite1, 2, 0.2)
        #déplacement de 4.65 cm sur Y
        self.pointBoite2 = self.deplacement_point(self.pointBoite1, 1, 0.0465)
        #déplacement de 4.65 cm sur Y
        self.pointBoite3 = self.deplacement_point(self.pointBoite2, 1, 0.0465)
        #déplacement de 4.65 cm sur Y
        self.pointBoite4 = self.deplacement_point(self.pointBoite3, 1, 0.0465)
        #déplacement de 4.65 cm sur Y
        self.pointBoite5 = self.deplacement_point(self.pointBoite4, 1, 0.0465)
        
        

        self.points=[
            #point0
            np.array([-0.048, -0.06, 0.08, 1]),
            #point1
            np.array([-0.048, 0.025, 0.08, 1]),
            #point2
            np.array([-0.048, -0.12, 0.08, 1]),
            #point3
            np.array([-0.048, -0.20, 0.30, 1]),
            #point4
            np.array([-0.048, -0.60, -0.023, 1]),
            #point5
            np.array([-0.048, -0.40, 0.05, 1]),
            #point6
            np.array([-0.048, -0.40, 0.30, 1]),
            #point7
            np.array(self.point1),
            #point8
            np.array(self.point2),
            #point9
            np.array(self.point1),
            #point10
            np.array(self.point3),
            #point11
            np.array(self.point4),
            #point12
            np.array(self.point3),
            #point13
            np.array(self.point1),
            #point14
            np.array(self.point5),
            #point15
            np.array(self.point1),
            #point16
            np.array(self.point3),
            #point17
            np.array(self.point4),
            #point18
            np.array(self.point3),
            #point19
            np.array(self.point1),
            #np.array20
            np.array(self.point2),
            #np.array21
            np.array(self.point1),
            #np.array22
            np.array([-0.048, -0.60, -0.023, 1]),
        ]
        '''
        self.pointBis =[
            #joints[0],
            np.array(self.move_actual_pose(1, -0.2)),
            np.array(self.pointCellule1),
            np.array(self.pointCellule2),
            np.array(self.pointCellule3),
            np.array(self.pointCellule4),
            #joints[1]
            np.array(self.pointCellule5),
            np.array(self.pointCellule4)
            #joints[2]

        ]
        

        self.joints=[
            [-1.5102294127093714, -1.214823071156637, 1.8769307136535645, -3.8390358130084437, -1.6932085196124476, -1.4776809851275843],
            [-0.034966293965474904, -1.4821026960956019, 0.7819619178771973, -0.8718579451190394, -1.4708979765521448, -1.5708215872394007],
            [-0.02170879045595342, -1.852485481892721, 1.634385585784912, -1.3550799528705042, -1.4711735884295862, -1.5699823538409632],
            #[-0.4743412176715296, -1.5091918150531214, 1.348893642425537, -1.3945730368243616, -1.4682758490191858, -2.0456507841693323],
            #[-0.07224733034242803, -1.9188702742206019, 1.8160276412963867, -1.4937194029437464, -1.4703596274005335, -1.5883601347552698],
            [-1.5707710425006312, -1.9037888685809534, 1.8204197883605957, -1.5371840635882776, -1.4706586042987269, -1.5850275198565882],
        ]
        '''

        self.joints =[
            [-1.5102294127093714, -1.214823071156637, 1.8769307136535645, -3.8390358130084437, -1.6932085196124476, -1.4776809851275843]
        ]
        
        self.pointsBoite=[
            self.pointBoite1,
            self.pointBoite2,
            self.pointBoite3,
            self.pointBoite4,
            self.pointBoite5,
        ]
