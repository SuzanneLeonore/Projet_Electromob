import rtde_receive
import rtde_control
import time
#from Transfo import create_matrice, matrice_to_pose
import numpy as np
from pince import Pince
#import cheminQRcode as QRcode

class Robot :

    ROBOT_IP = "10.2.30.60"

    def connexion(self):
        #print('on se connecte')
        self.robot_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
        self.robot_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)

    def deconnexion(self): 
        self.robot_c.disconnect()
        #print('on se deconnecte')

    def bougerJ(self, pos, speed=0.5, acceleration=0.5):
        """Permet de bouger le Robot avec les Joints"""
        self.connexion()
        self.robot_c.moveJ(pos, speed, acceleration)
        self.deconnexion()

    def bougerL(self, pos, speed = 0.2, acceleration =0.2) :
        """Permet de bouger le Robot avec la position TCP"""
        self.connexion()
        pose_target = [float(x) for x in pos[:3]] + self.robot_r.getActualTCPPose()[3:]
        self.robot_c.moveL(pose_target, speed, acceleration)
        self.deconnexion()

    def deplacement_point(self, point, indice, distance):
        """Permet de bouger le robot depuis la position actuelle"""
        self.connexion()
        if np.allclose(point[:3], [0, 0, 0]):
            point = self.robot_r.getActualTCPPose()
        arrivee=point.copy()
        arrivee[indice]+=distance
        self.deconnexion()
        return arrivee
    
    def get_position(self):
        self.connexion()
        pose_current = self.robot_r.getActualQ()
        self.deconnexion()
        return pose_current
    
    def deplacement_joint(self, indice, angle) :
        self.connexion()
        q_current=self.robot_r.getActualQ()
        q_current[indice]=angle
        self.deconnexion()
        return q_current
    
    def move_actual_pose(self, indice, distance  ):
        self.connexion()
        pose_current = self.robot_r.getActualTCPPose()[:4]
        pose_current[indice] += distance
        pose_current[3]=1
        self.deconnexion()
        return pose_current
    
    def balisation(self,indice, angle) :
        #self.connexion()
        pose_current = self.robot_r.getActualQ()
        pose_current[indice]+=angle
        self.bougerJ(pose_current)
        self.deconnexion()

    def __init__(self):

        self.pose_init_Q = [-0.842104736958639, -1.4243934790240687, 2.540102958679199, -4.270684067402975, -1.5935080687152308, -1.564348045979635] #moveJ
        self.pose_Q =[-1.5771964232074183, -1.145515267048971, 2.160745143890381, -4.118872944508688, -1.571120564137594, -1.5643118063556116]
        self.joints=[
            [-1.5102294127093714, -1.214823071156637, 1.8769307136535645, -3.8390358130084437, -1.6932085196124476, -1.4776809851275843],
            [-0.034966293965474904, -1.4821026960956019, 0.7819619178771973, -0.8718579451190394, -1.4708979765521448, -1.5708215872394007],
            [-0.02170879045595342, -1.852485481892721, 1.634385585784912, -1.3550799528705042, -1.4711735884295862, -1.5699823538409632],
            #[-0.4743412176715296, -1.5091918150531214, 1.348893642425537, -1.3945730368243616, -1.4682758490191858, -2.0456507841693323],
            #[-0.07224733034242803, -1.9188702742206019, 1.8160276412963867, -1.4937194029437464, -1.4703596274005335, -1.5883601347552698],
            [-1.5707710425006312, -1.9037888685809534, 1.8204197883605957, -1.5371840635882776, -1.4706586042987269, -1.5850275198565882],
        ]

if __name__ == "__main__" :
    robot = Robot()
    #robot.bougerJ(robot.pose_init_Q)
    for i in range(len(robot.joints)) :
        robot.connexion()
        print(f"envoie action :{i+1}")
        robot.robot_c.moveJ(robot.joints[i],0.5, 0.5)
        time.sleep(5)
        input("on passe à l'action suivante")
        robot.deconnexion()
        