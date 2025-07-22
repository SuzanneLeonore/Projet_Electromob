import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_control
import rtde_receive
import yaml
import time
from collections import deque
from scipy.spatial.transform import Rotation as R


class ArucoURController:
    def __init__(self, robot_ip="10.2.30.60"):
        # 1. Initialisation de la caméra RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        
        
        # 2. Chargement des paramètres intrinsèques et de calibration
        try:
            # Chargement des paramètres intrinsèques
            with open('/home/robot/Bureau/Stage/UR5_projetFinale/chemin_QRcode/intrinsics.yaml', 'r') as f:
                intrinsic_data = yaml.safe_load(f)
                self.camera_matrix = np.array(intrinsic_data['camera_matrix'], dtype=np.float32)
                self.dist_coeffs = np.array(intrinsic_data['dist_coeffs'], dtype=np.float32)
            print("Paramètres intrinsèques chargés avec succès")
            
            # Chargement de la transformation hand-eye
            with open('/home/robot/Bureau/Stage/UR5_projetFinale/chemin_QRcode/hand_eye.yaml', 'r') as f:
                hand_eye_data = yaml.safe_load(f)
                self.T_cam_to_tcp = np.array(hand_eye_data['T_cam_to_flange'], dtype=np.float32)
            print("Transformation hand-eye chargée avec succès")
        except Exception as e:
            print(f"Erreur lors du chargement des calibrations: {e}")
            raise
        
        # 3. Configuration du détecteur ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # 4 Ajustement des paramètres du détecteur pour améliorer la robustesse
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size = 0.04  # Taille du marqueur en mètres (ajuster selon vos marqueurs)
        self.update_cam_to_base_transform()
        # 5. Initialisation des buffers pour le filtrage temporel
        self.pose_history = {}  # Dictionnaire qui contiendra des deques pour chaque marqueur
        self.history_length = 5  # Nombre de poses à conserver pour chaque marqueur
        self.last_frame_markers = None  # Pour stocker les derniers marqueurs détectés
        
        # 6. Facteurs de confiance pour les différents marqueurs (ajustable)
        self.marker_weights = {}  # À remplir dynamiquement ou à partir d'un fichier
        
        # 7. Paramètres pour la validation du mouvement
        self.last_successful_movement = None  # Pour la continuité des mouvements
        self.max_allowed_deviation = 0.05  # 5cm de déviation maximale permise
        
        # 8. Paramètre pour la correction d'échelle (pour résoudre le problème de distances)
        self.scale_correction = 1.0  # Facteur initial, peut être ajusté
    
    def connexion(self):
        self.ur5_receive = rtde_receive.RTDEReceiveInterface("10.2.30.60")
        try:
            self.ur5_control = rtde_control.RTDEControlInterface("10.2.30.60")
        except Exception as e:
            print("Erreur pendant la connexion RTDEControl :", e)

    def deconnexion(self): 
        self.ur5_control.disconnect()
    
    def update_cam_to_base_transform(self):
        """Calcule la transformation caméra -> base du robot en utilisant hand-eye"""
        self.connexion()
        # Récupérer la pose actuelle du TCP
        tcp_pose = self.ur5_receive.getActualTCPPose()
        if tcp_pose is None or len(tcp_pose) < 6:
            raise ValueError("Erreur : la pose TCP reçue est invalide ou vide.")
        # Construire la matrice base -> TCP
        T_tcp_to_base = np.eye(4)
        T_tcp_to_base[:3, 3] = tcp_pose[:3]  # Position
        rx, ry, rz = tcp_pose[3:6]
        # Convertir rx, ry, rz en matrice de rotation
        rot_vector = np.array([rx, ry, rz])
        rot_matrix, _ = cv2.Rodrigues(rot_vector)
        T_tcp_to_base[:3, :3] = rot_matrix
        #print(f"Tcp to base : {T_tcp_to_base}")
        
        # Calculer base -> caméra en utilisant base -> TCP et TCP -> caméra
        self.T_cam_to_base = T_tcp_to_base @ self.T_cam_to_tcp
        #print(f"Transformation caméra -> base du robot mise à jour : {self.T_cam_to_base}")
        self.deconnexion()

    def detect_aruco_markers(self, strict=False):
        """Détecte les marqueurs ArUco et calcule leurs poses avec une meilleure précision"""
        # Capture d'une image
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            print("Images non disponibles")
            return None, None
        
        # Conversion en image OpenCV
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Prétraitement d'image pour améliorer la détection
        # Augmentation du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Détection des marqueurs avec l'image améliorée
        corners, ids, _ = self.detector.detectMarkers(enhanced_gray)
        
        if ids is None or len(ids) == 0:
            if not strict and self.last_frame_markers is not None:
                return color_image, self.last_frame_markers
            return color_image, None
        
        # Dessiner les marqueurs détectés
        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
        
        # Collecter les positions et orientations des marqueurs
        markers_data = []
        
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            
            # Estimer la pose avec solvePnP pour plus de précision
            marker_points = np.array([
                [-self.marker_size/2, self.marker_size/2, 0],
                [self.marker_size/2, self.marker_size/2, 0],
                [self.marker_size/2, -self.marker_size/2, 0],
                [-self.marker_size/2, -self.marker_size/2, 0]
            ], dtype=np.float32)
            
            # Utiliser SOLVEPNP_IPPE_SQUARE qui est plus précis pour les marqueurs carrés
            ret, rvec, tvec = cv2.solvePnP(
                marker_points, corner[0],
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            # Raffiner la pose en utilisant une optimisation itérative
            ret, rvec, tvec = cv2.solvePnP(
                marker_points, corner[0],
                self.camera_matrix, self.dist_coeffs,
                rvec=rvec, tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # Dessiner les axes
            cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs,
                             rvec, tvec, self.marker_size/2)
            # Convertir le vecteur de rotation en matrice
            rot_matrix_cam = np.eye(4)
            rot_matrix, _ = cv2.Rodrigues(rvec)
            
            rot_matrix_cam [:3,:3]=rot_matrix

            # Position du marqueur dans le repère caméra
            position = tvec.flatten()
            position=position.tolist()
            for i in range (3):
                rot_matrix_cam[i][3]=position[i]

            # Vérifier la cohérence de la profondeur avec le capteur de profondeur
            # Centre approximatif du marqueur en pixels
            center_x = int(np.mean([p[0] for p in corner[0]]))
            center_y = int(np.mean([p[1] for p in corner[0]]))
            
            # Obtenir la profondeur du marqueur (moyenne sur une petite zone)
            depth_region = depth_image[
                max(0, center_y-5):min(depth_image.shape[0], center_y+5),
                max(0, center_x-5):min(depth_image.shape[1], center_x+5)
            ]
            valid_depths = depth_region[depth_region > 0]
            
            if len(valid_depths) > 0:
                # Convertir en mètres
                depth_sensor = np.median(valid_depths) / 1000.0
                
                # Comparer avec la profondeur estimée par solvePnP
                depth_solvepnp = position[2]
                depth_diff = abs(depth_sensor - depth_solvepnp)
                
                # Si la différence est trop grande, ajuster la pose
                if depth_diff > 0.05:  # 5cm de différence
                    # Facteur de confiance entre les deux mesures (privilégier la profondeur du capteur)
                    alpha = 0.7
                    corrected_depth = alpha * depth_sensor + (1 - alpha) * depth_solvepnp
                    
                    # Mettre à jour la position Z
                    position[2] = corrected_depth
                    
                    # Recalculer tvec pour le dessin des axes
                    tvec[2] = corrected_depth
            
            # Orientation (axes X, Y, Z) dans le repère caméra
            x_axis = rot_matrix[:, 0]
            y_axis = rot_matrix[:, 1]
            z_axis = rot_matrix[:, 2]

            rot_matrix_gripper = self.T_cam_to_tcp @ rot_matrix_cam
            rpy = self.rotvec_to_rpy(rvec[0][0], rvec[1][0], rvec[2][0])

            # Transformer la position du repère caméra au repère base du robot
            pos_hom = np.ones(4)
            pos_hom[:3] = position
            position_base = (self.T_cam_to_base @ pos_hom)[:3]

            # Transformer les axes du repère caméra au repère base du robot
            R_cam_to_base = self.T_cam_to_base[:3, :3]
            x_axis_base = R_cam_to_base @ x_axis
            y_axis_base = R_cam_to_base @ y_axis
            z_axis_base = R_cam_to_base @ z_axis

            # Filtrage temporel: ajouter à l'historique
            if marker_id not in self.pose_history:
                self.pose_history[marker_id] = deque(maxlen=self.history_length)
            
            # Stocker les données du marqueur
            marker_data = {
                'id': marker_id,
                'position': position_base,
                'pos_marker': pos_hom[:3],
                'rvec' : rvec,
                'rpy' :rpy,
                'axes': {
                    'x': x_axis_base,
                    'y': y_axis_base,
                    'z': z_axis_base
                },
                'timestamp': time.time()
            }
            
            # Ajouter à l'historique
            self.pose_history[marker_id].append(marker_data)
            
            # Calculer une position filtrée (moyenne mobile)
            if len(self.pose_history[marker_id]) > 1:
                filtered_position = np.mean([m['position'] for m in self.pose_history[marker_id]], axis=0)
                # Filtre temporel aussi sur les axes
                filtered_x = np.mean([m['axes']['x'] for m in self.pose_history[marker_id]], axis=0)
                filtered_y = np.mean([m['axes']['y'] for m in self.pose_history[marker_id]], axis=0)
                filtered_z = np.mean([m['axes']['z'] for m in self.pose_history[marker_id]], axis=0)
                
                # Normaliser les axes filtrés
                filtered_x = filtered_x / np.linalg.norm(filtered_x)
                filtered_y = filtered_y / np.linalg.norm(filtered_y)
                filtered_z = filtered_z / np.linalg.norm(filtered_z)
                
                marker_data['position'] = filtered_position
                marker_data['axes']['x'] = filtered_x
                marker_data['axes']['y'] = filtered_y
                marker_data['axes']['z'] = filtered_z
            
            markers_data.append(marker_data)
            
            # Afficher l'ID et la position du marqueur
            cv2.putText(color_image, f"ID={marker_id}", (int(corner[0][0][0]), int(corner[0][0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #print(corner[0][0][0])
        
        # Mémoriser les marqueurs détectés pour la prochaine image
        self.last_frame_markers = markers_data
        
        return color_image, markers_data
    
    def compute_weighted_aruco_reference_frames(self, markers_data):
        """
        Calcule un repère de référence à partir des marqueurs détectés
        en tenant compte de leur fiabilité relative
        """
        if not markers_data or len(markers_data) == 0:
            return None
            
        # Tenir compte de la fiabilité relative des marqueurs (poids)
        # Par défaut, tous les marqueurs ont un poids égal
        total_weight = 0
        weighted_positions = np.zeros(3)
        weighted_x_axes = np.zeros(3)
        weighted_y_axes = np.zeros(3)
        weighted_z_axes = np.zeros(3)
        
        # Collecter les données de tous les marqueurs
        all_markers_axes = {}
        
        for marker in markers_data:
            marker_id = marker['id']
            marker_pos = marker['pos_marker']
            
            # Stocker les axes du marqueur actuel
            all_markers_axes[marker_id] = {
                'position': marker['position'],
                'axes': marker['axes'],
                'pos_marker': marker_pos,
                'rvec' : marker['rvec'],
                'rpy' :marker['rpy']
            }
            
            # Poids par défaut = 1.0, ou utiliser une valeur personnalisée si définie
            weight = self.marker_weights.get(marker_id, 1.0)
            
            # Ajustement du poids en fonction de la distance (les marqueurs proches sont plus fiables)
            distance = np.linalg.norm(marker['position'])
            distance_factor = np.exp(-distance / 0.5)  # Décroissance exponentielle avec la distance
            weight *= distance_factor
            
            # Appliquer le poids aux données du marqueur
            weighted_positions += weight * marker['position']
            weighted_x_axes += weight * marker['axes']['x']
            weighted_y_axes += weight * marker['axes']['y']
            weighted_z_axes += weight * marker['axes']['z']
            total_weight += weight
        
        # Normaliser par le poids total
        if total_weight > 0:
            mean_position = weighted_positions / total_weight
            mean_x = weighted_x_axes / total_weight
            mean_y = weighted_y_axes / total_weight
            mean_z = weighted_z_axes / total_weight
        else:
            # Fallback : moyennes simples
            positions = [marker['position'] for marker in markers_data]
            x_axes = [marker['axes']['x'] for marker in markers_data]
            y_axes = [marker['axes']['y'] for marker in markers_data]
            z_axes = [marker['axes']['z'] for marker in markers_data]
            mean_position = np.mean(positions, axis=0)
            mean_x = np.mean(x_axes, axis=0)
            mean_y = np.mean(y_axes, axis=0)
            mean_z = np.mean(z_axes, axis=0)
        
        # Orthonormaliser le repère
        mean_z = mean_z / np.linalg.norm(mean_z)  # Normaliser Z
        
        # Rendre Y perpendiculaire à Z
        mean_y = mean_y - np.dot(mean_y, mean_z) * mean_z
        mean_y = mean_y / np.linalg.norm(mean_y)  # Normaliser Y
        
        # X doit être perpendiculaire à Y et Z
        mean_x = np.cross(mean_y, mean_z)
        rot_matrices = [R.from_rotvec(marker['rvec'].flatten()).as_matrix() for marker in markers_data]
        R_avg = sum(rot_matrices) / len(rot_matrices)
        U, _, Vt = np.linalg.svd(R_avg)
        R_orthonorm = U @ Vt
        rvec_avg = R.from_matrix(R_orthonorm).as_rotvec()
        # Matrice de rotation pour corriger les axes des marqueurs
        # (Utilise une rotation de 180° autour de Z pour aligner avec le repère du robot)
        rot_z_180 = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        rotated_x = rot_z_180 @ mean_x
        rotated_y = rot_z_180 @ mean_y
        
        # Récupérer marker_id et marker_pos du premier marqueur (ou autre logique si nécessaire)
        if markers_data:
            marker_id = markers_data[0]['id']
            marker_pos = markers_data[0]['pos_marker']
        else:
            marker_id = None
            marker_pos = None
            
        return {
            'id': marker_id,
            'pos_marker': marker_pos,
            'position': mean_position,
            'axes': {
                'x': rotated_x,
                'y': rotated_y,
                'z': mean_z
            },
            'original_axes': {
                'x': mean_x,
                'y': mean_y,
                'z': mean_z
            },
            'rvec_avg' : rvec_avg,
            'all_markers': all_markers_axes  # Ajout de tous les axes des marqueurs
        }
   
    def load_marker_weights(self):
        """Charge les poids des marqueurs depuis un fichier"""
        try:
            with open('marker_weights.yaml', 'r') as f:
                data = yaml.safe_load(f)
                self.marker_weights = data.get('marker_weights', {})
            print("Poids des marqueurs chargés depuis marker_weights.yaml")
            return True
        except FileNotFoundError:
            print("Fichier marker_weights.yaml non trouvé. Utilisation des poids par défaut.")
            return False
        except Exception as e:
            print(f"Erreur lors du chargement des poids: {e}")
            return False

    def rpy_to_rotvec(self,roll, pitch, yaw):
        # Conversion RPY → vecteur de rotation
        rot = R.from_euler('xyz', [roll, pitch, yaw])
        rotvec = rot.as_rotvec()
        return rotvec

    def rotvec_to_rpy(self,rx, ry, rz):
        # Conversion vecteur de rotation → RPYdegrees = True 
        rot = R.from_rotvec([rx, ry, rz])
        rpy = rot.as_euler('xyz')
        rpy_1 = rot.as_euler('xyz', degrees = True)
        #print(rpy_1)
        return rpy
    
    def rotation_rvec(self, liste) :
        self.update_cam_to_base_transform()
        _, markers_data = self.detect_aruco_markers()
        selected_markers = [m for m in markers_data if m['id'] in liste]
        rot_matrices = [R.from_rotvec(marker['rvec'].flatten()).as_matrix() for marker in selected_markers]
        if len(rot_matrices)!=0 :
            R_avg = sum(rot_matrices) / len(rot_matrices)
        else :
            R_avg = sum(rot_matrices)
        U, _, Vt = np.linalg.svd(R_avg)
        R_orthonorm = U @ Vt
        rvec_avg = R.from_matrix(R_orthonorm).as_rotvec()
        rvec_avg =rvec_avg.tolist()
        rpy = self.rotvec_to_rpy(rvec_avg[0],rvec_avg[1],rvec_avg[2])
        if liste == [3,2] or liste == [2,3]:
            rotvec = self.rpy_to_rotvec(0,np.pi/2,np.pi/2)
            self.connexion()
            current_pose_TCP = self.ur5_receive.getActualTCPPose()
            for i in range(3) :
                current_pose_TCP[i+3]=rotvec[i]
            self.ur5_control.moveL(current_pose_TCP, 0.5,0.5)
            self.pose_history.clear()
            self.deconnexion()
        else :
            #rotvec = self.rpy_to_rotvec(np.pi,0,rpy[2])
            rotvec = self.rpy_to_rotvec(np.pi,0,0)
            print(rotvec)
            self.connexion()
            current_pose_TCP = self.ur5_receive.getActualTCPPose()
            for i in range(3) :
                current_pose_TCP[i+3]=rotvec[i]
            self.ur5_control.moveL(current_pose_TCP, 0.5,0.5)
            self.pose_history.clear()
            self.deconnexion()
    
    def rotation(self, liste) :
        self.update_cam_to_base_transform()
        _, markers_data = self.detect_aruco_markers()
        selected_markers = [m for m in markers_data if m['id'] in liste]

        rot_matrices = [R.from_rotvec(marker['rvec'].flatten()).as_matrix() for marker in selected_markers]

        if len(rot_matrices)!=0 :
            R_avg = sum(rot_matrices) / len(rot_matrices)
        else :
            R_avg = sum(rot_matrices)
        U, _, Vt = np.linalg.svd(R_avg)
        R_orthonorm = U @ Vt
        rvec_avg = R.from_matrix(R_orthonorm).as_rotvec()
        rvec_avg =rvec_avg.tolist()
        rpy = self.rotvec_to_rpy(rvec_avg[0],rvec_avg[1],rvec_avg[2])

        if liste == [3,2] or liste == [2,3]:
            rotvec = self.rpy_to_rotvec(0,np.pi/2,np.pi/2)
            self.connexion()
            current_pose_TCP = self.ur5_receive.getActualTCPPose()
            current_pose_Q = self.ur5_receive.getActualQ()
            for i in range(3) :
                current_pose_TCP[i+3]=rotvec[i]
            self.ur5_control.moveL(current_pose_TCP, 0.5,0.5)
            current_pose_Q = self.ur5_receive.getActualQ()
            if current_pose_Q[5] < 0 :
                current_pose_Q[5] +=np.pi
            #else :
                #current_pose_Q[5] -=np.pi
            self.ur5_control.moveJ(current_pose_Q, 0.8,0.8)
            self.pose_history.clear()
            self.deconnexion()

        elif liste ==[0] : 
            rotvec = self.rpy_to_rotvec(np.pi,0,0)
            self.connexion()
            current_pose_TCP = self.ur5_receive.getActualTCPPose()
            for i in range(3) :
                current_pose_TCP[i+3]=rotvec[i]
            self.ur5_control.moveL(current_pose_TCP, 0.5,0.5)
            current_pose_Q = self.ur5_receive.getActualQ()
            current_pose_Q[5] +=rpy[2]
            '''if current_pose_Q[5] < 0 :
                current_pose_Q[5] +=np.pi
                current_pose_Q[5] +=rpy[2]
            else :
                current_pose_Q[5] -=np.pi
                current_pose_Q[5] -=rpy[2]'''
            self.ur5_control.moveJ(current_pose_Q, 0.8,0.8)
            self.pose_history.clear()
            self.deconnexion()

        else :
            rotvec = self.rpy_to_rotvec(np.pi,0,0)
            print(rotvec)
            self.connexion()
            current_pose_TCP = self.ur5_receive.getActualTCPPose()
            for i in range(3) :
                current_pose_TCP[i+3]=rotvec[i]
            self.ur5_control.moveL(current_pose_TCP, 0.5,0.5)
            current_pose_Q = self.ur5_receive.getActualQ()
            if current_pose_Q[5] < 0 :
                current_pose_Q[5] +=np.pi
                current_pose_Q[5] +=rpy[2]
            else :
                current_pose_Q[5] -=np.pi
                current_pose_Q[5] +=rpy[2]
            self.ur5_control.moveJ(current_pose_Q, 0.8,0.8)
            self.pose_history.clear()
            self.deconnexion()

    def rotation_yaw(self, liste) :
        self.update_cam_to_base_transform()
        _, markers_data = self.detect_aruco_markers()
        selected_markers = [m for m in markers_data if m['id'] in liste]

        rot_matrices = [R.from_rotvec(marker['rvec'].flatten()).as_matrix() for marker in selected_markers]

        R_avg = sum(rot_matrices) / len(rot_matrices)
        U, _, Vt = np.linalg.svd(R_avg)
        R_orthonorm = U @ Vt
        rvec_avg = R.from_matrix(R_orthonorm).as_rotvec()
        rvec_avg =rvec_avg.tolist()
        rpy = self.rotvec_to_rpy(rvec_avg[0],rvec_avg[1],rvec_avg[2])
        if sorted(liste) == [2, 3]:
            self.connexion()
            current_pose_Q = self.ur5_receive.getActualQ()
            #current_pose_Q[5] = (current_pose_Q[5] + np.pi) % (2 * np.pi) - np.pi
            current_pose_Q[5]+=np.pi
            self.ur5_control.moveJ(current_pose_Q, 0.5,0.5)
            self.pose_history.clear()
            self.deconnexion()
        elif sorted(liste) == [20,21]:
            self.connexion()
            current_pose_Q = self.ur5_receive.getActualQ()
            current_pose_Q[5]+=rpy[2]
            self.ur5_control.moveJ(current_pose_Q, 0.5,0.5)
            self.pose_history.clear()
            self.deconnexion()
        else:
            rpy = self.rotvec_to_rpy(rvec_avg[0], rvec_avg[1], rvec_avg[2])
            self.connexion()
            current_pose_Q = self.ur5_receive.getActualQ()
            current_pose_Q[5]+=rpy[2]
            self.ur5_control.moveJ(current_pose_Q, 0.5,0.5)
            self.pose_history.clear()
            self.deconnexion()
    
    def deplacementPose(self, liste, offset) :
        self.pose_history.clear()
        
        self.update_cam_to_base_transform()
        _, markers_data = self.detect_aruco_markers()
        selected_markers = [m for m in markers_data if m['id'] in liste]
        selected_positions = [marker['position'] for marker in selected_markers]
        print(selected_positions)
        mean_pos = np.mean(np.stack(selected_positions, axis=0),axis=0)
        print(mean_pos)
        target =  mean_pos + offset 
        print(target)
        self.connexion()
        current_pose = self.ur5_receive.getActualTCPPose()
        for i in range(3):
            current_pose[i] = target[i]
        self.ur5_control.moveL(current_pose, 0.2, 0.2)
        self.deconnexion()

    def balisation (self, liste, indice, angle, joint=1) :
        print("on est dans la fonction")
        self.update_cam_to_base_transform()
        marker_ids = []

        while True:
            self.pose_history.clear()
            _, markers_data = self.detect_aruco_markers(strict=True)

            if markers_data:
                aruco_frames = self.compute_weighted_aruco_reference_frames(markers_data)
                marker_ids = list(aruco_frames['all_markers'].keys())
                print(marker_ids)
                #liste_inverse = liste[::-1]
                
            if set(liste).issubset(marker_ids):
                break
            if joint == 1 :
                self.connexion()
                print("joint")
                pose_current = self.ur5_receive.getActualQ()
                pose_current[indice] += angle
                self.ur5_control.moveJ(pose_current, 0.8, 0.8)
                self.pose_history.clear()
                self.deconnexion()
            if joint == 0 :
                self.connexion()
                print("TCP")
                pose_current = self.ur5_receive.getActualTCPPose()
                pose_current[indice] += angle
                self.ur5_control.moveL(pose_current, 0.5, 0.5)
                print(pose_current)
                self.pose_history.clear()
                self.deconnexion()
            self.pose_history.clear()

    def run_demo(self):
        """Démo: détecte les marqueurs et permet de déplacer le robot selon différents axes"""
        print("Démarrage de la démo améliorée...")
        print("Commandes:")
        # Charger les poids des marqueurs s'ils existent
        self.load_marker_weights()
        
        current_axis = 'z'  # Axe par défaut
        current_distance = 0.05  # 5cm par défaut
        
        while True:
            #time.sleep(3)
            # Détecter et afficher les marqueurs
            frame, markers_data = self.detect_aruco_markers()
            
            if frame is not None:
                # Afficher l'axe et la distance actuels
                #cv2.putText(frame, f"Axe: {current_axis}, Distance: {current_distance*100:.1f}cm, Scale: {self.scale_correction:.2f}",
                        #(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Calculer le repère de référence si des marqueurs sont détectés
                if markers_data:
                    
                    aruco_frames = self.compute_weighted_aruco_reference_frames(markers_data)
                    
                    markers_all_data = aruco_frames['all_markers']
                    marker_ids = list(markers_all_data.keys())
                    #print(aruco_frames['rvec_avg'])
                    # Afficher des informations sur le repère détecté
                    if aruco_frames:
                        poses = []
                        for i in range(len(marker_ids)):
                            current_marker = marker_ids[i]
                            pose = markers_all_data[current_marker]['pos_marker']
                            poses.append(pose)
                            rvec_avg = aruco_frames['rvec_avg']
                            rvec_avg =rvec_avg.tolist()
                            rpy = self.rotvec_to_rpy(rvec_avg[0],rvec_avg[1],rvec_avg[2])
                            roll = (rpy[0] + np.pi) % (2 * np.pi) - np.pi
                            pitch = (rpy[1] + np.pi) % (2 * np.pi) - np.pi
                            yaw = (rpy[2] + np.pi) % (2 * np.pi) - np.pi
                            rotvec = self.rpy_to_rotvec (0, np.pi/2, np.pi/2)

                        cv2.putText(frame, f"Position selon caméra : [{poses[0][0]:.3f}, {poses[0][1]:.3f}, {poses[0][2]:.3f}]",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        cv2.putText(frame, f"Position selon base :{poses[0][0]:.3f}, Axe y :{poses[0][1]:.3f}, Axe z :{poses[0][2]:.3f}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        #cv2.putText(frame, f"rotvec RX : {rotvec[0]:.3f}, RY :{rotvec[1]:.3f},  RZ :{rotvec[2]:.3f}",
                                #(10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Afficher le nombre de marqueurs utilisés
                        cv2.putText(frame, f"Marqueurs: {len(markers_data)}", 
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Afficher les IDs des marqueurs
                        marker_ids = [str(m['id']) for m in markers_data]
                        cv2.putText(frame, f"IDs: {', '.join(marker_ids)}",
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Afficher l'image
                cv2.imshow("ArUco Markers", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') :
                # 1) Mettre à jour la calibration caméra→base et redétecter
                self.update_cam_to_base_transform()
                _, markers_data = self.detect_aruco_markers()
                if not markers_data:
                    #print("Aucun marqueur détecté.")
                    continue

                # 2) Repère moyen (position + axes déjà corrigés) du(des) ArUco
                aruco_frame = self.compute_weighted_aruco_reference_frames(markers_data)
                if aruco_frame is None:
                    print("Impossible de calculer le repère ArUco.")
                    continue

                self.connexion()
                pose6d = self.ur5_receive.getActualTCPPose()
                pose_current = self.ur5_receive.getActualQ()
                
                print(f" la valeur de rvec est : {rotvec}")
                
                for i in range(3):
                    pose6d[i+3] = rotvec[i]
                #print(pose6d)
                self.ur5_control.moveL(pose6d,0.2,0.2)

                base_pos = [data['position'] for data in aruco_frame['all_markers'].values()]
                mean_pos = np.mean(np.stack(base_pos, axis=0),axis=0)
                print(mean_pos)

                offset = np.array([0.035, 0.01, 0.01])
                target_pos = mean_pos + offset

                for i in range (3) :
                    pose6d[i] = target_pos[i]
                #print(pose6d)

                #self.ur5_control.moveL(pose6d,0.2,0.2)
            
                time.sleep(4)
                #self.ur5_control.moveJ(pose_current, 0.5,0.5)
                self.deconnexion()
            elif key == ord('e') :
                self.rotation_rvec([0])
                self.rotation_yaw([0])
                self.deplacementPose([0],[0,0,0.01])

            elif key == ord('a') :
                self.connexion()
                current_pose = self.ur5_receive.getActualQ()
                print(current_pose)
                self.deconnexion()
            elif key == ord('o') :
                # 1) Mettre à jour la calibration caméra→base et redétecter
                self.update_cam_to_base_transform()
                _, markers_data = self.detect_aruco_markers()
                if not markers_data:
                    #print("Aucun marqueur détecté.")
                    continue

                # 2) Repère moyen (position + axes déjà corrigés) du(des) ArUco
                aruco_frame = self.compute_weighted_aruco_reference_frames(markers_data)
                if aruco_frame is None:
                    print("Impossible de calculer le repère ArUco.")
                    continue
                self.connexion()
                markers_all_data = aruco_frames['all_markers']
                rvec_avg = aruco_frames['rvec_avg']
                rvec_avg =rvec_avg.tolist()
                rpy = self.rotvec_to_rpy(rvec_avg[0],rvec_avg[1],rvec_avg[2])

                current_pose = self.ur5_receive.getActualQ()
                current_pose[5]=np.pi
                self.ur5_control.moveJ(current_pose, 0.5, 0.5)
                self.deconnexion()
            elif key ==ord('s') :
                self.rotation_rvec([2,3])
                self.rotation_yaw([2,3])

        
        # Nettoyage
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.ur5_control.disconnect()

if __name__ == "__main__":
    aruco =ArucoURController()
    aruco.run_demo()
    #aruco.rotation_rvec([12,9])
    #input("on passe au yaw...")
    #aruco.rotation_yaw([12,9])
    #aruco.rotation_rvec_v2([12,9])
    #aruco.deplacementPose([18,8],[0,0,0])
    
    
   
    #aruco.deplacementPose([0],[0,0,0.005])
    #aruco.balisation([2,3], 0, -0.05,0)