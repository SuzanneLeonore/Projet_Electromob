import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_control
import rtde_receive
import yaml
import time
from collections import deque

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
            with open('/home/robot/Bureau/Stage/electromob/testCamera/intrinsics_v2.yaml', 'r') as f:
                intrinsic_data = yaml.safe_load(f)
                self.camera_matrix = np.array(intrinsic_data['camera_matrix'], dtype=np.float32)
                self.dist_coeffs = np.array(intrinsic_data['dist_coeffs'], dtype=np.float32)
            print("Paramètres intrinsèques chargés avec succès")
            
            # Chargement de la transformation hand-eye
            with open('/home/robot/Bureau/Stage/electromob/testCamera/hand_eye.yaml', 'r') as f:
                hand_eye_data = yaml.safe_load(f)
                self.T_cam_to_TCP = np.array(hand_eye_data['T_cam_to_flange'], dtype=np.float32)
                self.T_cam_to_TCP =self.rotate_z_90(self.T_cam_to_TCP)
                self.T_cam_to_TCP =self.rotate_z_90(self.T_cam_to_TCP)
            print("Transformation hand-eye chargée avec succès")
        except Exception as e:
            print(f"Erreur lors du chargement des calibrations: {e}")
            raise
        
        # 3. Configuration du détecteur ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Ajustement des paramètres du détecteur pour améliorer la robustesse
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size = 0.02  # Taille du marqueur en mètres (ajuster selon vos marqueurs)
        
        # 4. Connexion au robot UR5
        try:
            self.ur5_control = rtde_control.RTDEControlInterface(robot_ip)
            self.ur5_receive = rtde_receive.RTDEReceiveInterface(robot_ip)
            print("Robot UR5 connecté avec succès")
            
            # 5. Calcul de la transformation caméra -> base du robot
            self.update_cam_to_base_transform()
        except Exception as e:
            print(f"Erreur lors de la connexion au robot: {e}")
            raise
            
        # 6. Initialisation des buffers pour le filtrage temporel
        self.pose_history = {}  # Dictionnaire qui contiendra des deques pour chaque marqueur
        self.history_length = 5  # Nombre de poses à conserver pour chaque marqueur
        self.last_frame_markers = None  # Pour stocker les derniers marqueurs détectés
        
        # 7. Facteurs de confiance pour les différents marqueurs (ajustable)
        self.marker_weights = {}  # À remplir dynamiquement ou à partir d'un fichier
        
        # 8. Paramètres pour la validation du mouvement
        self.last_successful_movement = None  # Pour la continuité des mouvements
        self.max_allowed_deviation = 0.05  # 5cm de déviation maximale permise
        
        # 9. Paramètre pour la correction d'échelle (pour résoudre le problème de distances)
        self.scale_correction = 1.0  # Facteur initial, peut être ajusté

    def rotate_z_90(self, T):
        R_flip = np.array([
            [0,  1,  0],
            [ -1, 0,  0],
            [ 0,  0,  1]
        ])
        T[:3, :3] = R_flip @ T[:3, :3]    # Appliquer la rotation à la matrice de rotation
        return T
    
    def rotate_x_90(self,T):
        R_flip = np.array([
            [1, 0,  0],
            [0, 0, -1],
            [0, 1,  0]
        ])
        T[:3, :3] = T[:3, :3] @ R_flip  # Appliquer la rotation à la matrice de rotation
        return T

    def rotate_y_90(self,T):
        R_flip = np.array([
            [1,  0,  0],
            [ 0, 0,  -1],
            [ 0,  1,  0]
        ])
        T[:3, :3] = T[:3, :3] @ R_flip  # Appliquer la rotation à la matrice de rotation
        return T
    
    def update_cam_to_base_transform(self):
        """Calcule la transformation caméra -> base du robot en utilisant hand-eye"""
        # Récupérer la pose actuelle du TCP
        tcp_pose = self.ur5_receive.getActualTCPPose()
        
        # Construire la matrice base -> TCP
        T_tcp_to_base = np.eye(4)
        T_tcp_to_base[:3, 3] = tcp_pose[:3]  # Position
        rx, ry, rz = tcp_pose[3:6]
        # Convertir rx, ry, rz en matrice de rotation
        rot_vector = np.array([rx, ry, rz])
        rot_matrix, _ = cv2.Rodrigues(rot_vector)
        T_tcp_to_base[:3, :3] = rot_matrix
        # Calculer base -> caméra en utilisant base -> TCP et TCP -> caméra
        self.T_cam_to_base = T_tcp_to_base @ self.T_cam_to_TCP
    
        # Calculer caméra -> base (inverse)
        #self.T_cam_to_base = np.linalg.inv(T_base_cam)
    
    def detect_aruco_markers(self):
        """Détecte les marqueurs ArUco et calcule leurs poses avec une meilleure précision"""
        # Capture d'une image
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        angle =[]
        position = {}
        
        if not color_frame or not depth_frame:
            print("Images non disponibles")
            return None, None, None, None
        
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
            print("Aucun marqueur détecté")
            # Utiliser la dernière détection réussie si disponible
            if self.last_frame_markers is not None:
                print("Utilisation des marqueurs de la dernière trame")
                return color_image, self.last_frame_markers,position,angle
            return color_image, None, None, None
        
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
            
            R_aruco_cam, _ = cv2.Rodrigues(rvec)
            T_aruco_cam = np.eye(4)
            T_aruco_cam[:3, :3] = R_aruco_cam
            T_aruco_cam[:3, 3] = tvec.flatten()

            # Transformation dans le repère base
            T_aruco_base = self.T_cam_to_base @ T_aruco_cam

            pos_base = T_aruco_base[:3, 3]
            R_base = T_aruco_base[:3, :3]
            rot_vec, _ = cv2.Rodrigues(R_base)

            pose_target = np.hstack([pos_base, rot_vec.flatten()])
            print(pose_target)
            try:
                self.ur5_control.moveL(pose_target, speed=0.1, acceleration=0.1)
                print("[INFO] Alignement robot ↔ ArUco réussi.")
            except Exception as e:
                print(f"[ERREUR] Échec du mouvement : {e}")
            # Convertir le vecteur de rotation en matrice
            rot_matrix, _ = cv2.Rodrigues(rvec)
            
            # Position du marqueur dans le repère caméra
            pos= tvec.flatten()
            position[marker_id]=pos

            
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
                depth_solvepnp = pos[2]
                depth_diff = abs(depth_sensor - depth_solvepnp)
                
                # Si la différence est trop grande, ajuster la pose
                if depth_diff > 0.05:  # 5cm de différence
                    # Facteur de confiance entre les deux mesures (privilégier la profondeur du capteur)
                    alpha = 0.7
                    corrected_depth = alpha * depth_sensor + (1 - alpha) * depth_solvepnp
                    # Mettre à jour la position Z
                    pos[2] = corrected_depth
                    
                    # Recalculer tvec pour le dessin des axes
                    tvec[2] = corrected_depth
            
            # Orientation (axes X, Y, Z) dans le repère caméra
            x_axis = rot_matrix[:, 0]
            y_axis = rot_matrix[:, 1]
            z_axis = rot_matrix[:, 2]
            x_axis_camera = np.array([1.0, 0.0, 0.0])  # axe X caméra dans son propre repère

            # Normalisation (au cas où)
            x_axis = x_axis / np.linalg.norm(x_axis)
            x_axis_camera = x_axis_camera / np.linalg.norm(x_axis_camera)

            # Calcul de l’angle en radians et degrés
            cos_theta = np.clip(np.dot(x_axis, x_axis_camera), -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)
            angle.append(angle_rad)
            angle_deg = np.degrees(angle_rad)

            # Transformer la position du repère caméra au repère base du robot
            pos_hom = np.ones(4)
            pos_hom[:3] = pos

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
        
        # Mémoriser les marqueurs détectés pour la prochaine image
        self.last_frame_markers = markers_data
        
        return color_image, markers_data, position,angle
    
    def compute_weighted_aruco_reference_frame(self, markers_data):
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
        
        for marker in markers_data:
            marker_id = marker['id']
            
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
        
        # Matrice de rotation pour corriger les axes des marqueurs
        # (Utilise une rotation de 180° autour de Z pour aligner avec le repère du robot)
        rot_z_180 = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        
        rotated_x = rot_z_180 @ mean_x
        rotated_y = rot_z_180 @ mean_y
        
        return {
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
            }
        }
    
    def move_along_axis(self, axis='y', distance=0.05, speed=0.1):
        """
        Déplace le robot le long d'un axe des marqueurs ArUco avec vérification et correction
        """
        # 1. Mettre à jour la transformation caméra-robot
        self.update_cam_to_base_transform()
        
        # 2. Détecter les marqueurs ArUco
        _, markers_data,_,_ = self.detect_aruco_markers()
        
        if not markers_data:
            print("Aucun marqueur détecté. Impossible de se déplacer.")
            return False
        
        # 3. Calculer le repère de référence des marqueurs avec pondération
        aruco_frame = self.compute_weighted_aruco_reference_frame(markers_data)
        
        if not aruco_frame:
            print("Impossible de calculer le repère ArUco. Arrêt.")
            return False
        
        # 4. Obtenir la pose actuelle du robot
        current_pose = self.ur5_receive.getActualTCPPose()
        
        # 5. Sélectionner l'axe de déplacement
        if axis.lower() == 'x':
            direction_vector = aruco_frame['axes']['x']
            print("Déplacement selon l'axe X des marqueurs")
        elif axis.lower() == 'y':
            direction_vector = aruco_frame['axes']['y']
            print("Déplacement selon l'axe Y des marqueurs")
        elif axis.lower() == 'z':
            direction_vector = aruco_frame['axes']['z']
            print("Déplacement selon l'axe Z des marqueurs")
        else:
            print(f"Axe '{axis}' non reconnu. Utilisation de l'axe Y par défaut")
            direction_vector = aruco_frame['axes']['y']
        
        # 6. Appliquer le facteur de correction d'échelle
        corrected_distance = distance * self.scale_correction
        
        # 7. Calculer la nouvelle position
        displacement = corrected_distance * direction_vector
        new_pose = current_pose.copy()
        new_pose[0] += displacement[0]
        new_pose[1] += displacement[1]
        new_pose[2] += displacement[2]
        
        # 8. Vérifier que le déplacement est cohérent avec les mouvements précédents
        if self.last_successful_movement is not None:
            # Calculer l'angle entre le déplacement précédent et le nouveau
            previous_direction = self.last_successful_movement
            cos_angle = np.dot(direction_vector, previous_direction) / (
                np.linalg.norm(direction_vector) * np.linalg.norm(previous_direction))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            
            # Si l'angle est trop grand (direction très différente), demander confirmation
            if angle > 45:  # Plus de 45 degrés de différence
                print(f"ATTENTION: Changement important de direction détecté ({angle:.1f}°).")
                print(f"Déplacement précédent: [{previous_direction[0]:.4f}, {previous_direction[1]:.4f}, {previous_direction[2]:.4f}]")
                print(f"Déplacement actuel: [{direction_vector[0]:.4f}, {direction_vector[1]:.4f}, {direction_vector[2]:.4f}]")
                confirm = input("Voulez-vous continuer? (o/n): ")
                if confirm.lower() != 'o':
                    print("Déplacement annulé par l'utilisateur.")
                    return False
        
        # 9. Exécuter le mouvement
        try:
            print(f"Déplacement de {distance}m (corrigé: {corrected_distance:.4f}m) selon l'axe {axis}")
            print(f"Vecteur de déplacement: [{displacement[0]:.4f}, {displacement[1]:.4f}, {displacement[2]:.4f}]")
            
            # Afficher les axes pour débogage
            print("Référentiel des marqueurs:")
            print(f"  X: [{aruco_frame['axes']['x'][0]:.4f}, {aruco_frame['axes']['x'][1]:.4f}, {aruco_frame['axes']['x'][2]:.4f}]")
            print(f"  Y: [{aruco_frame['axes']['y'][0]:.4f}, {aruco_frame['axes']['y'][1]:.4f}, {aruco_frame['axes']['y'][2]:.4f}]")
            print(f"  Z: [{aruco_frame['axes']['z'][0]:.4f}, {aruco_frame['axes']['z'][1]:.4f}, {aruco_frame['axes']['z'][2]:.4f}]")
            
            # Exécuter le mouvement
            self.ur5_control.moveL(new_pose, speed, 0.1)
            
            # Enregistrer ce mouvement comme réussi
            self.last_successful_movement = direction_vector
            
            print("Déplacement terminé avec succès")
            return True
        except Exception as e:
            print(f"Erreur lors du déplacement du robot: {e}")
            return False
    
    def calibrate_scale_correction(self):
        """
        Utilise une procédure de calibration pour corriger le facteur d'échelle
        """
        print("\n=== CALIBRATION DU FACTEUR D'ÉCHELLE ===")
        print("Cette procédure va déterminer le facteur de correction pour les distances.")
        print("Placez le robot à une position de départ, puis:")
        print("1. Le robot sera déplacé d'une distance connue")
        print("2. Mesurez la distance réelle parcourue")
        print("3. Le facteur de correction sera automatiquement calculé\n")
        
        input("Appuyez sur Entrée quand vous êtes prêt...")
        
        # Distance de test
        test_distance = 0.1  # 10cm
        axis = 'y'
        
        # Position initiale
        start_pose = self.ur5_receive.getActualTCPPose()
        
        # Déplacer le robot avec le facteur actuel
        print(f"Déplacement de {test_distance}m selon l'axe {axis}...")
        self.move_along_axis(axis, test_distance)
        
        # Position finale
        end_pose = self.ur5_receive.getActualTCPPose()
        
        # Calculer la distance parcourue dans l'espace 3D
        travelled_distance = np.linalg.norm(np.array(end_pose[:3]) - np.array(start_pose[:3]))
        
        # Demander la distance réellement parcourue
        print(f"Distance mesurée par le système: {travelled_distance:.4f}m")
        real_distance = float(input("Entrez la distance réellement parcourue (en mètres): "))
        
        # Calculer le facteur de correction
        if travelled_distance > 0:
            self.scale_correction = real_distance / test_distance
            print(f"Nouveau facteur de correction d'échelle: {self.scale_correction:.4f}")
        else:
            print("Erreur: distance parcourue nulle, impossible de calculer le facteur.")
        
        # Retour à la position initiale
        print("Retour à la position initiale...")
        self.ur5_control.moveL(start_pose, 0.1, 0.1)
        
        return self.scale_correction

    def adjust_marker_weights(self):
        """
        Permet d'ajuster les poids des différents marqueurs ArUco
        """
        print("\n=== AJUSTEMENT DES POIDS DES MARQUEURS ===")
        print("Cette procédure permet de définir la fiabilité relative de chaque marqueur.")
        print("Un poids plus élevé donne plus d'importance à un marqueur dans le calcul du repère.")
        
        # Détecter les marqueurs disponibles
        _, markers_data,_,_ = self.detect_aruco_markers()
        
        if not markers_data:
            print("Aucun marqueur détecté. Impossible d'ajuster les poids.")
            return
        
        # Afficher les marqueurs détectés et leurs poids actuels
        print("\nMarqueurs détectés:")
        for marker in markers_data:
            marker_id = marker['id']
            current_weight = self.marker_weights.get(marker_id, 1.0)
            print(f"ID {marker_id}: poids actuel = {current_weight}")
        
        # Demander les nouveaux poids
        print("\nEntrez les nouveaux poids (1.0 = poids normal, 0.5 = moins fiable, 2.0 = plus fiable)")
        for marker in markers_data:
            marker_id = marker['id']
            try:
                new_weight = float(input(f"Nouveau poids pour ID {marker_id} (Entrée pour conserver {self.marker_weights.get(marker_id, 1.0)}): ") or self.marker_weights.get(marker_id, 1.0))
                self.marker_weights[marker_id] = new_weight
            except ValueError:
                print("Valeur invalide, poids inchangé.")
        
        print("\nNouveaux poids des marqueurs:")
        for marker_id, weight in self.marker_weights.items():
            print(f"ID {marker_id}: {weight}")
        
        # Sauvegarder les poids dans un fichier
        try:
            with open('marker_weights.yaml', 'w') as f:
                yaml.dump({'marker_weights': self.marker_weights}, f)
            print("Poids sauvegardés dans marker_weights.yaml")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des poids: {e}")
    
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
        
    def move_to_aruco_center(self, markers_data, position, marker_cible, speed=0.1, accel=0.1):

        self.update_cam_to_base_transform()
        
        aruco_frame = self.compute_weighted_aruco_reference_frame(markers_data)

        if aruco_frame is None:
            print("[INFO] Aucun repère ArUco détecté")
            return
        pos =[]

        for e in marker_cible :
            pos.append(position[e])

        s=0       
        for i in range(len(marker_cible)) :
            s+=position[i]
        s=s/len(marker_cible)
        s_list = s.tolist()
        s_list.append(1)
        pose = self.T_cam_to_base @ s_list #position du marqueur cible dans le repère robot
        target = pose.tolist()
        p = self.ur5_receive.getActualTCPPose()
        for i in range (3):
            p[i]=target[i]
        p[2]+= 0.02
        target = np.array(p)
        try:
            self.ur5_control.moveL(target, speed = 0.2, accel=0.2)
        except Exception as e:
            print(f"[ERREUR] Déplacement échoué : {e}")
    
    def move_to_qrcode_center(self, markers_length=0.05, speed=0.1, accel=0.1):

        # 1. Capture une image depuis la caméra
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        marker_length = float(0.02)
        if not color_frame:
            print("[ERREUR] Aucun flux vidéo.")
            return

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        #print("Camera matrix:\n", self.camera_matrix)
        #print("Distortion coefficients:\n", self.dist_coeffs)


        # 2. Détecte les ArUco
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None or len(ids) == 0:
            print("[INFO] Aucun marqueur détecté.")
            return
        cv2.imwrite("debug_frame.png", color_image)

        # 3. Estime la pose du 1er ArUco
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            marker_length,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        # Pose du centre du QR code 0 (le premier détecté)
        rvec = rvecs[0][0]   # shape (3,)
        tvec = tvecs[0][0]   # shape (3,)

        # 4. Mise à jour de la transformation caméra → base robot
        self.update_cam_to_base_transform()

        # 5. Convertit la position du centre ArUco en repère base robot
        pos_cam = np.ones(4)
        pos_cam = np.array([tvec[0], tvec[1], tvec[2], 1.0])
        pos_base = self.T_cam_to_base @ pos_cam

        corrected_target_pos = pos_base[:3]  # [X, Y, Z] dans base robot

        # 6. Récupère l'orientation actuelle du robot (RX, RY, RZ)
        # Pose cible en matrice homogène (T_target)
        R, _ = cv2.Rodrigues(rvec)
        T_target = np.eye(4)
        T_target[:3, :3] = R
        T_target[:3, 3] = corrected_target_pos # en base robot

        # Convertir la matrice homogène en pose [x, y, z, rx, ry, rz]
        rot_vec, _ = cv2.Rodrigues(T_target[:3, :3])
        pose_target = np.hstack([T_target[:3, 3], rot_vec.flatten()])
        #pose_target[2]+=0.01
        pose_actuelle = self.ur5_receive.getActualTCPPose()
        # Résoudre la cinématique inverse
        self.ur5_control.moveL(pose_target, speed=0.25, acceleration=0.5)
        time.sleep(2)
        self.ur5_control.moveL(pose_actuelle, speed = 0.25, acceleration = 0.5)

        # Envoyer la commande
        #if joint_positions is not None:
            #self.ur5_control.moveJ(joint_positions, speed, accel)
        #else:
            #print("⚠️ Aucune solution IK trouvée pour cette orientation.")


        # 7. Déplace le robot
        #self.ur5_control.moveL(target_pose, speed, accel)

    def align_robot_to_aruco(self, rvec, tvec):
        """
        Déplace le robot pour que son outil se place à la position du ArUco et adopte son orientation.
        """

        self.update_cam_to_base_transform()

        # Construire transformation du ArUco dans caméra
        R_aruco_cam, _ = cv2.Rodrigues(rvec)
        T_aruco_cam = np.eye(4)
        T_aruco_cam[:3, :3] = R_aruco_cam
        T_aruco_cam[:3, 3] = tvec.flatten()

        # Transformation dans le repère base
        T_aruco_base = self.T_cam_to_base @ T_aruco_cam

        pos_base = T_aruco_base[:3, 3]
        R_base = T_aruco_base[:3, :3]
        rot_vec, _ = cv2.Rodrigues(R_base)

        pose_target = np.hstack([pos_base, rot_vec.flatten()])

        try:
            self.ur5_control.moveL(pose_target, speed=0.1, acceleration=0.1)
            print("[INFO] Alignement robot ↔ ArUco réussi.")
        except Exception as e:
            print(f"[ERREUR] Échec du mouvement : {e}")


    def run_demo(self):
        """Démo: détecte les marqueurs et permet de déplacer le robot selon différents axes"""
        print("Démarrage de la démo améliorée...")
        print("Commandes:")
        print("  'q' - Quitter")
        print("  'x' - Déplacer selon l'axe X des marqueurs")
        print("  'y' - Déplacer selon l'axe Y des marqueurs")
        print("  'z' - Déplacer selon l'axe Z des marqueurs")
        print("  '+' - Augmenter la distance (actuellement 5cm)")
        print("  '-' - Diminuer la distance")
        print("  'm' - Exécuter le mouvement")
        print("  'c' - Calibrer le facteur d'échelle")
        print("  'w' - Ajuster les poids des marqueurs")
        print("  'r' - Mettre à jour la transformation caméra-robot")
        
        # Charger les poids des marqueurs s'ils existent
        self.load_marker_weights()
        
        current_axis = 'y'  # Axe par défaut
        current_distance = 0.05  # 5cm par défaut
        
        while True:
            # Détecter et afficher les marqueurs
            frame, markers_data, position, angle_rad= self.detect_aruco_markers()
            
            if frame is not None:
                # Afficher l'axe et la distance actuels
                cv2.putText(frame, f"Axe: {current_axis}, Distance: {current_distance*100:.1f}cm, Scale: {self.scale_correction:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Calculer le repère de référence si des marqueurs sont détectés
                if markers_data:
                    aruco_frame = self.compute_weighted_aruco_reference_frame(markers_data)
                    
                    # Afficher des informations sur le repère détecté
                    if aruco_frame:
                        pos = aruco_frame['position']
                        cv2.putText(frame, f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Afficher le nombre de marqueurs utilisés
                        cv2.putText(frame, f"Marqueurs: {len(markers_data)}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Afficher les IDs des marqueurs
                        marker_ids = [str(m['id']) for m in markers_data]
                        cv2.putText(frame, f"IDs: {', '.join(marker_ids)}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                       
                        if all(str(i) in marker_ids for i in [6, 7, 8]):
                            R = np.column_stack([
                                aruco_frame['axes']['x'],
                                aruco_frame['axes']['y'],
                                aruco_frame['axes']['z']
                            ])
                            rvec, _ = cv2.Rodrigues(R)
                            p = [0, 0, 0, rvec[0][0], rvec[1][0], rvec[2][0]]
                            for i in range (3) :
                                p[i]=pos[i]
                            target = np.array(p)
                            self.ur5_control.moveL(target, 0.05, 0.1)
                        if all(str(i) in marker_ids for i in [9, 10]):
                            R = np.column_stack([
                                aruco_frame['axes']['x'],
                                aruco_frame['axes']['y'],
                                aruco_frame['axes']['z']
                            ])
                            rvec, _ = cv2.Rodrigues(R)
                            p = [0, 0, 0, rvec[0][0], rvec[1][0], rvec[2][0]]
                            for i in range (3) :
                                p[i]=pos[i]
                            target = np.array(p)
                            self.ur5_control.moveL(target, 0.05, 0.1)

                # Afficher l'image
                cv2.imshow("ArUco Markers", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key == ord('u') :
                pose = position.tolist()
                pose.append(1)
                target =self.T_cam_to_base @ pose
                target = target.tolist()
                target.pop()
                p = self.ur5_receive.getActualTCPPose()
                print(pose)
                print(angle_rad)
                for i in range (3):
                    p[i]=target[i]
                print(p)
                p[2]+=0.01
                target = np.array(p)
                self.ur5_control.moveL(target, 0.05, 0.1)
                p1= self.ur5_receive.getActualQ()
                time.sleep(2)
                p1[5]+=angle_rad[0]
                self.ur5_control.moveJ(p1, 0.1, 0.1)
            elif key == ord ('t') :
                p= self.ur5_receive.getActualTCPPose()
                print(pos)
                print(angle_rad)
                for i in range (3):
                    p[i]=pos[i]
                print(p)
                p[2]+=0.01
                target = np.array(p)
                self.ur5_control.moveL(target, 0.05, 0.1)
                p1= self.ur5_receive.getActualQ()
                time.sleep(2)
                p1[5]+=angle_rad[0]
                self.ur5_control.moveJ(p1, 0.1, 0.1)
            elif key == ord('p') :
                self.move_to_qrcode_center()
            elif key == ord('x'):
                current_axis = 'x'
                print(f"Axe sélectionné: {current_axis}")
            elif key == ord('y'):
                current_axis = 'y'
                print(f"Axe sélectionné: {current_axis}")
            elif key == ord('z'):
                current_axis = 'z'
                print(f"Axe sélectionné: {current_axis}")
            elif key == ord('+'):
                current_distance += 0.01  # +1cm
                print(f"Distance: {current_distance*100:.1f}cm")
            elif key == ord('-'):
                current_distance = max(0.01, current_distance - 0.01)  # minimum 1cm
                print(f"Distance: {current_distance*100:.1f}cm")
            elif key == ord('m'):
                # Déplacer le robot selon l'axe et la distance actuels
                self.move_along_axis(current_axis, current_distance)
            elif key == ord('c'):
                # Calibrer le facteur d'échelle
                self.calibrate_scale_correction()
            elif key == ord('w'):
                # Ajuster les poids des marqueurs
                self.adjust_marker_weights()
            elif key == ord('r'):
                # Mettre à jour la transformation caméra-robot
                self.update_cam_to_base_transform()
        
        # Nettoyage
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.ur5_control.disconnect()

if __name__ == "__main__":
    controller = ArucoURController()
    controller.run_demo()
