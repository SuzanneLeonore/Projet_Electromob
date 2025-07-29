import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
import time
import rtde_control
import rtde_receive
from collections import deque
from scipy.spatial.transform import Rotation as R
from pince import Pince
import threading
from robot import Robot

class ArucoURController:
    def __init__(self, robot_ip="10.2.30.60", smoothing=5, offset_x=0.011):
        # Init RealSense
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        # Load intrinsics
        with open('detectionPlus/intrinsics.yaml') as f:
            data = yaml.safe_load(f)
        self.camera_matrix = np.array(data['camera_matrix'], np.float32)
        self.dist_coeffs = np.array(data['dist_coeffs'], np.float32)

        # ArUco parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshConstant = 7
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        self.marker_size = 0.04  # m

        # smoothing history for projected center
        self.center_history = deque(maxlen=smoothing)
        self.last_avg_t = None

        # Offset in base X (positive)
        self.offset_x=0.0

        self.rot = 0

        # Compute static transform camera -> base
        self.connexion()
        self.update_cam_to_base_transform()

        self.pose_history = {}

        self.history_length = 5  # Nombre de poses à conserver pour chaque marqueur
        self.last_frame_markers = None  # Pour stocker les derniers marqueurs détectés

        # historique pour le centre projeté
        self.center_history = deque(maxlen=smoothing)
        # offset positif en X (m)
        self.offset_x = offset_x
        
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
        """
        Met à jour la transformation de la caméra vers la base du robot.

        Cette méthode calcule la matrice de transformation homogène entre la caméra
        et la base du robot, en combinant la position actuelle du TCP (Tool Center Point)
        avec la transformation prédéterminée entre la caméra et la bride de l’outil,
        lue depuis un fichier YAML.

        Étapes :
            1. Récupère la pose actuelle du TCP du robot (position + orientation en rotation de Rodrigues).
            2. Construit une matrice de transformation homogène du TCP vers la base.
            3. Lit la matrice de transformation de la caméra à la bride depuis un fichier YAML.
            4. Calcule la transformation finale de la caméra à la base du robot.

        Attributs modifiés :
            self.T_cam_to_base (np.ndarray): Matrice 4x4 représentant la transformation de la caméra à la base.

        Fichier requis :
            detectionPlus/hand_eye.yaml : Doit contenir une clé 'T_cam_to_flange' correspondant à une matrice 4x4.

        Exceptions :
            Lève une exception si le fichier YAML est manquant ou mal formé.
        """
        tcp = self.ur5_receive.getActualTCPPose()
        T_tcp_base = np.eye(4)
        T_tcp_base[:3,3] = tcp[:3]
        R_tcp, _ = cv2.Rodrigues(np.array(tcp[3:6]))
        T_tcp_base[:3,:3] = R_tcp
        with open('detectionPlus/hand_eye.yaml') as f:
            H = np.array(yaml.safe_load(f)['T_cam_to_flange'], np.float32)
        self.T_cam_to_base = T_tcp_base @ H

    def detect_aruco_markers(self, strict = False):
        """
        Détecte les marqueurs ArUco dans une image capturée par la caméra RealSense.

        Deux passes de détection sont effectuées :
        - Une première pour repérer rapidement les marqueurs et calculer un centre projeté moyen.
        - Une seconde pour une estimation précise de la pose de chaque marqueur avec solvePnP,
            affinée par CLAHE et stabilisée via un historique temporel.

        Si aucun marqueur n'est détecté, le comportement dépend du paramètre `strict` :
        - Si `strict=True` : retourne None pour les poses.
        - Si `strict=False` : retourne la dernière détection connue (si disponible).

        Returns:
            und (np.ndarray): Image couleur (BGR) non distordue avec les marqueurs, axes et cercles dessinés.
            proj_center (tuple[int, int] or None): Centre projeté moyen des marqueurs (x, y en pixels) ou None si non détecté.
            markers_data (list[dict] or None): Liste de dictionnaires contenant les informations de pose pour chaque marqueur :
                - 'id' (int): Identifiant du marqueur ArUco.
                - 'position' (np.ndarray): Position 3D du marqueur dans le repère de la base robot.
                - 'pos_marker' (list[float]): Position du marqueur dans le repère caméra.
                - 'rvec' (np.ndarray): Vecteur de rotation (Rodrigues).
                - 'rpy' (tuple[float, float, float]): Orientation (roll, pitch, yaw) en radians.
                - 'axes' (dict[str, np.ndarray]): Axes du repère du marqueur projetés en base robot ('x', 'y', 'z').
                - 'timestamp' (float): Horodatage de la détection (temps système).

        Args:
            strict (bool, optional): Si True, ne retourne aucun marqueur si la détection échoue.
                                        Si False, conserve les données de la dernière détection valide.
                                        Défaut : False.

        Modifie :
            self.last_avg_t (np.ndarray): Dernière position moyenne estimée (repère caméra).
            self.center_history (deque): Historique des centres projetés pour lissage.
            self.pose_history (dict): Historique des poses par ID pour filtrage temporel.
            self.last_frame_markers (list): Derniers marqueurs détectés.

        Dépendances :
            - Caméra RealSense initialisée avec `self.pipeline` et `self.align`.
            - Matrice de calibration `self.camera_matrix` et distorsion `self.dist_coeffs`.
            - Transformée `self.T_cam_to_base` pour conversion en base robot.
            - Taille des marqueurs définie par `self.marker_size`.

        Exceptions:
            Ne lève pas d’exception explicitement, mais retourne des None en cas d'échec d'acquisition ou de détection.
        """
        # 1) Acquisition RealSense
        frames       = self.pipeline.wait_for_frames()
        aligned      = self.align.process(frames)
        color_frame  = aligned.get_color_frame()
        depth_frame  = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            print("Images non disponibles")
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 2) Undistort + gris
        und  = cv2.undistort(color_image, self.camera_matrix, self.dist_coeffs)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        # --- Partie 1 : simple averaging pour proj_center ---
        corners, ids, _ = self.detector.detectMarkers(gray)
        proj_center = None
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(und, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            valid_t = []
            for rvec, tvec, c in zip(rvecs, tvecs, corners):
                if not (0.02 < np.linalg.norm(tvec) < 1.0):
                    continue
                cv2.drawFrameAxes(und, self.camera_matrix, self.dist_coeffs,
                                rvec, tvec, self.marker_size/2)
                cx = int(c[0][:,0].mean()); cy = int(c[0][:,1].mean())
                cv2.circle(und, (cx, cy), 4, (0,255,0), -1)
                valid_t.append(tvec.flatten())
            if valid_t:
                avg_t = np.mean(valid_t, axis=0).reshape(1,3)
                self.last_avg_t = avg_t.copy()
                proj, _ = cv2.projectPoints(
                    avg_t, np.zeros((3,1)), np.zeros((3,1)),
                    self.camera_matrix, self.dist_coeffs
                )
                px, py = int(proj[0,0,0]), int(proj[0,0,1])
                self.center_history.append((px,py))
                xs, ys = zip(*self.center_history)
                proj_center = (int(np.mean(xs)), int(np.mean(ys)))
                cv2.circle(und, proj_center, 7, (255,0,0), -1)

        # --- Partie 2 : pose fine + axes + filtrage historique ---
        clahe         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        corners2, ids2, _ = self.detector.detectMarkers(enhanced_gray)

        if ids2 is None or len(ids2) == 0:
            markers_data = None if strict or not hasattr(self, 'last_frame_markers') else self.last_frame_markers
        else:
            cv2.aruco.drawDetectedMarkers(und, corners2, ids2)
            markers_data = []
            for j, corner in enumerate(corners2):
                mid = int(ids2[j][0])
                # 3D corners du carré (en m)
                mp = np.array([
                    [-self.marker_size/2,  self.marker_size/2, 0],
                    [ self.marker_size/2,  self.marker_size/2, 0],
                    [ self.marker_size/2, -self.marker_size/2, 0],
                    [-self.marker_size/2, -self.marker_size/2, 0]
                ], dtype=np.float32)

                # solvePnP IPPE_SQUARE + refine
                _, rvec, tvec = cv2.solvePnP(mp, corner[0],
                                            self.camera_matrix, self.dist_coeffs,
                                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
                _, rvec, tvec = cv2.solvePnP(mp, corner[0],
                                            self.camera_matrix, self.dist_coeffs,
                                            rvec=rvec, tvec=tvec,
                                            useExtrinsicGuess=True,
                                            flags=cv2.SOLVEPNP_ITERATIVE)
                cv2.drawFrameAxes(und, self.camera_matrix, self.dist_coeffs,
                                rvec, tvec, self.marker_size/2)

                # Matrice homogène marqueur→caméra
                rot_cam = np.eye(4)
                R, _    = cv2.Rodrigues(rvec)
                rot_cam[:3,:3] = R
                pos = tvec.flatten().tolist()
                for k in range(3):
                    rot_cam[k,3] = pos[k]

                # Correction Z via capteur profondeur
                cx = int(np.mean(corner[0][:,0])); cy = int(np.mean(corner[0][:,1]))
                region = depth_image[
                    max(0,cy-5):min(depth_image.shape[0], cy+5),
                    max(0,cx-5):min(depth_image.shape[1], cx+5)
                ]
                vd = region[region>0]
                if len(vd) > 0:
                    depth_sensor = np.median(vd) / 1000.0
                    if abs(depth_sensor - pos[2]) > 0.05:
                        pos[2] = 0.7*depth_sensor + 0.3*pos[2]
                        tvec[2] = pos[2]

                # Passage en base robot
                p_h = np.array([pos[0],pos[1],pos[2],1.0])
                pos_base = (self.T_cam_to_base @ p_h)[:3]
                Rcb = self.T_cam_to_base[:3,:3]
                axes_base = {
                    'x': (Rcb @ R[:,0]),
                    'y': (Rcb @ R[:,1]),
                    'z': (Rcb @ R[:,2])
                }
                rpy = self.rotvec_to_rpy(float(rvec[0]), float(rvec[1]), float(rvec[2]))

                # Historique & filtrage
                data = {
                    'id': mid, 'position': pos_base, 'pos_marker': pos,
                    'rvec': rvec, 'rpy': rpy, 'axes': axes_base,
                    'timestamp': time.time()
                }
                if mid not in self.pose_history:
                    self.pose_history[mid] = deque(maxlen=self.history_length)
                self.pose_history[mid].append(data)
                if len(self.pose_history[mid]) > 1:
                    positions = [m['position'] for m in self.pose_history[mid] if 'position' in m]
                    if positions:
                        data['position'] = np.mean(positions, axis=0)

                    for ax in ['x','y','z']:
                        axes_vals = [m['axes'][ax] for m in self.pose_history[mid] if 'axes' in m and ax in m['axes']]
                        if axes_vals:
                            v = np.mean(axes_vals, axis=0)
                            data['axes'][ax] = v / np.linalg.norm(v)

                markers_data.append(data)
            self.last_frame_markers = markers_data

        return und, proj_center, markers_data

    def compute_weighted_aruco_reference_frames(self, markers_data):
        """
        Calcule un repère de référence global à partir de plusieurs marqueurs ArUco détectés,
        en moyennant leur position et leurs axes, puis en orthonormalisant les vecteurs.

        Ce repère peut servir de système de coordonnées stable pour une tâche de robotique.
        Une rotation de 180° autour de l'axe Z est appliquée pour aligner le repère avec
        le système robot (option typique en cas de caméra montée tête en bas).

        Args:
            markers_data (list[dict]): Liste de dictionnaires contenant les données de pose
                pour chaque marqueur détecté. Chaque dictionnaire doit inclure :
                - 'id' (int): Identifiant du marqueur.
                - 'position' (np.ndarray): Position 3D du marqueur dans la base robot.
                - 'pos_marker' (list[float]): Position du marqueur dans le repère caméra.
                - 'axes' (dict): Repère local du marqueur (x, y, z) dans la base robot.
                - 'rvec' (np.ndarray): Vecteur de rotation (Rodrigues).
                - 'rpy' (tuple): Angles roll-pitch-yaw.

        Returns:
            dict or None: Dictionnaire représentant le repère de référence global, ou None
            si aucun marqueur valide n’est fourni. Le dictionnaire contient :
                - 'id' (int): ID du premier marqueur utilisé comme référence.
                - 'pos_marker' (list[float]): Position d'origine du marqueur de référence.
                - 'position' (np.ndarray): Position moyenne 3D (dans la base robot).
                - 'axes' (dict): Repère orthonormé transformé ('x', 'y', 'z').
                - 'original_axes' (dict): Repère avant transformation (non corrigé).
                - 'rvec_avg' (np.ndarray): Moyenne des rotations (Rodrigues).
                - 'all_markers' (dict): Données brutes de chaque marqueur, indexées par ID.

        Notes:
            - Tous les marqueurs sont traités avec un poids égal (pondération possible à ajouter).
            - Le repère est redressé via une SVD moyenne des matrices de rotation.
            - Le vecteur Z est pris tel quel, Y est redressé pour être perpendiculaire à Z,
            et X est obtenu par produit vectoriel.
            - L'orientation est corrigée pour aligner avec un repère robotique standard.

        See Also:
            `scipy.spatial.transform.Rotation` pour les conversions entre rvec et matrice de rotation.
        """
        if not markers_data or len(markers_data) == 0:
            return None
            
        # Tenir compte de la fiabilité relative des marqueurs (poids)
        # Par défaut, tous les marqueurs ont un poids égal

        
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

    def rpy_to_rotvec(self, roll, pitch, yaw):
        """
        Convertit des angles RPY (roll, pitch, yaw) en vecteur de rotation (Rodrigues).

        Args:
            roll (float): Angle de roulis (rotation autour de l'axe X), en radians.
            pitch (float): Angle de tangage (rotation autour de Y), en radians.
            yaw (float): Angle de lacet (rotation autour de Z), en radians.

        Returns:
            np.ndarray: Vecteur de rotation (shape: (3,)) représentant l'orientation.
        """
        rot = R.from_euler('xyz', [roll, pitch, yaw])
        rotvec = rot.as_rotvec()
        return rotvec

    def rotvec_to_rpy(self, rx, ry, rz):
        """
        Convertit un vecteur de rotation (Rodrigues) en angles RPY (roll, pitch, yaw).

        Args:
            rx (float): Composante X du vecteur de rotation.
            ry (float): Composante Y du vecteur de rotation.
            rz (float): Composante Z du vecteur de rotation.

        Returns:
            tuple[float, float, float]: Angles roll, pitch, yaw (en radians).
        """
        rot = R.from_rotvec([rx, ry, rz])
        rpy = rot.as_euler('xyz')  # en radians
        return rpy
        
    def rotation_rvec(self, liste) :
        """
        Calcule et applique une orientation moyenne à l'outil du robot à partir d'un
        sous-ensemble de marqueurs ArUco spécifiés.

        Cette fonction :
        1. Met à jour la transformation caméra → base robot.
        2. Détecte les marqueurs ArUco visibles.
        3. Filtre ceux dont l'ID est dans `liste`.
        4. Moyenne leurs vecteurs de rotation (Rodrigues) via SVD.
        5. Convertit le résultat en RPY puis génère un vecteur de rotation ajusté.
        6. Applique une pose TCP avec cette orientation au robot UR5 en utilisant un mouvement linéaire.

        Un comportement particulier est défini si `liste == [24, 25]` ou `[25, 24]` :
        - Une orientation prédéfinie (π/2 autour de Y et Z) est utilisée à la place de celle estimée.
        - Le robot est déplacé avec cette orientation, et l'historique de pose est réinitialisé.

        Args:
            liste (list[int]): Liste d'identifiants de marqueurs ArUco à utiliser pour le calcul
                de l'orientation moyenne. Typiquement [24, 25] ou autres.

        Effects:
            - Le robot est connecté, déplacé vers une nouvelle orientation TCP, puis déconnecté.
            - L'historique `self.pose_history` est vidé après chaque mouvement.
            - Un mouvement linéaire (`moveL`) est effectué via `self.ur5_control`.

        Raises:
            Aucun sauf si les appels aux sous-systèmes échouent silencieusement (non gérés ici).

        Dependencies:
            - `detect_aruco_markers()` pour obtenir les poses des marqueurs.
            - `rotvec_to_rpy()` et `rpy_to_rotvec()` pour les conversions de rotation.
            - `ur5_receive` et `ur5_control` pour obtenir la pose TCP et commander le robot.
            - `connexion()` et `deconnexion()` pour gérer la session robot.
        """
        self.update_cam_to_base_transform()
        _, _, markers_data = self.detect_aruco_markers()
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
        if liste == [24,25] or liste == [25,24]:
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
        """
        Calcule une orientation moyenne à partir des marqueurs ArUco spécifiés, puis
        oriente le robot UR5 en conséquence en utilisant des mouvements linéaires (moveL)
        et articulaires (moveJ).

        Le comportement dépend de la composition de `liste` :
        - Si `liste == [24, 25]` (ou l’inverse), une rotation fixe de (0, π/2, π/2) est utilisée.
        - Si `liste == [0]`, la rotation est fixée à (π, 0, 0) en RPY avec un ajustement final en q[5].
        - Sinon, même rotation de base (π, 0, 0) mais ajustement conditionnel selon q[5].

        Args:
            liste (list[int]): Liste des IDs des marqueurs ArUco à prendre en compte pour
                calculer la rotation moyenne.

        Returns:
            None

        Effets de bord :
            - Connecte et déconnecte le robot UR5.
            - Déplace le robot selon l’orientation calculée ou imposée.
            - Vide l'historique de poses (`self.pose_history`) après mouvement.
            - Fait appel à `detect_aruco_markers()` et `update_cam_to_base_transform()`.

        Notes :
            - La rotation moyenne est calculée via SVD sur les matrices de rotation issues
            des vecteurs de rotation (Rodrigues) des marqueurs.
            - L’orientation est injectée dans les derniers 3 éléments de la pose TCP.
            - `current_pose_Q[5]` (axe 6) est ajusté pour corriger l'orientation finale du robot.
        """
        self.update_cam_to_base_transform()
        _, _, markers_data = self.detect_aruco_markers()
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
        if liste == [24,25] or liste == [25,24]:
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
            '''
            if current_pose_Q[5] < 0 :
                current_pose_Q[5] +=np.pi
                current_pose_Q[5] +=rpy[2]
            else :
                current_pose_Q[5] -=np.pi
                current_pose_Q[5] -=rpy[2]
            '''
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
        """
        Calcule l'angle de lacet (yaw) moyen à partir des marqueurs ArUco spécifiés, puis
        applique une rotation autour de l'axe Z du robot en modifiant la 6e articulation.

        Cette méthode ne modifie que la composante yaw (rotation autour de Z) en ajustant
        l’angle `q[5]` de la configuration articulaire du robot (UR5).

        Args:
            liste (list[int]): Liste d’identifiants des marqueurs ArUco à considérer
                pour le calcul de la rotation moyenne.

        Returns:
            None

        Effets de bord :
            - Connecte au robot UR5.
            - Modifie l’angle `q[5]` en l’incrémentant avec la composante yaw estimée.
            - Envoie une commande de mouvement articulaire (moveJ).
            - Vide l’historique `self.pose_history`.
            - Déconnecte du robot.

        Notes :
            - Le yaw est extrait à partir de la moyenne des matrices de rotation calculées
            sur les vecteurs de rotation (Rodrigues) des marqueurs sélectionnés.
            - Le système de coordonnées utilisé suppose un repère aligné avec le robot.
            - Aucun filtrage de validité des marqueurs n’est effectué ici.
        """
        self.update_cam_to_base_transform()
        _, _, markers_data = self.detect_aruco_markers()
        selected_markers = [m for m in markers_data if m['id'] in liste]

        rot_matrices = [R.from_rotvec(marker['rvec'].flatten()).as_matrix() for marker in selected_markers]

        R_avg = sum(rot_matrices) / len(rot_matrices)
        U, _, Vt = np.linalg.svd(R_avg)
        R_orthonorm = U @ Vt
        rvec_avg = R.from_matrix(R_orthonorm).as_rotvec()
        rvec_avg =rvec_avg.tolist()
        rpy = self.rotvec_to_rpy(rvec_avg[0],rvec_avg[1],rvec_avg[2])
        rpy = self.rotvec_to_rpy(rvec_avg[0], rvec_avg[1], rvec_avg[2])
        self.connexion()
        current_pose_Q = self.ur5_receive.getActualQ()
        current_pose_Q[5]+=rpy[2]
        self.ur5_control.moveJ(current_pose_Q, 0.5,0.5)
        self.pose_history.clear()
        self.deconnexion()

    def deplacementPose(self, liste, offset) :
        """
        Déplace le robot vers une position cible estimée à partir de marqueurs ArUco,
        avec un décalage appliqué (offset) à la position moyenne.

        Cette méthode :
        - Calcule la position moyenne des marqueurs spécifiés.
        - Applique un offset (translation).
        - Déplace le TCP du robot vers cette nouvelle position par interpolation linéaire.

        Args:
            liste (list[int]): Liste des IDs de marqueurs ArUco à considérer.
            offset (np.ndarray): Vecteur de décalage (shape: (3,)) à appliquer à la position cible.

        Returns:
            None

        Effets de bord :
            - Connecte et déconnecte le robot.
            - Envoie une commande `moveL` vers une nouvelle position TCP.
            - Vide l’historique de poses `self.pose_history`.

        Notes :
            - Utilise `compute_weighted_aruco_reference_frames` pour estimer les poses.
            - La rotation TCP reste inchangée.
        """
        self.pose_history.clear()
        
        self.update_cam_to_base_transform()
        _, _, markers_data = self.detect_aruco_markers()
        selected_markers = [m for m in markers_data if m['id'] in liste]
        if markers_data :
            aruco_frames = self.compute_weighted_aruco_reference_frames(markers_data)
    
        base_pos = [data['position'] for data in aruco_frames['all_markers'].values()]
        print(base_pos)
        mean_pos = np.mean(np.stack(base_pos, axis=0),axis=0)
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
        """
        Effectue un mouvement par étapes jusqu'à ce que tous les marqueurs de la liste soient détectés.

        Cette méthode :
        - Déplace le robot jusqu’à détecter tous les marqueurs ArUco spécifiés.
        - Répète un mouvement incrémental (soit articulaire, soit TCP) à chaque itération.
        - Utilise la détection stricte de `detect_aruco_markers(strict=True)`.

        Args:
            liste (list[int]): Liste des IDs de marqueurs ArUco à détecter.
            indice (int): Index de la coordonnée à modifier (0-5 pour TCP, 0-5 pour Q).
            angle (float): Incrément de déplacement (radian ou mètre selon le mode).
            joint (int, optional): 
                Mode de déplacement : 
                - 1 = mouvement articulaire (moveJ).
                - 0 = mouvement linéaire du TCP (moveL).
                Par défaut 1.

        Returns:
            None

        Effets de bord :
            - Effectue plusieurs connexions/déconnexions robot.
            - Envoie des commandes de mouvement à chaque itération.
            - Vide `self.pose_history` à chaque boucle.

        Notes :
            - La boucle est bloquante jusqu'à détection complète des marqueurs.
            - Aucun timeout ou sécurité n’est implémenté.
        """
        self.update_cam_to_base_transform()
        marker_ids = []

        while True:
            self.pose_history.clear()
            _, _, markers_data = self.detect_aruco_markers(strict=True)

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

    def doing_moveL(self, target, speed, acceleration):
        """
        Lance un déplacement linéaire (`moveL`) du robot vers la cible spécifiée
        dans un thread séparé, puis attend la fin de ce déplacement.

        Args:
            target (list or np.ndarray): Pose cible TCP (6 éléments : position + orientation).
            speed (float): Vitesse de déplacement.
            acceleration (float): Accélération du mouvement.

        Returns:
            None

        Note:
            - La fonction utilise un thread pour lancer la commande, mais attend
            immédiatement sa fin avec `join()`, ce qui bloque l'exécution jusqu'à
            la fin du mouvement.
        """
        move_thread = threading.Thread(
            target=self.ur5_control.moveL,
            args=(target, speed, acceleration)
        )
        move_thread.start()
        move_thread.join()   # <-- on attend ici


    def move_to_center(self, use_marker_axis_offset: bool = True, offset = None):
        print("on est là")
        def center_thread(use_marker_axis_offset = use_marker_axis_offset, offset = offset):
            print("on est là 2")
            """
            Déplace le robot vers le centre projeté, puis applique un offset :
            - soit selon l'axe X du robot (ancienne version),
            - soit selon l'axe X positif du premier marqueur détecté.
            """
            # Vérification
            if self.last_avg_t is None \
            or not hasattr(self, 'last_frame_markers') \
            or len(self.last_frame_markers) == 0:
                print("No valid 3D center to move to.")
                return

            # 1) Calcul de la position cible en repère base
            p_cam    = np.hstack([self.last_avg_t.flatten(), 1.0])
            p_base_h = self.T_cam_to_base @ p_cam        # homogène (4,)
            pos_base = p_base_h[:3].copy()               # (3,)

            # 2) Offset le long de l'axe du marqueur
            if use_marker_axis_offset:
                md      = self.last_frame_markers[0]      # premier marqueur
                axis_x  = np.array(md['axes']['x'])       # vecteur unitaire (3,)
                pos_base += axis_x * self.offset_x
            else:
                pos_base[0] += self.offset_x

            ####################
            #   OFFSET
            ###################

            if offset is None :
                offset = [0,0,0]
                current_offset = offset

            else :
                current_offset = offset

            for i in range(3):
                pos_base[i] += current_offset[i]

            # 3) Construction de la pose finale TCP (on garde l'orientation courante)
            tcp_pose = self.ur5_receive.getActualTCPPose()  # [x,y,z,rx,ry,rz]
            new_pose = [float(pos_base[i]) for i in range(3)] + tcp_pose[3:6]

            # 4) Envoi de la commande
            print(new_pose)
            self.ur5_control.moveL(new_pose, 0.5, 0.2)

        move_thread = threading.Thread(
            target=center_thread,
            args=(use_marker_axis_offset, offset)
        )
        move_thread.start()
        move_thread.join()

    def sequence_mouvement(self):

        ### Initialisation de la pince

        pince = Pince()
        robot = Robot()

        ##########################
        # Définitions des offsets
        ##########################

        offset_cellule           = [0.005,0.005,-0.005]
        offset_cadre             = [0.09, -0.005, 0.32]
        offset_injecteur         = [0.0015,0,0.2]
        offset_bac               = [0.006,0.0823,0.15]
        offset_inter_slots       = [0,0.0465,0]

        ##########################
        # Mouvements à exécuter
        ##########################

        ### 1) Se placer à la position initiale

        pince.lacher()
        self.deconnexion()

        self.connexion()
        self.ur5_control.moveJ([-0.842104736958639, -1.4243934790240687, 2.540102958679199, -4.270684067402975, -1.5935080687152308, -1.564348045979635],0.5, 0.5)

        ### 2) Se mettre devant le convoyeur

        self.update_cam_to_base_transform()
        self.balisation([24,25],0,-0.0872665)

        self.update_cam_to_base_transform()
        self.rotation([24,25])
        self.rotation_yaw([24,25])
        self.deconnexion()

        ### 3) Prendre la cellule
        self.connexion()
        self.update_cam_to_base_transform()
        self.move_to_center(use_marker_axis_offset=True,offset=offset_cellule)
        self.deconnexion()

        self.update_cam_to_base_transform()

        pince.prise()
        self.deconnexion()

        ### 4) Aller au l'ArUco 0

        robot.bougerL(robot.move_actual_pose(1, -0.2))
        
        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, 0.2))

        self.update_cam_to_base_transform()
        robot.bougerJ(robot.joints[3])

        self.update_cam_to_base_transform()
        self.balisation([0], 1, -0.05,0)

        self.update_cam_to_base_transform()
        self.deplacementPose([0],[0.0,0.0,0.4])

        ### 5) Amener la Tuile au cadre
        self.update_cam_to_base_transform()
        self.balisation([20,21],0, 0.174533 )

        self.update_cam_to_base_transform()
        self.deplacementPose([20,21],[0.0,0.0,0.4])

        self.update_cam_to_base_transform()
        self.rotation_yaw([20,21])

        self.update_cam_to_base_transform()
        self.rotation([20,21])
        
        self.update_cam_to_base_transform()
        self.rotation_yaw([20,21])
        
        self.update_cam_to_base_transform()
        self.deplacementPose([20,21],[0.16,0,0.4])

        self.update_cam_to_base_transform()
        self.deplacementPose([20,21],[0.16,0,0.092])

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, 0.3))
        self.deconnexion()
        
        ### 6) placer la tuile dans le cadre
        self.connexion()
        self.update_cam_to_base_transform()
        self.move_to_center(use_marker_axis_offset=True,offset=offset_cadre)
        self.deconnexion()

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, -0.02))

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, -0.045))

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, -0.05))

        pince.lacher()
        self.deconnexion()

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, 0.065))
        self.deconnexion()

        ### 7) Prendre l'injecteur 
        self.connexion()
        self.update_cam_to_base_transform()
        self.move_to_center(use_marker_axis_offset=True,offset=offset_injecteur)
        self.deconnexion()

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, -0.12))
        self.deconnexion()

        self.connexion()
        pince.prise()
        self.deconnexion()

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, 0.275))
        self.deconnexion()

        ### 8) Injecter la Cellule
        
        self.connexion()
        self.update_cam_to_base_transform()
        self.move_to_center(use_marker_axis_offset=True,offset=offset_cadre)
        self.deconnexion()

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, -0.1))
        robot.bougerL(robot.move_actual_pose(2, 0.1))
        self.deconnexion()
        
        self.connexion()
        self.update_cam_to_base_transform()
        self.move_to_center(use_marker_axis_offset=True,offset=offset_cadre)
        self.deconnexion()

        ### 9) Ranger l'injecteur

        self.connexion()
        self.update_cam_to_base_transform()
        self.move_to_center(use_marker_axis_offset=True,offset=offset_injecteur)
        self.deconnexion()

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, -0.12))
        self.deconnexion()

        self.connexion()
        pince.lacher()
        self.deconnexion()

        ### 10) Reprendre la cellule

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, 0.275))
        self.deconnexion()
        
        self.connexion()
        self.update_cam_to_base_transform()
        self.move_to_center(use_marker_axis_offset=True,offset=offset_cadre)
        self.deconnexion()

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, -0.2))
        self.deconnexion()

        self.connexion()
        pince.prise()
        self.deconnexion()

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, 0.2))
        self.deconnexion()

        ### 12) Amener la cellule au bac de rangement

        self.balisation([23,22],0, -0.174533)

        ### 13) Se mettre face au bac de rangement

        self.update_cam_to_base_transform()
        self.deplacementPose([23,22],[0.05,-0.04,0.3])

        self.update_cam_to_base_transform()
        self.rotation_yaw([23,22])

        self.update_cam_to_base_transform()
        self.rotation([23,22])

        self.update_cam_to_base_transform()
        self.rotation_yaw([23,22])

        self.update_cam_to_base_transform()
        self.rotation_yaw([23,22])

        self.update_cam_to_base_transform()
        self.balisation([0], 1, -0.05,0)

        self.update_cam_to_base_transform()
        self.deplacementPose([0],[0.0,0.0,0.111])
        self.update_cam_to_base_transform()
        self.deplacementPose([0],[0.0,0.2,0.3])

        ### 14) Ranger la cellule dans l'emplacement choisi

        offset_bac = np.array([0.008, 0.0826, 0.15])
        offset_inter_slots = np.array([0, 0.0465, 0])
        while True:
            try:
                x = int(input("Dans quel emplacement ranger le bac ? [0,1,2,3,4] "))
                if 0 <= x <= 4:
                    break
                else:
                    print("Veuillez entrer un nombre entre 0 et 4.")
            except ValueError:
                print("Entrée invalide. Veuillez entrer un nombre entier.")
        print(f"La tuile va être rangée dans l'emplacement n°{x}")

        new_offset = offset_bac - offset_inter_slots*x

        self.connexion()
        self.update_cam_to_base_transform()
        self.move_to_center(use_marker_axis_offset=True,offset=new_offset)
        self.deconnexion()

        input("on descend")
        
        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, -0.06))
        self.deconnexion()

        pince.lacher()
        self.deconnexion()

        self.update_cam_to_base_transform()
        robot.bougerL(robot.move_actual_pose(2, 0.2))
        ### 15) Reviens à la position initiale
        self.connexion()
        self.ur5_control.moveJ([-0.842104736958639, -1.4243934790240687, 2.540102958679199, -4.270684067402975, -1.5935080687152308, -1.564348045979635],0.5, 0.5)


    def run(self):
        print("Interactive ArUco UR5. Press 'i' to move with offset.")
        while True:
            frame, center, markers_data = self.detect_aruco_markers()
            if frame is not None:
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

                        cv2.putText(frame, f"Position selon caméra : [{poses[0][0]:.3f}, {poses[0][1]:.3f}, {poses[0][2]:.3f}]",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        cv2.putText(frame, f"Position selon base :{poses[0][0]:.3f}, Axe y :{poses[0][1]:.3f}, Axe z :{poses[0][2]:.3f}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
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

            elif key == ord('i'):
                # Ne pas bloquer la boucle vidéo : on démarre juste le thread
                if not hasattr(self, '_thread_i') or not self._thread_i.is_alive():
                    self._thread_i = threading.Thread(target=self.sequence_mouvement,
                                                    daemon=True)
                    self._thread_i.start()

            if center:
                cv2.putText(frame, f"Center: {center}", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.ur5_control.disconnect()

if __name__ == '__main__':
    aruco = ArucoURController()
    aruco.run()
    aruco.rotation([24,25])